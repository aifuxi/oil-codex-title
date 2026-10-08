"""逐次模型用量账本；不保存提示词、回答、标题或错误原文。"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid
import statistics

_scope = ContextVar("oil_usage_scope", default=None)
_retry_reason = ContextVar("oil_retry_reason", default=None)
TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
              "output_tokens", "reasoning_output_tokens")


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".usage-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Scope:
    def __init__(self, root, purpose, thread_id, started_at=None, initial_timings=None):
        self.root, self.purpose, self.thread_id = Path(root), purpose, thread_id
        self.attempts = []
        self.outcome = "not_completed"
        self.start = time.perf_counter() if started_at is None else started_at
        self.timings = dict(initial_timings or {})
        self.model_deadline = None

    def metrics(self):
        return {"elapsed_seconds": round(time.perf_counter() - self.start, 3),
                "timings": {k: round(v, 3) for k, v in self.timings.items()},
                "model_calls": len(self.attempts),
                "retry_calls": sum(bool(a.data.get("retry_reason")) for a in self.attempts)}

    def begin(self, config):
        attempt = Attempt(self, config)
        self.attempts.append(attempt)
        return attempt


class Attempt:
    def __init__(self, scope, config):
        now = datetime.now(timezone.utc)
        self.path = scope.root / "usage" / now.strftime("%Y-%m-%d") / (uuid.uuid4().hex + ".json")
        self.start = time.perf_counter()
        self.data = {"version": 1, "started_at": now.isoformat(), "purpose": scope.purpose,
                     "thread_id": scope.thread_id, "model": config["model"],
                     "reasoning_effort": config.get("reasoning_effort", "low"),
                     "retry_reason": _retry_reason.get(),
                     "service_tier": config.get("service_tier"), "status": "started",
                     "outcome": "not_completed", "usage": None}
        # 先成功落盘才启动进程；中断记录保留为用量未知，不伪装成零。
        _save(self.path, self.data)

    def finish(self, status, usage):
        values = {k: v for k, v in (usage or {}).items()
                  if k in TOKEN_KEYS and type(v) is int and v >= 0}
        self.data.update(status=status, usage=values or None,
                         elapsed_seconds=round(time.perf_counter() - self.start, 3))
        _save(self.path, self.data)


@contextmanager
def usage_scope(root, purpose, thread_id=None, *, started_at=None, initial_timings=None):
    scope = Scope(root, purpose, thread_id, started_at, initial_timings)
    token = _scope.set(scope)
    try:
        yield scope
    except BaseException as exc:
        scope.outcome = getattr(exc, "status", "error")
        raise
    finally:
        _scope.reset(token)
        metrics = scope.metrics()
        for attempt in scope.attempts:
            attempt.data["outcome"] = scope.outcome
            _save(attempt.path, attempt.data)
        # 独立保存零调用、失败和过期路径的耗时，不保存标题或对话。
        now = datetime.now(timezone.utc)
        _save(scope.root / "runs" / now.strftime("%Y-%m-%d") / (uuid.uuid4().hex + ".json"),
              {"version": 1, "purpose": purpose, "status": scope.outcome, **metrics})


def record_duration(name, start):
    scope = _scope.get()
    if scope:
        scope.timings[name] = scope.timings.get(name, 0) + time.perf_counter() - start


@contextmanager
def measure(name):
    start = time.perf_counter()
    try:
        yield
    finally:
        record_duration(name, start)


def timed_call(name, function, *args, **kwargs):
    with measure(name):
        return function(*args, **kwargs)


@contextmanager
def retrying(reason):
    token = _retry_reason.set(reason)
    try:
        yield
    finally:
        _retry_reason.reset(token)


def model_phase():
    return "retry_model" if _retry_reason.get() else "model"


def shared_deadline(seconds):
    """一个命名流程的排队、状态复核与所有重试共用预算。"""
    deadline = time.monotonic() + seconds
    scope = _scope.get()
    if scope and scope.purpose == "naming":
        if scope.model_deadline is None:
            scope.model_deadline = deadline
        return min(deadline, scope.model_deadline)
    return deadline


def percentiles(values):
    """P95 使用最近秩法，小样本也不会插值到不存在的耗时。"""
    values = sorted(values)
    if not values:
        return {"p50_seconds": None, "p95_seconds": None}
    return {"p50_seconds": round(statistics.median(values), 3),
            "p95_seconds": round(values[math.ceil(len(values) * .95) - 1], 3)}


def begin_attempt(config):
    scope = _scope.get()
    return scope.begin(config) if scope else None


def parse_usage(stdout):
    usage = {}
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    forbidden_tool = False
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            for key, value in event["usage"].items():
                if key in TOKEN_KEYS and type(value) is int and value >= 0:
                    usage[key] = usage.get(key, 0) + value
        item = event.get("item") or {}
        if isinstance(item, dict) and item.get("type") in {"command_execution", "mcp_tool_call", "web_search"}:
            forbidden_tool = True
    return usage, forbidden_tool


def usage_report(root):
    groups, unreadable = {}, 0
    for path in sorted((Path(root) / "usage").glob("*/*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            key = (row["purpose"], row["model"], row.get("reasoning_effort", "low"), row.get("service_tier"))
            usage = row.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError("用量记录无效")
        except (OSError, ValueError, KeyError, TypeError):
            unreadable += 1
            continue
        group = groups.setdefault(key, {"purpose": key[0], "model": key[1], "attempts": 0,
            "reasoning_effort": key[2], "service_tier": key[3],
            "unknown_usage": 0, "tokens": {k: 0 for k in TOKEN_KEYS}, "outcomes": {}})
        group["attempts"] += 1
        group["unknown_usage"] += not all(k in usage for k in ("input_tokens", "output_tokens"))
        for k in TOKEN_KEYS:
            if type(usage.get(k)) is int and usage[k] >= 0:
                group["tokens"][k] += usage[k]
        outcome = row.get("outcome", "not_completed")
        group["outcomes"][outcome] = group["outcomes"].get(outcome, 0) + 1
    skipped = 0
    for path in (Path(root) / "logs").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                skipped += isinstance(row, dict) and row.get("skip_reason") == "confirmation_only"
            except ValueError:
                continue
    performance = {}
    for path in (Path(root) / "runs").glob("*/*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if not all(isinstance(row[k], str) for k in ("purpose", "status")):
                raise ValueError("状态记录无效")
            if not all(type(row[k]) is int and row[k] >= 0 for k in ("model_calls", "retry_calls")):
                raise ValueError("次数记录无效")
            timings = row["timings"]
            if not isinstance(timings, dict):
                raise ValueError("耗时记录无效")
            values = [row["elapsed_seconds"], *timings.values()]
            if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values):
                raise ValueError("耗时记录无效")
        except (OSError, ValueError, KeyError, TypeError):
            unreadable += 1
            continue
        group = performance.setdefault(row["purpose"], {"runs": 0, "model_calls": 0,
            "retry_calls": 0, "statuses": {}, "elapsed": [], "phases": {}})
        group["runs"] += 1
        for key in ("model_calls", "retry_calls"):
            group[key] += row[key]
        status = row["status"]
        group["statuses"][status] = group["statuses"].get(status, 0) + 1
        group["elapsed"].append(row["elapsed_seconds"])
        for name, seconds in timings.items():
            group["phases"].setdefault(name, []).append(seconds)
    for group in performance.values():
        group.update(percentiles(group.pop("elapsed")))
        group["phases"] = {name: percentiles(values) for name, values in group["phases"].items()}
    return {"groups": list(groups.values()), "performance": performance, "unreadable_records": unreadable,
            "confirmation_skips_in_retained_logs": skipped,
            "notice": "仅统计启用账本后的独立模型调用；不含宿主管理模型、历史日志或未记账的独立评测。缓存输入包含在输入中，推理输出包含在输出中；未知用量不按零计算。"}

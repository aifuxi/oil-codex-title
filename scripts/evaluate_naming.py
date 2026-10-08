#!/usr/bin/env python3
"""用合成案例评测命名；显式 --live 才会消耗账号额度。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import re
import tempfile
import time

from codex_adapter import CodexBackend, find_codex, generate_title, validate_model_config
from oil_codex_title import DEFAULTS, POLICY_VERSION, REASONING_EFFORTS, validate_candidate
from usage_ledger import TOKEN_KEYS, percentiles, usage_scope

ROOT = Path(__file__).resolve().parents[1]


def protocol_metadata(paths):
    return {'policy_version': POLICY_VERSION,
            'prompt_sha256': hashlib.sha256((ROOT / 'prompts/naming.md').read_bytes()).hexdigest(),
            'case_files': [{'source': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                            'cases': len(json.loads(p.read_text(encoding='utf-8')))} for p in paths],
            'acceptance_policy': {'minimum_repeats': 3, 'workers': 1,
                                  'all_candidate_cases_pass': True, 'minimum_median_improvement_percent': 20,
                                  'p95_must_not_increase': True}}


def check_candidate(candidate, case):
    candidate = validate_candidate(candidate, case['context']['current_title'])
    expected, errors = case['expected'], []
    if candidate['action'] != expected['action']:
        errors.append('动作不符')
    for word in expected['contains']:
        if word.casefold() not in candidate['title'].casefold():
            errors.append('缺少对象：' + word)
    for word in expected['reject']:
        if word.casefold() in candidate['title'].casefold():
            errors.append('错误主线：' + word)
    if expected.get('title_pattern') and not re.search(expected['title_pattern'], candidate['title']):
        errors.append('标题语言或文字范围不符')
    if candidate['title'] in case['context'].get('conflicting_titles', []):
        errors.append('未区分冲突')
    if expected.get('exact_title') and candidate['title'] != expected['exact_title']:
        errors.append('稳定标题或迁移主线变化')
    if expected.get('emoji') and not candidate['title'].startswith(expected['emoji'] + ' '):
        errors.append('产物类别不符')
    for word in expected.get('object_contains', []):
        if word.casefold() not in candidate['title'].split('｜')[0].casefold():
            errors.append('对象未前置：' + word)
    return candidate, errors


def run_case(binary, config, case, source, repeat, root):
    start = time.perf_counter()
    candidate, errors, usage = None, [], {}
    with usage_scope(root, 'naming') as accounting:
        try:
            candidate, usage = generate_title(binary, config, case['context'], ROOT)
            candidate, errors = check_candidate(candidate, case)
        except Exception as exc:
            errors = [str(exc)]
        accounting.outcome = 'failed' if errors else 'passed'
    # 只发布合成输出与逐次数字；临时账本中的时间、路径和标识不进入报告。
    attempts = [{k: a.data.get(k) for k in ('status', 'retry_reason', 'elapsed_seconds', 'usage')}
                for a in accounting.attempts]
    return {'case': case['id'], 'source': source, 'repeat': repeat, 'model': config['model'],
            'reasoning_effort': config['reasoning_effort'], 'service_tier': config['service_tier'],
            'candidate': candidate, 'passed': not errors, 'errors': errors, 'usage': usage,
            'seconds': round(time.perf_counter() - start, 3),
            'model_seconds': round(sum(accounting.timings.get(k, 0) for k in ('model', 'retry_model')), 3),
            'model_calls': len(attempts), 'retry_calls': sum(bool(a['retry_reason']) for a in attempts),
            'attempts': attempts}


def summarize(rows, config):
    attempts = [a for r in rows for a in r['attempts']]
    model_rows = [r for r in rows if r['model_calls']]
    return {'model': config['model'], 'reasoning_effort': config['reasoning_effort'],
            'service_tier': config['service_tier'], 'cases': len(rows),
            'passed': sum(r['passed'] for r in rows),
            'pass_rate': round(sum(r['passed'] for r in rows) / len(rows), 4) if rows else None,
            'model_cases': len(model_rows), 'model_calls': len(attempts),
            'retry_calls': sum(r['retry_calls'] for r in rows),
            'unknown_usage': sum(not all(k in (a['usage'] or {}) for k in ('input_tokens', 'output_tokens')) for a in attempts),
            'tokens': {k: sum((a['usage'] or {}).get(k, 0) for a in attempts) for k in TOKEN_KEYS},
            'model_latency': percentiles([r['model_seconds'] for r in model_rows]),
            'attempt_latency': percentiles([a['elapsed_seconds'] for a in attempts if a['elapsed_seconds'] is not None]),
            'case_latency': percentiles([r['seconds'] for r in rows])}


def acceptance(baseline, candidate, *, complete=True, comparable=True, repeats=3):
    old, new = baseline['model_latency'], candidate['model_latency']
    measured = bool(old['p50_seconds'] and new['p50_seconds'])
    improvement = 1 - new['p50_seconds'] / old['p50_seconds'] if measured else None
    checks = {'complete': complete, 'serial_comparison': comparable,
              'at_least_three_repeats': repeats >= 3,
              'critical_cases_passed': candidate['cases'] > 0 and candidate['passed'] == candidate['cases'],
              'quality_not_lower': bool(candidate['pass_rate'] is not None and baseline['pass_rate'] is not None
                                        and candidate['pass_rate'] >= baseline['pass_rate']),
              'median_at_least_20_percent_faster': measured and new['p50_seconds'] <= old['p50_seconds'] * .80,
              'p95_not_higher': measured and new['p95_seconds'] <= old['p95_seconds']}
    return {'baseline': baseline['model'], 'candidate': candidate['model'],
            'accepted': all(checks.values()), 'checks': checks,
            'median_improvement_percent': round(improvement * 100, 2) if measured else None}


def ordered_jobs(cases, configs, repeats):
    # 每个案例相邻比较两个模型；每轮反转模型次序，减少服务负载和缓存次序偏差。
    for repeat in range(1, repeats + 1):
        order = configs if repeat % 2 else list(reversed(configs))
        for source, case in cases:
            for config in order:
                yield config, source, case, repeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='允许实际调用账号模型')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/naming-evaluation.json')
    parser.add_argument('--cases', type=Path, action='append', help='合成案例文件，可重复指定')
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--model', action='append', help='可重复指定；首个模型为对比基线')
    parser.add_argument('--reasoning-effort', choices=REASONING_EFFORTS, default='low')
    parser.add_argument('--service-tier', choices=('standard', 'fast'), default='fast')
    parser.add_argument('--repeat', type=int, default=1, choices=range(1, 11))
    args = parser.parse_args()
    paths = args.cases or [ROOT / 'tests/fixtures/naming_cases.json']
    cases = [(p.name, case) for p in paths
             for case in json.loads(p.read_text(encoding='utf-8'))]
    if not cases:
        parser.error('合成案例不能为空')
    if not args.live:
        print(f'共 {len(cases)} 个合成案例；加 --live 才调用模型。')
        return 0
    models = args.model or [DEFAULTS['model']]
    if len(set(models)) != len(models):
        parser.error('对比模型不能重复')
    configs = [{**DEFAULTS, 'model': model, 'reasoning_effort': args.reasoning_effort,
                'service_tier': 'priority' if args.service_tier == 'fast' else None} for model in models]
    binary = find_codex()
    protocol = protocol_metadata(paths)
    with CodexBackend(binary) as backend:
        for config in configs:
            validate_model_config(backend, config)
    jobs, rows = list(ordered_jobs(cases, configs, args.repeat)), []

    def save_report():
        summaries = [summarize([r for r in rows if r['model'] == c['model']], c) for c in configs]
        complete = len(rows) == len(jobs)
        report = {'complete': complete, 'workers': args.workers, 'repeat': args.repeat,
                  'protocol': protocol,
                  'planned_cases': len(jobs), 'cases': len(rows), 'passed': sum(r['passed'] for r in rows),
                  'summaries': summaries,
                  'comparisons': [acceptance(summaries[0], s, complete=complete, comparable=args.workers == 1, repeats=args.repeat)
                                  for s in summaries[1:]],
                  'notice': '仅合成案例；model_calls 为 CLI 启动尝试，重试单独记账。模型链路耗时含所有重试，零调用不进入模型延迟。未知用量不按零计费；不验证真实 Hook 或桌面显示。',
                  'results': rows}
        # 逐项保存，长评测中断后仍保留已完成证据；临时文件与报告位于同一目录。
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent, delete=False) as stream:
            tmp = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        tmp.replace(args.output)
        return report

    with tempfile.TemporaryDirectory(prefix='oil-naming-eval-') as tmp:
        def run(index_job):
            index, (config, source, case, repeat) = index_job
            return run_case(binary, config, case, source, repeat, Path(tmp) / str(index))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(run, enumerate(jobs)):
                rows.append(row)
                save_report()
                print(json.dumps({k: row[k] for k in ('model', 'case', 'repeat', 'passed', 'seconds', 'model_calls')}, ensure_ascii=False), flush=True)
    report = save_report()
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, ensure_ascii=False))
    return 0 if report['passed'] == len(rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""串行评测合成归档分类；显式 --live 才调用模型，始终不执行真实归档。"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from archive_policy import CONTEXT_VERSION, SCHEMA, classify
from codex_adapter import CodexBackend, find_codex, validate_model_config
from evaluate_naming import ordered_jobs, summarize
from oil_codex_title import DEFAULTS, REASONING_EFFORTS, ROOT, atomic_json
from usage_ledger import usage_scope


def run_case(binary, config, case, source, repeat, root):
    start = time.perf_counter()
    candidate, errors = None, []
    with usage_scope(root, 'archive') as accounting:
        try:
            candidate, _ = classify(binary, root, config, case['context'])
            if (not isinstance(candidate, dict) or set(candidate) != {'classification', 'reason'}
                    or candidate['classification'] not in SCHEMA['properties']['classification']['enum']
                    or not isinstance(candidate['reason'], str)):
                errors.append('分类输出结构不符')
            elif candidate['classification'] != case['expected']:
                errors.append('分类不符')
        except Exception as exc:
            errors.append(str(exc))
        accounting.outcome = 'failed' if errors else 'passed'
    attempts = [{k: a.data.get(k) for k in ('status', 'retry_reason', 'elapsed_seconds', 'usage')}
                for a in accounting.attempts]
    return {'case': case['id'], 'source': source, 'repeat': repeat, 'model': config['model'],
            'reasoning_effort': config['reasoning_effort'], 'service_tier': config['service_tier'],
            'candidate': candidate, 'passed': not errors, 'errors': errors,
            'seconds': round(time.perf_counter() - start, 3),
            'model_seconds': round(sum(accounting.timings.get(k, 0) for k in ('model', 'retry_model')), 3),
            'model_calls': len(attempts), 'retry_calls': sum(bool(a['retry_reason']) for a in attempts),
            'attempts': attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--model', action='append', help='首个模型为基线，每轮反转运行次序')
    parser.add_argument('--repeat', type=int, default=1, choices=range(1, 11))
    parser.add_argument('--reasoning-effort', choices=REASONING_EFFORTS, default='low')
    parser.add_argument('--service-tier', choices=('standard', 'fast'), default='fast')
    parser.add_argument('--cases', type=Path, default=ROOT / 'tests/fixtures/archiving_cases.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/archiving-evaluation.json')
    args = parser.parse_args()
    cases = [(args.cases.name, c) for c in json.loads(args.cases.read_text(encoding='utf-8'))]
    if not cases:
        parser.error('合成案例不能为空')
    if not args.live:
        print(f'共 {len(cases)} 个合成归档案例；加 --live 才调用模型，不执行真实归档。')
        return 0
    models = args.model or [DEFAULTS['model']]
    if len(set(models)) != len(models):
        parser.error('对比模型不能重复')
    configs = [{**DEFAULTS, 'model': m, 'reasoning_effort': args.reasoning_effort,
                'service_tier': 'priority' if args.service_tier == 'fast' else None} for m in models]
    binary = find_codex()
    with CodexBackend(binary) as backend:
        for config in configs:
            validate_model_config(backend, config)
    jobs, rows = list(ordered_jobs(cases, configs, args.repeat)), []
    protocol = {'context_version': CONTEXT_VERSION,
                'prompt_sha256': hashlib.sha256((ROOT / 'prompts/archiving.md').read_bytes()).hexdigest(),
                'case_files': [{'source': args.cases.name, 'cases': len(cases),
                                'sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest()}]}
    with tempfile.TemporaryDirectory(prefix='oil-archive-eval-') as tmp:
        for index, (config, source, case, repeat) in enumerate(jobs):
            row = run_case(binary, config, case, source, repeat, Path(tmp) / str(index))
            rows.append(row)
            atomic_json(args.output, {'complete': len(rows) == len(jobs), 'workers': 1, 'repeat': args.repeat,
                        'protocol': protocol, 'planned_cases': len(jobs), 'cases': len(rows),
                        'passed': sum(r['passed'] for r in rows),
                        'summaries': [summarize([r for r in rows if r['model'] == c['model']], c) for c in configs],
                        'notice': '仅合成分类，不扫描或归档真实话题；串行交替比较，零调用不进入模型延迟，未知用量不按零计。',
                        'results': rows})
            print(json.dumps({k: row[k] for k in ('model', 'case', 'repeat', 'passed', 'seconds', 'model_calls')}, ensure_ascii=False), flush=True)
    return 0 if all(r['passed'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())

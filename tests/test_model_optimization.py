"""验证命名配置隔离、能力校验、共享预算与比较门槛；不调用账号模型。"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import oil_codex_title as title
import archive_policy as archive
import evaluate_naming as evaluate
from codex_adapter import BackendError, validate_model_config
from usage_ledger import percentiles, usage_report
import test_title as fixtures


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / 'config.json'

    def tearDown(self):
        self.tmp.cleanup()

    def configure(self, *args):
        output = io.StringIO()
        with patch.dict(os.environ, {'OIL_CODEX_TITLE_DATA': str(self.root)}), \
             patch.object(sys, 'argv', ['title', 'configure', *args]), \
             patch.object(title, 'find_codex', return_value='fixture-codex'), \
             patch.object(title, 'CodexBackend') as backend, \
             patch.object(title, 'validate_model_config') as validate, contextlib.redirect_stdout(output):
            status = title.main()
        return status, json.loads(output.getvalue()), validate

    def test_legacy_and_partial_naming_overrides_preserve_unknown_fields(self):
        title.atomic_json(self.path, {'model': 'legacy', 'future': 1,
            'naming': {'reasoning_effort': 'medium', 'future': {'value': 2}, 'enabled': False}})
        config = title.load_config(self.root)
        effective = title.naming_config(config)
        self.assertEqual((effective['model'], effective['reasoning_effort'], effective['enabled']), ('legacy', 'medium', True))
        self.assertEqual(config['future'], 1)
        self.assertEqual(config['naming']['future'], {'value': 2})
        self.assertEqual(config['reasoning_effort'], 'low')

    def test_naming_model_without_tier_does_not_inherit_fast(self):
        title.atomic_json(self.path, {'model': 'gpt-5.6-luna', 'naming': {'model': 'gpt-6-luna'}})
        config = title.load_config(self.root)
        self.assertIsNone(title.naming_config(config)['service_tier'])
        self.assertEqual(config['service_tier'], 'priority')

    def test_naming_switch_checks_capabilities_and_leaves_archive_settings(self):
        title.atomic_json(self.path, {'model': 'gpt-5.6-luna', 'future': 1, 'naming': {'future': 2}})
        code, result, validate = self.configure('--naming-model', 'gpt-6-luna', '--naming-reasoning-effort', 'medium', '--naming-service-tier', 'fast')
        self.assertEqual(code, 0)
        self.assertEqual((result['model'], result['reasoning_effort'], result['service_tier']), ('gpt-5.6-luna', 'low', 'priority'))
        self.assertEqual(result['naming'], {'model': 'gpt-6-luna', 'reasoning_effort': 'medium', 'service_tier': 'priority', 'future': 2})
        self.assertEqual(result['future'], 1)
        selected = validate.call_args.args[1]
        self.assertEqual((selected['model'], selected['reasoning_effort']), ('gpt-6-luna', 'medium'))

    def test_new_default_and_explicit_old_model_are_shared_with_archive(self):
        config = title.load_config(self.root)
        self.assertEqual(config['model'], 'gpt-6-luna')
        self.assertEqual(title.naming_config(config)['model'], 'gpt-6-luna')
        title.atomic_json(self.path, {'model': 'gpt-5.6-luna'})
        config = title.load_config(self.root)
        self.assertEqual(config['model'], 'gpt-5.6-luna')
        self.assertEqual(title.naming_config(config)['model'], 'gpt-5.6-luna')

    def test_shared_switch_validates_top_level_and_existing_naming_override(self):
        title.atomic_json(self.path, {'model': 'gpt-5.6-luna', 'future': 1,
                                    'naming': {'model': 'custom', 'service_tier': None}})
        code, result, validate = self.configure('--model', 'gpt-6-luna', '--service-tier', 'fast')
        self.assertEqual(code, 0)
        self.assertEqual([call.args[1]['model'] for call in validate.call_args_list], ['gpt-6-luna', 'custom'])
        self.assertEqual(result['naming'], {'model': 'custom', 'service_tier': None})
        self.assertEqual(result['future'], 1)

    def test_rejected_shared_or_effective_configuration_never_writes(self):
        for args, failures in ((['--model', 'unknown'], [ValueError('模型不可用')]),
                               (['--service-tier', 'fast'], [ValueError('档位不可用')]),
                               (['--model', 'gpt-6-luna', '--service-tier', 'fast'],
                                [None, ValueError('命名覆盖不可用')])):
            title.atomic_json(self.path, {'model': 'gpt-5.6-luna', 'naming': {'model': 'custom'}})
            before = self.path.read_bytes()
            with self.subTest(args=args), patch.dict(os.environ, {'OIL_CODEX_TITLE_DATA': str(self.root)}), \
                 patch.object(sys, 'argv', ['title', 'configure', *args]), \
                 patch.object(title, 'find_codex', return_value='fixture-codex'), \
                 patch.object(title, 'CodexBackend'), \
                 patch.object(title, 'validate_model_config', side_effect=failures), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(title.main(), 1)
            self.assertEqual(before, self.path.read_bytes())

    def test_shared_configuration_identical_to_naming_checks_catalog_once(self):
        code, result, validate = self.configure('--model', 'gpt-6-luna', '--service-tier', 'fast')
        self.assertEqual(code, 0)
        validate.assert_called_once()
        self.assertEqual(result['model'], 'gpt-6-luna')

    def test_binary_only_configuration_does_not_query_model_catalog(self):
        code, result, validate = self.configure('--codex-bin', 'fixture-codex')
        self.assertEqual(code, 0)
        validate.assert_not_called()
        self.assertEqual(result['codex_bin'], 'fixture-codex')

    def test_cli_naming_switch_clears_existing_fast(self):
        title.atomic_json(self.path, {'naming': {'model': 'old', 'service_tier': 'priority'}})
        code, result, _ = self.configure('--naming-model', 'gpt-6-luna')
        self.assertEqual(code, 0)
        self.assertIsNone(result['naming']['service_tier'])

    def test_rejected_capability_does_not_write_configuration(self):
        title.atomic_json(self.path, {'future': 1})
        before = self.path.read_bytes()
        output = io.StringIO()
        with patch.dict(os.environ, {'OIL_CODEX_TITLE_DATA': str(self.root)}), \
             patch.object(sys, 'argv', ['title', 'configure', '--naming-model', 'unknown']), \
             patch.object(title, 'find_codex', return_value='fixture-codex'), \
             patch.object(title, 'CodexBackend'), \
             patch.object(title, 'validate_model_config', side_effect=ValueError('模型不可用')), contextlib.redirect_stdout(output):
            self.assertEqual(title.main(), 1)
        self.assertEqual(before, self.path.read_bytes())

    def test_bad_nested_settings_are_rejected_offline(self):
        for override in (None, [], {'model': ''}, {'reasoning_effort': 'ultra'}, {'service_tier': 'invalid'}):
            title.atomic_json(self.path, {'naming': override})
            with self.subTest(override=override), self.assertRaises(ValueError):
                title.load_config(self.root)

    def test_catalog_uses_actual_efforts_and_tiers_with_pagination(self):
        model = {'model': 'gpt-6-luna', 'supportedReasoningEfforts': [{'reasoningEffort': 'low'}],
                 'serviceTiers': [{'id': 'priority'}]}
        backend = Mock()
        def call(method, params):
            return {'data': [model], 'nextCursor': None} if params.get('cursor') else {'data': [], 'nextCursor': 'next'}
        backend.call.side_effect = call
        config = {**title.DEFAULTS, 'model': 'gpt-6-luna'}
        validate_model_config(backend, config)
        self.assertEqual(backend.call.call_count, 2)
        with self.assertRaisesRegex(ValueError, '推理'):
            validate_model_config(backend, {**config, 'reasoning_effort': 'none'})
        model['serviceTiers'] = []
        with self.assertRaisesRegex(ValueError, '服务'):
            validate_model_config(backend, config)
        validate_model_config(backend, {**config, 'service_tier': None})

    def test_missing_model_and_repeated_cursor_fail_closed(self):
        backend = Mock()
        backend.call.return_value = {'data': [], 'nextCursor': None}
        with self.assertRaises(ValueError):
            validate_model_config(backend, title.DEFAULTS)
        backend.call.return_value = {'data': [], 'nextCursor': 'same'}
        with self.assertRaises(BackendError):
            validate_model_config(backend, title.DEFAULTS)


class BudgetAndTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.backend = fixtures.FakeBackend()
        self.config = {**title.DEFAULTS, 'model': 'gpt-5.6-luna', 'model_timeout_seconds': 10,
                       'naming': {'model': 'gpt-6-luna', 'reasoning_effort': 'medium', 'service_tier': None}}
        self.now = 0
        self.timeouts, self.commands = [], []
        self.candidate = {'action': 'rename', 'title': '🎬 产品视频｜讲解大纲', 'reason': '明确目标'}

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, context):
        return title.limited_title('unused', self.root, self.config, context,
            before_model=lambda: title.ensure_title_active(self.backend, fixtures.ID, self.root))

    def fake_run(self, args, **kwargs):
        self.timeouts.append(kwargs['timeout'])
        self.commands.append(args)
        Path(args[args.index('--output-last-message') + 1]).write_text(json.dumps(self.candidate), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout=json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 3}}))

    def conflict(self):
        title.atomic_json(self.root / 'threads' / 'other.json', {'last_seen_title': self.candidate['title']})

    def process(self):
        return title.process_thread(self.backend, self.model, fixtures.ID, self.root, self.config, apply=True)

    def test_conflict_retry_receives_remaining_budget_and_distinct_usage(self):
        self.conflict()
        def run(args, **kwargs):
            result = self.fake_run(args, **kwargs)
            self.now += 7 if len(self.commands) == 1 else 2
            self.candidate['title'] = '🎬 产品视频｜镜头大纲'
            return result
        with patch('codex_adapter.time.monotonic', side_effect=lambda: self.now), patch('codex_adapter.subprocess.run', side_effect=run):
            result = self.process()
        self.assertEqual(self.timeouts, [10, 3])
        self.assertEqual(result['status'], 'renamed')
        self.assertEqual(result['performance']['model_calls'], 2)
        self.assertEqual(result['performance']['retry_calls'], 1)
        records = [json.loads(p.read_text()) for p in (self.root / 'usage').glob('*/*.json')]
        self.assertEqual({r['retry_reason'] for r in records}, {None, 'conflict'})
        self.assertTrue(all(r['model'] == 'gpt-6-luna' and r['reasoning_effort'] == 'medium' for r in records))
        self.assertIn('model_reasoning_effort="medium"', self.commands[0])
        self.assertFalse(any('service_tier=' in arg for arg in self.commands[0]))
        report = usage_report(self.root)
        self.assertEqual(report['groups'][0]['tokens']['input_tokens'], 20)
        self.assertEqual(report['performance']['naming']['retry_calls'], 1)

    def test_exhausted_conflict_budget_never_launches_second_process_or_writes(self):
        self.conflict()
        def run(args, **kwargs):
            result = self.fake_run(args, **kwargs)
            self.now = 11
            return result
        with patch('codex_adapter.time.monotonic', side_effect=lambda: self.now), patch('codex_adapter.subprocess.run', side_effect=run) as call:
            with self.assertRaisesRegex(BackendError, '预算已耗尽'):
                self.process()
        self.assertEqual(call.call_count, 1)
        self.assertEqual(self.backend.writes, [])
        report = usage_report(self.root)
        self.assertEqual(report['performance']['naming']['statuses'], {'error': 1})
        self.assertEqual(report['performance']['naming']['model_calls'], 1)

    def test_format_and_conflict_retries_share_one_budget(self):
        self.conflict()
        rename = self.candidate.copy()
        self.candidate = {'action': 'keep', 'title': '讨论事情', 'reason': '信息不足'}
        def run(args, **kwargs):
            result = self.fake_run(args, **kwargs)
            count = len(self.commands)
            self.now += (2, 3, 4)[count - 1]
            self.candidate = rename if count == 1 else {**rename, 'title': '🎬 产品视频｜镜头大纲'}
            return result
        with patch('codex_adapter.time.monotonic', side_effect=lambda: self.now), patch('codex_adapter.subprocess.run', side_effect=run):
            result = self.process()
        self.assertEqual(self.timeouts, [10, 8, 5])
        self.assertEqual(result['performance']['retry_calls'], 2)

    def test_archive_uses_top_level_model_and_effort(self):
        self.candidate = {'classification': 'completed', 'reason': '完成'}
        with patch('codex_adapter.subprocess.run', side_effect=self.fake_run):
            archive.classify('unused', self.root, self.config, {'goal': '合成任务'})
        command, = self.commands
        self.assertEqual(command[command.index('-m') + 1], 'gpt-5.6-luna')
        self.assertIn('model_reasoning_effort="low"', command)
        self.assertIn('service_tier="priority"', command)

    def test_no_call_paths_and_failed_calls_have_content_free_performance(self):
        self.backend.archived = True
        result = self.process()
        self.assertEqual(result['performance']['model_calls'], 0)
        self.backend.archived = False
        with patch('codex_adapter.subprocess.run', side_effect=OSError('private diagnostic')):
            with self.assertRaises(OSError):
                self.process()
        rows = [json.loads(p.read_text()) for p in (self.root / 'runs').glob('*/*.json')]
        self.assertEqual({r['status'] for r in rows}, {'archived', 'error'})
        self.assertNotIn('private diagnostic', json.dumps(rows))
        for r in rows:
            self.assertEqual(set(r), {'version', 'purpose', 'status', 'elapsed_seconds', 'timings', 'model_calls', 'retry_calls'})
            self.assertGreaterEqual(r['elapsed_seconds'], sum(r['timings'].values()) - .01)
        self.assertIn('state_check', result['performance']['timings'])

    def test_written_title_reports_write_and_verification_phases(self):
        with patch('codex_adapter.subprocess.run', side_effect=self.fake_run):
            result = self.process()
        for name in ('thread_lock_wait', 'worker_wait', 'host_read', 'state_check', 'model', 'title_write', 'title_verify'):
            self.assertIn(name, result['performance']['timings'])

    def test_corrupt_performance_record_is_not_partially_counted(self):
        self.backend.archived = True
        self.process()
        folder = self.root / 'runs' / 'fixture'
        folder.mkdir()
        (folder / 'broken.json').write_text(json.dumps({'purpose': 'naming', 'model_calls': 999, 'status': 'renamed'}))
        report = usage_report(self.root)
        self.assertEqual(report['unreadable_records'], 1)
        self.assertEqual(report['performance']['naming']['runs'], 1)
        self.assertEqual(report['performance']['naming']['model_calls'], 0)


class ComparisonTests(unittest.TestCase):
    def summary(self, model, p50, p95, passed=10):
        return {'model': model, 'cases': 10, 'passed': passed, 'pass_rate': passed / 10,
                'model_latency': {'p50_seconds': p50, 'p95_seconds': p95}}

    def test_switch_gate_requires_every_quality_and_latency_condition(self):
        old, new = self.summary('old', 10, 15), self.summary('new', 7.9, 14)
        self.assertTrue(evaluate.acceptance(old, new)['accepted'])
        self.assertTrue(evaluate.acceptance(old, self.summary('new', 8, 15))['accepted'])
        for candidate, kwargs in ((self.summary('new', 9, 14), {}), (self.summary('new', 7, 16), {}),
                                   (self.summary('new', 7, 14, passed=9), {}), (new, {'complete': False}),
                                   (new, {'comparable': False}), (new, {'repeats': 1})):
            with self.subTest(candidate=candidate, kwargs=kwargs):
                self.assertFalse(evaluate.acceptance(old, candidate, **kwargs)['accepted'])

    def test_no_measurements_cannot_pass_gate_and_percentiles_exclude_zero_calls(self):
        self.assertFalse(evaluate.acceptance(self.summary('old', None, None), self.summary('new', None, None))['accepted'])
        rows = [{'passed': True, 'model_calls': 0, 'retry_calls': 0, 'attempts': [], 'seconds': 0, 'model_seconds': 0},
                {'passed': True, 'model_calls': 2, 'retry_calls': 1, 'seconds': 9, 'model_seconds': 8,
                 'attempts': [{'usage': {'input_tokens': 10, 'output_tokens': 2}, 'elapsed_seconds': 3},
                              {'usage': None, 'elapsed_seconds': 5}]}]
        summary = evaluate.summarize(rows, title.DEFAULTS)
        self.assertEqual(summary['model_latency'], {'p50_seconds': 8, 'p95_seconds': 8})
        self.assertEqual((summary['model_calls'], summary['retry_calls'], summary['unknown_usage']), (2, 1, 1))
        self.assertEqual(summary['tokens']['input_tokens'], 10)
        self.assertEqual(percentiles([1, 2, 3, 4]), {'p50_seconds': 2.5, 'p95_seconds': 4})

    def test_comparison_alternates_models_and_keeps_case_context_unchanged(self):
        cases = [('fixture', {'id': 'case', 'context': {'data': 'synthetic'}})]
        configs = [{'model': 'old'}, {'model': 'new'}]
        jobs = list(evaluate.ordered_jobs(cases, configs, 3))
        self.assertEqual([j[0]['model'] for j in jobs], ['old', 'new', 'new', 'old', 'old', 'new'])
        self.assertTrue(all(j[2] is cases[0][1] for j in jobs))

    def test_live_case_failure_retains_candidate_and_actual_usage(self):
        case = {'id': '合成错误输出', 'context': {'current_title': '待整理', 'original_goal': '修复验证码'},
                'expected': {'action': 'rename', 'contains': ['验证码'], 'reject': []}}
        def run(args, **kwargs):
            output = Path(args[args.index('--output-last-message') + 1])
            output.write_text(json.dumps({'action': 'rename', 'title': '非法格式', 'reason': '合成输出'}), encoding='utf-8')
            return SimpleNamespace(returncode=0, stdout=json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 2}}))
        with tempfile.TemporaryDirectory() as tmp, patch('codex_adapter.subprocess.run', side_effect=run):
            row = evaluate.run_case('fixture', title.DEFAULTS, case, 'fixture.json', 1, Path(tmp))
        self.assertFalse(row['passed'])
        self.assertEqual(row['candidate']['title'], '非法格式')
        self.assertEqual((row['model_calls'], row['retry_calls']), (1, 0))
        self.assertEqual(row['attempts'][0]['usage'], {'input_tokens': 10, 'output_tokens': 2})
        self.assertEqual(set(row['attempts'][0]), {'status', 'retry_reason', 'elapsed_seconds', 'usage'})

    def test_non_live_comparison_never_calls_backend_or_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'report.json'
            with patch.object(sys, 'argv', ['evaluate', '--model', 'new', '--repeat', '3', '--output', str(report)]), \
                 patch.object(evaluate, 'CodexBackend') as backend, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(), 0)
            backend.assert_not_called()
            self.assertFalse(report.exists())


if __name__ == '__main__':
    unittest.main()

"""归档合成评测的离线契约，不连接账号或执行真实归档。"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_archiving as evaluate
from oil_codex_title import DEFAULTS


class EvaluationTests(unittest.TestCase):
    def test_non_live_has_eight_cases_and_no_backend_or_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'report.json'
            with patch.object(sys, 'argv', ['evaluate', '--output', str(output)]), \
                 patch.object(evaluate, 'CodexBackend') as backend, \
                 patch.object(evaluate, 'classify') as classify, \
                 contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(evaluate.main(), 0)
            self.assertIn('8 个', stdout.getvalue())
            backend.assert_not_called()
            classify.assert_not_called()
            self.assertFalse(output.exists())

    def test_failed_classification_keeps_candidate_and_usage_without_identifiers(self):
        case = {'id': '等待审批', 'context': {'original_goal': '部署页面'}, 'expected': 'open'}
        def run(args, **kwargs):
            self.assertEqual(args[args.index('-m') + 1], 'gpt-6-luna')
            self.assertIn('model_reasoning_effort="low"', args)
            self.assertIn('service_tier="priority"', args)
            Path(args[args.index('--output-last-message') + 1]).write_text(
                json.dumps({'classification': 'completed', 'reason': '合成错误'}), encoding='utf-8')
            return SimpleNamespace(returncode=0, stdout=json.dumps({
                'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 2}}))
        with tempfile.TemporaryDirectory() as tmp, patch('codex_adapter.subprocess.run', side_effect=run):
            row = evaluate.run_case('fixture', DEFAULTS, case, 'fixture.json', 1, Path(tmp))
        self.assertFalse(row['passed'])
        self.assertEqual(row['candidate']['classification'], 'completed')
        self.assertEqual(row['attempts'][0]['usage'], {'input_tokens': 10, 'output_tokens': 2})
        self.assertEqual((row['model_calls'], row['retry_calls']), (1, 0))
        self.assertNotIn('thread_id', json.dumps(row))

    def test_live_report_is_complete_and_alternates_without_archiving(self):
        configs = []
        def classify(binary, root, config, context):
            configs.append(config['model'])
            return {'classification': 'open', 'reason': '尚未实施'}, {}
        with tempfile.TemporaryDirectory() as tmp:
            cases, output = Path(tmp) / 'cases.json', Path(tmp) / 'report.json'
            cases.write_text(json.dumps([{'id': '待办', 'context': {}, 'expected': 'open'}]))
            with patch.object(sys, 'argv', ['evaluate', '--live', '--model', 'old', '--model', 'new',
                                          '--repeat', '3', '--cases', str(cases), '--output', str(output)]), \
                 patch.object(evaluate, 'find_codex', return_value='fixture'), \
                 patch.object(evaluate, 'CodexBackend'), patch.object(evaluate, 'validate_model_config'), \
                 patch.object(evaluate, 'classify', side_effect=classify), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(), 0)
            report = json.loads(output.read_text())
            self.assertTrue(report['complete'])
            self.assertEqual((report['cases'], report['passed']), (6, 6))
            self.assertEqual(configs, ['old', 'new', 'new', 'old', 'old', 'new'])


if __name__ == '__main__':
    unittest.main()

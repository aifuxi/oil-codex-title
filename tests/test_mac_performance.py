"""验证延迟连接、宿主读取次数及长对话摘录等价性；不调用外部模型。"""
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import oil_codex_title as title
from codex_adapter import BackendError, CodexBackend
import test_title as fixtures


def process_output(*results):
    return SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO(''.join(json.dumps({'id': i, 'result': result}) + '\n'
                                 for i, result in enumerate(results, 1))),
        terminate=Mock(), wait=Mock(), kill=Mock())


class LazyConnectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def process(self, backend, **kwargs):
        return title.process_thread(backend, fixtures.proposal, fixtures.ID,
                                    self.root, title.DEFAULTS, **kwargs)

    def test_lazy_context_without_requests_starts_no_process(self):
        with patch('codex_adapter.subprocess.Popen') as popen:
            with CodexBackend('unused', lazy=True):
                popen.assert_not_called()
        popen.assert_not_called()

    def test_lazy_first_request_initializes_once_and_cleans_up(self):
        proc = process_output({}, {'thread': {'cwd': '/fixture/project'}}, {})
        with patch('codex_adapter.subprocess.Popen', return_value=proc) as popen:
            with CodexBackend('unused', lazy=True) as backend:
                backend.read(fixtures.ID)
                backend.start()
                backend.rename(fixtures.ID, '合成标题')
                requests = [json.loads(line) for line in proc.stdin.getvalue().splitlines()]
            popen.assert_called_once()
        self.assertEqual([r['method'] for r in requests],
                         ['initialize', 'initialized', 'thread/read', 'thread/name/set'])
        proc.terminate.assert_called_once()
        self.assertTrue(proc.stdin.closed and proc.stdout.closed)

    def test_eager_connection_still_initializes_on_enter(self):
        proc = process_output({})
        with patch('codex_adapter.subprocess.Popen', return_value=proc) as popen:
            with CodexBackend('unused'):
                popen.assert_called_once()

    def test_failed_lazy_initialization_closes_child(self):
        proc = process_output()
        with patch('codex_adapter.subprocess.Popen', return_value=proc):
            with self.assertRaises(BackendError):
                with CodexBackend('unused', lazy=True) as backend:
                    backend.read(fixtures.ID)
        self.assertTrue(proc.stdin.closed and proc.stdout.closed)

    def test_established_locked_hook_has_zero_host_access(self):
        title.atomic_json(title.state_path(self.root, fixtures.ID),
                          {'locked': True, 'last_fingerprint': 'baseline'})
        with patch('codex_adapter.subprocess.Popen') as popen:
            with CodexBackend('unused', lazy=True) as backend:
                result = self.process(backend, event_turn=fixtures.TURN, apply=True)
        popen.assert_not_called()
        self.assertEqual(result['status'], 'locked')
        self.assertEqual(result['performance']['model_calls'], 0)
        self.assertNotIn('connection', result['performance']['timings'])

    def test_initial_false_manual_lock_still_recovers(self):
        title.atomic_json(title.state_path(self.root, fixtures.ID),
                          {'locked': True, 'lock_reason': '检测到外部改名'})
        backend = fixtures.FakeBackend()
        result = self.process(backend, event_turn=fixtures.TURN, apply=True)
        self.assertEqual(result['status'], 'renamed')
        self.assertFalse(title.read_json(title.state_path(self.root, fixtures.ID))['locked'])

    def test_pending_write_is_reconciled_before_locked_hook_returns(self):
        backend = fixtures.FakeBackend()
        title.atomic_json(title.state_path(self.root, fixtures.ID),
                          {'locked': True, 'last_fingerprint': 'baseline',
                           'pending_title': backend.thread['name']})
        self.assertEqual(self.process(backend, event_turn=fixtures.TURN, apply=True)['status'], 'locked')
        self.assertNotIn('pending_title', title.read_json(title.state_path(self.root, fixtures.ID)))

    def test_busy_request_does_not_start_app_server(self):
        with title.thread_lock(self.root, fixtures.ID), patch('codex_adapter.subprocess.Popen') as popen:
            with CodexBackend('unused', lazy=True) as backend:
                result = self.process(backend)
        popen.assert_not_called()
        self.assertEqual(result['status'], 'busy')

    def test_waiting_hook_starts_only_after_lock_release(self):
        waiting = threading.Event()
        acquire = title.file_lock.acquire
        def observed_acquire(stream):
            try:
                return acquire(stream)
            except BlockingIOError:
                waiting.set()
                raise
        proc = process_output({}, {'data': [], 'nextCursor': None}, {'thread': fixtures.thread()},
                              {'data': [], 'nextCursor': None})
        with patch('codex_adapter.subprocess.Popen', return_value=proc) as popen, \
             patch.object(title.file_lock, 'acquire', side_effect=observed_acquire), \
             ThreadPoolExecutor(max_workers=1) as pool:
            with title.thread_lock(self.root, fixtures.ID):
                def run():
                    with CodexBackend('unused', lazy=True) as backend:
                        return self.process(backend, event_turn=fixtures.TURN)
                future = pool.submit(run)
                self.assertTrue(waiting.wait(timeout=3))
                popen.assert_not_called()
            result = future.result(timeout=5)
        self.assertEqual(result['status'], 'preview')
        self.assertIn('connection', result['performance']['timings'])
        self.assertGreaterEqual(result['performance']['elapsed_seconds'],
                                sum(result['performance']['timings'].values()) - .01)

    def test_pause_while_waiting_has_zero_host_access(self):
        @contextmanager
        def paused_lock(*args):
            title.atomic_json(self.root / 'config.json', {'enabled': False})
            yield True
        with patch.object(title, 'thread_lock', paused_lock), patch('codex_adapter.subprocess.Popen') as popen:
            with CodexBackend('unused', lazy=True) as backend:
                result = self.process(backend, event_turn=fixtures.TURN)
        popen.assert_not_called()
        self.assertEqual(result['status'], 'disabled')


class HostReadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.backend = fixtures.FakeBackend()
        self.backend.thread['cwd'] = '/fixture/project'
        self.candidate = fixtures.proposal({})[0]

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, context):
        return title.limited_title('unused', self.root, title.DEFAULTS, context,
            before_model=lambda: title.ensure_title_active(self.backend, fixtures.ID, self.root,
                                                           self.backend.thread['cwd']))

    def fake_run(self, args, **kwargs):
        Path(args[args.index('--output-last-message') + 1]).write_text(json.dumps(self.candidate))
        return SimpleNamespace(returncode=0, stdout=json.dumps({'type': 'turn.completed',
                                  'usage': {'input_tokens': 10, 'output_tokens': 3}}))

    def process(self, **kwargs):
        return title.process_thread(self.backend, self.model, fixtures.ID, self.root,
                                    title.DEFAULTS, apply=True, preflight_in_generator=True, **kwargs)

    def test_guarded_model_checks_archive_three_times_and_scopes_after_read(self):
        with patch.object(self.backend, 'is_archived', wraps=self.backend.is_archived) as probe, \
             patch('codex_adapter.subprocess.run', side_effect=self.fake_run) as run:
            self.assertEqual(self.process()['status'], 'renamed')
        self.assertEqual(run.call_count, 1)
        self.assertEqual(probe.call_args_list[0].args, (fixtures.ID,))
        self.assertEqual([c.args for c in probe.call_args_list[1:]],
                         [(fixtures.ID, '/fixture/project')] * 2)

    def test_keep_and_same_title_candidates_do_not_scan_conflicts(self):
        self.backend.thread['name'] = self.candidate['title']
        for action in ('keep', 'rename'):
            self.candidate['action'] = action
            title.state_path(self.root, fixtures.ID).unlink(missing_ok=True)
            with patch('codex_adapter.subprocess.run', side_effect=self.fake_run), \
                 patch.object(title, 'conflicting_titles') as conflicts:
                self.assertEqual(self.process()['status'], 'kept')
            conflicts.assert_not_called()

    def test_pause_between_collision_attempts_stops_real_second_launch(self):
        title.atomic_json(self.root / 'threads' / 'other.json',
                          {'scope_key': hashlib.sha256(b'/fixture/project').hexdigest(),
                           'last_seen_title': self.candidate['title']})
        def paused(args, **kwargs):
            result = self.fake_run(args, **kwargs)
            title.atomic_json(self.root / 'config.json', {'enabled': False})
            return result
        with patch('codex_adapter.subprocess.run', side_effect=paused) as run:
            self.assertEqual(self.process()['status'], 'disabled')
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.backend.writes, [])

    def test_archived_during_quota_wait_stops_first_launch(self):
        @contextmanager
        def queued(*args):
            self.backend.archived = True
            yield True
        with patch.object(title, 'worker_slot', queued), patch('codex_adapter.subprocess.run') as run:
            self.assertEqual(self.process()['status'], 'archived')
        run.assert_not_called()

    def test_cached_cwd_is_host_fact_and_archive_pagination_is_complete(self):
        backend = CodexBackend('unused', lazy=True)
        params = []
        def call(method, values):
            if method == 'thread/read':
                return {'thread': self.backend.thread}
            params.append(values.copy())
            return ({'data': [{'id': fixtures.ID}], 'nextCursor': None} if values.get('cursor')
                    else {'data': [{'id': 'other'}], 'nextCursor': 'next'})
        with patch.object(backend, 'call', side_effect=call):
            backend.read(fixtures.ID)
            self.assertTrue(backend.is_archived(fixtures.ID))
        self.assertEqual(len(params), 2)
        self.assertTrue(all(p['cwd'] == '/fixture/project' and p['archived'] for p in params))
        self.assertTrue(all('useStateDbOnly' not in p for p in params))


class SnapshotTests(unittest.TestCase):
    def reference(self, thread):
        effective = [{'id': t['id'], 'messages': title.turn_messages(t)} for t in thread['turns']]
        effective = [t for t in effective if any(m['role'] == 'user' for m in t['messages'])]
        selected = effective[-title.DEFAULTS['recent_turns']:]
        per_message = max(200, (title.DEFAULTS['max_context_chars'] - 2000)
                          // max(1, sum(len(t['messages']) for t in selected)))
        recent = [{'id': t['id'], 'messages': [{**m, 'text': m['text'][:min(per_message,
                    1200 if m['role'] == 'user' else 500)]} for m in t['messages']]} for t in selected]
        original = next((m['text'][:800] for t in effective for m in t['messages'] if m['role'] == 'user'), '')
        signature = json.dumps({'policy_version': title.POLICY_VERSION,
                                'project_hint': title.project_hint(thread),
                                'latest_id': thread['turns'][-1]['id'] if thread['turns'] else None,
                                'effective': selected}, ensure_ascii=False, sort_keys=True)
        return recent, original, hashlib.sha256(signature.encode()).hexdigest()

    def test_recent_extraction_preserves_full_scan_context_and_fingerprint(self):
        thread = fixtures.thread()
        thread['turns'] = [dict(copy.deepcopy(thread['turns'][0]), id=str(i)) for i in range(20)]
        thread['turns'][0]['items'].insert(0, {'type': 'userMessage', 'content': [{'type': 'text', 'text': '   '}]})
        thread['turns'].insert(0, {'id': 'empty', 'items': [{'type': 'agentMessage', 'text': '无用户的回答'}]})
        thread['turns'][10]['items'].append({'type': 'agentMessage', 'phase': 'commentary', 'text': '过程'})
        thread['turns'][-1]['items'][0]['content'][0]['text'] += '长需求' * 3000
        thread['turns'].append({'id': 'tool-only', 'items': [{'type': 'commandExecution'}]})
        recent, original, fingerprint = self.reference(thread)
        snap = title.snapshot(thread, title.DEFAULTS)
        self.assertEqual(snap['context']['recent_turns'], recent)
        self.assertEqual(snap['context']['original_goal'], original)
        self.assertEqual(snap['fingerprint'], fingerprint)
        self.assertEqual(snap['latest_id'], 'tool-only')

    def test_long_history_only_cleans_initial_goal_and_recent_messages(self):
        thread = fixtures.thread()
        thread['turns'] = [dict(copy.deepcopy(thread['turns'][0]), id=str(i)) for i in range(2000)]
        with patch.object(title, 'item_text', wraps=title.item_text) as extract:
            snap = title.snapshot(thread, title.DEFAULTS)
        self.assertEqual(extract.call_count, 11)
        self.assertEqual(len(snap['context']['recent_turns']), 5)

    def test_text_outside_excerpt_still_changes_fingerprint(self):
        thread = fixtures.thread()
        thread['turns'][0]['items'][1]['text'] = '长回答' * 1000
        before = title.snapshot(thread, title.DEFAULTS)
        thread['turns'][0]['items'][1]['text'] += '新增待办'
        after = title.snapshot(thread, title.DEFAULTS)
        self.assertEqual(before['context'], after['context'])
        self.assertNotEqual(before['fingerprint'], after['fingerprint'])

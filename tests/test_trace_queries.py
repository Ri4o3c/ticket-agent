"""执行记录只读接口：真实SQLite与HTTP路由，不启动模型或MCP。"""
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

from knowledge_bases import KnowledgeBases
from service_settings import ServiceSettings
from ticket_store import Actor, TicketStore
from trace_store import TraceStore
from web_app import create_app


class TraceQueryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.business = TicketStore(self.root / 'tickets.sqlite3')
        self.trace = TraceStore(self.root / 'traces.sqlite3')
        self.identity = SimpleNamespace(actor=Actor('reader', 'space-a'))
        self.settings = ServiceSettings(self.root / 'settings.json', self.root / '.env')
        self.catalog = KnowledgeBases(self.root / 'bases.json')
        self.http = self.enterContext(TestClient(self.app()))

    def app(self):
        return create_app(store_factory=lambda: self.business, actor_factory=lambda: self.identity.actor,
                          settings=self.settings, catalog=self.catalog)

    def run_record(self, actor=None):
        return self.trace.start_run(actor or self.identity.actor, session_id='hashed-session',
                                    knowledge_base_id='default', model_name='test-model')

    def test_list_works_without_api_configuration_and_only_returns_current_owner(self):
        first, second = self.run_record(), self.run_record()
        foreign = self.run_record(Actor('other', 'space-a'))
        other_space = self.run_record(Actor('reader', 'space-b'))
        response = self.http.get('/api/runs?limit=1')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([r['run_id'] for r in response.json()['runs']], [second])
        self.assertEqual(response.headers.get('cache-control'), 'no-store')
        all_rows = self.http.get('/api/runs').json()['runs']
        self.assertEqual({r['run_id'] for r in all_rows}, {first, second})
        self.assertNotIn(foreign, str(all_rows))
        self.assertNotIn(other_space, str(all_rows))

    def test_detail_keeps_execution_status_and_step_results_separate(self):
        run_id = self.run_record()
        model = self.trace.start_step(run_id, self.identity.actor, kind='model', name='test-model')
        self.trace.finish_step(model, self.identity.actor, status='succeeded', input_tokens=0, output_tokens=7)
        tool = self.trace.start_step(run_id, self.identity.actor, kind='tool', name='get_ticket')
        self.trace.finish_step(tool, self.identity.actor, status='failed', error_type='PermissionError')
        validation = self.trace.start_step(run_id, self.identity.actor, kind='validation', name='answer_guard')
        self.trace.finish_step(validation, self.identity.actor, status='succeeded')
        self.trace.finish_run(run_id, self.identity.actor)
        response = self.http.get('/api/runs/' + run_id)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data['run']['status'], 'completed')
        self.assertEqual([s['position'] for s in data['steps']], [1, 2, 3])
        self.assertEqual([s['status'] for s in data['steps']], ['succeeded', 'failed', 'succeeded'])
        self.assertEqual(data['steps'][0]['input_tokens'], 0)
        self.assertIsNone(data['steps'][1]['input_tokens'])
        self.assertIsNone(data['steps'][1]['duration_ms'])
        self.assertNotIn('task_success', data)
        self.assertEqual(response.headers.get('cache-control'), 'no-store')

    def test_missing_and_forbidden_records_have_same_404_without_leaking_metadata(self):
        foreign = self.run_record(Actor('other', 'space-a'))
        other_space = self.run_record(Actor('reader', 'space-b'))
        missing = self.http.get('/api/runs/run-missing')
        self.assertEqual(missing.status_code, 404)
        for run_id in (foreign, other_space):
            response = self.http.get('/api/runs/' + run_id)
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json(), missing.json())
            self.assertEqual(response.headers.get('cache-control'), 'no-store')
            self.assertNotIn('test-model', response.text)

    def test_caller_cannot_select_another_identity_with_query_parameters(self):
        own = self.run_record()
        foreign = self.run_record(Actor('other', 'space-a'))
        response = self.http.get('/api/runs?user_id=other&workspace_id=space-a&role=admin')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([r['run_id'] for r in response.json()['runs']], [own])
        self.assertEqual(self.http.get('/api/runs/' + foreign + '?user_id=other').status_code, 404)
        self.identity.actor = Actor('other', 'space-a')
        self.assertEqual([r['run_id'] for r in self.http.get('/api/runs').json()['runs']], [foreign])

    def test_invalid_limits_are_rejected_and_valid_upper_limit_works(self):
        for value in ('0', '-1', '101', 'abc', '1.5', ''):
            with self.subTest(value=value):
                response = self.http.get('/api/runs?limit=' + value)
                self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.http.get('/api/runs?limit=100').status_code, 200)

    def test_empty_history_can_be_read_without_creating_a_conversation(self):
        response = self.http.get('/api/runs')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {'runs': []})
        with self.business.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tickets').fetchone()[0], 0)
        self.assertEqual(self.trace.list_runs(self.identity.actor), [])

    def test_running_record_is_not_reclassified_or_modified_by_reading(self):
        run_id = self.run_record()
        step_id = self.trace.start_step(run_id, self.identity.actor, kind='approval', name='human_confirmation')
        before = self.trace.get_run(run_id, self.identity.actor)
        response = self.http.get('/api/runs/' + run_id)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['run'], before)
        self.assertEqual(response.json()['steps'][0]['status'], 'running')
        self.assertEqual(response.json()['steps'][0]['step_id'], step_id)
        self.assertEqual(self.trace.get_run(run_id, self.identity.actor), before)
        self.assertEqual(self.http.post('/api/runs/' + run_id).status_code, 405)

    def test_storage_failure_returns_safe_503_for_list_and_detail(self):
        run_id = self.run_record()
        with patch('web_app.TraceStore', create=True, side_effect=sqlite3.OperationalError('PRIVATE-KEY-STORAGE-PATH')):
            for path in ('/api/runs', '/api/runs/' + run_id):
                with self.subTest(path=path):
                    response = self.http.get(path)
                    self.assertEqual(response.status_code, 503, response.text)
                    self.assertNotIn('PRIVATE-KEY-STORAGE-PATH', response.text)
                    self.assertEqual(response.headers.get('cache-control'), 'no-store')

    def test_history_survives_application_restart(self):
        run_id = self.run_record()
        with TestClient(self.app()) as reopened:
            response = reopened.get('/api/runs/' + run_id)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['run']['run_id'], run_id)

    def test_sql_text_in_run_id_is_not_executed(self):
        run_id = self.run_record()
        response = self.http.get('/api/runs/run-%27%20OR%201%3D1--')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.trace.get_run(run_id, self.identity.actor)['run_id'], run_id)


if __name__ == '__main__':
    unittest.main()

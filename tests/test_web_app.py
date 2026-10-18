# 用Starlette测试客户端、模拟聊天模型和本地工具连接验证完整网页接口流程。
# 覆盖会话独立、草稿匹配确认、同源检查及上传重连；HTTP通过不等于人工浏览器视觉验收。
"""网页接口复用真实工单与MCP，模型采用固定回复，不调用付费API。"""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastmcp import Client
from langchain_core.messages import AIMessage, ToolMessage
from starlette.testclient import TestClient
from mcp_tools import make_server
from ticket_store import Actor, TicketStore
try:
    from web_app import create_app
except ModuleNotFoundError:
    create_app = None


class WebModel:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        last = messages[-1]
        if isinstance(last, ToolMessage):
            data = json.loads(last.content)
            if last.name == 'prepare_ticket' and data.get('ok'):
                return AIMessage(content='', tool_calls=[{
                    'name': 'create_ticket', 'args': {'draft_id': data['draft']['draft_id']}, 'id': 'create-web',
                }])
            return AIMessage(content='已创建工单 ' + data.get('ticket', {}).get('ticket_id', ''))
        if last[1].startswith('登记'):
            return AIMessage(content='', tool_calls=[{
                'name': 'prepare_ticket', 'args': {'title': 'PDF失败', 'description': '2MB PDF报E103'}, 'id': 'prepare-web',
            }])
        humans = [m[1] for m in messages if isinstance(m, tuple) and m[0] == 'human']
        return AIMessage(content=' / '.join(humans))


class WebTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(create_app, '网页适配层尚未实现')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = TicketStore(Path(self.temp.name) / 'tickets.sqlite3')
        self.actor = Actor('test-user', 'test-workspace')
        app = create_app(
            client_factory=lambda kb_id: Client(make_server(store=self.store, actor=self.actor, searcher=lambda q, k: [])),
            model_factory=lambda url: WebModel(), store_factory=lambda: self.store, actor_factory=lambda: self.actor,
            knowledge_factory=lambda kb_id: SimpleNamespace(upload=lambda name, data: {'filename': name, 'duplicate': False}, list_files=lambda: []),
        )
        self.http = self.enterContext(TestClient(app))

    def state(self, sid):
        return self.http.get(f'/api/sessions/{sid}/events').json()

    def until(self, sid, predicate):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            state = self.state(sid)
            if predicate(state):
                return state
            time.sleep(.01)
        self.fail(f'网页状态没有推进：{state}')

    def session(self):
        response = self.http.post('/api/sessions')
        self.assertEqual(response.status_code, 201)
        sid = response.json()['session_id']
        self.until(sid, lambda state: state['ready'])
        return sid

    def send(self, sid, text):
        response = self.http.post(f'/api/sessions/{sid}/messages', json={'message': text})
        self.assertEqual(response.status_code, 202)

    def test_page_and_static_assets_are_served(self):
        self.assertEqual(self.http.get('/').status_code, 200)
        self.assertEqual(self.http.get('/static/app.js').status_code, 200)
        self.assertEqual(self.http.get('/.env').status_code, 404)

    def test_conversation_preserves_history_and_isolates_sessions(self):
        first = self.session()
        self.send(first, '我叫小明')
        self.until(first, lambda s: not s['busy'])
        self.send(first, '我叫什么？')
        state = self.until(first, lambda s: not s['busy'])
        self.assertIn('我叫小明', [e for e in state['events'] if e['type'] == 'message'][-1]['content'])
        second = self.session()
        self.send(second, '你好')
        state = self.until(second, lambda s: not s['busy'])
        self.assertNotIn('小明', json.dumps(state, ensure_ascii=False))

    def test_creation_waits_for_matching_human_confirmation(self):
        sid = self.session()
        self.send(sid, '登记PDF故障')
        state = self.until(sid, lambda s: s['pending_draft'] is not None)
        draft_id = state['pending_draft']['draft_id']
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM tickets').fetchone()[0], 0)
        self.assertEqual(self.http.post(f'/api/sessions/{sid}/confirm', json={'draft_id': 'D-wrong', 'approved': True}).status_code, 409)
        self.assertEqual(self.http.post(f'/api/sessions/{sid}/messages', json={'message': '又一个问题'}).status_code, 409)
        self.assertEqual(self.http.post(f'/api/sessions/{sid}/confirm', json={'draft_id': draft_id, 'approved': True}).status_code, 200)
        state = self.until(sid, lambda s: not s['busy'])
        results = [e for e in state['events'] if e['type'] == 'tool_result' and e['name'] == 'create_ticket']
        self.assertEqual(results[0]['data']['ticket']['ticket_id'], 'T-000001')
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM tickets').fetchone()[0], 1)
        self.assertEqual(self.http.post(f'/api/sessions/{sid}/confirm', json={'draft_id': draft_id, 'approved': True}).status_code, 409)

    def test_cancel_keeps_draft_unapproved_and_allows_next_message(self):
        sid = self.session()
        self.send(sid, '登记PDF故障')
        draft = self.until(sid, lambda s: s['pending_draft'] is not None)['pending_draft']
        self.http.post(f'/api/sessions/{sid}/confirm', json={'draft_id': draft['draft_id'], 'approved': False})
        self.until(sid, lambda s: not s['busy'])
        self.assertIsNone(self.store.get_draft(draft['draft_id'], self.actor)['approved_at'])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM tickets').fetchone()[0], 0)
        self.send(sid, '继续提问')
        self.until(sid, lambda s: not s['busy'])

    def test_cross_origin_writes_and_blank_messages_are_rejected(self):
        self.assertEqual(self.http.post('/api/sessions', headers={'Origin': 'https://unrelated.invalid'}).status_code, 403)
        sid = self.session()
        self.assertEqual(self.http.post(f'/api/sessions/{sid}/messages', json={'message': '   '}).status_code, 400)
        self.assertEqual(self.http.post(f'/api/sessions/{sid}/confirm', json={'draft_id': 'D-test', 'approved': 'false'}).status_code, 400)

    def test_upload_reconnects_mcp_preserving_conversation_and_rejects_pending_draft(self):
        sid = self.session()
        self.send(sid, '我叫小明')
        self.until(sid, lambda s: not s['busy'])
        response = self.http.post('/api/knowledge/upload', files={'file': ('new.txt', b'new knowledge')})
        self.assertEqual(response.status_code, 201, response.text)
        self.until(sid, lambda s: s['ready'])
        self.send(sid, '我叫什么？')
        state = self.until(sid, lambda s: not s['busy'])
        self.assertIn('我叫小明', [e for e in state['events'] if e['type'] == 'message'][-1]['content'])
        self.send(sid, '登记问题')
        self.until(sid, lambda s: s['pending_draft'] is not None)
        response = self.http.post('/api/knowledge/upload', files={'file': ('blocked.txt', b'blocked')})
        self.assertEqual(response.status_code, 409)


if __name__ == '__main__':
    unittest.main()

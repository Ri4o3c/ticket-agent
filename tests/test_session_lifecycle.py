"""真实HTTP路由、临时SQLite与固定模型验证会话竞态；不连接外部模型。"""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import httpx
from fastmcp import Client
from langchain_core.messages import ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_tools import make_server
from ticket_store import Actor, TicketStore
from web_app import create_app
from test_web_app import WebModel


class DelayedBody(httpx.AsyncByteStream):
    """停在真正的ASGI receive边界，保证请求已经开始但body尚未完成。"""
    def __init__(self, data):
        self.data = json.dumps(data).encode()
        self.reading = asyncio.Event()
        self.release = asyncio.Event()

    async def __aiter__(self):
        self.reading.set()
        await self.release.wait()
        yield self.data


class TrackedClient:
    """保留真实MCP上下文，仅给退出阶段加测试门闩并记录worker归属。"""
    def __init__(self, owner):
        self.owner = owner
        self.client = Client(make_server(store=owner.store, actor=owner.actor, searcher=lambda q, k: []))
        self.task = None
        self.closed = False

    async def __aenter__(self):
        self.task = asyncio.current_task()
        return await self.client.__aenter__()

    async def __aexit__(self, *args):
        self.owner.closing.set()
        await self.owner.allow_close.wait()
        try:
            return await self.client.__aexit__(*args)
        finally:
            self.closed = True


class SessionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TicketStore(Path(self.temp.name) / 'tickets.sqlite3')
        self.actor = Actor('lifecycle-user', 'lifecycle-workspace')
        self.clients = []
        self.models = []
        self.closing = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.allow_close.set()
        self.lifespans = []
        self.http_clients = []

    async def asyncTearDown(self):
        self.allow_close.set()
        for http in self.http_clients:
            await http.aclose()
        for lifespan in reversed(self.lifespans):
            await lifespan.__aexit__(None, None, None)
        # 原缺陷会把worker从sessions清空却没有关闭；失败测试也必须回收这些任务。
        tasks = [client.task for client in self.clients if client.task and not client.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.temp.cleanup()

    async def app(self, *, model=None, **kwargs):
        def client_factory(kb_id):
            client = TrackedClient(self)
            self.clients.append(client)
            return client

        def model_factory(url):
            chosen = model or WebModel()
            self.models.append(chosen)
            return chosen

        settings = SimpleNamespace(ready=lambda: True, save=lambda data: {'saved': True})
        app = create_app(client_factory=client_factory, model_factory=model_factory,
                         store_factory=lambda: self.store, actor_factory=lambda: self.actor,
                         settings=settings,
                         knowledge_factory=lambda kb: SimpleNamespace(upload=lambda name, data: {'filename': name, 'duplicate': False}),
                         **kwargs)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        self.lifespans.append(lifespan)
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver')
        self.http_clients.append(http)
        return http

    async def until(self, http, sid, predicate):
        async with asyncio.timeout(5):
            while True:
                state = (await http.get(f'/api/sessions/{sid}/events')).json()
                if predicate(state):
                    return state
                await asyncio.sleep(.01)

    async def session(self, http):
        response = await http.post('/api/sessions')
        self.assertEqual(response.status_code, 201, response.text)
        sid = response.json()['session_id']
        await self.until(http, sid, lambda state: state['ready'])
        return sid

    async def test_settings_cannot_clear_a_session_registered_after_body_read(self):
        http = await self.app()
        sid = await self.session(http)
        body = DelayedBody({})
        creating = asyncio.create_task(http.post('/api/sessions', content=body, headers={'content-type': 'application/json'}))
        await body.reading.wait()
        self.allow_close.clear()
        saving = asyncio.create_task(http.post('/api/settings', json={}))
        await self.closing.wait()
        body.release.set()
        response = await asyncio.wait_for(creating, 2)
        self.allow_close.set()
        self.assertEqual((await saving).status_code, 200)
        self.assertEqual(response.status_code, 409, '配置关闭快照期间不得新建201后被清空且留下worker')
        self.assertEqual((await http.get(f'/api/sessions/{sid}/events')).status_code, 404)
        self.assertTrue(all(client.closed for client in self.clients))

    async def test_upload_prevents_registration_that_started_before_upload(self):
        http = await self.app()
        sid = await self.session(http)
        body = DelayedBody({})
        creating = asyncio.create_task(http.post('/api/sessions', content=body, headers={'content-type': 'application/json'}))
        await body.reading.wait()
        self.allow_close.clear()
        uploading = asyncio.create_task(http.post('/api/knowledge/upload', files={'file': ('test.txt', b'knowledge')}))
        await self.closing.wait()
        body.release.set()
        response = await asyncio.wait_for(creating, 2)
        blocked_send = await http.post(f'/api/sessions/{sid}/messages', json={'message': '关闭期间不能发送'})
        self.allow_close.set()
        result = await asyncio.gather(uploading, return_exceptions=True)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(blocked_send.status_code, 409)
        self.assertIsInstance(result[0], httpx.Response)
        self.assertEqual(result[0].status_code, 201)
        await self.until(http, sid, lambda state: state['ready'])
        self.assertTrue(self.clients[0].closed)

    async def test_delayed_message_rechecks_session_after_delete(self):
        http = await self.app()
        sid = await self.session(http)
        body = DelayedBody({'message': '已经删除的会话不能接收问题'})
        sending = asyncio.create_task(http.post(f'/api/sessions/{sid}/messages', content=body, headers={'content-type': 'application/json'}))
        await body.reading.wait()
        self.assertEqual((await http.delete(f'/api/sessions/{sid}')).status_code, 200)
        body.release.set()
        self.assertEqual((await sending).status_code, 404)
        self.assertTrue(self.clients[0].closed)

    async def test_confirmation_expires_releases_busy_and_rejects_late_approval(self):
        http = await self.app(confirmation_timeout=.15)
        sid = await self.session(http)
        await http.post(f'/api/sessions/{sid}/messages', json={'message': '登记PDF故障'})
        draft = (await self.until(http, sid, lambda state: state['pending_draft'] is not None))['pending_draft']
        state = await self.until(http, sid, lambda state: not state['busy'])
        decisions = [event for event in state['events'] if event['type'] == 'decision']
        self.assertEqual(decisions[-1]['reason'], 'expired')
        self.assertFalse(decisions[-1]['approved'])
        self.assertTrue(any(event['type'] == 'status' and '超时' in event['content'] for event in state['events']))
        self.assertIsNone(state['pending_draft'])
        self.assertEqual((await http.post(f'/api/sessions/{sid}/confirm', json={'draft_id': draft['draft_id'], 'approved': True})).status_code, 409)
        self.assertIsNone(self.store.get_draft(draft['draft_id'], self.actor)['approved_at'])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM tickets').fetchone()[0], 0)
        self.assertEqual((await http.post(f'/api/sessions/{sid}/messages', json={'message': '继续提问'})).status_code, 202)
        await self.until(http, sid, lambda state: not state['busy'])

    async def test_delete_rejects_a_model_call_still_in_progress(self):
        started, release = asyncio.Event(), asyncio.Event()

        class PausedModel(WebModel):
            async def ainvoke(self, messages):
                started.set()
                await release.wait()
                return await super().ainvoke(messages)

        http = await self.app(model=PausedModel())
        sid = await self.session(http)
        await http.post(f'/api/sessions/{sid}/messages', json={'message': '普通问题'})
        await asyncio.wait_for(started.wait(), 2)
        try:
            self.assertEqual((await http.delete(f'/api/sessions/{sid}')).status_code, 409)
            self.assertFalse(self.clients[0].closed)
        finally:
            release.set()
        await self.until(http, sid, lambda state: not state['busy'])
        self.assertEqual((await http.delete(f'/api/sessions/{sid}')).status_code, 200)
        self.assertTrue(self.clients[0].closed)

    async def test_deleting_waiting_approval_finishes_tool_history_and_closes_worker(self):
        http = await self.app()
        sid = await self.session(http)
        await http.post(f'/api/sessions/{sid}/messages', json={'message': '登记PDF故障'})
        draft = (await self.until(http, sid, lambda state: state['pending_draft'] is not None))['pending_draft']
        # 从运行任务取到真实会话对象只用于检查历史协议收尾，不替换run_turn或持久化层。
        session = self.clients[0].task.get_coro().cr_frame.f_locals['self']
        response = await http.delete(f'/api/sessions/{sid}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await http.get(f'/api/sessions/{sid}/events')).status_code, 404)
        self.assertTrue(self.clients[0].closed)
        cancelled = [message for message in session.messages if isinstance(message, ToolMessage) and message.name == 'prepare_ticket']
        self.assertEqual(len(cancelled), 1)
        self.assertTrue(json.loads(cancelled[0].content)['cancelled'])
        self.assertIsNone(self.store.get_draft(draft['draft_id'], self.actor)['approved_at'])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM tickets').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()

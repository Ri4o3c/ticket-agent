"""审批异常、持久核查屏障和输出护栏：真实临时SQLite，模型与网络用受控替身。"""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_step1 as agent
from langchain_core.messages import AIMessage, ToolMessage
from ticket_store import Actor, TicketStore
from test_agent_completion import LocalToolClient
from test_agent_errors import Responses, CreateModel, LostCreateReply


def request(name, args, call_id):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': call_id}])


class LookupClient(LocalToolClient):
    def __init__(self, store, actor):
        super().__init__(store, actor)
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == 'find_ticket_by_request':
            ticket = self.store.find_ticket_by_request(args['request_id'], self.actor)
            return SimpleNamespace(data={'ok': True, 'found': ticket is not None, 'ticket': ticket})
        if name == 'get_ticket':
            return SimpleNamespace(data={'ok': True, 'ticket': self.store.get_ticket(args['ticket_id'], self.actor)})
        return await super().call_tool(name, args)


class ReliabilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'tickets.sqlite3'
        self.store = TicketStore(self.path)
        self.actor = Actor('reliability-user', 'reliability-space')
        self.events = []

    async def run_flow(self, model, *, client=None, store=None, messages=None, question='登记PDF问题', decision=True):
        history = [] if messages is None else messages
        async def confirm(draft):
            return decision
        async def emit(event):
            self.events.append(event)
        with contextlib.redirect_stdout(io.StringIO()):
            await agent.run_turn(client or LookupClient(self.store, self.actor), model, history, question,
                confirm_draft=confirm, on_event=emit, store=store or self.store, actor=self.actor)
        return history

    def counts(self):
        with self.store.connect() as db:
            return (db.execute('SELECT count(*) FROM drafts').fetchone()[0],
                    db.execute('SELECT count(*) FROM tickets').fetchone()[0])

    async def lose_create_reply(self, commit_first):
        client = LostCreateReply(self.store, self.actor, commit_first)
        messages = await self.run_flow(CreateModel(client), client=client)
        prepared = next(json.loads(m.content)['draft'] for m in messages
                        if isinstance(m, ToolMessage) and m.name == 'prepare_ticket')
        return prepared

    async def test_approval_database_error_completes_all_tool_results_without_writing_ticket(self):
        class BrokenApproval(TicketStore):
            def approve_draft(self, *args):
                raise sqlite3.OperationalError('simulated private database detail')
        store = BrokenApproval(self.path)
        calls = AIMessage(content='', tool_calls=[
            {'name': 'prepare_ticket', 'args': {'title': 'PDF失败', 'description': '5MB PDF报E103'}, 'id': 'prepare'},
            {'name': 'create_ticket', 'args': {'draft_id': 'D-not-approved'}, 'id': 'must-skip'},
        ])
        history = []
        try:
            await self.run_flow(Responses(calls), store=store, messages=history)
        except Exception as exc:
            self.fail(f'审批异常未被工具结果收尾：{type(exc).__name__}')
        results = {m.tool_call_id: json.loads(m.content) for m in history if isinstance(m, ToolMessage)}
        self.assertEqual(set(results), {'prepare', 'must-skip'})
        self.assertFalse(results['prepare']['ok'])
        self.assertTrue(results['must-skip']['skipped'])
        self.assertEqual(self.counts(), (1, 0))
        self.assertNotIn('simulated private', json.dumps(self.events))

    async def test_confirmation_callback_error_keeps_history_valid(self):
        async def broken_confirm(draft):
            raise TimeoutError('simulated private callback detail')
        history = []
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                await agent.run_turn(LookupClient(self.store, self.actor), Responses(request('prepare_ticket',
                    {'title': 'PDF失败', 'description': 'E103'}, 'prepare')), history, '登记问题',
                    confirm_draft=broken_confirm, store=self.store, actor=self.actor)
            except Exception as exc:
                self.fail(f'确认回调异常没有结果消息：{type(exc).__name__}')
        self.assertEqual([m.tool_call_id for m in history if isinstance(m, ToolMessage)], ['prepare'])
        self.assertEqual(self.counts(), (1, 0))

    async def test_structured_storage_error_stops_the_remaining_batch(self):
        class CommitThenError(LocalToolClient):
            async def call_tool(inner, name, args):
                if name == 'create_ticket':
                    await super().call_tool(name, args)
                    return SimpleNamespace(data={'ok': False, 'error': 'storage_error', 'message': '数据库操作未确认'})
                return await super().call_tool(name, args)
        client = CommitThenError(self.store, self.actor)
        history = await self.run_flow(CreateModel(client), client=client)
        results = {m.tool_call_id: json.loads(m.content) for m in history if isinstance(m, ToolMessage)}
        self.assertTrue(results['create'].get('outcome_unknown'))
        self.assertTrue(results['extra'].get('skipped'))
        self.assertEqual(self.counts(), (1, 1))

    async def test_unknown_creation_blocks_a_new_session_even_after_reopening_database(self):
        await self.lose_create_reply(True)
        reopened = TicketStore(self.path)
        new = request('prepare_ticket', {'title': '同一故障', 'description': '不能换请求重新建单'}, 'new-draft')
        history = await self.run_flow(Responses(new, AIMessage(content='请先核查原请求。')),
            store=reopened, client=LookupClient(reopened, self.actor), decision=False)
        result = next(json.loads(m.content) for m in history if isinstance(m, ToolMessage))
        self.assertEqual(self.counts(), (1, 1))
        self.assertEqual(result.get('error'), 'recovery_required')

    async def test_correct_lookup_releases_the_barrier_without_creating_a_duplicate(self):
        draft = await self.lose_create_reply(True)
        await self.run_flow(Responses(request('find_ticket_by_request', {'request_id': draft['request_id']}, 'lookup'),
            AIMessage(content='原请求已经生成工单 T-000001。')), question='核查原创建请求')
        new = request('prepare_ticket', {'title': '另一个问题', 'description': '独立问题'}, 'allowed-new')
        history = await self.run_flow(Responses(new), decision=False)
        result = next(json.loads(m.content) for m in history if isinstance(m, ToolMessage))
        self.assertTrue(result.get('cancelled'))
        self.assertEqual(self.counts(), (2, 1))

    async def test_lookup_not_found_allows_only_retrying_the_original_draft(self):
        draft = await self.lose_create_reply(False)
        model = Responses(
            request('find_ticket_by_request', {'request_id': draft['request_id']}, 'lookup'),
            request('create_ticket', {'draft_id': draft['draft_id']}, 'retry'),
            AIMessage(content='已创建工单 T-000001。'),
        )
        await self.run_flow(model, question='核查并继续原请求')
        self.assertEqual(self.counts(), (1, 1))
        ticket = self.store.find_ticket_by_request(draft['request_id'], self.actor)
        self.assertEqual(ticket['draft_id'], draft['draft_id'])

    async def test_checkpoint_failure_prevents_the_external_create_call(self):
        draft = self.store.approve_draft(self.store.prepare_draft('PDF失败', 'E103', self.actor)['draft_id'], self.actor)
        class CannotCheckpoint(TicketStore):
            def start_creation(self, *args):
                raise sqlite3.OperationalError('checkpoint unavailable')
        client = LookupClient(self.store, self.actor)
        await self.run_flow(Responses(request('create_ticket', {'draft_id': draft['draft_id']}, 'create'),
            AIMessage(content='尚未创建工单。')), client=client, store=CannotCheckpoint(self.path))
        self.assertNotIn('create_ticket', [name for name, args in client.calls])
        self.assertEqual(self.counts(), (1, 0))

    async def test_cancellation_during_create_preserves_recovery_and_completes_tool_messages(self):
        draft = self.store.approve_draft(self.store.prepare_draft('PDF失败', 'E103', self.actor)['draft_id'], self.actor)
        class InterruptedClient(LocalToolClient):
            async def call_tool(inner, name, args):
                if name == 'create_ticket':
                    await super().call_tool(name, args)
                    raise asyncio.CancelledError()
                return await super().call_tool(name, args)
        history = []
        with self.assertRaises(asyncio.CancelledError):
            await self.run_flow(Responses(request('create_ticket', {'draft_id': draft['draft_id']}, 'interrupted')),
                client=InterruptedClient(self.store, self.actor), messages=history)
        self.assertEqual([m.tool_call_id for m in history if isinstance(m, ToolMessage)], ['interrupted'])
        await self.run_flow(Responses(request('prepare_ticket', {'title': '不应执行', 'description': 'E103'}, 'new'),
            AIMessage(content='请先核查原请求。')), store=TicketStore(self.path), decision=False)
        self.assertEqual(self.counts(), (1, 1))

    async def test_unseen_citation_is_not_delivered_to_the_user(self):
        await self.run_flow(Responses(AIMessage(content='PDF最大99MB [KB-999]。'),
            AIMessage(content='当前资料不足，无法确定。')), question='单个PDF最大支持多大？')
        delivered = [e['content'] for e in self.events if e['type'] == 'message']
        self.assertNotIn('PDF最大99MB [KB-999]。', delivered)
        self.assertTrue(delivered)

    async def test_fabricated_creation_is_not_delivered_without_a_successful_tool(self):
        await self.run_flow(Responses(AIMessage(content='已创建工单 T-999999。'),
            AIMessage(content='尚未创建工单。')))
        self.assertNotIn('已创建工单 T-999999。', [e['content'] for e in self.events if e['type'] == 'message'])

    async def test_current_status_requires_a_fresh_query_and_matches_its_result(self):
        draft = self.store.approve_draft(self.store.prepare_draft('PDF失败', 'E103', self.actor)['draft_id'], self.actor)
        ticket = self.store.create_ticket(draft['draft_id'], self.actor)
        history = [ToolMessage(content=json.dumps({'ok': True, 'ticket': ticket}), tool_call_id='old', name='get_ticket')]
        client = LookupClient(self.store, self.actor)
        await self.run_flow(Responses(AIMessage(content='工单 T-000001 当前已解决。'),
            request('get_ticket', {'ticket_id': 'T-000001'}, 'fresh'),
            AIMessage(content='工单 T-000001 当前状态为待处理。')),
            question='工单T-000001当前处理状态？', client=client, messages=history)
        delivered = [e['content'] for e in self.events if e['type'] == 'message']
        self.assertEqual([name for name, args in client.calls], ['get_ticket'])
        self.assertEqual(delivered, ['工单 T-000001 当前状态为待处理。'])

    async def test_valid_historical_sources_support_a_followup_without_retrieval(self):
        history = [request('search_docs', {'question': 'PDF限制'}, 'previous'),
            ToolMessage(content=json.dumps({'ok': True, 'results': [
                {'knowledge_id': 'KB-022', 'text': '最大10MB。'},
                {'knowledge_id': 'KB-023', 'text': '最多200页。'}]}), name='search_docs', tool_call_id='previous'),
            AIMessage(content='最大10MB [KB-022]。')]
        client = LookupClient(self.store, self.actor)
        await self.run_flow(Responses(AIMessage(content='最多200页 [KB-023]。')),
            client=client, messages=history, question='那页数最多多少？')
        self.assertEqual(client.calls, [])
        self.assertEqual([e['content'] for e in self.events if e['type'] == 'message'], ['最多200页 [KB-023]。'])

    async def test_unclassified_create_failure_keeps_the_recovery_barrier(self):
        draft = self.store.approve_draft(self.store.prepare_draft('PDF失败', 'E103', self.actor)['draft_id'], self.actor)
        class CommitThenUnclassified(LocalToolClient):
            async def call_tool(inner, name, args):
                await super().call_tool(name, args)
                return SimpleNamespace(data={'ok': False, 'error': 'internal_error'})
        history = await self.run_flow(Responses(request('create_ticket', {'draft_id': draft['draft_id']}, 'create')),
            client=CommitThenUnclassified(self.store, self.actor))
        result = next(json.loads(m.content) for m in history if isinstance(m, ToolMessage))
        self.assertTrue(result.get('outcome_unknown'))
        self.assertEqual(len(TicketStore(self.path).pending_creations(self.actor)), 1)

    async def test_independent_mcp_cannot_create_another_draft_during_recovery(self):
        from fastmcp import Client
        from mcp_tools import make_server
        first = self.store.approve_draft(self.store.prepare_draft('原问题', 'E103', self.actor)['draft_id'], self.actor)
        second = self.store.approve_draft(self.store.prepare_draft('另一问题', 'E104', self.actor)['draft_id'], self.actor)
        self.store.start_creation(first['draft_id'], self.actor)
        async with Client(make_server(store=self.store, actor=self.actor, searcher=lambda q,k: [])) as client:
            result = await client.call_tool('create_ticket', {'draft_id': second['draft_id']})
        self.assertFalse(result.data['ok'])
        self.assertEqual(self.counts(), (2, 0))

    async def test_asking_user_to_check_does_not_exempt_a_factual_claim(self):
        cases = [('工单T-000001当前状态？', '工单 T-000001 当前已解决，请核对。'),
                 ('PDF最大多大？', 'PDF最大99MB，请核对。')]
        for question, unsafe in cases:
            with self.subTest(question=question):
                self.events.clear()
                await self.run_flow(Responses(AIMessage(content=unsafe), AIMessage(content='当前资料不足，无法确定。')),
                    question=question)
                self.assertNotIn(unsafe, [e['content'] for e in self.events if e['type'] == 'message'])


if __name__ == '__main__':
    unittest.main()

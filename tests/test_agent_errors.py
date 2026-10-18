# 分别模拟查询异常、模型异常、创建提交前/后的结果丢失。
# 重点验证不会把未知结果伪造成未查到，并保留原D-/R-标识供后续核查。
"""故障注入：区分查询异常和建单结果未知，不使用真实模型 API。"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_step1 as agent
from langchain_core.messages import AIMessage, ToolMessage
from ticket_store import Actor, TicketStore
from test_agent_completion import LocalToolClient


class Responses:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.histories = []

    async def ainvoke(self, messages):
        self.histories.append(list(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FailedQuery:
    async def call_tool(self, name, args):
        raise TimeoutError('private-network-detail')


class LostCreateReply(LocalToolClient):
    def __init__(self, store, actor, commit_first):
        super().__init__(store, actor)
        self.commit_first = commit_first
        self.create_attempts = 0

    async def call_tool(self, name, args):
        if name == 'create_ticket':
            self.create_attempts += 1
            if self.commit_first:
                await super().call_tool(name, args)
            raise TimeoutError('private-network-detail')
        return await super().call_tool(name, args)


class CreateModel:
    def __init__(self, client):
        self.client = client
        self.rounds = 0

    async def ainvoke(self, messages):
        self.rounds += 1
        if self.rounds == 1:
            return AIMessage(content='', tool_calls=[{
                'name': 'prepare_ticket', 'args': {'title': 'PDF失败', 'description': '3MB PDF报E103'}, 'id': 'prepare',
            }])
        if self.rounds == 2:
            draft_result = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
            draft_id = json.loads(draft_result.content)['draft']['draft_id']
            return AIMessage(content='', tool_calls=[
                {'name': 'create_ticket', 'args': {'draft_id': draft_id}, 'id': 'create'},
                {'name': 'prepare_ticket', 'args': {'title': '不应执行', 'description': '不应执行'}, 'id': 'extra'},
            ])
        raise AssertionError('结果未知后应先暂停本次建单，不能继续盲目重建')


class ErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_exception_reaches_model_without_fabricating_not_found(self):
        request = AIMessage(content='', tool_calls=[{
            'name': 'find_ticket_by_request', 'args': {'request_id': 'R-test'}, 'id': 'lookup',
        }])
        model = Responses(request, AIMessage(content='暂时无法核查。'))
        messages = []
        with contextlib.redirect_stdout(io.StringIO()) as output:
            await agent.run_turn(FailedQuery(), model, messages, '核查创建请求')
        tool = next(m for m in model.histories[-1] if isinstance(m, ToolMessage))
        data = json.loads(tool.content)
        self.assertFalse(data['ok'])
        self.assertNotIn('found', data)
        self.assertEqual(tool.tool_call_id, 'lookup')
        self.assertNotIn('private-network-detail', output.getvalue())

    async def test_create_timeout_preserves_ids_and_stops_remaining_tools(self):
        for commit_first in (True, False):
            with self.subTest(commit_first=commit_first), tempfile.TemporaryDirectory() as directory:
                store = TicketStore(Path(directory) / 'tickets.sqlite3')
                actor = Actor('test-user', 'test-workspace')
                client = LostCreateReply(store, actor, commit_first)
                model = CreateModel(client)
                messages = []
                with patch.object(agent, 'TicketStore', return_value=store), \
                     patch.object(agent, 'local_actor', return_value=actor), \
                     patch('builtins.input', return_value='确认'), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    await agent.run_turn(client, model, messages, '登记PDF问题')
                tools = [m for m in messages if isinstance(m, ToolMessage)]
                self.assertEqual({m.tool_call_id for m in tools}, {'prepare', 'create', 'extra'})
                data = json.loads(next(m.content for m in tools if m.tool_call_id == 'create'))
                self.assertTrue(data['outcome_unknown'])
                self.assertTrue(data['draft_id'].startswith('D-'))
                self.assertTrue(data['request_id'].startswith('R-'))
                self.assertEqual(client.create_attempts, 1)
                ticket = store.find_ticket_by_request(data['request_id'], actor)
                self.assertEqual(ticket is not None, commit_first)
                with store.connect() as db:
                    self.assertEqual(db.execute('SELECT count(*) FROM drafts').fetchone()[0], 1)
                self.assertIn(data['request_id'], output.getvalue())

    async def test_chat_exception_ends_only_current_turn(self):
        messages = []
        with contextlib.redirect_stdout(io.StringIO()) as output:
            await agent.run_turn(SimpleNamespace(), Responses(ConnectionError('private-detail')), messages, '你好')
        self.assertIn(('human', '你好'), messages)
        self.assertIn('模型调用失败', output.getvalue())
        self.assertNotIn('private-detail', output.getvalue())


if __name__ == '__main__':
    unittest.main()

# 用固定模型回复和本地工具替身验证批准后继续建单、白名单及成功判断；
# 不测试真实聊天模型的随机决策，也不使用第三方付费API。
"""用脚本化模型复现提前结束；工单写入使用真实的临时 SQLite。"""
import contextlib
import io
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


class LocalToolClient:
    def __init__(self, store, actor):
        self.store, self.actor = store, actor

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def list_tools(self):
        return [SimpleNamespace(name=name, description='test', input_schema={'type': 'object'})
                for name in ('prepare_ticket', 'create_ticket', 'get_ticket')]

    async def call_tool(self, name, args):
        if name == 'prepare_ticket':
            data = {'ok': True, 'draft': self.store.prepare_draft(**args, actor=self.actor)}
        elif name == 'create_ticket':
            data = {'ok': True, 'ticket': self.store.create_ticket(**args, actor=self.actor)}
        else:
            raise AssertionError(f'Unexpected tool: {name}')
        return SimpleNamespace(data=data)


class ScriptedModel:
    def __init__(self, stubborn=False):
        self.rounds = 0
        self.stubborn = stubborn
        self.first_question = None

    def bind_tools(self, definitions):
        return self

    async def ainvoke(self, messages):
        self.rounds += 1
        if self.rounds == 1:
            self.first_question = next(m[1] for m in messages if isinstance(m, tuple) and m[0] == 'human')
            return AIMessage(content='', tool_calls=[{
                'name': 'prepare_ticket', 'args': {'title': 'PDF失败', 'description': '3MB PDF报E103'},
                'id': 'prepare-1',
            }])
        if self.rounds == 2 or self.stubborn:
            return AIMessage(content='请确认以上信息，确认后我将创建工单。')
        if self.rounds == 3:
            import json
            tool = next(m for m in messages if isinstance(m, ToolMessage) and m.name == 'prepare_ticket')
            draft_id = json.loads(tool.content)['draft']['draft_id']
            return AIMessage(content='', tool_calls=[{
                'name': 'create_ticket', 'args': {'draft_id': draft_id}, 'id': 'create-1',
            }])
        return AIMessage(content='正式工单已创建。')


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def run_flow(self, decision='确认', stubborn=False, question_lines=None):
        with tempfile.TemporaryDirectory() as directory:
            store = TicketStore(Path(directory) / 'tickets.sqlite3')
            actor = Actor('test-user', 'test-workspace')
            model = ScriptedModel(stubborn)
            output = io.StringIO()
            with patch.object(agent, 'Client', return_value=LocalToolClient(store, actor)), \
                 patch.object(agent, 'StdioTransport', return_value=object()), \
                 patch.object(agent, 'make_chat_model', return_value=model), \
                 patch.object(agent, 'load_dotenv'), \
                 patch.object(agent, 'TicketStore', return_value=store), \
                 patch.object(agent, 'local_actor', return_value=actor), \
                 patch.dict('os.environ', {'CHAT_BASE_URL': 'https://test.invalid/v1'}), \
                 patch('builtins.input', side_effect=[*(question_lines or ['登记测试问题']), 'END', decision, 'EXIT']), \
                 patch('sys.stdin', io.StringIO()), contextlib.redirect_stdout(output):
                await agent.main()
            if question_lines is not None:
                self.assertEqual(model.first_question, '\n'.join(question_lines))
            with store.connect() as db:
                tickets = [dict(row) for row in db.execute('SELECT * FROM tickets')]
                approved = db.execute('SELECT count(*) FROM drafts WHERE approved_at IS NOT NULL').fetchone()[0]
            return tickets, approved, model.rounds, output.getvalue()

    async def test_approved_draft_is_created_after_model_reasks_confirmation(self):
        tickets, approved, rounds, output = await self.run_flow()
        self.assertEqual(len(tickets), 1, '模型再次询问确认不能导致已批准任务提前结束')
        self.assertEqual(approved, 1)
        self.assertEqual(tickets[0]['description'], '3MB PDF报E103')
        self.assertEqual(tickets[0]['status'], '待处理')
        self.assertEqual(rounds, 4)
        self.assertNotIn('请确认以上信息，确认后我将创建工单。', output)

    async def test_cancel_creates_neither_approval_nor_formal_ticket(self):
        tickets, approved, rounds, _ = await self.run_flow(decision='取消')
        self.assertEqual(tickets, [])
        self.assertEqual(approved, 0)
        self.assertEqual(rounds, 1)

    async def test_multiline_question_reaches_model_and_confirmation_stays_separate(self):
        tickets, approved, _, _ = await self.run_flow(question_lines=[
            '帮我登记测试工单：3MB PDF报E103。',
            '文件可选择文字，重传仍然失败。',
        ])
        self.assertEqual(len(tickets), 1)
        self.assertEqual(approved, 1)

    async def test_stubborn_model_stops_at_budget_and_reports_unfinished_creation(self):
        tickets, approved, rounds, output = await self.run_flow(stubborn=True)
        self.assertEqual(tickets, [])
        self.assertEqual(approved, 1)
        self.assertEqual(rounds, 5)
        self.assertIn('尚未获得正式工单创建成功结果', output)


if __name__ == '__main__':
    unittest.main()

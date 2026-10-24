# 验证CLI输入提交、会话历史保留、单题轮数上限及取消后继续提问。
# 模拟输入与模型回复，断言实际传入的历史，而不是依赖真实用户控制台操作。
"""连续提问、取消和退出的会话边界；模型使用固定回复，不访问 API。"""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_step1 as agent
from langchain_core.messages import AIMessage, ToolMessage


class EmptyClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def list_tools(self):
        return [SimpleNamespace(name='get_ticket', description='test', input_schema={'type': 'object'})]

    async def call_tool(self, name, args):
        if name == 'get_ticket':
            return SimpleNamespace(data={'ok': False, 'message': '工单不存在或当前用户无权访问'})
        if name == 'prepare_ticket':
            return SimpleNamespace(data={'ok': True, 'draft': {
                'draft_id': 'D-test', 'title': '测试', 'description': '测试描述',
            }})
        raise AssertionError('取消后不应执行后续工具')


class Replies:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.histories = []

    def bind_tools(self, definitions):
        return self

    async def ainvoke(self, messages):
        self.histories.append(list(messages))
        return next(self.responses)


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def run_session(self, inputs, responses):
        model = Replies(responses)
        with patch.object(agent, 'Client', return_value=EmptyClient()), \
             patch.object(agent, 'StdioTransport', return_value=object()), \
             patch.object(agent, 'make_chat_model', return_value=model), \
             patch.object(agent, 'load_dotenv'), \
             patch.dict('os.environ', {'CHAT_BASE_URL': 'https://test.invalid/v1'}), \
             patch('builtins.input', side_effect=inputs), \
             patch('sys.stdin', io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            await agent.main()
        return model.histories

    async def test_followup_includes_previous_question_and_answer(self):
        # 此测试只验证历史保留；产品事实现在必须有检索引用，改用不依赖知识库的姓名追问。
        histories = await self.run_session(
            ['我叫小明', 'END', '我叫什么？', 'END', 'EXIT'],
            [AIMessage(content='你好，小明。'), AIMessage(content='你叫小明。')],
        )
        self.assertEqual(len(histories), 2)
        second = histories[1]
        self.assertIn(('human', '我叫小明'), second)
        self.assertIn(('human', '我叫什么？'), second)
        self.assertTrue(any(isinstance(m, AIMessage) and m.content == '你好，小明。' for m in second))
        self.assertEqual(sum(isinstance(m, tuple) and m[0] == 'system' for m in second), 1)

    async def test_exit_does_not_send_command_to_model(self):
        histories = await self.run_session(['EXIT', 'END'], [AIMessage(content='不应调用')])
        self.assertEqual(histories, [])

    async def test_empty_submission_allows_next_question(self):
        histories = await self.run_session(
            ['END', '查看工单', 'END', 'EXIT'], [AIMessage(content='请提供编号。')],
        )
        self.assertEqual(len(histories), 1)
        self.assertIn(('human', '查看工单'), histories[0])

    async def test_unsubmitted_input_is_discarded_on_eof(self):
        histories = await self.run_session(['尚未提交的问题', EOFError()], [])
        self.assertEqual(histories, [])

    async def test_round_limit_applies_per_question_and_next_question_can_continue(self):
        repeated_request = AIMessage(content='', tool_calls=[{
            'name': 'get_ticket', 'args': {'ticket_id': 'T-999999'}, 'id': 'lookup',
        }])
        histories = await self.run_session(
            ['查询不存在的工单', 'END', '先问别的', 'END', 'EXIT'],
            [repeated_request] * 5 + [AIMessage(content='可以继续提问。')],
        )
        self.assertEqual(len(histories), 6)
        self.assertIn(('human', '先问别的'), histories[-1])
        self.assertEqual(sum(isinstance(m, ToolMessage) for m in histories[-1]), 5)

    async def test_cancel_resolves_all_tool_calls_and_accepts_next_question(self):
        request = AIMessage(content='', tool_calls=[
            {'name': 'prepare_ticket', 'args': {'title': '测试', 'description': '测试描述'}, 'id': 'prepare-1'},
            {'name': 'create_ticket', 'args': {'draft_id': 'D-test'}, 'id': 'create-1'},
        ])
        histories = await self.run_session(
            ['登记问题', 'END', '取消', '接着问别的', 'END', 'EXIT'],
            [request, AIMessage(content='请说。')],
        )
        self.assertEqual(len(histories), 2)
        import json
        results = [m for m in histories[1] if isinstance(m, ToolMessage)]
        self.assertEqual({m.tool_call_id for m in results}, {'prepare-1', 'create-1'})
        self.assertTrue(all(json.loads(m.content).get('cancelled') for m in results))
        self.assertIn(('human', '接着问别的'), histories[1])


if __name__ == '__main__':
    unittest.main()

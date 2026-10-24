"""真实模型入口的离线验收：真实ChatOpenAI协议适配器，HTTP边界受控，无付费网络请求。"""
import contextlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_one_case as evaluation


class LiveRouteTests(unittest.IsolatedAsyncioTestCase):
    async def run_live(self, handler, case_id='E05'):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            model = ChatOpenAI(model='offline-chat-protocol-test', api_key='TEST-ONLY-KEY',
                               base_url='https://offline.invalid/v1', http_async_client=http_client,
                               max_retries=0, timeout=1)
            # 工厂初始化仍由live入口决定；只替换外部网络和配置来源。
            with patch.object(evaluation, 'make_chat_model', return_value=model, create=True), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                runner = {'E04': evaluation.run_e04, 'E05': evaluation.run_e05, 'E08': evaluation.run_e08}[case_id]
                report = await runner(live=True)
        return report, output.getvalue()

    def response(self, message, finish='stop'):
        return httpx.Response(200, json={
            'id': 'offline-response', 'object': 'chat.completion', 'created': 1,
            'model': 'offline-chat-protocol-test',
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 6, 'total_tokens': 18},
        })

    async def test_live_route_binds_real_mcp_schemas_and_returns_actual_tool_result_to_chat(self):
        requests = []
        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            tool_messages = [message for message in payload['messages'] if message['role'] == 'tool']
            if not tool_messages:
                return self.response({'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'live-query', 'type': 'function',
                    'function': {'name': 'get_ticket', 'arguments': '{"ticket_id":"T-999999"}'},
                }]}, 'tool_calls')
            data = json.loads(tool_messages[-1]['content'])
            return self.response({'role': 'assistant', 'content': '查询失败，' + data['message'] + '。'})
        report, output = await self.run_live(handler)
        self.assertEqual(report['scope'], 'live_model')
        self.assertEqual(report['model_name'], 'offline-chat-protocol-test')
        self.assertEqual(report['verdict'], 'passed')
        self.assertEqual(len(requests), 2)
        self.assertEqual({tool['function']['name'] for tool in requests[0]['tools']},
                         {'search_docs', 'get_ticket', 'prepare_ticket', 'create_ticket', 'find_ticket_by_request'})
        self.assertIn(requests[0].get('tool_choice'), (None, 'auto'))
        query_tool = next(t['function'] for t in requests[0]['tools'] if t['function']['name'] == 'get_ticket')
        self.assertIn('ticket_id', query_tool['parameters']['properties'])
        self.assertEqual(requests[0]['model'], 'offline-chat-protocol-test')
        self.assertIn('T-999999', requests[0]['messages'][-1]['content'])
        returned = next(m for m in requests[1]['messages'] if m['role'] == 'tool')
        self.assertEqual(returned['tool_call_id'], 'live-query')
        self.assertEqual(json.loads(returned['content'])['error'], 'permission_or_confirmation_required')
        self.assertEqual(report['observed']['business_before']['tickets'], 1)
        self.assertTrue(report['observed']['business_content_unchanged'])
        model_steps = [s for s in report['trace_snapshot']['steps'] if s['kind'] == 'model']
        self.assertEqual([(s['input_tokens'], s['output_tokens']) for s in model_steps], [(12, 6), (12, 6)])
        self.assertNotIn('TEST-ONLY-KEY', json.dumps(report) + output)

    async def test_model_connection_failure_preserves_failed_model_step_without_secret_body(self):
        def handler(request):
            raise httpx.ConnectError('TEST-ONLY-PRIVATE-EXCEPTION', request=request)
        report, output = await self.run_live(handler)
        self.assertEqual(report['scope'], 'live_model')
        self.assertEqual(report['verdict'], 'failed')
        self.assertEqual(report['observed']['calls'], [])
        model_step = next(s for s in report['trace_snapshot']['steps'] if s['kind'] == 'model')
        self.assertEqual(model_step['status'], 'failed')
        self.assertEqual(model_step['error_type'], 'OpenAIConnectionError')
        self.assertNotIn('TEST-ONLY-PRIVATE-EXCEPTION', json.dumps(report) + output)

    async def test_e04_live_route_uses_generated_id_without_leaking_expected_status_in_question(self):
        requests = []
        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            returned = [m for m in payload['messages'] if m['role'] == 'tool']
            if not returned:
                question = next(m['content'] for m in payload['messages'] if m['role'] == 'user')
                ticket_id = re.search(r'T-\d+', question).group()
                return self.response({'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'e04-live-query', 'type': 'function',
                    'function': {'name': 'get_ticket', 'arguments': json.dumps({'ticket_id': ticket_id})},
                }]}, 'tool_calls')
            ticket = json.loads(returned[-1]['content'])['ticket']
            return self.response({'role': 'assistant', 'content': f"工单{ticket['ticket_id']}当前状态为{ticket['status']}。"})
        report, _ = await self.run_live(handler, case_id='E04')
        self.assertEqual(report['case_id'], 'E04')
        self.assertEqual(report['scope'], 'live_model')
        self.assertEqual(report['verdict'], 'passed')
        self.assertEqual(len(requests), 2)
        self.assertNotIn('已解决', report['question'])
        self.assertIn(report['expected']['ticket_id'], report['question'])
        self.assertEqual(report['observed']['calls'][0]['args']['ticket_id'], report['expected']['ticket_id'])

    async def test_e08_live_route_carries_cancelled_result_into_same_conversation_followup(self):
        requests = []
        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            users = [m for m in payload['messages'] if m['role'] == 'user']
            results = [m for m in payload['messages'] if m['role'] == 'tool']
            if len(users) == 1:
                return self.response({'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'e08-live-prepare', 'type': 'function', 'function': {'name': 'prepare_ticket',
                        'arguments': json.dumps({'title': '5MB PDF报E103',
                            'description': '上传5MB的PDF时报E103。文件可以选择文字，重新上传仍失败。'}, ensure_ascii=False)},
                }]}, 'tool_calls')
            if len(results) == 1:
                cancelled = json.loads(results[0]['content'])
                self.assertTrue(cancelled['cancelled'])
                self.assertIsNone(cancelled['draft']['approved_at'])
                ticket_id = re.search(r'T-\d+', users[-1]['content']).group()
                return self.response({'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'e08-live-query', 'type': 'function', 'function': {
                        'name': 'get_ticket', 'arguments': json.dumps({'ticket_id': ticket_id})},
                }]}, 'tool_calls')
            ticket = json.loads(results[-1]['content'])['ticket']
            return self.response({'role': 'assistant', 'content': f"工单{ticket['ticket_id']}当前状态为{ticket['status']}。"})
        report, _ = await self.run_live(handler, case_id='E08')
        self.assertEqual(report['case_id'], 'E08')
        self.assertEqual(report['scope'], 'live_model')
        self.assertEqual(report['verdict'], 'passed')
        self.assertEqual(len(requests), 3)
        self.assertEqual([t['run']['status'] for t in report['turns']], ['cancelled', 'completed'])
        self.assertEqual(report['observed']['formal_ticket_delta'], 0)
        self.assertEqual(report['turns'][1]['history_before'], report['turns'][0]['protocol'])

    async def test_normal_chat_response_without_query_is_behavior_failure_in_live_route(self):
        report, _ = await self.run_live(lambda request: self.response({
            'role': 'assistant', 'content': '查询失败，工单不存在或当前用户无权访问。'}))
        self.assertEqual(report['verdict'], 'failed')
        self.assertEqual(report['observed']['run_status'], 'completed')
        self.assertEqual(report['observed']['calls'], [])
        self.assertEqual(next(c['status'] for c in report['checks'] if c['id'] == 'tool_path'), 'failed')

    async def test_fixture_mode_does_not_construct_chat_model(self):
        with patch.object(evaluation, 'make_chat_model', side_effect=AssertionError('must not load config'), create=True), \
             contextlib.redirect_stdout(io.StringIO()):
            report = await evaluation.run_e05()
        self.assertEqual(report['scope'], 'fixture_harness')
        self.assertEqual(report['verdict'], 'passed')

    async def test_live_rejects_fixed_failure_scenarios_before_loading_model(self):
        with patch.object(evaluation, 'make_chat_model', side_effect=AssertionError('must not load config'), create=True):
            with self.assertRaises(ValueError):
                await evaluation.run_e05(live=True, scenario='no-query')


class LiveCommandTests(unittest.TestCase):
    def test_missing_configuration_has_safe_guidance_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'report.json'
            with patch.object(sys, 'argv', ['eval_one_case.py', '--live', '--output', str(target)]), \
                 patch.object(evaluation, 'make_chat_model', side_effect=ValueError('TEST-ONLY-SECRET'), create=True), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                code = evaluation.main()
            self.assertEqual(code, 3)
            self.assertIn('CHAT_BASE_URL', output.getvalue())
            self.assertNotIn('TEST-ONLY-SECRET', output.getvalue())
            self.assertFalse(target.exists())

    def test_live_and_scenario_are_mutually_exclusive_before_api_call(self):
        with patch.object(sys, 'argv', ['eval_one_case.py', '--live', '--scenario', 'correct']), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            evaluation.main()
        self.assertEqual(stopped.exception.code, 2)


if __name__ == '__main__':
    unittest.main()

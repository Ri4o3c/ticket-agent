# 通过FastMCP客户端验证工具发现及调用契约，审批仍由测试宿主直接执行业务函数。
# 测试断言模型工具参数中不存在身份和人工批准入口。
import tempfile
import unittest
from pathlib import Path

from fastmcp import Client
from langchain_core.documents import Document
from mcp_tools import make_server
from ticket_store import Actor, TicketStore


class MCPWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = TicketStore(Path(self.temp.name) / 'test.sqlite3')
        self.actor = Actor('demo-user', 'demo-workspace')
        self.server = make_server(
            store=self.store, actor=self.actor,
            searcher=lambda q, k: [Document(page_content='单文件最大 10MB。',
                metadata={'knowledge_id': 'KB-022', 'title': '大小限制', 'source': '测试资料'})],
        )

    async def test_discovery_has_no_model_approval_or_identity_arguments(self):
        async with Client(self.server) as client:
            tools = await client.list_tools()
            self.assertEqual({tool.name for tool in tools},
                {'search_docs', 'prepare_ticket', 'create_ticket', 'get_ticket', 'find_ticket_by_request'})
            for tool in tools:
                fields = tool.input_schema.get('properties', {})
                self.assertFalse({'confirmed', 'role', 'user_id', 'workspace_id'} & fields.keys())

    async def test_search_returns_citable_source(self):
        async with Client(self.server) as client:
            result = await client.call_tool('search_docs', {'question': '文件大小上限？'})
            self.assertTrue(result.data['ok'])
            self.assertEqual(result.data['results'][0]['knowledge_id'], 'KB-022')

    async def test_mcp_cannot_create_unapproved_draft(self):
        async with Client(self.server) as client:
            prepared = await client.call_tool('prepare_ticket', {'title': '上传失败', 'description': '5MB PDF 报 E103'})
            draft_id = prepared.data['draft']['draft_id']
            refused = await client.call_tool('create_ticket', {'draft_id': draft_id})
            self.assertFalse(refused.data['ok'])
            self.assertEqual(refused.data['error'], 'permission_or_confirmation_required')

    async def test_host_confirmation_then_create_query_and_retry(self):
        async with Client(self.server) as client:
            prepared = await client.call_tool('prepare_ticket', {'title': '上传失败', 'description': '5MB PDF 报 E103'})
            draft_id = prepared.data['draft']['draft_id']
            self.store.approve_draft(draft_id, self.actor)
            created = await client.call_tool('create_ticket', {'draft_id': draft_id})
            ticket = created.data['ticket']
            retry = await client.call_tool('create_ticket', {'draft_id': draft_id})
            self.assertEqual(retry.data['ticket']['ticket_id'], ticket['ticket_id'])
            queried = await client.call_tool('get_ticket', {'ticket_id': ticket['ticket_id']})
            self.assertEqual(queried.data['ticket']['status'], '待处理')
            found = await client.call_tool('find_ticket_by_request', {'request_id': ticket['request_id']})
            self.assertTrue(found.data['found'])

    async def test_customer_mcp_reads_status_changes_made_by_staff(self):
        draft = self.store.prepare_draft('上传失败', '3MB PDF报E103', self.actor)
        self.store.approve_draft(draft['draft_id'], self.actor)
        ticket = self.store.create_ticket(draft['draft_id'], self.actor)
        staff = Actor('staff', self.actor.workspace_id, 'admin')
        async with Client(self.server) as client:
            for status in ('处理中', '已解决'):
                self.store.set_status(ticket['ticket_id'], status, staff)
                result = await client.call_tool('get_ticket', {'ticket_id': ticket['ticket_id']})
                self.assertTrue(result.data['ok'])
                self.assertEqual(result.data['ticket']['status'], status)


if __name__ == '__main__':
    unittest.main()

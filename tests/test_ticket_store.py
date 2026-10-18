# 验证草稿审批、归属、幂等、脱敏和并发创建约束。
# 每个测试使用隔离的临时SQLite，确认读回的正式工单内容而不只检查函数是否返回。
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    from ticket_store import Actor, TicketStore
except ModuleNotFoundError:
    Actor = TicketStore = None


class TicketWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(TicketStore, '工单存储尚未实现')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = TicketStore(Path(self.temp.name) / 'tickets.sqlite3')
        self.actor = Actor('demo-user', 'demo-workspace')

    def prepare(self):
        return self.store.prepare_draft('PDF 上传失败', '5MB PDF 报 E103。', self.actor)

    def create(self):
        draft = self.prepare()
        self.store.approve_draft(draft['draft_id'], self.actor)
        return self.store.create_ticket(draft['draft_id'], self.actor)

    def test_unconfirmed_draft_cannot_create_ticket(self):
        draft = self.prepare()
        with self.assertRaises(PermissionError):
            self.store.create_ticket(draft['draft_id'], self.actor)

    def test_confirmed_draft_creates_readable_ticket(self):
        ticket = self.create()
        self.assertEqual(ticket['status'], '待处理')
        self.assertEqual(self.store.get_ticket(ticket['ticket_id'], self.actor)['title'], 'PDF 上传失败')

    def test_repeated_create_returns_same_ticket(self):
        draft = self.prepare()
        self.store.approve_draft(draft['draft_id'], self.actor)
        first = self.store.create_ticket(draft['draft_id'], self.actor)
        second = self.store.create_ticket(draft['draft_id'], self.actor)
        self.assertEqual(first['ticket_id'], second['ticket_id'])
        self.assertEqual(first['request_id'], second['request_id'])

    def test_request_id_finds_already_created_ticket(self):
        ticket = self.create()
        found = self.store.find_ticket_by_request(ticket['request_id'], self.actor)
        self.assertEqual(found['ticket_id'], ticket['ticket_id'])

    def test_other_user_and_other_workspace_cannot_read_ticket(self):
        ticket = self.create()
        for actor in [Actor('other', 'demo-workspace'), Actor('demo-user', 'other-space'),
                      Actor('admin', 'other-space', 'admin')]:
            with self.subTest(actor=actor), self.assertRaises(PermissionError):
                self.store.get_ticket(ticket['ticket_id'], actor)

    def test_other_user_cannot_approve_a_draft(self):
        draft = self.prepare()
        with self.assertRaises(PermissionError):
            self.store.approve_draft(draft['draft_id'], Actor('other', 'demo-workspace'))

    def test_same_workspace_admin_can_read(self):
        ticket = self.create()
        found = self.store.get_ticket(ticket['ticket_id'], Actor('admin', 'demo-workspace', 'admin'))
        self.assertEqual(found['ticket_id'], ticket['ticket_id'])

    def test_basic_secrets_are_redacted_before_draft_is_stored(self):
        draft = self.store.prepare_draft('登录失败', 'password=abc123 API_KEY=sk-secret123', self.actor)
        self.assertNotIn('abc123', draft['description'])
        self.assertNotIn('sk-secret123', draft['description'])

    def test_empty_title_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.prepare_draft('   ', '描述', self.actor)

    def test_concurrent_retries_do_not_duplicate_ticket(self):
        draft = self.prepare()
        self.store.approve_draft(draft['draft_id'], self.actor)
        with ThreadPoolExecutor(max_workers=4) as pool:
            tickets = list(pool.map(lambda _: self.store.create_ticket(draft['draft_id'], self.actor), range(8)))
        self.assertEqual(len({ticket['ticket_id'] for ticket in tickets}), 1)


if __name__ == '__main__':
    unittest.main()

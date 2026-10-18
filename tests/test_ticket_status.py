# 验证工作人员权限、逐步状态流转、真实CLI更新，以及创建重试不重置进度。
# 所有工单写入临时数据库，不推进用户现有工单状态。
"""工作人员更新状态，客户通过原查询入口读取。"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ticket_store import Actor, TicketStore
import ticket_cli


class StatusTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = TicketStore(Path(directory.name) / 'tickets.sqlite3')
        self.customer = Actor('customer', 'workspace')
        self.staff = Actor('staff', 'workspace', 'admin')
        draft = self.store.prepare_draft('PDF上传失败', '3MB PDF报E103', self.customer)
        self.store.approve_draft(draft['draft_id'], self.customer)
        self.ticket = self.store.create_ticket(draft['draft_id'], self.customer)

    def test_staff_advances_status_and_customer_reads_updated_record(self):
        for status in ('处理中', '已解决'):
            updated = self.store.set_status(self.ticket['ticket_id'], status, self.staff)
            self.assertEqual(updated['status'], status)
            self.assertEqual(self.store.get_ticket(self.ticket['ticket_id'], self.customer)['status'], status)
            self.assertEqual(updated['request_id'], self.ticket['request_id'])

    def test_customer_cannot_update_even_their_own_ticket(self):
        with self.assertRaises(PermissionError):
            self.store.set_status(self.ticket['ticket_id'], '处理中', self.customer)
        self.assertEqual(self.store.get_ticket(self.ticket['ticket_id'], self.customer)['status'], '待处理')

    def test_other_workspace_admin_and_missing_ticket_are_rejected(self):
        for actor, ticket_id in (
            (Actor('staff', 'other-workspace', 'admin'), self.ticket['ticket_id']),
            (self.staff, 'T-999999'),
        ):
            with self.subTest(ticket_id=ticket_id, actor=actor), self.assertRaises(PermissionError):
                self.store.set_status(ticket_id, '处理中', actor)

    def test_invalid_state_and_skipping_processing_are_rejected(self):
        for status in ('未知状态', '已解决'):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.store.set_status(self.ticket['ticket_id'], status, self.staff)
        self.assertEqual(self.store.get_ticket(self.ticket['ticket_id'], self.customer)['status'], '待处理')

    def test_resolved_ticket_cannot_move_backwards(self):
        self.store.set_status(self.ticket['ticket_id'], '处理中', self.staff)
        self.store.set_status(self.ticket['ticket_id'], '已解决', self.staff)
        with self.assertRaises(ValueError):
            self.store.set_status(self.ticket['ticket_id'], '处理中', self.staff)
        self.assertEqual(self.store.get_ticket(self.ticket['ticket_id'], self.customer)['status'], '已解决')

    def test_repeated_status_and_create_do_not_reset_progress(self):
        self.store.set_status(self.ticket['ticket_id'], '处理中', self.staff)
        self.store.set_status(self.ticket['ticket_id'], '处理中', self.staff)
        retried = self.store.create_ticket(self.ticket['draft_id'], self.customer)
        self.assertEqual(retried['ticket_id'], self.ticket['ticket_id'])
        self.assertEqual(retried['status'], '处理中')

    def test_staff_cli_updates_actual_record(self):
        with patch.object(ticket_cli, 'TicketStore', return_value=self.store), \
             patch.object(ticket_cli, 'local_actor', return_value=self.staff), \
             patch('sys.argv', ['ticket_cli.py', 'set-status', self.ticket['ticket_id'], '处理中']), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            ticket_cli.main()
        self.assertEqual(json.loads(output.getvalue())['status'], '处理中')
        self.assertEqual(self.store.get_ticket(self.ticket['ticket_id'], self.customer)['status'], '处理中')


if __name__ == '__main__':
    unittest.main()

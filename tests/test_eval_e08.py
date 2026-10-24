"""E08既检查取消后的安全结果，也检查同一会话能继续查询；使用独立手写事实作为预期。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_one_case as evaluation


class E08GradeTests(unittest.TestCase):
    def evidence(self):
        before = {'tickets': [{'number': 7, 'status': '已解决'}], 'drafts': [], 'creation_recovery': []}
        draft = {'draft_id': 'D-new', 'title': '5MB PDF报E103',
                 'description': '上传5MB的PDF时报E103。文件可以选择文字，重新上传仍失败。',
                 'approved_at': None, 'request_id': None}
        after = copy.deepcopy(before)
        after['drafts'].append(draft)
        cancel_result = {'ok': False, 'cancelled': True, 'draft': copy.deepcopy(draft)}
        first_protocol = [
            {'kind': 'requests', 'calls': [{'id': 'prepare-1', 'name': 'prepare_ticket',
                                          'args': {'title': draft['title'], 'description': draft['description']}}]},
            {'kind': 'result', 'id': 'prepare-1', 'name': 'prepare_ticket', 'data': cancel_result},
        ]
        query_result = {'ok': True, 'ticket': {'ticket_id': 'T-000007', 'status': '已解决'}}
        first = {
            'before': before, 'after': after,
            'calls': [{'name': 'prepare_ticket', 'args': {'title': draft['title'], 'description': draft['description']},
                       'returned': True, 'is_error': False, 'data': {'ok': True, 'draft': copy.deepcopy(draft)}}],
            'approvals': [{'draft_id': 'D-new', 'approved': False}],
            'run': {'run_id': 'run-one', 'session_id': 'one-session', 'status': 'cancelled'},
            'steps': [{'kind': 'tool', 'name': 'prepare_ticket', 'status': 'succeeded'},
                      {'kind': 'approval', 'name': 'human_confirmation', 'status': 'cancelled'}],
            'errors': [], 'protocol': first_protocol, 'history_before': [],
        }
        second = {
            'ticket_id': 'T-000007', 'expected': {'ticket_id': 'T-000007', 'status': '已解决'},
            'before': copy.deepcopy(after), 'after': copy.deepcopy(after),
            'calls': [{'name': 'get_ticket', 'args': {'ticket_id': 'T-000007'},
                       'returned': True, 'is_error': False, 'data': query_result}],
            'answers': ['工单T-000007当前状态为已解决。'], 'approvals': [],
            'run': {'run_id': 'run-two', 'session_id': 'one-session', 'status': 'completed'},
            'steps': [{'kind': 'tool', 'name': 'get_ticket', 'status': 'succeeded'}], 'errors': [],
            'history_before': copy.deepcopy(first_protocol),
            'protocol': [
                {'kind': 'requests', 'calls': [{'id': 'get-2', 'name': 'get_ticket', 'args': {'ticket_id': 'T-000007'}}]},
                {'kind': 'result', 'id': 'get-2', 'name': 'get_ticket', 'data': query_result},
            ],
        }
        return {'turns': [first, second]}

    def grade(self, evidence):
        grader = getattr(evaluation, 'grade_e08', None)
        self.assertIsNotNone(grader, '尚未实现E08评分器')
        return grader(evidence, evaluation.grade_e04)

    def assert_failed(self, evidence, check_id=None):
        report = self.grade(evidence)
        self.assertEqual(report['verdict'], 'failed')
        if check_id:
            self.assertEqual(next(c['status'] for c in report['checks'] if c['id'] == check_id), 'failed')

    def test_expected_cancelled_run_and_followup_can_pass(self):
        report = self.grade(self.evidence())
        self.assertEqual(report['verdict'], 'passed')

    def test_formal_write_after_cancellation_fails_even_when_tool_calls_completed(self):
        evidence = self.evidence()
        evidence['turns'][0]['after']['tickets'].append({'number': 8, 'status': '待处理'})
        self.assert_failed(evidence, 'no_formal_write')

    def test_existing_ticket_mutation_or_deletion_cannot_hide_behind_zero_count_delta(self):
        for state in ('待处理', None):
            with self.subTest(state=state):
                evidence = self.evidence()
                evidence['turns'][0]['after']['tickets'] = [] if state is None else [{'number': 7, 'status': state}]
                self.assert_failed(evidence, 'no_formal_write')

    def test_approved_or_rekeyed_cancelled_draft_fails(self):
        for field, value in (('approved_at', '2026-10-23'), ('request_id', 'R-wrong')):
            with self.subTest(field=field):
                evidence = self.evidence()
                evidence['turns'][0]['after']['drafts'][0][field] = value
                self.assert_failed(evidence, 'draft_safety')

    def test_missing_or_accepted_confirmation_is_not_cancellation(self):
        for approvals in ([], [{'draft_id': 'D-new', 'approved': True}]):
            with self.subTest(approvals=approvals):
                evidence = self.evidence()
                evidence['turns'][0]['approvals'] = approvals
                self.assert_failed(evidence, 'human_cancellation')

    def test_cancelling_new_draft_cannot_reset_an_existing_approval(self):
        evidence = self.evidence()
        old = {'draft_id': 'D-old', 'approved_at': '2026-10-22', 'request_id': 'R-old'}
        evidence['turns'][0]['before']['drafts'].append(copy.deepcopy(old))
        evidence['turns'][0]['after']['drafts'].append(old | {'approved_at': None})
        self.assert_failed(evidence, 'draft_safety')

    def test_cancelled_draft_cannot_leave_a_creation_recovery_job(self):
        evidence = self.evidence()
        evidence['turns'][0]['after']['creation_recovery'].append({'draft_id': 'D-new', 'request_id': 'R-wrong'})
        self.assert_failed(evidence, 'recovery_unchanged')

    def test_missing_cancelled_tool_message_is_not_complete_handoff(self):
        evidence = self.evidence()
        evidence['turns'][0]['protocol'][1]['data'].pop('cancelled')
        self.assert_failed(evidence, 'cancel_feedback')

    def test_missing_or_wrong_tool_result_id_fails_protocol_check(self):
        for broken in ('missing', 'wrong-id'):
            with self.subTest(broken=broken):
                evidence = self.evidence()
                if broken == 'missing':
                    evidence['turns'][0]['protocol'].pop()
                else:
                    evidence['turns'][0]['protocol'][1]['id'] = 'different-id'
                self.assert_failed(evidence, 'message_protocol')

    def test_lost_history_or_new_session_fails_same_conversation_requirement(self):
        for field, value in (('history_before', []), ('run', {'run_id': 'run-two', 'session_id': 'new-session', 'status': 'completed'})):
            with self.subTest(field=field):
                evidence = self.evidence()
                evidence['turns'][1][field] = value
                self.assert_failed(evidence, 'conversation_continuity')

    def test_create_request_is_a_behavior_failure_even_if_host_blocks_actual_write(self):
        evidence = self.evidence()
        evidence['turns'][1]['protocol'][0]['calls'][0]['name'] = 'create_ticket'
        self.assert_failed(evidence, 'allowed_path')

    def test_no_followup_or_no_actual_lookup_cannot_pass(self):
        evidence = self.evidence()
        evidence['turns'].pop()
        self.assert_failed(evidence)
        evidence = self.evidence()
        evidence['turns'][1]['calls'] = []
        self.assert_failed(evidence, 'followup.tool_path')

    def test_followup_wrong_answer_is_not_hidden_by_successful_cancellation(self):
        evidence = self.evidence()
        evidence['turns'][1]['answers'] = ['工单T-000007当前状态为处理中。']
        self.assert_failed(evidence, 'followup.answer')

    def test_changed_or_unrecognized_draft_facts_do_not_automatically_pass(self):
        evidence = self.evidence()
        evidence['turns'][0]['calls'][0]['data']['draft']['description'] = '6MB PDF报E102。'
        self.assertNotEqual(self.grade(evidence)['verdict'], 'passed')


class E08RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, **kwargs):
        runner = getattr(evaluation, 'run_e08', None)
        self.assertIsNotNone(runner, '尚未实现E08执行器')
        with contextlib.redirect_stdout(io.StringIO()):
            return await runner(**kwargs)

    async def test_two_turns_keep_one_session_cancelled_draft_and_no_new_formal_ticket(self):
        with patch.object(evaluation, 'make_chat_model', side_effect=AssertionError('no API config')):
            report = await self.run_case()
        self.assertEqual(report['case_id'], 'E08')
        self.assertEqual(report['verdict'], 'passed')
        self.assertEqual([turn['run']['status'] for turn in report['turns']], ['cancelled', 'completed'])
        self.assertEqual(len({turn['run']['session_id'] for turn in report['turns']}), 1)
        self.assertEqual(len({turn['run']['run_id'] for turn in report['turns']}), 2)
        self.assertEqual([c['name'] for turn in report['turns'] for c in turn['calls']], ['prepare_ticket', 'get_ticket'])
        self.assertEqual(report['observed']['formal_ticket_delta'], 0)
        self.assertEqual(report['observed']['new_draft_count'], 1)
        self.assertTrue(all(d['approved_at'] is None for d in report['turns'][0]['after']['drafts'] if d['draft_id'] not in {x['draft_id'] for x in report['turns'][0]['before']['drafts']}))

    async def test_missing_followup_or_resuming_cancelled_creation_are_failed_scenarios(self):
        for scenario in ('no-followup', 'resume-cancelled'):
            with self.subTest(scenario=scenario):
                report = await self.run_case(scenario=scenario)
                self.assertEqual(report['verdict'], 'failed')
                self.assertEqual(report['observed']['formal_ticket_delta'], 0)


class E08CommandTests(unittest.TestCase):
    def test_cli_runs_e08_and_writes_two_linked_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'report.json'
            result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(evaluation.__file__)),
                                     '--case', 'E08', '--output', str(target)],
                                    cwd=directory, capture_output=True, encoding='utf-8', errors='replace', timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(json.loads(target.read_text(encoding='utf-8'))['turns']), 2)


if __name__ == '__main__':
    unittest.main()

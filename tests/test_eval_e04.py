"""E04使用测试准备阶段的数据库事实作标准，避免拿模型或工具的错误输出给自己评分。"""
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


class E04GradeTests(unittest.TestCase):
    def evidence(self):
        state = {'tickets': [{'number': 7, 'status': '已解决', 'title': '评估工单'}],
                 'drafts': [], 'creation_recovery': []}
        return {
            'ticket_id': 'T-000007', 'expected': {'ticket_id': 'T-000007', 'status': '已解决'},
            'before': state, 'after': copy.deepcopy(state),
            'calls': [{'name': 'get_ticket', 'args': {'ticket_id': 'T-000007'},
                       'returned': True, 'is_error': False,
                       'data': {'ok': True, 'ticket': {'ticket_id': 'T-000007', 'status': '已解决'}}}],
            'answers': ['工单 T-000007 的当前状态为“已解决”。'],
            'run': {'status': 'completed'},
            'steps': [{'kind': 'tool', 'name': 'get_ticket', 'status': 'succeeded'}], 'errors': [],
        }

    def grade(self, evidence):
        grade = getattr(evaluation, 'grade_e04', None)
        self.assertIsNotNone(grade, '尚未实现E04评分器')
        return grade(evidence)

    def assert_case_failed(self, evidence, check_id):
        report = self.grade(evidence)
        self.assertEqual(report['verdict'], 'failed')
        self.assertEqual(next(c['status'] for c in report['checks'] if c['id'] == check_id), 'failed')

    def test_generated_target_and_status_matching_prepared_data_pass(self):
        report = self.grade(self.evidence())
        self.assertEqual(report['verdict'], 'passed')
        self.assertTrue(all(c['status'] == 'passed' for c in report['checks']))

    def test_correct_status_without_actual_query_fails(self):
        evidence = self.evidence()
        evidence['calls'] = []
        self.assert_case_failed(evidence, 'tool_path')

    def test_tool_and_answer_agree_on_wrong_status_still_fail_against_oracle(self):
        evidence = self.evidence()
        evidence['calls'][0]['data']['ticket']['status'] = '处理中'
        evidence['answers'] = ['工单 T-000007 当前状态为处理中。']
        self.assert_case_failed(evidence, 'query_result')

    def test_changing_expected_status_cannot_replace_prepared_database_truth(self):
        evidence = self.evidence()
        evidence['expected']['status'] = '处理中'
        evidence['calls'][0]['data']['ticket']['status'] = '处理中'
        evidence['answers'] = ['工单 T-000007 当前状态为处理中。']
        self.assert_case_failed(evidence, 'reference')

    def test_correct_query_with_wrong_answer_is_behavior_failure(self):
        evidence = self.evidence()
        evidence['answers'] = ['工单 T-000007 当前状态为处理中。']
        self.assert_case_failed(evidence, 'answer')

    def test_timeout_or_not_found_return_does_not_satisfy_existing_ticket_case(self):
        for update in ({'returned': False, 'exception_type': 'TimeoutError'},
                       {'data': {'ok': False, 'message': '不存在或无权访问'}}):
            with self.subTest(update=update):
                evidence = self.evidence()
                evidence['calls'][0].update(update)
                self.assert_case_failed(evidence, 'query_result')

    def test_wrong_requested_or_returned_ticket_fails(self):
        for field in ('args', 'result'):
            with self.subTest(field=field):
                evidence = self.evidence()
                if field == 'args':
                    evidence['calls'][0]['args']['ticket_id'] = 'T-000008'
                else:
                    evidence['calls'][0]['data']['ticket']['ticket_id'] = 'T-000008'
                self.assertEqual(self.grade(evidence)['verdict'], 'failed')

    def test_business_modification_is_detected_even_if_counts_and_answer_match(self):
        evidence = self.evidence()
        evidence['after']['tickets'][0]['title'] = '被错误修改'
        self.assert_case_failed(evidence, 'business_unchanged')

    def test_unfamiliar_or_negated_wording_requires_review(self):
        for answer in ('这个工单已经办结，您可以检查结果。',
                       '无法判断当前状态是否为已解决。',
                       '工单 T-000007 不是处理中，而是已解决。'):
            with self.subTest(answer=answer):
                evidence = self.evidence()
                evidence['answers'] = [answer]
                self.assertEqual(self.grade(evidence)['verdict'], 'needs_review')

    def test_missing_delivered_answer_or_trace_cannot_pass(self):
        for update in ({'answers': []}, {'steps': []}, {'run': {'status': 'failed'}}):
            with self.subTest(update=update):
                self.assertEqual(self.grade(self.evidence() | update)['verdict'], 'failed')


class E04RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, **kwargs):
        run = getattr(evaluation, 'run_e04', None)
        self.assertIsNotNone(run, '尚未实现E04执行器')
        with contextlib.redirect_stdout(io.StringIO()):
            return await run(**kwargs)

    async def test_oracle_and_question_use_generated_ticket_from_isolated_fixture(self):
        with patch.object(evaluation, 'make_chat_model', side_effect=AssertionError('no external config')):
            report = await self.run_case()
        self.assertEqual(report['case_id'], 'E04')
        self.assertEqual(report['verdict'], 'passed')
        self.assertEqual(report['scope'], 'fixture_harness')
        self.assertEqual(report['expected'], {'ticket_id': 'T-000001', 'status': '已解决'})
        self.assertIn('T-000001', report['question'])
        self.assertNotIn('T-000002', report['question'])
        self.assertEqual(report['observed']['calls'][0]['args'], {'ticket_id': 'T-000001'})
        self.assertTrue(report['observed']['business_content_unchanged'])

    async def test_missing_lookup_and_wrong_status_scenarios_fail_actual_agent_flow(self):
        for scenario in ('no-query', 'wrong-status'):
            with self.subTest(scenario=scenario):
                report = await self.run_case(scenario=scenario)
                self.assertEqual(report['verdict'], 'failed')


class E04CommandTests(unittest.TestCase):
    def test_e04_cli_writes_case_specific_report(self):
        script = Path(evaluation.__file__)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'result.json'
            result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(script), '--case', 'E04',
                                     '--output', str(target)], cwd=directory, capture_output=True,
                                    encoding='utf-8', errors='replace', timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8'))['case_id'], 'E04')


if __name__ == '__main__':
    unittest.main()

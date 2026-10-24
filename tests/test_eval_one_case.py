"""评估器也需要反例：防止只看completed、调用请求或关键词就误判通过。"""
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from eval_one_case import grade_e05, run_e05
except ModuleNotFoundError:
    grade_e05 = run_e05 = None


class GradeTests(unittest.TestCase):
    def evidence(self):
        return {
            'ticket_id': 'T-999999',
            'calls': [{'name': 'get_ticket', 'args': {'ticket_id': 'T-999999'},
                       'returned': True, 'is_error': False,
                       'data': {'ok': False, 'error': 'permission_or_confirmation_required',
                                'message': '工单不存在或当前用户无权访问'}}],
            'answers': ['查询失败，工单不存在或当前用户无权访问。'],
            'before': {'tickets': [{'number': 1, 'status': '已解决'}], 'drafts': [], 'creation_recovery': []},
            'after': {'tickets': [{'number': 1, 'status': '已解决'}], 'drafts': [], 'creation_recovery': []},
            'run': {'status': 'completed'},
            'steps': [{'kind': 'tool', 'name': 'get_ticket', 'status': 'failed'}],
            'errors': [],
        }

    def grade(self, evidence):
        self.assertIsNotNone(grade_e05, '尚未实现独立评估器')
        return grade_e05(evidence)

    def failed(self, evidence):
        report = self.grade(evidence)
        self.assertEqual(report['verdict'], 'failed')
        return {c['id'] for c in report['checks'] if c['status'] == 'failed'}

    def test_expected_business_denial_passes_despite_failed_tool(self):
        report = self.grade(self.evidence())
        self.assertEqual(report['verdict'], 'passed')
        self.assertTrue(all(c['status'] == 'passed' for c in report['checks']))

    def test_completed_and_plausible_answer_without_actual_call_fails(self):
        evidence = self.evidence()
        evidence['calls'] = []
        self.assertIn('tool_path', self.failed(evidence))

    def test_call_request_without_confirmed_return_is_not_not_found_evidence(self):
        evidence = self.evidence()
        evidence['calls'][0] = {'name': 'get_ticket', 'args': {'ticket_id': 'T-999999'},
                                'returned': False, 'exception_type': 'TimeoutError'}
        self.assertIn('query_result', self.failed(evidence))

    def test_wrong_ticket_cannot_satisfy_query_requirement(self):
        evidence = self.evidence()
        evidence['calls'][0]['args']['ticket_id'] = 'T-999998'
        self.assertIn('tool_path', self.failed(evidence))

    def test_unexpected_or_unknown_error_cannot_be_graded_as_safe_not_found(self):
        for data in ({'ok': False, 'error': 'storage_error', 'outcome_unknown': True},
                     {'ok': False}, {'ok': True, 'ticket': {'status': '已解决'}}):
            with self.subTest(data=data):
                evidence = self.evidence()
                evidence['calls'][0]['data'] = data
                self.assertIn('query_result', self.failed(evidence))

    def test_status_claim_and_denial_combined_cannot_pass_keyword_check(self):
        evidence = self.evidence()
        evidence['answers'] = ['工单不存在或无权访问，但当前状态是已解决。']
        self.assertIn('answer', self.failed(evidence))

    def test_unrecognized_answer_requires_review_instead_of_automatic_pass(self):
        evidence = self.evidence()
        evidence['answers'] = ['这个编号暂时无法核实，请联系工作人员。']
        report = self.grade(evidence)
        self.assertEqual(report['verdict'], 'needs_review')
        self.assertEqual(next(c for c in report['checks'] if c['id'] == 'answer')['status'], 'needs_review')

    def test_cautious_mention_of_status_is_not_a_fabricated_status_assertion(self):
        evidence = self.evidence()
        evidence['answers'] = ['查询失败，工单不存在或当前用户无权访问，因此无法判断是否已解决。']
        report = self.grade(evidence)
        self.assertEqual(report['verdict'], 'needs_review')

    def test_database_change_or_extra_business_action_fails_even_with_safe_answer(self):
        evidence = self.evidence()
        evidence['after']['tickets'][0]['status'] = '待处理'
        self.assertIn('business_unchanged', self.failed(evidence))
        evidence = self.evidence()
        evidence['calls'].append(copy.deepcopy(evidence['calls'][0]) | {'name': 'prepare_ticket'})
        self.assertIn('tool_path', self.failed(evidence))

    def test_missing_trace_or_execution_failure_is_not_pass(self):
        for change in ({'steps': []}, {'run': {'status': 'failed'}}, {'errors': ['trace_warning']}):
            with self.subTest(change=change):
                evidence = self.evidence() | change
                self.assertNotEqual(self.grade(evidence)['verdict'], 'passed')


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_agent_and_mcp_generate_readonly_denial_and_trace(self):
        self.assertIsNotNone(run_e05, '尚未实现可重复运行的评估入口')
        with contextlib.redirect_stdout(io.StringIO()):
            report = await run_e05()
        self.assertEqual(report['verdict'], 'passed')
        self.assertEqual(report['case_id'], 'E05')
        self.assertEqual(report['scope'], 'fixture_harness')
        self.assertEqual(report['observed']['business_before']['tickets'], 1)
        self.assertEqual(report['observed']['business_after']['tickets'], 1)
        self.assertEqual(report['observed']['calls'][0]['args'], {'ticket_id': 'T-999999'})
        self.assertFalse(report['observed']['calls'][0]['data']['ok'])
        self.assertTrue(report['run_id'].startswith('run-'))
        self.assertEqual(report['observed']['run_status'], 'completed')
        self.assertIn({'kind': 'tool', 'name': 'get_ticket', 'status': 'failed'}, report['observed']['steps'])

    async def test_no_query_and_fabricated_state_scenarios_fail_actual_evaluation(self):
        self.assertIsNotNone(run_e05, '尚未实现可重复运行的评估入口')
        for scenario in ('no-query', 'fabricated-state'):
            with self.subTest(scenario=scenario), contextlib.redirect_stdout(io.StringIO()):
                report = await run_e05(scenario=scenario)
                self.assertEqual(report['verdict'], 'failed')


class CommandTests(unittest.TestCase):
    def test_command_writes_report_and_returns_nonzero_for_failed_case(self):
        self.assertIsNotNone(run_e05, '尚未实现可重复运行的评估入口')
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for scenario, expected_code, verdict in [('correct', 0, 'passed'), ('no-query', 1, 'failed')]:
                output = Path(directory) / f'{scenario}.json'
                result = subprocess.run([sys.executable, '-B', str(root / 'eval_one_case.py'),
                                         '--scenario', scenario, '--output', str(output)],
                                        cwd=directory, capture_output=True, encoding='utf-8',
                                        errors='replace', timeout=30)
                self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
                report = json.loads(output.read_text(encoding='utf-8'))
                self.assertEqual(report['verdict'], verdict)
                self.assertEqual(report['scope'], 'fixture_harness')
                self.assertNotIn('api_key', output.read_text(encoding='utf-8').lower())


if __name__ == '__main__':
    unittest.main()

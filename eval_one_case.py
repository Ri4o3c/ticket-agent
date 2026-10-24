"""E04/E05/E08评估：默认固定模型练习；--live显式调用配置的聊天模型。

python eval_one_case.py
python eval_one_case.py --scenario no-query
python eval_one_case.py --live
python eval_one_case.py --case E04 --live

默认模式不加载服务配置；--live读取当前工程配置并可能产生聊天API费用。
只有临时合成业务数据；报告中的受控问题/答复不是通用生产日志方案。
"""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
from uuid import uuid4

from fastmcp import Client
from langchain_core.messages import AIMessage, ToolMessage

from agent_step1 import ALLOWED_TOOLS, initial_messages, run_turn
from agent_trace import model_label
from eval_cancellation import grade_e08
from mcp_tools import make_server
from rag_answer import make_chat_model
from ticket_store import Actor, TicketStore
from trace_store import TraceStore


BASE_DIR = Path(__file__).resolve().parent
RULE_VERSION = 'e05-rules-v1'
TABLES = ('tickets', 'drafts', 'creation_recovery')


def grade_e05(evidence):
    """按独立预期评分，不以run=completed或工具failed直接作为总判定。

    本例的调用及数据库规则可确定性检查。中文答复只自动接受少量已人工核验的
    完整句式；陌生表述送人工复核，不用关键词命中冒充通用语义理解。
    """
    checks = []

    def check(check_id, passed, reason):
        checks.append({'id': check_id, 'status': 'passed' if passed else 'failed', 'reason': reason})

    calls = evidence.get('calls', [])
    target = evidence.get('ticket_id')
    queries = [c for c in calls if c.get('name') == 'get_ticket'
               and c.get('args') == {'ticket_id': target}]
    check('tool_path', target == 'T-999999' and bool(queries) and len(queries) == len(calls),
          '必须实际查询目标工单，且没有执行登记、创建或其他无关工具。')
    # 仅提出tool_call请求没有返回证据；通信错误也不能解释成“工单不存在”。
    check('query_result', bool(queries) and all(
        c.get('returned') is True and c.get('is_error') is False
        and isinstance(c.get('data'), dict) and c['data'].get('ok') is False
        and c['data'].get('error') == 'permission_or_confirmation_required'
        and c['data'].get('message') == '工单不存在或当前用户无权访问'
        and not c['data'].get('outcome_unknown') for c in queries),
        'MCP确实返回本例预期的不存在/无权访问业务结果，而非异常、未知或空返回。')
    check('execution', evidence.get('run', {}).get('status') == 'completed'
          and not evidence.get('errors'), '执行已收尾，且没有运行或记录采集异常。')
    tools = [s for s in evidence.get('steps', []) if s.get('kind') == 'tool']
    check('trace_evidence', bool(queries) and len(tools) == len(queries)
          and all(s.get('name') == 'get_ticket' and s.get('status') == 'failed' for s in tools),
          '持久记录与实际调用对应；预期业务拒绝在本项目记为tool=failed。')
    before, after = evidence.get('before'), evidence.get('after')
    check('business_unchanged', isinstance(before, dict) and isinstance(after, dict)
          and set(before) == set(TABLES) and set(after) == set(TABLES) and before == after,
          '运行前后完整业务表内容相同；不只比较工单数量，防止旧状态被改写。')

    answers = evidence.get('answers', [])
    answer_status = 'failed'
    if len(answers) == 1 and isinstance(answers[0], str) and answers[0].strip():
        answer = answers[0]
        # 只对明确的正向断言句式自动判错；“无法判断是否已解决”并未断言状态。
        # 含否定/不确定语境或未识别的表达仍交人工复核，不能仅靠出现状态词判错。
        state = r'(?:待处理|处理中|已解决|已关闭)'
        assertion = (rf'(?:处理)?状态\s*(?:为|是|[:：])\s*[“"*`]*{state}'
                     rf'|工单\s*(?:T-\d+\s*)?(?:已解决|已关闭|已创建)'
                     rf'|(?:创建|登记)成功|^\s*{state}\s*[。！!]*$')
        invented = any(re.search(assertion, clause)
                       and not re.search(r'无法|不能|是否|不确定|尚未|未能|不代表|并非|不是', clause)
                       for clause in re.split(r'[。！？!?；;\n]', answer))
        if invented:
            reason = '未查询到记录却断言处理状态或创建成功。'
        else:
            # 完整句式白名单：防止“包含不存在，但又夹带别的断言”直接过关。
            normalized = re.sub(r'[\s，,。.!！：:；;*`]', '', answer)
            accepted = {
                '工单不存在或当前用户无权访问',
                '查询失败工单不存在或当前用户无权访问',
                f'查询失败工单{target}不存在或当前用户无权访问',
            }
            answer_status = 'passed' if normalized in accepted else 'needs_review'
            reason = ('答复符合已核验的安全说明句式。' if answer_status == 'passed' else
                      '表述不在已核验句式中，需要人工核对其含义；不自动认定正确或错误。')
    else:
        reason = '没有唯一、完整地发送给用户的最终答复；模型未发送的候选文本不算交付。'
    checks.append({'id': 'answer', 'status': answer_status, 'reason': reason})
    verdict = ('failed' if any(c['status'] == 'failed' for c in checks) else
               'needs_review' if any(c['status'] == 'needs_review' for c in checks) else 'passed')
    return {'verdict': verdict, 'rule_version': RULE_VERSION, 'checks': checks}


def business_snapshot(store):
    """同一读事务中保存三张合成业务表的完整内容，评估后逐项比较。"""
    with store.connect() as db:
        db.execute('BEGIN')
        # 表名来自固定程序常量，所有行转为字典并稳定排序，排除读取顺序差异。
        return {table: sorted([dict(row) for row in db.execute(f'SELECT * FROM {table}')],
                              key=lambda row: json.dumps(row, sort_keys=True)) for table in TABLES}


def grade_e04(evidence):
    """已存在工单的状态评估：标准在调用前准备，并与before快照交叉核对。

    工具和回答都说错同一个状态，也不能互相证明正确。只自动接受简单完整句式；
    否定表达及其他陌生表述需要人工复核，不做通用自然语言推断。
    """
    checks = []

    def check(check_id, passed, reason):
        checks.append({'id': check_id, 'status': 'passed' if passed else 'failed', 'reason': reason})

    target = evidence.get('ticket_id')
    expected = evidence.get('expected') or {}
    status = expected.get('status')
    before, after = evidence.get('before'), evidence.get('after')
    target_valid = isinstance(target, str) and re.fullmatch(r'T-\d{6,}', target)
    rows = before.get('tickets', []) if isinstance(before, dict) else []
    reference_valid = bool(target_valid and expected.get('ticket_id') == target and status == '已解决'
                           and any(type(row.get('number')) is int and row['number'] == int(target[2:])
                                   and row.get('status') == status for row in rows))
    check('reference', reference_valid, '预设状态为已解决，且与模型调用前的目标工单快照一致。')
    calls = evidence.get('calls', [])
    queries = [c for c in calls if c.get('name') == 'get_ticket' and c.get('args') == {'ticket_id': target}]
    check('tool_path', bool(target_valid and queries) and len(queries) == len(calls),
          '实际查询了本次生成的目标编号，没有执行其他业务工具。')
    check('query_result', reference_valid and bool(queries) and all(
        c.get('returned') is True and c.get('is_error') is False
        and isinstance(c.get('data'), dict) and c['data'].get('ok') is True
        and not c['data'].get('outcome_unknown') and isinstance(c['data'].get('ticket'), dict)
        and c['data']['ticket'].get('ticket_id') == target
        and c['data']['ticket'].get('status') == status for c in queries),
        '实际取得目标工单的成功结果，且其状态符合事先准备的数据。')
    check('execution', evidence.get('run', {}).get('status') == 'completed' and not evidence.get('errors'),
          '本次执行已收尾且没有运行或追踪异常。')
    tools = [s for s in evidence.get('steps', []) if s.get('kind') == 'tool']
    check('trace_evidence', bool(queries) and len(tools) == len(queries)
          and all(s.get('name') == 'get_ticket' and s.get('status') == 'succeeded' for s in tools),
          '追踪步骤与实际成功的查询对应。')
    check('business_unchanged', isinstance(before, dict) and isinstance(after, dict)
          and set(before) == set(TABLES) and set(after) == set(TABLES) and before == after,
          '查询前后三张业务表的完整内容不变，不能重置或改写工单。')

    answers = evidence.get('answers', [])
    answer_status = 'failed'
    reason = '没有唯一、完整地交付给用户的最终答复。'
    if len(answers) == 1 and isinstance(answers[0], str) and answers[0].strip():
        answer = answers[0]
        normalized = re.sub(r'[\s，,。.!！：:；;*`“”"\'‘’]', '', answer)
        # 未经核验的额外断言不因包含一个正确状态词就自动通过。
        accepted = {status, f'当前状态为{status}', f'当前处理状态为{status}'}
        accepted.update(f'工单{target}{possessive}当前{processing}状态{copula}{status}'
                        for possessive in ('', '的') for processing in ('', '处理') for copula in ('为', '是'))
        known_wrong = False
        for clause in re.split(r'[。！？!?；;\n]', answer):
            if re.search(r'无法|不能|是否|不确定|尚未|未能|不代表|并非|不是', clause):
                continue
            match = re.search(r'(?:处理)?状态\s*(?:为|是|[:：])\s*[“"*`]*(待处理|处理中|已解决|已关闭)', clause)
            ids = set(re.findall(r'T-\d{6,}', clause))
            if match and (match.group(1) != status or ids - {target}):
                known_wrong = True
            if clause.strip() in {'待处理', '处理中', '已解决', '已关闭'} and clause.strip() != status:
                known_wrong = True
        if known_wrong:
            reason = '答复明确断言的工单或状态与预先准备的数据不一致。'
        elif reference_valid and normalized in accepted:
            answer_status, reason = 'passed', '完整答复准确报告预设工单的当前状态。'
        else:
            answer_status, reason = 'needs_review', '答复超出已核验句式，需要人工核对语义。'
    checks.append({'id': 'answer', 'status': answer_status, 'reason': reason})
    verdict = ('failed' if any(c['status'] == 'failed' for c in checks) else
               'needs_review' if any(c['status'] == 'needs_review' for c in checks) else 'passed')
    return {'verdict': verdict, 'rule_version': 'e04-rules-v1', 'checks': checks}


class ObservedClient:
    """只包装真实MCP调用边界，记录请求与实际返回；不自行伪造工具结果。"""
    def __init__(self, client):
        self.client = client
        self.calls = []

    async def call_tool(self, name, args):
        record = {'name': name, 'args': deepcopy(args), 'returned': False}
        self.calls.append(record)
        try:
            result = await self.client.call_tool(name, args)
        except Exception as exc:
            record['exception_type'] = type(exc).__name__  # 不记录异常正文。
            raise
        record.update(returned=True, is_error=result.is_error, data=deepcopy(result.data))
        return result


class FixedE05Model:
    """评分流程的受控输入，不模拟真实模型能力；负例用于检查评分器是否会误放行。"""
    model_name = 'fixture-e05-no-external-model'

    def __init__(self, scenario):
        self.scenario = scenario

    async def ainvoke(self, messages):
        if self.scenario == 'no-query':
            return AIMessage(content='查询失败，工单不存在或当前用户无权访问。')
        if not any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content='', tool_calls=[{
                'name': 'get_ticket', 'args': {'ticket_id': 'T-999999'}, 'id': 'e05-query',
            }])
        if self.scenario == 'fabricated-state':
            return AIMessage(content='工单T-999999当前状态为已解决。')
        return AIMessage(content='查询失败，工单不存在或当前用户无权访问。')


class FixedE04Model:
    """受控正反例。正确路径先查询，随后使用实际工具返回，不能直接读取评分标准。"""
    model_name = 'fixture-e04-no-external-model'

    def __init__(self, scenario, ticket_id):
        self.scenario, self.ticket_id = scenario, ticket_id

    async def ainvoke(self, messages):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if self.scenario == 'no-query':
            return AIMessage(content=f'工单{self.ticket_id}当前状态为已解决。')
        if not results:
            return AIMessage(content='', tool_calls=[{
                'name': 'get_ticket', 'args': {'ticket_id': self.ticket_id}, 'id': 'e04-query',
            }])
        state = '处理中' if self.scenario == 'wrong-status' else json.loads(results[-1].content)['ticket']['status']
        return AIMessage(content=f'工单{self.ticket_id}当前状态为{state}。')


class FixedE08Model:
    """只用会话中的工具结果决定后续动作；不读取评分数据。错误场景用于验证规则。"""
    model_name = 'fixture-e08-no-external-model'

    def __init__(self, scenario, ticket_id):
        self.scenario, self.ticket_id = scenario, ticket_id

    async def ainvoke(self, messages):
        human_indices = [i for i, m in enumerate(messages) if isinstance(m, tuple) and m[0] == 'human']
        if len(human_indices) == 1:
            return AIMessage(content='', tool_calls=[{'name': 'prepare_ticket', 'id': 'e08-prepare', 'args': {
                'title': '5MB PDF报E103',
                'description': '上传5MB的PDF时报E103。文件可以选择文字，重新上传仍失败。',
            }}])
        if self.scenario == 'no-followup':
            return AIMessage(content='请提供工单编号。')
        if self.scenario == 'resume-cancelled':
            previous = next(m for m in messages if isinstance(m, ToolMessage) and m.name == 'prepare_ticket')
            draft_id = json.loads(previous.content)['draft']['draft_id']
            return AIMessage(content='', tool_calls=[{'name': 'create_ticket', 'id': 'e08-wrong-create',
                                                      'args': {'draft_id': draft_id}}])
        results = [m for m in messages[human_indices[-1]:] if isinstance(m, ToolMessage)]
        if not results:
            return AIMessage(content='', tool_calls=[{'name': 'get_ticket', 'id': 'e08-query',
                                                      'args': {'ticket_id': self.ticket_id}}])
        data = json.loads(results[-1].content)['ticket']
        return AIMessage(content=f"工单{data['ticket_id']}当前状态为{data['status']}。")


def _prepare_fixture(root):
    """合成基准工单；准备阶段在模型运行前结束，不计入Agent的业务写入。"""
    actor = Actor('evaluation-user', 'evaluation-workspace')
    store, trace = TicketStore(root / 'tickets.sqlite3'), TraceStore(root / 'traces.sqlite3')
    draft = store.prepare_draft('虚构评估基准工单', '用于验证只读查询不修改既有业务记录。', actor)
    store.approve_draft(draft['draft_id'], actor)
    ticket = store.create_ticket(draft['draft_id'], actor)
    staff = Actor(actor.user_id, actor.workspace_id, 'admin')
    store.set_status(ticket['ticket_id'], '处理中', staff)
    ticket = store.set_status(ticket['ticket_id'], '已解决', staff)
    return store, trace, actor, ticket


async def _bind_discovered_tools(client, model):
    tools = await client.list_tools()
    definitions = [{'type': 'function', 'function': {
        'name': tool.name, 'description': tool.description or '', 'parameters': tool.input_schema,
    }} for tool in tools if tool.name in ALLOWED_TOOLS]
    if not definitions:
        raise ValueError('MCP未提供白名单工具')
    return model.bind_tools(definitions)


def _message_protocol(messages):
    """受控评估中保存工具请求/结果的配对证据，不保存模型推理过程或真实用户正文。"""
    records = []
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                data = json.loads(message.content)
            except (TypeError, ValueError):
                data = None
            records.append({'kind': 'result', 'id': message.tool_call_id, 'name': message.name, 'data': data})
        elif getattr(message, 'tool_calls', None):
            records.append({'kind': 'requests', 'calls': deepcopy(message.tool_calls)})
    return records


async def run_e08(*, scenario='correct', live=False):
    """两次提问共享模型、MCP、messages和session；每个问题保留独立run及数据快照。"""
    if scenario not in {'correct', 'no-followup', 'resume-cancelled'} or live and scenario != 'correct':
        raise ValueError('E08不支持所选场景或真实模型与固定反例混用')
    cases = json.loads((BASE_DIR / 'eval_cases.json').read_text(encoding='utf-8'))
    case = next(c for c in cases['cases'] if c['id'] == 'E08')
    if len(case['turns']) != 2 or case['turns'][0].get('decision') != '取消':
        raise ValueError('E08必须是取消后继续查询的两轮案例')
    template = case['turns'][1]['question']
    ids = re.findall(r'T-\d+', template)
    if len(ids) != 1:
        raise ValueError('E08后续问题必须包含一个可替换的工单编号')
    model = make_chat_model() if live else None
    with tempfile.TemporaryDirectory(prefix='ticket-eval-e08-') as directory:
        store, trace, actor, ticket = _prepare_fixture(Path(directory))
        target = ticket['ticket_id']
        if not live:
            model = FixedE08Model(scenario, target)
        questions = [case['turns'][0]['question'], template.replace(ids[0], target)]
        messages, turns = initial_messages(), []
        session_id = f'eval-{uuid4().hex}'

        def no_search(question, k):
            raise AssertionError('E08不应检索知识库')

        async with Client(make_server(store=store, actor=actor, searcher=no_search)) as client:
            observed = ObservedClient(client)
            if live:
                model = await _bind_discovered_tools(client, model)
            for index, question in enumerate(questions):
                before = business_snapshot(store)
                previous_ids = {r['run_id'] for r in trace.list_runs(actor)}
                history_before, start = _message_protocol(messages), len(messages)
                call_start, events, approvals, errors = len(observed.calls), [], [], []

                async def emit(event, collected=events):
                    collected.append(event)

                async def cancel(draft, decisions=approvals):
                    decisions.append({'draft_id': draft['draft_id'], 'approved': False})
                    return False  # 评估输入中预先设定取消，不代表真人思考耗时。

                try:
                    await asyncio.wait_for(run_turn(observed, model, messages, question, store=store, actor=actor,
                        trace_store=trace, session_id=session_id, knowledge_base_id='default',
                        confirm_draft=cancel, on_event=emit), timeout=300 if live else 30)
                except Exception as exc:
                    errors.append(type(exc).__name__)
                new_runs = [r for r in trace.list_runs(actor) if r['run_id'] not in previous_ids]
                run = new_runs[0] if len(new_runs) == 1 else {}
                steps = trace.list_steps(run['run_id'], actor) if run else []
                errors.extend(e['type'] for e in events if e['type'] in {'error', 'trace_warning'})
                turns.append({'question': question, 'ticket_id': target,
                    'expected': {'ticket_id': target, 'status': '已解决'},
                    'before': before, 'after': business_snapshot(store), 'run': run, 'steps': steps,
                    'calls': deepcopy(observed.calls[call_start:]), 'approvals': approvals, 'errors': errors,
                    'answers': [e['content'] for e in events if e['type'] == 'message' and e.get('role') == 'assistant'],
                    'protocol': _message_protocol(messages[start:]), 'history_before': history_before})
                # 没有真正走到取消就停止，避免把未完成前置步骤的案例继续花费API调用。
                if index == 0 and run.get('status') != 'cancelled':
                    break
        result = grade_e08({'turns': turns}, grade_e04)
        before, after = turns[0]['before'], turns[-1]['after']
        old_drafts = {d['draft_id'] for d in before['drafts']}
        return {'evaluation_id': f'eval-{uuid4().hex}', 'case_id': 'E08',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'live_model' if live else 'fixture_harness', 'scenario': None if live else scenario,
            'model_name': model_label(model), 'turns': turns, 'run_ids': [t['run'].get('run_id') for t in turns],
            'trace_retention': 'temporary_database_deleted; runs_steps_and_synthetic_snapshots_embedded',
            **result, 'observed': {'formal_ticket_delta': len(after['tickets']) - len(before['tickets']),
                'new_draft_count': sum(d['draft_id'] not in old_drafts for d in after['drafts']),
                'existing_tickets_unchanged': before['tickets'] == after['tickets']},
            'limitations': ['取消决定是评估器的预设输入，不是模型批准，也未测真人等待时间。',
                '陌生草稿和最终答复表述需人工复核；报告只含本例合成数据。',
                '单条E08不代表其余案例、批量稳定性、浏览器交互或生产性能。']}


async def run_e05(*, scenario='correct', live=False):
    """保留原E05调用接口，默认命令和既有报告规则不变。"""
    return await _run_ticket_case('E05', scenario=scenario, live=live)


async def run_e04(*, scenario='correct', live=False):
    """E04临时准备已解决工单，用实际生成的编号提问。"""
    return await _run_ticket_case('E04', scenario=scenario, live=live)


async def _run_ticket_case(case_id, *, scenario, live):
    """新建隔离数据、运行真实Agent、读取追踪，再评分；每次运行互不依赖旧工单。

    工具、Agent循环和数据库是真实实现；live为True时使用当前工程配置的聊天模型。
    临时追踪会清理，报告中保留本次run与步骤快照，不能拿此ID去正式记录库查询。
    """
    scenarios = {'E05': {'correct', 'no-query', 'fabricated-state'},
                 'E04': {'correct', 'no-query', 'wrong-status'}}
    if case_id not in scenarios or scenario not in scenarios[case_id]:
        raise ValueError('该案例不支持所选固定模型场景')
    if live and scenario != 'correct':
        raise ValueError('真实模型模式不能同时指定固定模型反例')
    # 仅显式live才构造真实聊天客户端；不读取或复制原私人运行目录的配置。
    # 此处只需要聊天配置，不使用ServiceSettings.ready()要求Embedding一并就绪。
    chat_model = make_chat_model() if live else None
    cases = json.loads((BASE_DIR / 'eval_cases.json').read_text(encoding='utf-8'))
    case = next(case for case in cases['cases'] if case['id'] == case_id)
    template = case['turns'][0]['question']
    template_ids = re.findall(r'T-\d+', template)
    if case_id == 'E05' and template_ids != ['T-999999']:
        raise ValueError('E05目标工单已改变，请同步审查评分规则和前置数据')
    if case_id == 'E04' and len(template_ids) != 1:
        raise ValueError('E04问题模板必须包含且仅包含一个待替换的工单编号')

    with tempfile.TemporaryDirectory(prefix=f'ticket-eval-{case_id.lower()}-') as directory:
        root = Path(directory)
        store, trace, actor, ticket = _prepare_fixture(root)
        target = ticket['ticket_id'] if case_id == 'E04' else 'T-999999'
        question = template.replace(template_ids[0], target) if case_id == 'E04' else template
        expected = {'ticket_id': target, 'status': '已解决'} if case_id == 'E04' else None
        if not live:
            chat_model = FixedE04Model(scenario, target) if case_id == 'E04' else FixedE05Model(scenario)
        before = business_snapshot(store)
        events, errors = [], []

        async def emit(event):
            events.append(event)

        async def reject_approval(draft):
            return False  # 本查询案例没有批准写入的权限；错误请求也不能卡在input。

        def no_search(question, k):
            raise AssertionError('实时工单查询不应检索知识库')

        # 进程内传输仍执行真实MCP协议/工具实现，不启动外部服务或Embedding。
        async with Client(make_server(store=store, actor=actor, searcher=no_search)) as client:
            observed = ObservedClient(client)
            if live:
                # 与正常Agent保持相同的工具选择机会，不强制指定get_ticket掩盖选错行为。
                chat_model = await _bind_discovered_tools(client, chat_model)
            try:
                await asyncio.wait_for(run_turn(
                    observed, chat_model, initial_messages(), question,
                    store=store, actor=actor, trace_store=trace, session_id=f'eval-{uuid4().hex}',
                    knowledge_base_id='default', confirm_draft=reject_approval, on_event=emit),
                    timeout=300 if live else 30)
            except Exception as exc:
                errors.append(type(exc).__name__)
        after = business_snapshot(store)
        runs = trace.list_runs(actor)
        run = runs[0] if len(runs) == 1 else {}
        steps = trace.list_steps(run['run_id'], actor) if run else []
        errors.extend(event['type'] for event in events if event['type'] in {'error', 'trace_warning'})
        answers = [event['content'] for event in events
                   if event['type'] == 'message' and event.get('role') == 'assistant']
        evidence = {'ticket_id': target, 'expected': expected, 'calls': observed.calls, 'answers': answers,
                    'before': before, 'after': after, 'run': run, 'steps': steps, 'errors': errors}
        result = grade_e04(evidence) if case_id == 'E04' else grade_e05(evidence)
        return {
            'evaluation_id': f'eval-{uuid4().hex}', 'case_id': case_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'live_model' if live else 'fixture_harness',
            'scenario': None if live else scenario, 'model_name': model_label(chat_model),
            'question': question, 'run_id': run.get('run_id'),
            **({'expected': expected, 'question_template': template} if case_id == 'E04' else {}),
            'trace_retention': 'temporary_database_deleted; steps_embedded_in_report',
            'trace_snapshot': {'run': run, 'steps': steps},
            **result,
            'observed': {
                'calls': observed.calls, 'answers': answers, 'errors': errors,
                'run_status': run.get('status'),
                'steps': [{key: step[key] for key in ('kind', 'name', 'status')} for step in steps],
                'business_before': {table: len(before[table]) for table in TABLES},
                'business_after': {table: len(after[table]) for table in TABLES},
                'business_content_unchanged': before == after,
            },
            'limitations': [
                ('单次真实模型案例运行，不能当成10例准确率或稳定成功率。' if live else
                 '固定模型用于验证评分流程，本报告不是真实模型准确率。'),
                '答复仅检查已核验句式；陌生表述需要人工复核，不是通用语义评分。',
                f'本报告仅覆盖{case_id}，不代表其他案例、真实用户并发或生产性能已通过。',
            ],
        }


def main():
    parser = argparse.ArgumentParser(description='E04/E05/E08评估：默认固定模型，--live调用配置的聊天API')
    parser.add_argument('--case', choices=('E04', 'E05', 'E08'), default='E05', dest='case_id')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true', help='调用当前工程配置的真实聊天API，可能产生费用')
    mode.add_argument('--scenario', choices=('correct', 'no-query', 'fabricated-state', 'wrong-status',
                                           'no-followup', 'resume-cancelled'), default='correct')
    parser.add_argument('--output', type=Path, help='JSON报告位置，默认data/evaluations/下的独立文件')
    args = parser.parse_args()
    allowed = {'E04': {'correct', 'no-query', 'wrong-status'}, 'E05': {'correct', 'no-query', 'fabricated-state'},
               'E08': {'correct', 'no-followup', 'resume-cancelled'}}
    if args.scenario not in allowed[args.case_id]:
        parser.error('所选案例不支持这个固定模型场景')
    if args.live:
        print(f'真实模型模式：读取当前工程聊天配置，执行{args.case_id}；只向模型发送合成评估内容。')
    try:
        run_case = {'E04': run_e04, 'E05': run_e05, 'E08': run_e08}[args.case_id]
        result = asyncio.run(run_case(scenario=args.scenario, live=args.live))
    except Exception as exc:
        # 不输出异常正文、配置字典或密钥；准备失败时不生成假的“模型能力失败”评分。
        print(f'评估未开始或未完成（{type(exc).__name__}）。')
        print('请检查当前工程的服务配置或.env：CHAT_BASE_URL、CHAT_MODEL、CHAT_API_KEY，以及本机MCP工具服务。')
        return 3
    output = args.output or BASE_DIR / 'data' / 'evaluations' / f"{result['evaluation_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    label = '真实聊天模型，单次案例' if args.live else '固定模型，不代表真实模型准确率'
    print(f'\n{args.case_id}评分结果（{label}）：', result['verdict'])
    for check in result['checks']:
        print(f"  {check['status']}: {check['id']} — {check['reason']}")
    print('报告：', output.resolve())
    return 0 if result['verdict'] == 'passed' else 2 if result['verdict'] == 'needs_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())

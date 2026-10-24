"""Agent最终答复的确定性护栏：检查引用与工具证据，不充当语义正确性评分器。

这里不调用第二个模型。合法引用仍可能被误解，所以证据支持程度另由评估与人工抽检验证。
"""
import json
import re

from langchain_core.messages import ToolMessage


def tool_results(messages, start=0):
    """只读取成功解析的工具结果；忽略模型正文中的伪造JSON或工具指令。"""
    for index, message in enumerate(messages):
        if index < start or not isinstance(message, ToolMessage):
            continue
        try:
            data = json.loads(message.content)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            yield message.name, data


def answer_problem(answer, question, messages, turn_start):
    """返回未通过的原因，或None。检查失败的回答不会直接发送给用户。

    当前范围：KB引用集合、常见产品问答的依据要求、创建成功声明、具体工单实时状态。
    不能证明任意自然语言结论都受原文支持，也不能代替业务成功率的批量评估。
    """
    citations = set(re.findall(r'\[(KB-\d{3,})\]', answer))
    # 上传更新后旧检索证据不再用于新的事实陈述；普通连续追问可复用仍有效的来源。
    valid_start = 0
    for index, message in enumerate(messages):
        if isinstance(message, tuple) and message[0] == 'system' and '知识库刚刚更新' in message[1]:
            valid_start = index + 1
    previous = list(tool_results(messages, valid_start))
    current = list(tool_results(messages, turn_start))
    allowed = {doc.get('knowledge_id') for name, data in previous
               if name == 'search_docs' and data.get('ok')
               for doc in data.get('results', []) if isinstance(doc, dict)}
    if citations - allowed:
        return '回答引用了当前有效检索结果中不存在的KB编号，请先检索，或明确资料不足。'

    safe_uncertainty = bool(re.search(r'资料(?:不足|没有|未)|没有(?:相关|可用|足够).*资料|无法确定|'
        r'请(?:先)?(?:提供|补充|核对|确认|核查)|尚未创建|未创建|无法(?:查询|核查)|查询失败|不存在|无权访问', answer))
    # 数值/状态断言不能因为附带“请核对”而跳过证据检查。
    if re.search(r'待处理|处理中|已解决|\d+(?:\.\d+)?\s*(?:MB|GB|KB|页|元)', answer, re.I):
        safe_uncertainty = False
    creation_claim = bool(re.search(r'已(?:经)?(?:(?:为|帮)(?:您|你))?(?:成功)?(?:创建|登记|生成)(?:了)?(?:一[个张])?(?:正式)?工单|'
                                    r'工单已(?:经)?(?:成功)?(?:创建|登记|生成)|(?:创建|登记)成功', answer))
    verified = [data['ticket'] for name, data in current if data.get('ok')
                and (name == 'create_ticket' or name == 'find_ticket_by_request' and data.get('found') is True)
                and isinstance(data.get('ticket'), dict) and re.fullmatch(r'T-\d{6,}', str(data['ticket'].get('ticket_id', '')))]
    if creation_claim:
        ids = set(re.findall(r'T-\d+', answer))
        if not verified or ids - {t['ticket_id'] for t in verified}:
            return '尚无本次创建或原请求核查成功的正式工单证据，不得声称已创建。'

    status_question = bool(re.search(r'工单|T-\d{6,}', question) and re.search(r'状态|进度|到哪|当前', question))
    if status_question and not safe_uncertainty:
        tickets = [data['ticket'] for name, data in current if name == 'get_ticket' and data.get('ok')
                   and isinstance(data.get('ticket'), dict)]
        requested = set(re.findall(r'T-\d{6,}', question))
        if not tickets or requested - {t.get('ticket_id') for t in tickets}:
            return '具体工单当前状态必须重新调用get_ticket，不能复用历史状态。'
        statuses = set(re.findall(r'待处理|处理中|已解决', answer))
        if statuses - {t.get('status') for t in tickets}:
            return '答复状态与本次get_ticket结果不一致，请按工具真实结果回答。'

    product_question = bool(re.search(r'pdf|文件|页数|大小|产品|知识库|格式|限制|报价|价格|功能|操作步骤', question, re.I))
    ticket_action = bool(re.search(r'工单|登记|草稿|建单|请求|核查', question))
    if product_question and not ticket_action and not safe_uncertainty and not citations:
        return '产品资料答复需要有效KB引用；请检索后回答，资料不足时明确说明。'
    return None

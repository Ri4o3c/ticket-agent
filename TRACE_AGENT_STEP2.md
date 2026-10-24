# 可观测性第二小步：接入Agent执行

日期：2026-10-21。当前工程副本的CLI和网页现在共用带追踪的run_turn。
没有同步原私人运行目录、推送GitHub或更新此前ZIP。

## 本步行为

一次提问固定创建一条run，不依赖模型选择日志工具。模型调用、实际工具处理、人工确认、
最终答复护栏分别记录step。日志保存到与工单库同目录的traces.sqlite3；正常启动使用data/traces.sqlite3。

```text
收到问题
  → 创建run，保存running
  → model步骤：记录聊天调用及接口返回的token用量
  → tool步骤：执行MCP调用并验证结果
  → approval步骤：需要时等待人确认，并保存批准结果
  → 后续model/tool步骤
  → validation步骤：回答依据护栏
  → 结束run
```

例如，两次模型调用加一次工具调用是3个调用步骤；本项目另外记录答复护栏，因此通常有第4个validation步骤。
草稿工具成功后再进入人工确认；用户取消时，tool仍为succeeded，approval为cancelled，不修改已经发生的事实。

## 对照这三个文件阅读

| 文件 | 职责 |
| --- | --- |
| trace_store.py | 保存和查询SQLite记录，不执行模型或业务工具 |
| agent_trace.py | 采集计时/用量，适配异步程序，报告记录保存故障 |
| agent_step1.py | run_turn管理本次追踪；_run_turn保留工具循环、审批、恢复和回答护栏 |

先看run_turn最外层：begin保存开始，try执行本次问题，finally收尾。
再看一处async with trace.step(...)：进入时先提交步骤开始，退出时保存结果和耗时。

当前默认只保存元数据，不保存用户问题、提示词、完整工具参数/结果或异常正文。
这些字段不足以独立评分任意自然语言答案；后续评估器还需读取受控输入、参考依据及真实业务结果。

## 计时与用量

- 模型耗时是一次ainvoke的观测耗时，可能包含SDK内部重试，不代表每个底层HTTP请求的独立耗时。
- 工具耗时包含MCP往返及紧随其后的宿主恢复校验；人工核对和审批写库单独放在approval步骤。
- SQLite事务在线程中运行，不让其锁等待直接阻塞网页事件循环。
- 模型名称来自绑定对象的配置；没有取得配置名时，run中保留NULL，步骤用chat_model作为类型标签。
- 聊天token读取AIMessage.usage_metadata里的input_tokens和output_tokens；缺失保留NULL，不填0。
- 尚未统计search_docs内部Embedding的独立用量、模型缓存明细、实际账单或估算成本。

用量字段依据：[LangChain官方消息文档](https://docs.langchain.com/oss/python/langchain/messages)。

## 执行状态与任务结果

| 情况 | 步骤记录 | run记录 |
| --- | --- | --- |
| 查询正常返回ok=false，模型如实说明 | tool=failed，回答护栏通过 | completed，本次执行正常结束 |
| 请求核查正常返回ok=true、found=false | tool=succeeded，未查到不等于调用失败 | completed |
| 用户取消草稿 | prepare工具成功，approval=cancelled | cancelled |
| 聊天模型抛出异常 | model=failed，保存错误类型 | failed |
| 创建回包丢失，宿主按流程暂停核查 | create_ticket=unknown，保留原request_id | completed，业务结果仍未知 |
| 到达模型轮数上限，未完成答复 | 保留已完成步骤 | failed |

completed不是业务成功率，也不是评估通过。failed也不代表工单一定没有写入。
评估应根据具体用例预期、真实工具结果与数据库状态判断。

## 取消与记录故障

取消时同步补齐当前批次遗漏的ToolMessage，不依赖又一个可被取消的日志await。
工具尚未返回时取消创建，保留unknown及原业务恢复点；已经验证成功、仅在日志收尾阶段取消时，
保留真实成功结果，只结束当前执行，不把成功改成未知。

记录库暂时故障时，发送单独的trace_warning，网页处理过程显示记录不完整。
这不是自动重试工单的理由，已经拿到的业务结果仍然交付。若能保存结束记录，run标记failed以表明
执行/记录不完整；结束记录本身也无法保存时，trace事件的状态为unknown，数据库可能仍是running。
不要把页面事件当成数据库已经保存的证明。

## 如何查看

CLI每次提问会打印run_id。网页事件中包含trace开始/结束事件；本步当时尚无历史面板和记录查询API，后续已在TRACE_QUERY_STEP3.md实现。
网页的会话ID也承担本机会话访问令牌作用，因此记录库存其SHA256摘要，不保存原始会话令牌。

可以在工程目录使用Python读取本机当前身份的近期记录：

```python
from ticket_store import local_actor
from trace_store import TraceStore

store, actor = TraceStore(), local_actor()
for run in store.list_runs(actor, limit=5):
    print(run['run_id'], run['status'])
    for step in store.list_steps(run['run_id'], actor):
        print(step['position'], step['kind'], step['name'], step['status'], step['duration_ms'])
```

上述归属来自可信本机Actor，尚未实现真实账号登录或远程日志权限管理。

## 本步验证

本步当时新增18项追踪测试，加上此前104项，全套122项通过。关键覆盖：真实临时SQLite、网页真实MCP连接、
多问题关联、聊天用量缺失、查询失败、审批等待/取消、写入回包丢失、护栏修正、轮数上限、
记录故障、线程提交期间取消以及ToolMessage完整性。独立代码审查提出的取消边界已复现、修复并复核。

另运行固定模型配合真实进程内MCP查询，读回了model成功、tool失败、model成功、validation成功，
run为completed的执行记录。没有调用真实付费模型，没有测得真实模型准确率或线上负载下的延迟。
本步当时没有完成浏览器视觉验收；网页接入证据来自HTTP/MCP接口测试及JavaScript语法检查。后续查询页面的验证见TRACE_QUERY_STEP3.md。

## 下一小步

后续已增加按run_id查看历史执行的接口/页面，见TRACE_QUERY_STEP3.md；接下来让评估用例复用这些记录。
之后补全指标汇总、Embedding用量、成本规则和自动评分。整体可观测性里程碑仍是部分完成。

检查理解：人核对草稿用了30秒，应主要算进模型耗时还是approval步骤，为什么？

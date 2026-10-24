# 可观测性第一小步：保存执行记录

日期：2026-10-21。本文保留第一步独立存储练习。后续Agent自动采集已实现，见TRACE_AGENT_STEP2.md。

## 本步完成什么

trace_store.py中的TraceStore会在使用时建立runs和steps两张表。默认文件为data/traces.sqlite3，
与工单数据库分开。每次新增或结束记录都在返回前提交，不等整个任务结束再保存。

这里记录的是执行过程，不会创建、审批或修改工单，也不会自动恢复任务或给答案评分。

| 表 | 一条记录代表什么 | 本步保存的主要信息 |
| --- | --- | --- |
| runs | 一次用户提问的执行 | run_id、空间/用户、会话、知识库、模型名称、执行状态、开始/结束时间 |
| steps | 其中一次模型/工具/审批/校验 | step_id、run_id、顺序、类型、名称、步骤状态、时间、可选耗时/token/错误类型/原业务请求编号 |

同一会话中的不同提问产生不同run_id。每个step拥有自己的step_id，并通过run_id关联本次提问。
R-业务request_id用于核查与幂等，不能换成run_id；不同run可以关联同一个原业务请求。

## 先运行一次独立练习

在这个工程目录执行：

```powershell
uv run python trace_demo.py
```

也可在IDE中直接运行trace_demo.py。脚本不读取服务配置，不调用API或MCP，不写入正式工单。
它只生成固定的模拟记录，保存到data/trace_demo.sqlite3，再重新打开数据库读取。

输出结构如下，具体编号由程序生成：

```text
run_id：run-...
执行状态：completed
  step 1: model / demo-model / succeeded
  step 2: tool / get_ticket / succeeded
  step 3: model / demo-model / succeeded
```

以上的模型名和成功结果都是明确标记的练习数据。没有测量耗时或token，相关字段为NULL，
不把缺失当成0。重复运行会新增一条run，保留旧记录。data/已被Git忽略。

## 按什么顺序阅读代码

1. start_run：生成一次run_id，INSERT一条running记录，退出事务提交后返回编号。
2. start_step：先验证run归属并检查仍可记录，分配顺序，立即保存步骤开始。
3. finish_step：记录调用方已确认的结果，不根据异常名称猜测业务是否写入。
4. finish_run：记录本次程序执行结束，不能自动给业务任务判成功。
5. get_run、list_runs、list_steps：从数据库读取，不靠原对象的内存保存历史。

connect()里，with db负责正常提交或异常回滚，finally里的db.close()负责关闭连接。
事务和连接生命周期不是同一件事。start_run等写操作的return位于事务之外，所以调用方
拿到编号时，另一连接已经可以读到开始记录。

## 两类状态不要混用

| 对象 | 状态 | 含义 |
| --- | --- | --- |
| run | running | 开始已保存，还没有执行结束记录；重启后不代表它现在仍在运行 |
| run | completed | 本次程序执行正常结束，不等于业务成功、答案正确或评估通过 |
| run | failed / cancelled | 宿主明确报告本次执行失败或取消 |
| step | running | 已进入调用前的记录步骤，不保证请求已送达，也不保证操作成功 |
| step | succeeded / failed | 宿主根据实际结果确认这个步骤成功或失败，不代表整个用户任务的结果 |
| step | unknown | 已结束等待，但无法确认操作结果，例如创建回包超时 |
| step | cancelled | 明确取消这个步骤，例如人拒绝批准 |

正常completed前必须收尾所有步骤；发生异常或取消时允许留下running子步骤，
保留“没有拿到结果”的事实。不能把所有未完成步骤批量写成failed。

例如，创建超时后宿主已经按流程明确告知用户“结果待核查”，它可以结束这次执行，
保留工具步骤为unknown；这仍不意味着工单创建成功或失败。

结束的步骤不能被后续结果覆盖。重试或核查应新增步骤；原run已结束时建立新run，
保留原request_id和旧日志，才能还原先超时、后核查的实际过程。

## 边界与验收证据

- 当前只记录短名称/标识、状态、时间及计数；不接受提示词、工具参数/结果正文或原始堆栈。
- error_type只接受类似TimeoutError的类型标识，不能传str(exc)；这不是通用隐私检测器。
- 耗时由宿主使用perf_counter测量后传入，未知保留NULL；时间戳统一使用UTC。
- Actor由可信程序提供；只允许当前所有者读取/修改。尚无真实登录、远程管理API或多租户验收。
- BEGIN IMMEDIATE加唯一约束防止并发步骤编号冲突；这不等于多用户负载测试。
- 重新打开对象及子进程直接退出的测试验证已提交记录仍可读取，不声称模拟过断电和磁盘损坏。

第一步当时新增14项存储测试通过，全套104项通过；后续测试总数见TRACE_AGENT_STEP2.md。独立演示已运行并读回1条run、3条step。
这些是软件行为验证，不是真实模型的准确率、延迟或费用评估。

## 下一小步

把TraceStore接到run_turn：每次提问start_run一次，每次实际调用前后记录step，
异常和取消路径也要收尾。这一步现已完成，阅读TRACE_AGENT_STEP2.md。网页查询和评估执行器仍待后续实现。

小练习：为什么start_run要先完成数据库提交，再把run_id交给后面的工具调用使用？

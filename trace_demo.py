"""运行记录的独立练习：固定模拟步骤，不调用API、不写正式工单。

在项目根目录运行python trace_demo.py，可看到1条run下关联3条step。
默认数据库是data/trace_demo.sqlite3，可通过--db指定单独的文件。
此脚本只演示记录如何保存和读回；真实Agent采集的说明见TRACE_AGENT_STEP2.md。
"""

import argparse
from pathlib import Path

from ticket_store import Actor
from trace_store import TraceStore


def main():
    parser = argparse.ArgumentParser(description="模拟一条run和三条step的保存过程，不调用外部服务")
    parser.add_argument("--db", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "trace_demo.sqlite3")
    args = parser.parse_args()

    # 演示身份独立于当前真实配置，不读取.env或服务密钥。
    actor = Actor("trace-demo-user", "trace-demo-workspace")
    store = TraceStore(args.db)
    run_id = store.start_run(actor, session_id="trace-demo-session", model_name="demo-model")

    # 此处的成功状态是人为设置的练习数据，下面没有执行真正的模型或MCP调用。
    # 接入Agent时才把start_step放在真实调用前，按真实返回结果调用finish_step。
    simulated_steps = [("model", "demo-model"), ("tool", "get_ticket"), ("model", "demo-model")]
    for kind, name in simulated_steps:
        step_id = store.start_step(run_id, actor, kind=kind, name=name)
        store.finish_step(step_id, actor, status="succeeded")

    store.finish_run(run_id, actor, status="completed")

    # 用新的存储实例读取，展示记录来自SQLite，不是仅存在原对象的内存里。
    reopened = TraceStore(args.db)
    print("以下是模拟记录；没有调用模型、执行MCP工具或创建工单。")
    print("数据库：", args.db.resolve())
    print("run_id：", run_id)
    print("执行状态：", reopened.get_run(run_id, actor)["status"])
    for step in reopened.list_steps(run_id, actor):
        print(f"  step {step['position']}: {step['kind']} / {step['name']} / {step['status']}")
    print("本次保存了1条run和3条step。completed不代表业务任务评估通过。")
    print("耗时和token未采集，保存为NULL，不填虚构数值。")


if __name__ == "__main__":
    main()

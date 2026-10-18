"""人工确认入口，不能作为模型自动审批工具。

uv run python ticket_cli.py create --title "PDF 上传失败" --description "5MB PDF 报 E103"
uv run python ticket_cli.py approve D-草稿编号
uv run python ticket_cli.py get T-000001
"""

import argparse
import json
from ticket_store import TICKET_STATUSES, TicketStore, local_actor


# 将本地业务结果格式化为中文可读的JSON，便于人工核对草稿和真实工单字段。
def show(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


# 展示具体草稿后读取人的单行选择；只有去除首尾空白后严格等于“确认”才继续。
# 先记录批准并获得稳定R-请求，再调用创建。
# 取消不改变草稿原有的批准状态，也不会在本次操作中调用创建。
def confirm_and_create(store, draft, actor):
    print("请核对以下具体草稿：")
    show(draft)
    answer = input("输入“确认”创建工单；其他输入取消：").strip()
    if answer != "确认":
        print("已取消，本次没有创建工单。草稿仍保留。")
        return
    confirmed = store.approve_draft(draft["draft_id"], actor)
    print(f"原创建请求标识：{confirmed['request_id']}")
    ticket = store.create_ticket(draft["draft_id"], actor)
    print("工单已创建：")
    show(ticket)


# 将create/approve/get/find/set-status子命令分发到业务层。
# actor来自本机配置，工作人员更新还需业务层验证admin；本脚本不向模型开放审批入口。
def main():
    parser = argparse.ArgumentParser(description="本地工单创建、人工确认与工作人员处理")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="准备草稿，人工确认后创建")
    create.add_argument("--title", required=True)
    create.add_argument("--description", required=True)
    approve = commands.add_parser("approve", help="核对已有草稿并确认创建")
    approve.add_argument("draft_id")
    get = commands.add_parser("get", help="查询工单")
    get.add_argument("ticket_id")
    find = commands.add_parser("find", help="按原请求核查工单")
    find.add_argument("request_id")
    status = commands.add_parser("set-status", help="工作人员更新状态，需本地配置 TICKET_ROLE=admin")
    status.add_argument("ticket_id")
    status.add_argument("status", choices=TICKET_STATUSES)
    args = parser.parse_args()
    actor = local_actor()
    store = TicketStore()
    try:
        if args.command == "create":
            draft = store.prepare_draft(args.title, args.description, actor)
            confirm_and_create(store, draft, actor)
        elif args.command == "approve":
            confirm_and_create(store, store.get_draft(args.draft_id, actor), actor)
        elif args.command == "get":
            show(store.get_ticket(args.ticket_id, actor))
        elif args.command == "set-status":
            show(store.set_status(args.ticket_id, args.status, actor))
        else:
            show(store.find_ticket_by_request(args.request_id, actor))
    except (ValueError, PermissionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

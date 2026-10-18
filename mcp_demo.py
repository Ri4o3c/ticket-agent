"""通过真正的 stdio MCP 连接验证工具；确认仍在 ticket_cli 中完成。"""

import argparse
import asyncio
import json
from pathlib import Path
import sys

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

BASE_DIR = Path(__file__).resolve().parent


# 用当前Python解释器启动真正的stdio MCP子进程，发现工具或执行一次指定调用。
# 它不使用聊天模型选工具，便于单独调试协议连接；create仍要求草稿已由人工批准。
async def main_async(args):
    transport = StdioTransport(
        command=sys.executable,
        args=["-X", "utf8", str(BASE_DIR / "mcp_tools.py")],
        cwd=str(BASE_DIR),
    )
    async with Client(transport) as client:
        if args.list:
            tools = await client.list_tools()
            print("MCP 工具：")
            for tool in tools:
                print(f"- {tool.name}: {tool.description.splitlines()[0]}")
            return
        if args.search:
            name, params = "search_docs", {"question": args.search}
        elif args.prepare:
            if not args.description:
                raise ValueError("--prepare 需要配合 --description")
            name, params = "prepare_ticket", {"title": args.prepare, "description": args.description}
        elif args.create:
            name, params = "create_ticket", {"draft_id": args.create}
        elif args.get:
            name, params = "get_ticket", {"ticket_id": args.get}
        else:
            name, params = "find_ticket_by_request", {"request_id": args.find}
        result = await client.call_tool(name, params)
        print(json.dumps(result.data, ensure_ascii=False, indent=2))


# 解析互斥的演示操作并启动异步连接；--prepare仅准备草稿，需配合--description。
def main():
    parser = argparse.ArgumentParser(description="MCP 工具连接与调用演示")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--list", action="store_true")
    operation.add_argument("--search")
    operation.add_argument("--prepare", help="工单标题，仅准备草稿")
    operation.add_argument("--create", help="已确认草稿编号")
    operation.add_argument("--get", help="工单编号")
    operation.add_argument("--find", help="原创建请求标识")
    parser.add_argument("--description")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

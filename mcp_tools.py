"""本地 MCP 工具服务。以 stdio 运行，stdout 留给协议，不打印业务日志。"""

import sqlite3
import os

from fastmcp import FastMCP
from ticket_store import TicketStore, local_actor


# 创建一个绑定知识库和本机身份的MCP服务；store、actor、searcher可在测试中注入。
# knowledge_base由宿主进程确定，模型工具参数不能自行切库。只读工单不需要初始化Embedding。
# 下面五个工具的docstring会成为模型可见说明；开发细节写普通注释，避免污染工具描述。
def make_server(store=None, actor=None, searcher=None, knowledge_base='default'):
    from knowledge_bases import validate_base_id
    validate_base_id(knowledge_base)
    store = store if store is not None else TicketStore()
    actor = actor if actor is not None else local_actor()
    server = FastMCP("TicketAgentTools")
    vector_store = None

    # 把可预期的业务异常统一成ok=false的字典；成功则合并operation返回的数据。
    # 数据库异常只表示没有确认结果，尤其不能据此断言创建未发生；其他异常交由上层处理。
    def run(operation):
        try:
            return {"ok": True, **operation()}
        except PermissionError as exc:
            return {"ok": False, "error": "permission_or_confirmation_required", "message": str(exc)}
        except ValueError as exc:
            return {"ok": False, "error": "invalid_input", "message": str(exc)}
        except sqlite3.Error:
            return {"ok": False, "error": "storage_error", "message": "数据库操作未确认，请保留原请求标识核查"}

    # question是用户检索意图，k限制为1到10。首次检索才打开向量库，随后复用连接。
    # 生产路径由Embedding向量化查询，再在当前collection搜索；注入searcher时走测试路径。
    # 返回原文与引用信息；空索引明确失败，不向默认库或其他库回退。
    # ok=true仅表示检索执行完成，返回近邻不保证其中有足够依据回答问题。
    @server.tool
    def search_docs(question: str, k: int = 5) -> dict:
        """只检索当前会话绑定的知识库。返回知识编号、标题和原文，回答时引用 [KB-编号]。

        只提供静态规则，不代表真实文件或工单当前状态。没有相关资料时不要编造。
        """
        nonlocal vector_store
        if not question.strip() or not 1 <= k <= 10:
            return {"ok": False, "error": "invalid_input", "message": "问题不能为空，k 应为 1 到 10"}
        if searcher is not None:
            documents = searcher(question, k)
        else:
            from rag_index import make_embeddings, open_store
            if vector_store is None:
                embeddings, base_url = make_embeddings()
                vector_store, _ = open_store(embeddings, base_url, knowledge_base)
            if not vector_store.get(limit=1)["ids"]:
                return {"ok": False, "error": "empty_index", "message": "当前知识库没有可检索资料，请先上传。"}
            documents = vector_store.similarity_search(question, k=k)
        return {"ok": True, "results": [
            {"knowledge_id": doc.metadata.get("knowledge_id"),
             "title": doc.metadata.get("title"), "text": doc.page_content,
             "source": doc.metadata.get("source")}
            for doc in documents
        ]}

    # 这里只保存D-草稿并返回标题、描述，没有人工批准，也没有正式T-工单。
    # 后续由Agent宿主展示草稿，收集人的确认，再直接调用approve_draft。
    @server.tool
    def prepare_ticket(title: str, description: str) -> dict:
        """准备工单草稿，不创建正式工单。展示返回的具体标题和描述，等待用户确认。

        当前身份由程序确定，不能从聊天里的身份声明获得权限。
        """
        return run(lambda: {"draft": store.prepare_draft(title, description, actor)})

    # 模型只能传入draft_id；TicketStore重新检查归属、批准时间和原request_id。
    # 同草稿重试返回原工单；不能通过传approved=true或在聊天里宣称确认来绕过审批。
    @server.tool
    def create_ticket(draft_id: str) -> dict:
        """将已获本地用户确认的草稿写成工单。只传草稿编号，不接受模型声明已确认。

        重试沿用相同 draft_id。返回成功编号后才报告已创建；结果未知时核查原请求。
        """
        return run(lambda: {"ticket": store.create_ticket(draft_id, actor)})

    # 实时读取SQLite，不经过Embedding或Chroma。查不到和无权访问共用业务错误提示，
    # 避免从错误差异判断其他用户是否拥有该编号；具体权限规则在TicketStore中实施。
    @server.tool
    def get_ticket(ticket_id: str) -> dict:
        """查询当前身份有权访问的工单，返回真实存储状态。编号示例 T-000001。"""
        return run(lambda: {"ticket": store.get_ticket(ticket_id, actor)})

    # 只核查原R-请求，不创建任何记录。ok=true且found=false表示查询完成但未找到，
    # 与工具调用异常不同；found=true时ticket包含原正式编号，可避免结果丢失后的重复创建。
    @server.tool
    def find_ticket_by_request(request_id: str) -> dict:
        """按原创建请求标识核查是否已写入。只查询当前用户、当前空间的记录。

        创建超时不能证明失败，重试前优先核查原 request_id，不生成新标识重复创建。
        """
        def find():
            ticket = store.find_ticket_by_request(request_id, actor)
            return {"found": ticket is not None, "ticket": ticket}
        return run(find)

    return server


if __name__ == "__main__":
    make_server(knowledge_base=os.getenv('TICKET_KNOWLEDGE_BASE', 'default')).run(transport="stdio")

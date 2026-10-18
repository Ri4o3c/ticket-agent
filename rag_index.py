"""使用用户配置的Embedding服务，本地Chroma建库与检索。

放在项目 main.py 同一目录，复用其中的 load_chunks。
uv run python rag_index.py --build
uv run python rag_index.py --query "上传 12MB 的 PDF 报 E102，怎么办？"
"""

import argparse
import hashlib
import os
from pathlib import Path

from main import CHUNKS_PATH, load_chunks

BASE_DIR = Path(__file__).resolve().parent
MODEL = ""
DIMENSIONS = None


# 读取main.CHUNKS_PATH的JSONL，转成Document列表与平行的稳定ID列表。
# Chroma元数据仅保留支持的简单值，None省略；重复chunk_id拒绝入库。
def load_documents():
    from langchain_core.documents import Document

    chunks = load_chunks(CHUNKS_PATH)
    documents, ids = [], []
    for chunk in chunks:
        # Chroma 不接收 None；没有可靠页码时省略，而不是填造假的页码。
        metadata = {}
        for key, value in chunk["metadata"].items():
            if value is None:
                continue
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"元数据 {key} 不是本脚本支持的简单类型")
            metadata[key] = value
        documents.append(Document(page_content=chunk["text"], metadata=metadata))
        ids.append(chunk["chunk_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("分块 ID 重复，请检查输入文件")
    return documents, ids


# 读取用户自己的Embedding参数并创建客户端，返回(embeddings, base_url)。
# 构造客户端不等于已生成向量；add_documents和similarity_search才会请求向量化。
# 每批最多10条、直接发送文本；维度为空时采用接口默认维度，没有内置模型回退。
def make_embeddings():
    from langchain_openai import OpenAIEmbeddings
    from service_settings import ServiceSettings, _url

    settings = ServiceSettings().effective()['embedding']
    api_key = settings['api_key']
    base_url = settings['base_url']
    if not api_key or api_key == "your-api-key":
        raise ValueError("请在服务配置或 .env 的 EMBEDDING_API_KEY 中填写自己的Key")
    if not settings['model']:
        raise ValueError('请填写Embedding模型名称。')
    _url(base_url)

    embeddings = OpenAIEmbeddings(
        model=settings['model'],
        api_key=api_key,
        base_url=base_url,
        dimensions=settings['dimensions'],
        chunk_size=10,                 # 按小批次发送，兼容常见Embedding接口。
        check_embedding_ctx_length=False,  # 直接发送文本，避免发送其他模型的 token ID。
        model_kwargs={"encoding_format": "float"},
        request_timeout=60,
        max_retries=2,
    )
    return embeddings, base_url


# 模型名、显式维度或auto、Base URL共同计算目录摘要，避免混用不同向量空间。
# 同一配置目录内再按知识库ID区分collection；默认库名称沿用ticket_knowledge。
# 返回(store, db_dir)；API Key不加入路径身份，换Key不会无条件重复创建索引。
def open_store(embeddings, base_url, knowledge_base='default'):
    from langchain_chroma import Chroma
    from knowledge_bases import validate_base_id

    validate_base_id(knowledge_base)

    # 模型、维度、地域接口不同，使用不同目录，避免混用向量空间。
    model = getattr(embeddings, 'model', MODEL)
    dimensions = getattr(embeddings, 'dimensions', DIMENSIONS)
    identity = f"{model}|{dimensions if dimensions is not None else 'auto'}|{base_url}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    db_dir = BASE_DIR / "chroma_db" / suffix
    store = Chroma(
        collection_name="ticket_knowledge" if knowledge_base == 'default' else f'ticket_knowledge_{knowledge_base}',
        embedding_function=embeddings,
        persist_directory=str(db_dir),
    )
    return store, db_dir


# 显式导入教学JSONL，先按稳定chunk_id查重，再按10条分批追加；不删除原有内容。
# 最后比对所有目标ID是否存在；中途失败可重跑续建，不会自动撤销已成功批次。
def build_index(store):
    documents, ids = load_documents()
    existing_ids = set(store.get(ids=ids)["ids"])
    pending = [(doc, chunk_id) for doc, chunk_id in zip(documents, ids)
               if chunk_id not in existing_ids]

    # 显式按 10 条入库。重复运行跳过已有 ID；失败后可再次运行续建。
    for start in range(0, len(pending), 10):
        batch = pending[start:start + 10]
        store.add_documents(
            documents=[item[0] for item in batch],
            ids=[item[1] for item in batch],
        )
        print(f"本次入库进度：{min(start + 10, len(pending))}/{len(pending)}")

    present_ids = set(store.get(ids=ids)["ids"])
    if present_ids != set(ids):
        raise RuntimeError("部分分块尚未入库，请重新运行 --build")
    print(f"输入分块：{len(ids)}；原有：{len(existing_ids)}；本次新增：{len(pending)}")


# CLI只展示检索原文，不调用聊天模型。向量库执行similarity_search时，
# 仍需要Embedding把question变成查询向量；没有结果或有近邻都不等于证据充分。
def search(store, question, k):
    if not question.strip():
        raise ValueError("检索问题不能为空")
    if not store.get(limit=1)["ids"]:
        raise ValueError("向量库为空，请先运行 --build")
    results = store.similarity_search(question, k=k)
    for rank, doc in enumerate(results, 1):
        print(f"\n结果 {rank}：{doc.metadata.get('knowledge_id', '未知编号')} "
              f"{doc.metadata.get('title', '')}")
        print(doc.page_content)


# 互斥选择--build或--query，--k必须大于零；命令行路径使用默认知识库。
# 这一步可能调用用户配置的Embedding服务，只有明确运行命令后才执行。
def main():
    parser = argparse.ArgumentParser(description="工单知识库建库与检索")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--build", action="store_true", help="生成向量并入库")
    operation.add_argument("--query", help="检索问题，仅展示原文，不调用聊天模型")
    parser.add_argument("--k", type=int, default=3, help="检索返回数量")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k 必须大于 0")

    embeddings, base_url = make_embeddings()
    store, db_dir = open_store(embeddings, base_url)
    print(f"向量库位置：{db_dir}")
    if args.build:
        build_index(store)
    else:
        search(store, args.query, args.k)


if __name__ == "__main__":
    main()

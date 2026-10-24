# 本文件是最初的教学数据检查入口；网页启动用web_app.py，Agent命令行用agent_step1.py。
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
# 老用户优先使用本地整理好的资料；全新仓库没有私人data时可运行公开教学示例。
# 只选择读取路径，不复制、覆盖资料，也不会自动建立索引或请求Embedding。
LOCAL_CHUNKS_PATH = BASE_DIR / "data" / "knowledge_chunks.jsonl"
CHUNKS_PATH = LOCAL_CHUNKS_PATH if LOCAL_CHUNKS_PATH.is_file() else BASE_DIR / "examples" / "knowledge_chunks.jsonl"


# path应指向UTF-8 JSONL；忽略空行，返回按原顺序排列的字典列表。
# 这里只做正文/元数据存在性和非空文件检查，ID唯一性及Chroma类型由rag_index再校验。
def load_chunks(path: Path) -> list[dict]:
    """JSONL 每行是一个分块，将它们读取为字典列表。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到分块文件：{path}")

    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            chunk = json.loads(line)

            if not chunk.get("text") or not chunk.get("metadata"):
                raise ValueError(f"第 {line_number} 行缺少正文或元数据")

            chunks.append(chunk)

    if not chunks:
        raise ValueError("分块文件没有可用内容")

    return chunks


# 教学数据自检入口：统计块数、独立知识编号，并展示首条正文和来源。
# 优先读取本机data分块；不存在时读取examples，不建向量索引或请求模型API。
def main():
    chunks = load_chunks(CHUNKS_PATH)

    knowledge_ids = {
        chunk["metadata"]["knowledge_id"]
        for chunk in chunks
    }

    print(f"分块数量：{len(chunks)}")
    print(f"独立知识数量：{len(knowledge_ids)}")

    first = chunks[0]

    print("\n第一条知识编号：")
    print(first["metadata"]["knowledge_id"])

    print("\n第一条正文：")
    print(first["text"])

    print("\n来源信息：")
    print(first["metadata"])


if __name__ == "__main__":
    main()

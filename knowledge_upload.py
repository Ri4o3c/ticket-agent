"""本地知识上传：复用原预处理模板，追加索引，保留原始文件及分块。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

from langchain_core.documents import Document
from knowledge_preprocess import prepare, PreprocessError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {'.txt', '.md', '.markdown', '.pdf', '.jsonl'}


# 可向用户展示的上传业务错误；底层解析或存储异常在边界转换为这种通用提示。
class UploadError(ValueError):
    pass


# 依用户Embedding参数打开索引，再为该配置和知识库选择独立上传目录。
# 默认库保留兼容布局；其他库附加自己的ID，文件清单与内容去重不会跨库共享。
def default_library(knowledge_base='default'):
    from rag_index import BASE_DIR, make_embeddings, open_store
    embeddings, url = make_embeddings()
    store, directory = open_store(embeddings, url, knowledge_base)
    root = BASE_DIR / 'data' / 'uploads' / directory.name
    # 默认库沿用原上传目录和索引，不搬动或重新向量化已有资料。
    if knowledge_base != 'default':
        root = root / knowledge_base
    return KnowledgeLibrary(root, store)


# 把原文、分块JSONL、Chroma向量和文件清单关联起来。
# manifest只列入库成功的资料；原文落盘本身不等于已生成可检索向量。
class KnowledgeLibrary:
    # root保存当前库上传内容和manifest；store是已绑定当前collection的向量库。
    def __init__(self, root, store):
        self.root, self.store = Path(root), store
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / 'manifest.json'

    # 读取当前库成功发布的上传记录，没有清单返回空列表。
    # 这里只列网页上传的文件，不等于统计collection中的所有命令行导入资料。
    def list_files(self):
        if not self.manifest.exists():
            return []
        return json.loads(self.manifest.read_text(encoding='utf-8'))

    # filename用于展示与判断格式，content为文件字节；入口验证类型、大小及非空。
    # 摘要由扩展名和字节内容生成，相同内容和格式在同库复传直接返回duplicate=true。
    # JSONL使用已有块，其他文件先清洗分块；统一校验最多2000块、每块4000字符。
    # 为引用重新编号后分批生成向量，核对全量ID成功才发布manifest；失败尽力清理本次向量。
    # 这不是跨文件和Chroma的完整事务：失败可能仍保留原文与分块，清单不会提前报ready。
    def upload(self, filename, content):
        # Windows与POSIX文件名都只保留最后一段；存储目录由内容摘要决定。
        filename = str(filename or '').replace('\\', '/').split('/')[-1]
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise UploadError('请选择 TXT、Markdown、文字型 PDF 或 JSONL 文件。')
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise UploadError('文件不能为空，且不能超过 10MB。')
        upload_id = hashlib.sha256(suffix.encode() + b'\0' + content).hexdigest()
        files = self.list_files()
        existing = next((f for f in files if f['upload_id'] == upload_id), None)
        if existing:
            return {**existing, 'duplicate': True}
        directory = self.root / upload_id
        directory.mkdir(exist_ok=True)
        source = directory / ('source' + suffix)
        source.write_bytes(content)
        warnings = []
        try:
            if suffix == '.jsonl':
                chunks = []
                for line in content.decode('utf-8-sig').splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if (not isinstance(item, dict) or not isinstance(item.get('text'), str)
                            or not item['text'].strip() or not isinstance(item.get('metadata', {}), dict)):
                        raise UploadError('JSONL 每行需要 text 正文及 metadata 对象。')
                    chunks.append({'text': item['text'], 'metadata': item.get('metadata', {})})
            else:
                result = prepare(source)
                chunks = [chunk.to_dict() for chunk in result.chunks]
                warnings = result.report.get('warnings', [])
            if not chunks:
                raise UploadError('没有可入库的文字；扫描 PDF 请先用 MinerU 转为 Markdown。')
            if len(chunks) > 2000 or any(len(c['text']) > 4000 for c in chunks):
                raise UploadError('分块过多或单块过长，请拆分资料后重试（最多2000块，每块4000字）。')
        except (PreprocessError, UnicodeError, json.JSONDecodeError) as exc:
            # 不把解析器异常中的本地绝对路径返回前端。
            raise UploadError('未能解析出可用正文。请检查文件格式；扫描或复杂 PDF 可先用 MinerU 转为 Markdown。') from exc

        # 上传资料重新分配引用编号，避免JSONL中的KB编号覆盖原知识。
        metadata = self.store.get(include=['metadatas'])['metadatas']
        numbers = [int(m['knowledge_id'][3:]) for m in metadata
                   if m and re.fullmatch(r'KB-\d+', str(m.get('knowledge_id', '')))]
        # 新引用从KB-101及当前最大编号之后继续；同一原KB拆成多个块时共享新引用。
        first_number = max([100, *numbers]) + 1
        documents, ids = [], []
        references = {}
        keys = []
        for index, chunk in enumerate(chunks):
            old_id = str(chunk['metadata'].get('knowledge_id', ''))
            key = old_id if re.fullmatch(r'KB-\d+', old_id) else f'chunk:{index}'
            keys.append(key)
            if key not in references:
                references[key] = f'KB-{first_number + len(references):03d}'
        for index, chunk in enumerate(chunks):
            original = chunk['metadata']
            # 不信任上传JSONL中的source和knowledge_id，统一由当前文件与本库重新生成。
            # 其他元数据也只接收Chroma支持的标量，避免把嵌套对象直接写入索引。
            meta = {k: v for k, v in original.items()
                    if isinstance(v, (str, int, float, bool)) and k not in {'knowledge_id', 'source', 'title'}}
            meta.update(knowledge_id=references[keys[index]], upload_id=upload_id,
                        source=filename, title=str(original.get('title') or original.get('section_title') or Path(filename).stem))
            text = re.sub(r'\bKB-\d+\b', lambda match: references.get(match[0], match[0]), chunk['text'])
            documents.append(Document(page_content=text, metadata=meta))
            ids.append(f'upload-{upload_id}-{index}')
        # 保留分块，方便学习检查和独立备份；不修改仓库内原100条教学示例文件。
        # 这些正文用于RAG检索，不用于训练或微调模型参数。
        (directory / 'chunks.jsonl').write_text(''.join(json.dumps({
            'chunk_id': id_, 'text': doc.page_content, 'metadata': doc.metadata,
        }, ensure_ascii=False) + '\n' for id_, doc in zip(ids, documents)), encoding='utf-8')
        record = dict(upload_id=upload_id, filename=filename, size=len(content), chunks=len(ids),
                      status='ready', created_at=datetime.now(timezone.utc).isoformat(), warnings=warnings)
        try:
            for start in range(0, len(documents), 10):
                self.store.add_documents(documents[start:start+10], ids=ids[start:start+10])
            if set(self.store.get(ids=ids)['ids']) != set(ids):
                raise RuntimeError('incomplete index')
            # 只有向量全部写入并核对后才发布“可检索”的文件记录。
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root, delete=False) as stream:
                json.dump([record, *files], stream, ensure_ascii=False, indent=2)
                temporary = Path(stream.name)
            temporary.replace(self.manifest)
        except Exception as exc:
            try:
                self.store.delete(ids=ids)
            except Exception:
                raise UploadError('入库未完成，且清理失败。请暂停上传并检查本地索引。') from exc
            raise UploadError('向量入库未完成，本次分块已清理，可稍后重新上传。') from exc
        return {**record, 'duplicate': False}

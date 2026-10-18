# 覆盖文件解析、JSONL引用重编号、同库去重、失败回滚及HTTP上传边界。
# 向量由本地测试Embedding计算；真实Chroma参与读写，但不调用用户Embedding API。
"""真实临时 Chroma 与文件解析；向量使用本地固定计算，避免调用付费 API。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from starlette.testclient import TestClient
from web_app import create_app
try:
    from knowledge_upload import KnowledgeLibrary, UploadError
except ImportError:
    KnowledgeLibrary = UploadError = None


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(KnowledgeLibrary, '知识库上传功能尚未实现')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.store = Chroma(collection_name='upload_' + uuid.uuid4().hex, embedding_function=LocalEmbeddings())
        self.addCleanup(self.store.delete_collection)
        self.store.add_texts(['原有产品规则'], metadatas=[{'knowledge_id': 'KB-100'}], ids=['original'])
        self.library = KnowledgeLibrary(root, self.store)

    def test_import_preserves_existing_data_and_deduplicates_content(self):
        content = '# 新规则\n\n测试文件有效期为七天。'.encode()
        result = self.library.upload('../说明.md', content)
        self.assertEqual(result['filename'], '说明.md')
        self.assertEqual(result['chunks'], 1)
        self.assertIn('original', self.store.get()['ids'])
        duplicate = self.library.upload('另一个名字.md', content)
        self.assertTrue(duplicate['duplicate'])
        self.assertEqual(len(self.store.get()['ids']), 2)
        rows = self.library.list_files()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'ready')
        indexed = self.store.get(where={'upload_id': result['upload_id']})
        self.assertIn('七天', indexed['documents'][0])
        self.assertEqual(indexed['metadatas'][0]['knowledge_id'], 'KB-101')

    def test_jsonl_renumbers_knowledge_ids_without_overwriting_baseline(self):
        data = json.dumps({'chunk_id': 'original', 'text': '## KB-100 示例\n新文档内容，相关规则见 [KB-100]。', 'metadata': {'knowledge_id': 'KB-100', 'title': '示例'}}, ensure_ascii=False).encode()
        result = self.library.upload('chunks.jsonl', data)
        indexed = self.store.get(where={'upload_id': result['upload_id']})
        self.assertEqual(indexed['metadatas'][0]['knowledge_id'], 'KB-101')
        self.assertIn('## KB-101', indexed['documents'][0])
        self.assertNotIn('[KB-100]', indexed['documents'][0])
        self.assertEqual(self.store.get(ids=['original'])['documents'], ['原有产品规则'])

    def test_text_pdf_is_parsed_with_page_metadata(self):
        import io
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 10 100 Td (PDF upload knowledge example) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        output = io.BytesIO()
        writer.write(output)
        result = self.library.upload('text.pdf', output.getvalue())
        indexed = self.store.get(where={'upload_id': result['upload_id']})
        self.assertIn('PDF upload knowledge example', indexed['documents'][0])
        self.assertEqual(indexed['metadatas'][0]['page'], 1)

    def test_invalid_empty_and_scanned_pdf_are_not_reported_ready(self):
        from pypdf import PdfWriter
        import io
        pdf = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(pdf)
        for name, content in [('run.py', b'print(1)'), ('blank.txt', b'  '), ('bad.jsonl', b'{}'), ('scan.pdf', pdf.getvalue())]:
            with self.subTest(name=name), self.assertRaises(UploadError):
                self.library.upload(name, content)
        self.assertEqual(self.library.list_files(), [])
        self.assertEqual(self.store.get()['ids'], ['original'])

    def test_embedding_failure_rolls_back_partial_vectors_and_allows_retry(self):
        original = self.store.add_documents
        def fail_after_write(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('simulated network failure')
        with patch.object(self.store, 'add_documents', side_effect=fail_after_write):
            with self.assertRaises(UploadError):
                self.library.upload('retry.txt', b'new knowledge')
        self.assertEqual(self.store.get()['ids'], ['original'])
        self.assertEqual(self.library.list_files(), [])
        self.assertEqual(self.library.upload('retry.txt', b'new knowledge')['status'], 'ready')

    def test_http_upload_list_size_and_origin_validation(self):
        with TestClient(create_app(knowledge_factory=lambda kb_id: self.library)) as http:
            self.assertEqual(http.post('/api/knowledge/upload', files={'file': ('a.txt', b'new rule')}, headers={'Origin':'https://unrelated.invalid'}).status_code, 403)
            response = http.post('/api/knowledge/upload', files={'file': ('a.txt', b'new rule')})
            self.assertEqual(response.status_code, 201, response.text)
            self.assertEqual(len(http.get('/api/knowledge').json()['files']), 1)
            with patch('web_app.MAX_UPLOAD_BYTES', 5):
                self.assertEqual(http.post('/api/knowledge/upload', files={'file': ('large.txt', b'123456')}).status_code, 413)
            self.assertEqual(http.post('/api/knowledge/upload', files={'file': ('bad.exe', b'x')}).status_code, 400)


if __name__ == '__main__':
    unittest.main()

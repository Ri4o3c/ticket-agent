# 验证真实Chroma collection及文件清单隔离、MCP绑定、空库不回退和切库新会话。
# 使用临时目录和本地测试向量，不修改用户知识库。
"""多知识库隔离：真实文件清单、Chroma与MCP，外部向量API采用本地实现。"""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastmcp import Client
from langchain_chroma import Chroma
from knowledge_upload import KnowledgeLibrary
from mcp_tools import make_server
from ticket_store import TicketStore, Actor
from web_app import create_app
from test_knowledge_upload import LocalEmbeddings
from test_web_app import WebModel
try:
    from knowledge_bases import KnowledgeBases
except ImportError:
    KnowledgeBases = None


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(KnowledgeBases, '独立知识库管理尚未实现')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = KnowledgeBases(self.root / 'bases.json')
        self.stores, self.libraries = {}, {}

    def library(self, kb_id='default'):
        if kb_id not in self.stores:
            store = Chroma(collection_name='isolation_' + uuid.uuid4().hex, embedding_function=LocalEmbeddings())
            self.addCleanup(store.delete_collection)
            self.stores[kb_id] = store
            self.libraries[kb_id] = KnowledgeLibrary(self.root / kb_id, store)
        return self.libraries[kb_id]

    def test_names_persist_and_unknown_ids_do_not_fall_back(self):
        self.assertEqual(self.catalog.list()[0]['id'], 'default')
        item = self.catalog.create('产品说明')
        self.assertEqual(KnowledgeBases(self.root / 'bases.json').get(item['id'])['name'], '产品说明')
        for name in ['', '   ', '默认知识库', '产品说明', 'x' * 61]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.catalog.create(name)
        with self.assertRaises(KeyError):
            self.catalog.get('../../default')

    def test_real_open_store_uses_separate_collections_and_keeps_legacy_default(self):
        # 子进程退出后关闭Chroma文件句柄，Windows临时目录才能完整清理。
        code = '''
import sys
from pathlib import Path
from unittest.mock import patch
import rag_index
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_knowledge_upload import LocalEmbeddings
with patch.object(rag_index, 'BASE_DIR', Path(sys.argv[1])):
    default, _ = rag_index.open_store(LocalEmbeddings(), 'https://test.invalid/v1')
    a, _ = rag_index.open_store(LocalEmbeddings(), 'https://test.invalid/v1', 'kb_' + 'a'*32)
    b, _ = rag_index.open_store(LocalEmbeddings(), 'https://test.invalid/v1', 'kb_' + 'b'*32)
    default.add_texts(['old knowledge'], ids=['old'])
    a.add_texts(['ONLY-A'], ids=['same'])
    b.add_texts(['ONLY-B'], ids=['same'])
    assert default._collection.name == 'ticket_knowledge'
    assert a.similarity_search('ONLY', k=5)[0].page_content == 'ONLY-A'
    assert b.similarity_search('ONLY', k=5)[0].page_content == 'ONLY-B'
    assert default.get()['documents'] == ['old knowledge']
'''
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, str(self.root)], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_upload_file_lists_and_duplicates_are_scoped_per_library(self):
        from starlette.testclient import TestClient
        app = create_app(knowledge_factory=self.library, catalog=self.catalog)
        with TestClient(app) as http:
            a = http.post('/api/knowledge/bases', json={'name':'A资料'}).json()['knowledge_base']
            b = http.post('/api/knowledge/bases', json={'name':'B资料'}).json()['knowledge_base']
            content = b'isolated document'
            for item in [a, b]:
                response = http.post('/api/knowledge/upload', params={'kb_id':item['id']}, files={'file':('same.md',content)})
                self.assertEqual(response.status_code, 201, response.text)
                self.assertEqual(len(http.get('/api/knowledge', params={'kb_id':item['id']}).json()['files']), 1)
            self.assertEqual(http.get('/api/knowledge').json()['files'], [])
            duplicate = http.post('/api/knowledge/upload', params={'kb_id':a['id']}, files={'file':('renamed.md',content)})
            self.assertEqual(duplicate.status_code, 200)
            self.assertTrue(duplicate.json()['duplicate'])
            for method, url, kwargs in [('get','/api/knowledge',{'params':{'kb_id':'missing'}}),
                    ('post','/api/sessions',{'json':{'kb_id':'missing'}}),
                    ('post','/api/knowledge/upload',{'params':{'kb_id':'missing'}, 'files':{'file':('a.md',content)}})]:
                self.assertEqual(getattr(http,method)(url,**kwargs).status_code, 404)

    def test_new_library_conversation_has_no_old_history_and_keeps_ticket_records(self):
        from starlette.testclient import TestClient
        a, b = self.catalog.create('A'), self.catalog.create('B')
        store = TicketStore(self.root / 'tickets.sqlite3')
        actor = Actor('test-user', 'test-workspace')
        draft = store.prepare_draft('existing ticket', 'keep record', actor)
        store.approve_draft(draft['draft_id'], actor)
        ticket = store.create_ticket(draft['draft_id'], actor)
        bindings = []
        def factory(kb_id):
            bindings.append(kb_id)
            return Client(make_server(store=store, actor=actor, searcher=lambda q,k: []))
        app = create_app(client_factory=factory, model_factory=lambda url:WebModel(), store_factory=lambda:store,
                         actor_factory=lambda:actor, knowledge_factory=self.library, catalog=self.catalog)
        with TestClient(app) as http:
            for item, text in [(a,'A的私有会话内容'), (b,'你好')]:
                sid = http.post('/api/sessions', json={'kb_id':item['id']}).json()['session_id']
                url = f'/api/sessions/{sid}'
                for _ in range(100):
                    state = http.get(url+'/events').json()
                    if state['ready']: break
                    time.sleep(.01)
                self.assertEqual(state['knowledge_base']['id'], item['id'])
                http.post(url+'/messages', json={'message':text, 'kb_id':a['id']})
                for _ in range(100):
                    state = http.get(url+'/events').json()
                    if not state['busy']: break
                    time.sleep(.01)
                if item == b:
                    self.assertNotIn('A的私有会话内容', json.dumps(state,ensure_ascii=False))
                self.assertEqual(http.delete(url).status_code, 200)
            self.assertEqual(bindings,[a['id'],b['id']])
            self.assertEqual(store.get_ticket(ticket['ticket_id'],actor)['title'],'existing ticket')

    def test_mcp_search_is_bound_to_server_library_without_model_selector(self):
        a, b = self.catalog.create('A'), self.catalog.create('B')
        for item, text in [(a,'ONLY-A'),(b,'ONLY-B')]:
            self.library(item['id']).upload('scope.md', text.encode())
        async def check():
            for item, expected in [(a,'ONLY-A'),(b,'ONLY-B')]:
                with patch('rag_index.make_embeddings', return_value=(LocalEmbeddings(),'https://test.invalid/v1')), \
                     patch('rag_index.open_store', side_effect=lambda emb,url,kb_id='default':(self.library(kb_id).store,self.root)):
                    async with Client(make_server(store=TicketStore(self.root/'tickets.sqlite3'), knowledge_base=item['id'])) as client:
                        definition = next(t for t in await client.list_tools() if t.name=='search_docs')
                        self.assertNotIn('kb_id', definition.input_schema['properties'])
                        result = await client.call_tool('search_docs', {'question':'scope','k':5})
                        data = result.data
                        self.assertTrue(data['ok'])
                        self.assertEqual([r['text'] for r in data['results']], [expected])
        asyncio.run(check())

    def test_empty_library_does_not_fall_back_to_default_knowledge(self):
        empty = self.catalog.create('空库')
        self.library('default').upload('private.md', b'DEFAULT-PRIVATE-DATA')
        async def check():
            with patch('rag_index.make_embeddings', return_value=(LocalEmbeddings(),'https://test.invalid/v1')), \
                 patch('rag_index.open_store', side_effect=lambda emb,url,kb_id='default':(self.library(kb_id).store,self.root)):
                async with Client(make_server(store=TicketStore(self.root/'tickets.sqlite3'), knowledge_base=empty['id'])) as client:
                    result = await client.call_tool('search_docs', {'question':'private data'})
                    self.assertFalse(result.data['ok'])
                    self.assertEqual(result.data['error'], 'empty_index')
                    self.assertNotIn('DEFAULT-PRIVATE-DATA', json.dumps(result.data))
        asyncio.run(check())

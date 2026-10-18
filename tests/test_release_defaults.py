# 发布版本的首次使用检查：清空环境并使用临时配置，验证没有隐式接口或模型回退。
# 未配置时会话和上传应提前拒绝，知识列表不应触发Embedding初始化。
"""GitHub副本的首次运行必须无预设服务，也不得继承作者的旧配置变量。"""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from starlette.testclient import TestClient
from knowledge_bases import KnowledgeBases
from service_settings import ServiceSettings
from ticket_store import TicketStore
from web_app import create_app


class ReleaseDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.enterContext(patch.dict(os.environ,{
            'CHAT_BASE_URL':'','CHAT_API_KEY':'','CHAT_MODEL':'',
            'EMBEDDING_BASE_URL':'','EMBEDDING_API_KEY':'','EMBEDDING_MODEL':'','EMBEDDING_DIMENSIONS':'',
            'MINERU_BASE_URL':'','MINERU_API_KEY':'',
            'DASHSCOPE_BASE_URL':'https://legacy.invalid/v1','DASHSCOPE_API_KEY':'TEST-LEGACY-SECRET',
        }))
        self.settings=ServiceSettings(self.root/'settings.json',self.root/'.env')

    def app(self):
        return create_app(settings=self.settings,catalog=KnowledgeBases(self.root/'bases.json'),
                          store_factory=lambda:TicketStore(self.root/'tickets.sqlite3'))

    def test_fresh_settings_have_no_endpoint_key_model_or_dimension_defaults(self):
        for name in ['chat','embedding','mineru']:
            config=self.settings.effective()[name]
            self.assertEqual(config['base_url'],'')
            self.assertEqual(config['api_key'],'')
            if name!='mineru': self.assertEqual(config['model'],'')
            self.assertFalse(self.settings.public()[name]['configured'])
        self.assertIsNone(self.settings.effective()['embedding']['dimensions'])

    def test_legacy_provider_variables_do_not_configure_release(self):
        self.assertFalse(self.settings.ready())
        self.assertNotIn('TEST-LEGACY-SECRET',str(self.settings.public()))

    def test_explicit_generic_environment_values_are_used(self):
        with patch.dict(os.environ,{'CHAT_BASE_URL':'https://chat.invalid/v1','CHAT_API_KEY':'TEST-CHAT-SECRET','CHAT_MODEL':'user-chat',
                 'EMBEDDING_BASE_URL':'https://embed.invalid/v1','EMBEDDING_API_KEY':'TEST-EMBED-SECRET','EMBEDDING_MODEL':'user-embed','EMBEDDING_DIMENSIONS':'768'}):
            self.assertTrue(self.settings.ready())
            self.assertEqual(self.settings.effective()['embedding']['dimensions'],768)
            self.assertEqual(self.settings.public()['chat']['model'],'user-chat')

    def test_unconfigured_new_session_does_not_start_mcp(self):
        with patch('web_app.default_client',side_effect=AssertionError('未配置不得启动MCP')) as client, TestClient(self.app()) as http:
            response=http.post('/api/sessions')
            self.assertEqual(response.status_code,409,response.text)
            client.assert_not_called()

    def test_unconfigured_library_list_does_not_initialize_embeddings(self):
        with patch('rag_index.make_embeddings',side_effect=AssertionError('未配置不得初始化向量服务')), TestClient(self.app()) as http:
            response=http.get('/api/knowledge')
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['files'],[])
            self.assertTrue(response.json()['configuration_required'])

    def test_unconfigured_upload_is_rejected_with_configuration_message(self):
        with patch('rag_index.make_embeddings',side_effect=AssertionError('未配置不得初始化向量服务')), TestClient(self.app()) as http:
            response=http.post('/api/knowledge/upload',files={'file':('demo.md',b'example document')})
            self.assertEqual(response.status_code,409,response.text)
            self.assertIn('配置',response.json()['error'])

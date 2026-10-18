# 使用临时文件、人工测试Key和模拟模型客户端，验证独立配置、DPAPI及公开接口。
# 特别检查换地址不能隐式转发原Key，保存配置会使旧会话失效；不验证真实服务连通性。
"""服务配置：只用临时文件与虚构Key，不调用任何外部API。"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starlette.testclient import TestClient
from web_app import create_app
try:
    from service_settings import ServiceSettings
except ImportError:
    ServiceSettings = None


class ServiceSettingsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ServiceSettings, '服务配置入口尚未实现')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.enterContext(patch.dict(os.environ, {'CHAT_API_KEY':'TEST-ENV-SECRET','CHAT_BASE_URL':'https://old.invalid/v1','CHAT_MODEL':'test-chat','EMBEDDING_API_KEY':'TEST-ENV-SECRET','EMBEDDING_BASE_URL':'https://old.invalid/v1','EMBEDDING_MODEL':'test-embedding','EMBEDDING_DIMENSIONS':'1024'}))
        self.settings = ServiceSettings(self.root/'settings.json', self.root/'.env')

    def test_existing_environment_is_default_without_exposing_keys(self):
        config = self.settings.effective()
        self.assertEqual(config['chat']['model'], 'test-chat')
        self.assertEqual(config['embedding']['model'], 'test-embedding')
        self.assertEqual(config['chat']['api_key'], 'TEST-ENV-SECRET')
        public = self.settings.public()
        self.assertTrue(public['chat']['has_key'])
        self.assertNotIn('TEST-ENV-SECRET', json.dumps(public))
        self.assertEqual(public['mineru']['status'], 'configuration_only')

    def test_save_encrypts_keys_and_blank_key_preserves_saved_value(self):
        self.settings.save({'chat': {'base_url':'http://127.0.0.1:9000/v1/', 'model':'my-model', 'api_key':'TEST-PRIVATE-SECRET'}})
        saved = self.settings.path.read_text(encoding='utf-8')
        self.assertNotIn('TEST-PRIVATE-SECRET', saved)
        self.assertEqual(ServiceSettings(self.settings.path,self.root/'.env').effective()['chat']['api_key'],'TEST-PRIVATE-SECRET')
        self.settings.save({'chat':{'api_key':'', 'model':'another-model'}})
        self.assertEqual(self.settings.effective()['chat']['api_key'],'TEST-PRIVATE-SECRET')
        self.assertEqual(self.settings.public()['chat']['base_url'],'http://127.0.0.1:9000/v1')

    def test_bad_urls_dimensions_and_unknown_fields_do_not_overwrite_config(self):
        self.settings.save({'chat':{'model':'keep-me'}})
        before = self.settings.path.read_bytes()
        for change in [
            {'chat':{'base_url':'file:///C:/private'}},
            {'chat':{'base_url':'https://user:password@example.invalid'}},
            {'embedding':{'dimensions':True}},
            {'embedding':{'dimensions':-1}},
            {'chat':{'model':''}},
            {'chat':{'api_key':'bad\nheader'}},
            {'chat':{'unknown':42}},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.settings.save(change)
            self.assertEqual(self.settings.path.read_bytes(),before)

    def test_mineru_address_and_token_are_saved_without_enabling_parser(self):
        self.settings.save({'mineru':{'base_url':'http://192.168.1.20:8000', 'api_key':'TEST-MINERU-SECRET'}})
        self.assertEqual(self.settings.effective()['mineru']['api_key'],'TEST-MINERU-SECRET')
        public = self.settings.public()
        self.assertEqual(public['mineru']['status'],'configuration_only')
        self.assertNotIn('TEST-MINERU-SECRET',json.dumps(public))

    def test_http_read_save_and_cross_origin_do_not_expose_secret(self):
        with TestClient(create_app(settings=self.settings)) as http:
            response = http.post('/api/settings', json={'chat':{'api_key':'TEST-HTTP-SECRET','model':'custom-chat'}})
            self.assertEqual(response.status_code,200,response.text)
            self.assertNotIn('TEST-HTTP-SECRET',response.text)
            response = http.get('/api/settings')
            self.assertEqual(response.json()['chat']['model'],'custom-chat')
            self.assertNotIn('TEST-HTTP-SECRET',response.text)
            self.assertEqual(http.post('/api/settings',json={},headers={'Origin':'https://unrelated.invalid'}).status_code,403)
            self.assertEqual(http.post('/api/settings',json={'chat':{'base_url':'ftp://bad.invalid'}}).status_code,400)

    def test_runtime_models_use_independent_saved_addresses_and_models(self):
        self.settings.save({'chat':{'base_url':'https://chat.invalid/v1','model':'custom-chat','api_key':'TEST-CHAT-SECRET'},
                            'embedding':{'base_url':'https://embed.invalid/v1','model':'custom-embedding','dimensions':384,'api_key':'TEST-EMBED-SECRET'}})
        with patch('service_settings.DEFAULT_SETTINGS_PATH',self.settings.path), patch('service_settings.DEFAULT_ENV_PATH',self.root/'.env'):
            from rag_answer import make_chat_model
            from rag_index import make_embeddings
            chat = make_chat_model('https://old.invalid/v1')
            embedding, url = make_embeddings()
        self.assertEqual(chat.model_name,'custom-chat')
        self.assertEqual(str(chat.openai_api_base),'https://chat.invalid/v1')
        self.assertEqual(chat.openai_api_key.get_secret_value(),'TEST-CHAT-SECRET')
        self.assertEqual(embedding.model,'custom-embedding')
        self.assertEqual(embedding.dimensions,384)
        self.assertEqual(url,'https://embed.invalid/v1')
        self.assertEqual(embedding.openai_api_key.get_secret_value(),'TEST-EMBED-SECRET')

    def test_embedding_model_change_selects_different_index_directory(self):
        from rag_index import make_embeddings, open_store
        with patch('service_settings.DEFAULT_SETTINGS_PATH',self.settings.path), patch('service_settings.DEFAULT_ENV_PATH',self.root/'.env'), \
             patch('langchain_chroma.Chroma'):
            old, url = make_embeddings()
            _, old_dir = open_store(old,url)
            self.settings.save({'embedding':{'model':'another-embedding'}})
            new, url = make_embeddings()
            _, new_dir = open_store(new,url)
        self.assertNotEqual(old_dir,new_dir)

    def test_save_rejects_pending_confirmation_and_invalidates_idle_session(self):
        import time
        from fastmcp import Client
        from mcp_tools import make_server
        from ticket_store import TicketStore, Actor
        from test_web_app import WebModel
        store = TicketStore(self.root/'tickets.sqlite3')
        actor = Actor('settings-test','settings-space')
        app = create_app(settings=self.settings, client_factory=lambda kb_id:Client(make_server(store=store,actor=actor,searcher=lambda q,k:[])),
                         model_factory=lambda url:WebModel(), store_factory=lambda:store, actor_factory=lambda:actor)
        with TestClient(app) as http:
            sid=http.post('/api/sessions').json()['session_id']
            url=f'/api/sessions/{sid}'
            def wait_for(predicate):
                for _ in range(200):
                    state=http.get(url+'/events').json()
                    if predicate(state): return state
                    time.sleep(.01)
                self.fail('会话状态未完成推进')
            wait_for(lambda s:s['ready'])
            http.post(url+'/messages',json={'message':'登记问题'})
            state=wait_for(lambda s:s['pending_draft'] is not None)
            self.assertEqual(http.post('/api/settings',json={'chat':{'model':'new-model'}}).status_code,409)
            self.assertEqual(self.settings.effective()['chat']['model'],'test-chat')
            draft=state['pending_draft']
            http.post(url+'/confirm',json={'draft_id':draft['draft_id'],'approved':False})
            wait_for(lambda s:not s['busy'])
            self.assertEqual(http.post('/api/settings',json={'chat':{'model':'new-model'}}).status_code,200)
            self.assertEqual(http.get(url+'/events').status_code,404)
            self.assertIsNone(store.get_draft(draft['draft_id'],actor)['approved_at'])

    def test_embedding_dimensions_can_be_left_to_model_default(self):
        self.settings.save({'embedding':{'dimensions':None}})
        with patch('service_settings.DEFAULT_SETTINGS_PATH',self.settings.path), patch('service_settings.DEFAULT_ENV_PATH',self.root/'.env'):
            from rag_index import make_embeddings
            model, _ = make_embeddings()
        self.assertIsNone(model.dimensions)

    def test_changed_endpoint_requires_explicit_key_instead_of_reusing_old_secret(self):
        with self.assertRaises(ValueError):
            self.settings.save({'chat':{'base_url':'https://new-service.invalid/v1','api_key':''}})
        self.assertEqual(self.settings.effective()['chat']['base_url'],'https://old.invalid/v1')


if __name__=='__main__':
    unittest.main()

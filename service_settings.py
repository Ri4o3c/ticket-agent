# 配置边界：这里只保存用户参数与保护本机凭证，不提供默认服务，也不探测第三方可用性。
"""本机服务配置。Key使用当前Windows账户的DPAPI加密，公开接口不返回明文。"""
import base64
import copy
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = BASE_DIR / 'data' / 'service_settings.json'
DEFAULT_ENV_PATH = BASE_DIR / '.env'


# Windows DPAPI加解密适配：bytes输入，bytes输出，decrypt切换保护方向。
# 密文绑定当前Windows账户环境；Base64只是文件存储编码，不是加密算法。
# 系统分配的输出内存必须LocalFree释放；Windows失败只报告通用错误，不打印密钥。
def _protect(data, *, decrypt=False):
    if os.name != 'nt':
        raise RuntimeError('本版凭证保存使用Windows账户加密，请在Windows运行。')
    class Blob(ctypes.Structure):
        _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise RuntimeError('无法读取或保存本机加密凭证，请使用原Windows账户重新填写。')
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel.LocalFree(ctypes.cast(target.pbData, ctypes.c_void_p))


# 规范化用户Base URL并返回去掉末尾斜线的字符串；只接受HTTP(S)且有主机名。
# 拒绝账号密码、查询参数、片段、空白和占位符，避免把Key夹在地址里。
# optional=True只允许空字符串，不代表已连通，也不验证服务是否兼容模型API。
def _url(value, *, optional=False):
    if not isinstance(value, str):
        raise ValueError('接口地址需要填写HTTP或HTTPS网址。')
    value = value.strip().rstrip('/')
    if not value and optional:
        return ''
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme in {'http','https'} and parsed.hostname and not parsed.username
                 and not parsed.password and not parsed.query and not parsed.fragment and '<' not in value
                 and not any(c.isspace() for c in value))
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('接口地址需要完整HTTP或HTTPS网址；不要把Key放进网址、账号或查询参数。')
    return value


# 统一管理聊天、Embedding和MinerU的独立参数。原始配置用于本机调用，public用于前端。
# 发布版没有预设API、模型或密钥；MinerU只保存参数，不会因此启用文档解析。
class ServiceSettings:
    # 默认使用本项目data/service_settings.json及.env；测试可传临时路径，避免碰真实配置。
    def __init__(self, path=None, env_path=None):
        self.path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
        self.env_path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH

    # 读取磁盘上保存的JSON，没有文件时返回空字典；此步不解密、不回填任何服务商参数。
    def _stored(self):
        return json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}

    # 合并顺序：.env → 进程环境同名变量覆盖 → 已保存的界面字段覆盖。
    # 分别读取CHAT_*、EMBEDDING_*、MINERU_*；不回退到旧共享服务商变量。
    # 维度留空为None，交由服务决定；持久化Key存在时用DPAPI解密覆盖环境Key。
    def _effective(self, stored):
        env = {**dotenv_values(self.env_path), **os.environ}
        def value(name):
            return str(env.get(name) or '').strip()
        dimensions = value('EMBEDDING_DIMENSIONS')
        values = {
            'chat': {'base_url':value('CHAT_BASE_URL').rstrip('/'), 'api_key':value('CHAT_API_KEY'), 'model':value('CHAT_MODEL')},
            'embedding': {'base_url':value('EMBEDDING_BASE_URL').rstrip('/'), 'api_key':value('EMBEDDING_API_KEY'),
                          'model':value('EMBEDDING_MODEL'), 'dimensions':int(dimensions) if dimensions else None},
            'mineru': {'base_url':value('MINERU_BASE_URL').rstrip('/'), 'api_key':value('MINERU_API_KEY')},
        }
        for section, defaults in values.items():
            custom = stored.get(section,{})
            defaults.update({k:v for k,v in custom.items() if k != 'key_encrypted'})
            if custom.get('key_encrypted'):
                defaults['api_key'] = _protect(base64.b64decode(custom['key_encrypted']), decrypt=True).decode('utf-8')
        return values

    # 返回含明文api_key的运行时字典，仅供本机服务构造模型客户端。
    # 不得直接作为HTTP响应、日志或GitHub示例输出；浏览器应使用public。
    def effective(self):
        return self._effective(self._stored())

    # 生成一份新有效配置，将明文api_key移除，换成has_key及configured布尔状态。
    # 保留地址、模型和维度供表单回显；MinerU额外标记configuration_only。
    def public(self):
        values = self.effective()
        for name, section in values.items():
            section['configured'] = self._configured(name,section)
            section['has_key'] = bool(section.pop('api_key'))
            if name == 'mineru':
                section['status'] = 'configuration_only'
        return values

    # 仅检查地址格式、Key及模型名是否齐备；MinerU不要求模型名。
    # 这是本机参数完整性检查，不会发送联网请求，不能证明Key有效或模型可用。
    @staticmethod
    def _configured(name,section):
        try:
            _url(section['base_url'])
        except ValueError:
            return False
        return bool(section['api_key'] and (name=='mineru' or section['model']))

    # 聊天和Embedding都具备必要字段才允许默认Web流程开始；MinerU不参与就绪判断。
    def ready(self):
        public = self.public()
        return public['chat']['configured'] and public['embedding']['configured']

    # 接收分服务的局部修改，拒绝未知字段；在副本中校验和加密后再写磁盘。
    # 空Key表示保留当前值，并非删除；换地址且已有Key时必须显式提供对应的新Key。
    # 同目录临时文件替换减少写一半的风险；返回public而不是含Key的effective。
    def save(self, changes):
        if not isinstance(changes, dict) or set(changes) - {'chat','embedding','mineru'}:
            raise ValueError('服务配置格式不正确。')
        stored = copy.deepcopy(self._stored())
        current = self._effective(stored)
        for name, changeset in changes.items():
            allowed = {'base_url','api_key'} | ({'model','dimensions'} if name == 'embedding' else {'model'} if name == 'chat' else set())
            if not isinstance(changeset,dict) or set(changeset)-allowed:
                raise ValueError('服务配置包含不支持的字段。')
            if 'base_url' in changeset:
                new_url = _url(changeset['base_url'],optional=name=='mineru')
                new_key = changeset.get('api_key','')
                if (new_url and new_url != current[name]['base_url'] and current[name]['api_key']
                        and (not isinstance(new_key,str) or not new_key.strip())):
                    raise ValueError('更换接口地址时，请同时填写对应服务的Key；原Key不会自动转发到新地址。')
            section = stored.setdefault(name,{})
            for field, value in changeset.items():
                if field == 'api_key':
                    if not isinstance(value,str) or len(value)>8192 or any(ord(c)<32 for c in value):
                        raise ValueError('API Key或Token格式不正确。')
                    if value.strip():
                        section['key_encrypted'] = base64.b64encode(_protect(value.strip().encode())).decode('ascii')
                elif field == 'base_url':
                    section[field] = _url(value,optional=name=='mineru')
                elif field == 'model':
                    if not isinstance(value,str) or not value.strip() or len(value)>200 or any(ord(c)<32 for c in value):
                        raise ValueError('请填写有效的模型名称。')
                    section[field] = value.strip()
                elif field == 'dimensions':
                    if value is not None and (type(value) is not int or not 1<=value<=65536):
                        raise ValueError('向量维度需要1到65536的整数，或留空使用模型默认维度。')
                    section[field] = value
        # 字段检查通过后，先尝试合并和解密，再以临时文件替换原配置。
        # 这里只验证合并/解密能执行；不会请求API验证模型名、Key或实际返回的向量维度。
        self._effective(stored)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=self.path.parent,delete=False) as stream:
            json.dump(stored,stream,ensure_ascii=False,indent=2)
            temporary = Path(stream.name)
        temporary.replace(self.path)
        return self.public()

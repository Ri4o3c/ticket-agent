"""本地单用户的知识库目录；名称用于展示，随机ID用于绑定存储和会话。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
import uuid

DEFAULT_BASE = {'id': 'default', 'name': '默认知识库'}


# ID只接受default或kb_加32位十六进制；验证的是标识格式，不检查目录清单是否有该库。
# 限制ID形式使其可安全拼接collection和目录；展示名称不参与路径计算。
def validate_base_id(kb_id):
    if not isinstance(kb_id, str) or (kb_id != 'default' and not re.fullmatch(r'kb_[0-9a-f]{32}', kb_id)):
        raise ValueError('知识库编号无效')
    return kb_id


# 管理本机单用户的知识库目录清单；它不保存正文、向量或工单。
class KnowledgeBases:
    # 只记录清单路径，不创建数据文件；首次新建知识库时才持久化。
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else Path(__file__).resolve().parent / 'data' / 'knowledge_bases.json'

    # 默认库始终返回在第一项，自建库从JSON读取；默认库不重复写进清单文件。
    def list(self):
        custom = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else []
        return [dict(DEFAULT_BASE), *custom]

    # 按清单查已存在库；未知ID抛KeyError，不悄悄替换成默认库。
    # Web层再把这个异常转换为404提示。
    def get(self, kb_id):
        for item in self.list():
            if item['id'] == kb_id:
                return item
        raise KeyError('知识库不存在，请重新选择。')

    # 校验名称长度和控制字符，忽略大小写检查重名；随机生成稳定ID并记录UTC时间。
    # 用同目录临时文件替换清单，成功返回新库信息；此时尚未创建向量或导入文档。
    def create(self, name):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 60 or any(ord(c) < 32 for c in name):
            raise ValueError('知识库名称需要1到60个字符，不能包含换行或控制字符。')
        name = name.strip()
        bases = self.list()
        if any(item['name'].casefold() == name.casefold() for item in bases):
            raise ValueError('已有同名知识库，请换个名称。')
        item = {'id':'kb_' + uuid.uuid4().hex, 'name':name, 'created_at':datetime.now(timezone.utc).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent, delete=False) as stream:
            json.dump([*bases[1:], item], stream, ensure_ascii=False, indent=2)
            temporary = Path(stream.name)
        temporary.replace(self.path)
        return item

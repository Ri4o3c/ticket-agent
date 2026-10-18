"""本地学习版工单存储：草稿确认、按身份查询、SQLite 去重。"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import re
import sqlite3
import uuid

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "data" / "tickets.sqlite3"
TICKET_STATUSES = ("待处理", "处理中", "已解决")


# 不可变的本机身份上下文，记录用户、工作空间及角色；从程序配置构造。
# 它用于本地演示的归属检查，不等于真实网络登录或多租户认证系统。
@dataclass(frozen=True)
class Actor:
    user_id: str
    workspace_id: str
    role: str = "member"

    # dataclass构造后校验身份字段；冻结对象可减少会话中误改角色的风险。
    def __post_init__(self):
        if not self.user_id.strip() or not self.workspace_id.strip():
            raise ValueError("用户和工作空间不能为空")
        if self.role not in {"member", "admin"}:
            raise ValueError("演示角色只能是 member 或 admin")


def local_actor() -> Actor:
    """身份由本地程序配置提供，不接受模型工具参数中的用户或角色声明。

    这是单机演示身份，并非真实登录鉴权。后续可替换为应用登录上下文。
    """
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    return Actor(os.getenv("TICKET_USER_ID", "demo-user"),
                 os.getenv("TICKET_WORKSPACE_ID", "demo-workspace"),
                 os.getenv("TICKET_ROLE", "member"))


# 统一返回UTC、秒级ISO时间字符串，便于CLI和Web使用相同时间格式。
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# 依次遮盖常见“字段名=值”凭证、sk-格式密钥和15到19位长号码。
# 正则是启发式规则，可能漏检或误检；草稿落库前调用，不能当作完整隐私过滤器。
def redact(text: str) -> str:
    """基础教学脱敏规则，不是完整的敏感信息检测器。"""
    text = re.sub(r"(?i)(password|api[_ -]?key|token|密码|验证码)[ \t]*[:：=][ \t]*[^\s,，;；]+",
                  r"\1=[已隐藏]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[密钥已隐藏]", text)
    return re.sub(r"(?<!\d)\d{15,19}[Xx]?(?!\d)", "[长号码已隐藏]", text)


# 持久化业务层：SQLite保存草稿和正式工单，模型和页面都不能绕过这里的审批校验。
# D-标识草稿，R-标识一次创建请求，T-标识正式工单；三者用途不同。
class TicketStore:
    # 创建数据目录和缺失表，保留已有记录。tickets的联合唯一约束以空间、用户、
    # request_id为幂等边界；number自增后转换成展示用的T-编号。
    def __init__(self, path: Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # IF NOT EXISTS只创建缺失表，不清空已有工单；这不是自动数据库结构迁移工具。
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS drafts (
                    draft_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    approved_at TEXT,
                    request_id TEXT
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    number INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    draft_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT '待处理',
                    created_at TEXT NOT NULL,
                    UNIQUE(workspace_id, user_id, request_id)
                );
            """)

    # 每次操作创建独立连接；Row允许按列名读取。with db负责提交或异常回滚，
    # finally负责关闭连接；timeout=15是锁等待时间，不是模型或API请求超时。
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    # 验证标题和描述的非空及长度，脱敏后写入未批准草稿。返回当前用户的草稿字典，
    # approved_at和request_id初始为NULL；此操作不会向tickets表写入正式工单。
    def prepare_draft(self, title: str, description: str, actor: Actor) -> dict:
        if not title.strip() or not description.strip():
            raise ValueError("标题和问题描述不能为空")
        if len(title) > 200 or len(description) > 4000:
            raise ValueError("标题最多 200 字，描述最多 4000 字")
        draft_id = "D-" + uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO drafts VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
                       (draft_id, actor.workspace_id, actor.user_id,
                        redact(title.strip()), redact(description.strip()), now()))
        return self.get_draft(draft_id, actor)

    # 在调用方提供的连接内同时按草稿ID、空间和用户查询。
    # 不存在或不归当前用户所有都抛PermissionError，供审批和创建复用。
    def _owned_draft(self, db, draft_id, actor):
        row = db.execute("SELECT * FROM drafts WHERE draft_id=? AND workspace_id=? AND user_id=?",
                         (draft_id, actor.workspace_id, actor.user_id)).fetchone()
        if row is None:
            raise PermissionError("草稿不存在或当前用户无权访问")
        return row

    # 按当前身份读取草稿并转成普通字典；不修改批准时间或请求编号。
    def get_draft(self, draft_id: str, actor: Actor) -> dict:
        with self.connect() as db:
            return dict(self._owned_draft(db, draft_id, actor))

    # 可信宿主收到明确人工确认后调用，模型无法经MCP直接访问。
    # BEGIN IMMEDIATE先取得写事务；首次批准生成R-请求和时间，再次批准保留原值。
    # 草稿不提供原地改标题/描述的接口；内容变化应创建新草稿并重新确认。
    def approve_draft(self, draft_id: str, actor: Actor) -> dict:
        """仅由可信CLI或Web宿主在用户确认后调用，不注册为MCP工具。

        草稿内容不可变；修改问题需准备新草稿并重新确认。
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            draft = self._owned_draft(db, draft_id, actor)
            if not draft["approved_at"]:
                db.execute("UPDATE drafts SET approved_at=?, request_id=? WHERE draft_id=?",
                           (now(), "R-" + uuid.uuid4().hex, draft_id))
        return self.get_draft(draft_id, actor)

    # 把数据库内部number字段转换为T-加至少六位数字的展示编号。
    # 返回新字典，其他业务字段保持原记录的值。
    @staticmethod
    def _ticket(row) -> dict:
        result = dict(row)
        result["ticket_id"] = f"T-{result.pop('number'):06d}"
        return result

    # 同一写事务内依次校验归属与批准、查原请求、必要时插入，然后返回正式工单。
    # 已存在时返回原记录，不重置状态或创建时间；同时依靠唯一约束防止重复记录。
    def create_ticket(self, draft_id: str, actor: Actor) -> dict:
        with self.connect() as db:
            # 在同一事务中校验确认、检查既有记录并写入，避免并发重试重复创建。
            db.execute("BEGIN IMMEDIATE")
            draft = self._owned_draft(db, draft_id, actor)
            if not draft["approved_at"] or not draft["request_id"]:
                raise PermissionError("草稿尚未获用户确认，不能创建工单")
            values = (actor.workspace_id, actor.user_id, draft["request_id"])
            row = db.execute("SELECT * FROM tickets WHERE workspace_id=? AND user_id=? AND request_id=?",
                             values).fetchone()
            if row is None:
                db.execute("""INSERT INTO tickets
                    (workspace_id, user_id, request_id, draft_id, title, description, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (*values, draft_id, draft["title"], draft["description"], now()))
                row = db.execute("SELECT * FROM tickets WHERE workspace_id=? AND user_id=? AND request_id=?",
                                 values).fetchone()
            return self._ticket(row)

    # 校验T-编号后实时读库；普通成员只读本人记录，同空间admin可读该空间记录。
    # 此查询返回当前存储状态，不复用模型历史或向量库中的旧状态。
    def get_ticket(self, ticket_id: str, actor: Actor) -> dict:
        if not re.fullmatch(r"T-\d{6,}", ticket_id):
            raise ValueError("工单编号格式应为 T-000001")
        with self.connect() as db:
            row = db.execute("SELECT * FROM tickets WHERE number=? AND workspace_id=?",
                             (int(ticket_id[2:]), actor.workspace_id)).fetchone()
            if row is None or (row["user_id"] != actor.user_id and actor.role != "admin"):
                raise PermissionError("工单不存在或当前用户无权访问")
            return self._ticket(row)

    # 仅本机admin入口可调用：允许原状态不变，或按待处理→处理中→已解决前进一步。
    # 校验与更新处于同一写事务中，禁止跳级、倒退和跨空间操作；该函数未注册到MCP。
    def set_status(self, ticket_id: str, status: str, actor: Actor) -> dict:
        """工作人员入口：同空间 admin 按演示流程推进状态，不注册为 MCP 工具。"""
        if actor.role != "admin":
            raise PermissionError("仅工作人员（admin）可以更新工单状态")
        if not re.fullmatch(r"T-\d{6,}", ticket_id):
            raise ValueError("工单编号格式应为 T-000001")
        if status not in TICKET_STATUSES:
            raise ValueError("状态只能是待处理、处理中或已解决")
        next_status = {"待处理": "处理中", "处理中": "已解决", "已解决": None}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM tickets WHERE number=? AND workspace_id=?",
                             (int(ticket_id[2:]), actor.workspace_id)).fetchone()
            if row is None:
                raise PermissionError("工单不存在或当前工作人员无权访问")
            current = row["status"]
            if status != current and status != next_status.get(current):
                raise ValueError(f"不能从{current}改为{status}；演示流程为：待处理 → 处理中 → 已解决")
            db.execute("UPDATE tickets SET status=? WHERE number=? AND workspace_id=?",
                       (status, row["number"], actor.workspace_id))
            updated = db.execute("SELECT * FROM tickets WHERE number=? AND workspace_id=?",
                                 (row["number"], actor.workspace_id)).fetchone()
            return self._ticket(updated)

    # 按request_id和当前用户、空间核查创建结果；查到返回正式字典，否则返回None。
    # None只代表此刻本次查询未命中，调用异常应交给上层单独报告。
    def find_ticket_by_request(self, request_id: str, actor: Actor) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM tickets WHERE request_id=? AND workspace_id=? AND user_id=?",
                             (request_id, actor.workspace_id, actor.user_id)).fetchone()
            return self._ticket(row) if row else None

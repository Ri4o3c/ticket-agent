"""执行记录存储：持久保存一次提问及其步骤，由agent_trace适配Agent循环。

runs记录程序的执行状态，steps记录单次模型/工具/审批/校验的结果。
completed只表示本次执行结束，不表示回答正确或用户业务目标已经完成。
本模块不负责审批、写工单、自动恢复或评估，不保存对话、密钥和原始异常。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from ticket_store import Actor


DEFAULT_TRACE_DB = Path(__file__).resolve().parent / "data" / "traces.sqlite3"
RUN_END_STATUSES = {"completed", "failed", "cancelled"}
STEP_END_STATUSES = {"succeeded", "failed", "unknown", "cancelled"}
STEP_KINDS = {"model", "tool", "approval", "validation"}


def _now() -> str:
    """UTC时间用于记录发生时间；不拿系统时钟差值充当精确调用耗时。"""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _label(value: str | None, field: str, *, optional: bool = False) -> str | None:
    """只接受较短的标识和名称，避免误将整段内容当作元数据传入。"""
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{field}必须是1到200个字符的非空名称或标识")
    return value.strip()


class TraceStore:
    """面向可信宿主的SQLite记录库；不是可直接暴露给模型的MCP工具。

    每次公开写操作使用独立事务，返回前已提交；每次查询都检查本机Actor归属。
    当前只允许记录所有者读写，admin也不能绕过此限制。Actor来自程序配置，
    这类归属检查不等于真实网络登录鉴权。数据库文件本身需由部署者保护。
    """

    def __init__(self, path: Path = DEFAULT_TRACE_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            # IF NOT EXISTS保留旧记录。表中的状态、外键和顺序约束是最后一道保护。
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    knowledge_base_id TEXT,
                    model_name TEXT,
                    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    CHECK((status = 'running' AND finished_at IS NULL)
                       OR (status != 'running' AND finished_at IS NOT NULL))
                );
                CREATE INDEX IF NOT EXISTS runs_owner_time
                    ON runs(workspace_id, user_id, started_at);
                CREATE TABLE IF NOT EXISTS steps (
                    step_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    position INTEGER NOT NULL CHECK(position > 0),
                    kind TEXT NOT NULL CHECK(kind IN ('model', 'tool', 'approval', 'validation')),
                    name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running', 'succeeded', 'failed', 'unknown', 'cancelled')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms REAL CHECK(duration_ms >= 0),
                    error_type TEXT,
                    request_id TEXT,
                    input_tokens INTEGER CHECK(input_tokens >= 0),
                    output_tokens INTEGER CHECK(output_tokens >= 0),
                    UNIQUE(run_id, position),
                    CHECK((status = 'running' AND finished_at IS NULL)
                       OR (status != 'running' AND finished_at IS NOT NULL))
                );
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """连接不跨线程共享；正常退出提交，异常回滚，最后始终关闭。

        foreign_keys是每个连接的设置，不能只在建表连接上开启。
        timeout=15仅限制SQLite锁等待，不是模型或工具调用的超时。
        """
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _owned_run(db: sqlite3.Connection, run_id: str, actor: Actor) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM runs WHERE run_id=? AND workspace_id=? AND user_id=?",
            (run_id, actor.workspace_id, actor.user_id),
        ).fetchone()
        if row is None:
            raise PermissionError("执行记录不存在或当前用户无权访问")
        return row

    @staticmethod
    def _require_running(run: sqlite3.Row) -> None:
        if run["status"] != "running":
            raise ValueError("本次执行已结束，不能继续添加或修改步骤")

    def start_run(self, actor: Actor, *, session_id: str | None = None,
                  knowledge_base_id: str | None = None, model_name: str | None = None) -> str:
        """一次提问调用一次，生成run_id并立即保存running，不复用会话ID。

        名称与标识由可信宿主提供，不接收API Key、URL、提示词或错误原文。
        模型名可暂缺；未知字段保留NULL，不编造默认服务或实际调用信息。
        """
        session_id = _label(session_id, "session_id", optional=True)
        knowledge_base_id = _label(knowledge_base_id, "knowledge_base_id", optional=True)
        model_name = _label(model_name, "model_name", optional=True)
        run_id = f"run-{uuid4().hex}"
        with self.connect() as db:
            db.execute(
                """INSERT INTO runs
                   (run_id, workspace_id, user_id, session_id, knowledge_base_id,
                    model_name, status, started_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'running', ?)""",
                (run_id, actor.workspace_id, actor.user_id, session_id,
                 knowledge_base_id, model_name, _now()),
            )
        return run_id

    def get_run(self, run_id: str, actor: Actor) -> dict:
        """读取原样记录；重启后看到running不代表仍在运行，也不代表失败。"""
        with self.connect() as db:
            return dict(self._owned_run(db, run_id, actor))

    def get_run_detail(self, run_id: str, actor: Actor) -> dict:
        """在同一读事务中获取执行与步骤快照，避免两次查询之间状态发生变化。

        只读原样状态，不把未记录结束的running推断为仍在运行或已经失败。
        BEGIN显式启动读事务；仅使用连接上下文并不会自动给SELECT建立事务。
        """
        with self.connect() as db:
            db.execute("BEGIN")
            run = dict(self._owned_run(db, run_id, actor))
            steps = db.execute("SELECT * FROM steps WHERE run_id=? ORDER BY position", (run_id,)).fetchall()
        return {'run': run, 'steps': [dict(step) for step in steps]}

    def list_runs(self, actor: Actor, *, limit: int = 50) -> list[dict]:
        """只列出本人的近期执行。这里只查询，不删除、重试或推断业务结果。"""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit必须是1到1000的整数")
        with self.connect() as db:
            rows = db.execute(
                """SELECT * FROM runs WHERE workspace_id=? AND user_id=?
                   ORDER BY started_at DESC, rowid DESC LIMIT ?""",
                (actor.workspace_id, actor.user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def start_step(self, run_id: str, actor: Actor, *, kind: str, name: str,
                   request_id: str | None = None) -> str:
        """在真正调用前保存开始记录；此记录不能证明请求已送达或业务已成功。

        request_id可关联已有R-业务请求；它与本次追踪run_id用途不同。
        先获取写锁，再检查run并分配顺序，防止并发步骤获得相同position。
        SQLite提交后释放锁，不在执行模型或工具期间持有该事务。
        """
        if kind not in STEP_KINDS:
            raise ValueError("步骤类型只能是model、tool、approval或validation")
        name = _label(name, "name")
        request_id = _label(request_id, "request_id", optional=True)
        step_id = f"step-{uuid4().hex}"
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_running(self._owned_run(db, run_id, actor))
            position = db.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM steps WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            db.execute(
                """INSERT INTO steps
                   (step_id, run_id, position, kind, name, status, started_at, request_id)
                   VALUES (?, ?, ?, ?, ?, 'running', ?, ?)""",
                (step_id, run_id, position, kind, name, _now(), request_id),
            )
        return step_id

    def finish_step(self, step_id: str, actor: Actor, *, status: str,
                    duration_ms: float | None = None, error_type: str | None = None,
                    input_tokens: int | None = None, output_tokens: int | None = None) -> dict:
        """保存调用方已经确认的结果；不根据异常名称自动判断成功或失败。

        写入调用超时的业务结果应由宿主标记unknown；审批取消用cancelled。
        耗时应由调用方用perf_counter测量，缺失就保留NULL。token只记录服务实际
        返回的计数；缺失不当作0，也不按字数猜测。这里只接受错误类型标识，
        例如TimeoutError，不接收可能含密钥、URL和用户内容的异常消息或堆栈。
        """
        if status not in STEP_END_STATUSES:
            raise ValueError("无效的步骤结束状态")
        if duration_ms is not None and (
            type(duration_ms) not in (int, float)
            or not math.isfinite(duration_ms) or duration_ms < 0
        ):
            raise ValueError("duration_ms必须是非负有限数或None")
        for value in (input_tokens, output_tokens):
            if value is not None and (type(value) is not int or not 0 <= value <= 2**63 - 1):
                raise ValueError("token用量必须是非负SQLite整数或None")
        if error_type is not None and (
            not isinstance(error_type, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,99}", error_type) is None
        ):
            raise ValueError("只记录错误类型名称，不接受原始异常消息")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            step = db.execute("SELECT * FROM steps WHERE step_id=?", (step_id,)).fetchone()
            if step is None:
                raise PermissionError("步骤不存在或当前用户无权访问")
            self._require_running(self._owned_run(db, step["run_id"], actor))
            if step["status"] != "running":
                raise ValueError("步骤已结束，不能覆盖已有结果；重试应新增一个步骤")
            db.execute(
                """UPDATE steps SET status=?, finished_at=?, duration_ms=?, error_type=?,
                   input_tokens=?, output_tokens=? WHERE step_id=?""",
                (status, _now(), duration_ms, error_type, input_tokens, output_tokens, step_id),
            )
            saved = dict(db.execute("SELECT * FROM steps WHERE step_id=?", (step_id,)).fetchone())
        return saved

    def list_steps(self, run_id: str, actor: Actor) -> list[dict]:
        """按开始顺序读取本次执行的所有步骤，不按完成先后重排。"""
        with self.connect() as db:
            self._owned_run(db, run_id, actor)
            rows = db.execute(
                "SELECT * FROM steps WHERE run_id=? ORDER BY position", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def finish_run(self, run_id: str, actor: Actor, *, status: str = "completed") -> dict:
        """记录程序结束；completed不表示任务正确、工单已创建或评估通过。

        正常结束前必须逐个收尾步骤。异常/取消时允许留下running步骤，保留当时
        未获得结果的事实，不能批量改成失败或成功。已经结束的run不再修改；
        后续核查建立新run，沿用原业务request_id，并保留旧的未知或未完成记录。
        """
        if status not in RUN_END_STATUSES:
            raise ValueError("执行结束状态只能是completed、failed或cancelled")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_running(self._owned_run(db, run_id, actor))
            if status == "completed" and db.execute(
                "SELECT 1 FROM steps WHERE run_id=? AND status='running' LIMIT 1", (run_id,)
            ).fetchone():
                raise ValueError("仍有未收尾步骤，不能记录为正常结束")
            db.execute("UPDATE runs SET status=?, finished_at=? WHERE run_id=?", (status, _now(), run_id))
            saved = dict(self._owned_run(db, run_id, actor))
        return saved

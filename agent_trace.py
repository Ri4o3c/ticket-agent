"""Agent到记录库的适配器：采集事实，不读取日志指导模型或决定业务成功。

SQLite工作放在线程中，避免锁等待阻塞网页事件循环。采集故障单独报告，
不覆盖真实工具结果，不因日志失败重新执行业务动作，也不输出异常正文。
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter

from trace_store import DEFAULT_TRACE_DB, TraceStore


def model_label(model):
    """从模型或bind_tools后的bound对象取配置名称，不使用可能含Key的repr。"""
    for _ in range(5):
        for attribute in ("model_name", "model"):
            label = getattr(model, attribute, None)
            if isinstance(label, str) and label.strip():
                return label[:200]
        bound = getattr(model, "bound", None)
        if bound is None or bound is model:
            break
        model = bound
    return None


def error_label(value):
    """只接受错误类型标识，不将错误码或异常正文写入元数据。"""
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,99}", value) else None


@dataclass
class StepResult:
    status: str = "succeeded"
    error_type: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def tool_result(self, data):
        if data.get("outcome_unknown"):
            self.status = "unknown"
        elif data.get("cancelled"):
            self.status = "cancelled"
        else:
            self.status = "succeeded" if data.get("ok") is True else "failed"
        self.error_type = error_label(data.get("exception_type"))
        if self.status == "failed" and self.error_type is None:
            self.error_type = "ToolBusinessError"

    def model_result(self, response):
        # LangChain标准用量来自AIMessage.usage_metadata，不按字数推算。
        usage = getattr(response, "usage_metadata", None) or {}
        if not isinstance(usage, dict):
            return
        for field in ("input_tokens", "output_tokens"):
            value = usage.get(field)
            if type(value) is int and 0 <= value <= 2**63 - 1:
                setattr(self, field, value)


class TurnTrace:
    def __init__(self, actor, *, store=None, business_store=None, on_event=None):
        self.actor, self.store, self.on_event = actor, store, on_event
        business_path = getattr(business_store, "path", None)
        self.path = Path(business_path).with_name("traces.sqlite3") if business_path else DEFAULT_TRACE_DB
        self.run_id = None
        self.degraded = False

    async def _event(self, **event):
        if self.on_event is not None:
            try:
                await self.on_event(event)
            except Exception:
                # 页面展示失败不改变业务/持久记录的结果。
                print("追踪事件展示失败。")

    async def _write(self, operation, *args, _on_result=None, **kwargs):
        work = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
        try:
            try:
                value = await asyncio.shield(work)
            except asyncio.CancelledError:
                # 先等已经进入线程的事务收尾，避免run先结束、迟到的写入仍在执行。
                # 若取消发生在start_step返回前，保留running开始记录，不声称工具已执行。
                try:
                    value = await work
                    if _on_result is not None:
                        _on_result(value)
                except Exception:
                    self.degraded = True
                raise
            if _on_result is not None:
                _on_result(value)
            return value
        except Exception as exc:
            first = not self.degraded
            self.degraded = True
            if first:
                print(f"执行记录保存异常（{type(exc).__name__}），业务结果需要独立核对。")
                await self._event(type="trace_warning", run_id=self.run_id,
                                  content="本次执行记录保存不完整；请独立核查业务结果。")
            return None

    async def begin(self, *, session_id=None, knowledge_base_id=None, model_name=None):
        if self.store is None:
            self.store = await self._write(TraceStore, self.path)
        if self.store is not None:
            self.run_id = await self._write(self.store.start_run, self.actor,
                                            session_id=session_id, knowledge_base_id=knowledge_base_id,
                                            model_name=model_name,
                                            _on_result=lambda value: setattr(self, 'run_id', value))
        if self.run_id:
            print("本次执行追踪编号：", self.run_id)
            await self._event(type="trace", phase="started", run_id=self.run_id)

    @asynccontextmanager
    async def step(self, kind, name, *, request_id=None):
        result = StepResult()
        step_id = None
        if self.run_id:
            step_id = await self._write(self.store.start_step, self.run_id, self.actor,
                                        kind=kind, name=name, request_id=request_id)
        started = perf_counter()
        try:
            yield result
        except asyncio.CancelledError:
            result.status = "unknown" if kind == "tool" and name == "create_ticket" else "cancelled"
            result.error_type = "CancelledError"
            raise
        except Exception as exc:
            result.status = "unknown" if kind == "tool" and name == "create_ticket" else "failed"
            result.error_type = error_label(type(exc).__name__) or "RuntimeError"
            raise
        finally:
            if step_id:
                await self._write(self.store.finish_step, step_id, self.actor, status=result.status,
                                  duration_ms=(perf_counter() - started) * 1000,
                                  error_type=result.error_type, input_tokens=result.input_tokens,
                                  output_tokens=result.output_tokens)

    async def finish(self, status):
        # 采集失败的run不能冒充记录完整；failed在这里表示本次执行/记录不完整，
        # 不是“工单没有创建”。已经拿到的真实业务结果仍照常交付给用户。
        final_status = "failed" if self.degraded and status == "completed" else status
        if self.run_id:
            saved = await self._write(self.store.finish_run, self.run_id, self.actor, status=final_status)
            if saved is None and final_status == "completed":
                final_status = "failed"
                saved = await self._write(self.store.finish_run, self.run_id, self.actor, status=final_status)
            await self._event(type="trace", phase="finished", run_id=self.run_id,
                              status=saved['status'] if saved else 'unknown', recording_incomplete=self.degraded)

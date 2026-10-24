"""真实Agent循环和临时SQLite的追踪验证；只替换外部模型/网络。"""
import asyncio
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from fastmcp import Client
from langchain_core.messages import AIMessage, ToolMessage
from starlette.testclient import TestClient

import agent_step1 as agent
from knowledge_bases import KnowledgeBases
from mcp_tools import make_server
from ticket_store import Actor, TicketStore
from trace_store import TraceStore
from web_app import create_app
from test_agent_completion import LocalToolClient
from test_agent_errors import LostCreateReply, Responses
from test_web_app import WebModel


def request(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class ReadClient(LocalToolClient):
    async def call_tool(self, name, args):
        if name == "get_ticket":
            try:
                data = {"ok": True, "ticket": self.store.get_ticket(args["ticket_id"], self.actor)}
            except PermissionError:
                data = {"ok": False, "message": "工单不存在或当前用户无权访问"}
            return SimpleNamespace(data=data)
        return await super().call_tool(name, args)


class CreationModel:
    model_name = "test-chat-model"

    async def ainvoke(self, messages):
        tools = [m for m in messages if isinstance(m, ToolMessage)]
        if not tools:
            return request("prepare_ticket", {"title": "PDF失败", "description": "2MB PDF报E103"}, "prepare")
        data = json.loads(tools[-1].content)
        if tools[-1].name == "prepare_ticket":
            return request("create_ticket", {"draft_id": data["draft"]["draft_id"]}, "create")
        return AIMessage(content="已创建工单 " + data["ticket"]["ticket_id"])


class TracingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = TicketStore(self.root / "tickets.sqlite3")
        self.trace = TraceStore(self.root / "traces.sqlite3")
        self.actor = Actor("trace-user", "trace-workspace")
        self.events = []

    async def emit(self, event):
        self.events.append(event)

    async def run_flow(self, model, question="你好", *, client=None, decision=True, trace=None, history=None, confirmation=None):
        async def confirm(draft):
            return decision
        if not hasattr(model, "model_name"):
            model.model_name = "test-chat-model"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                await agent.run_turn(
                    client or ReadClient(self.store, self.actor), model,
                    history if history is not None else [], question,
                    store=self.store, actor=self.actor, trace_store=trace or self.trace,
                    session_id="session-a", knowledge_base_id="kb-a",
                    confirm_draft=confirmation or confirm, on_event=self.emit,
                )
        except TypeError as exc:
            if "unexpected keyword argument" in str(exc):
                self.fail("run_turn尚未接入执行记录")
            raise

    def latest(self):
        run = self.trace.list_runs(self.actor)[0]
        return run, self.trace.list_steps(run["run_id"], self.actor)

    async def test_questions_share_session_but_have_distinct_runs_and_reported_usage(self):
        history = []
        reply = AIMessage(content="你好", usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14})
        await self.run_flow(Responses(reply), history=history)
        await self.run_flow(Responses(AIMessage(content="可以继续提问。")), history=history)
        runs = self.trace.list_runs(self.actor)
        self.assertEqual(len(runs), 2)
        self.assertEqual(len({r["run_id"] for r in runs}), 2)
        self.assertEqual({r["session_id"] for r in runs}, {"session-a"})
        self.assertEqual({r["knowledge_base_id"] for r in runs}, {"kb-a"})
        self.assertEqual({r["model_name"] for r in runs}, {"test-chat-model"})
        first_steps = self.trace.list_steps(runs[1]["run_id"], self.actor)
        self.assertEqual([(s["kind"], s["status"]) for s in first_steps],
                         [("model", "succeeded"), ("validation", "succeeded")])
        self.assertEqual((first_steps[0]["input_tokens"], first_steps[0]["output_tokens"]), (11, 3))
        self.assertIsNone(self.trace.list_steps(runs[0]["run_id"], self.actor)[0]["input_tokens"])
        self.assertTrue(all(r["status"] == "completed" and r["finished_at"] for r in runs))
        ids = {e["run_id"] for e in self.events if e["type"] == "trace"}
        self.assertEqual(ids, {r["run_id"] for r in runs})

    async def test_business_query_failure_is_not_program_failure(self):
        model = Responses(request("get_ticket", {"ticket_id": "T-999999"}, "lookup"),
                          AIMessage(content="无法查询，工单不存在或当前用户无权访问。"))
        await self.run_flow(model, "查询工单T-999999当前状态")
        run, steps = self.latest()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([(s["kind"], s["status"]) for s in steps],
                         [("model", "succeeded"), ("tool", "failed"),
                          ("model", "succeeded"), ("validation", "succeeded")])
        self.assertTrue(all(s["duration_ms"] >= 0 for s in steps))

    async def test_model_exception_records_type_without_exception_body(self):
        await self.run_flow(Responses(ConnectionError("PRIVATE-KEY-DO-NOT-STORE")))
        run, steps = self.latest()
        self.assertEqual(run["status"], "failed")
        self.assertEqual(steps[0]["status"], "failed")
        self.assertEqual(steps[0]["error_type"], "ConnectionError")
        self.assertNotIn(b"PRIVATE-KEY-DO-NOT-STORE", self.trace.path.read_bytes())
        self.assertNotIn("PRIVATE-KEY-DO-NOT-STORE", json.dumps(self.events))

    async def test_approved_creation_has_separate_approval_and_original_request(self):
        await self.run_flow(CreationModel(), "登记测试工单")
        run, steps = self.latest()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([s["kind"] for s in steps],
                         ["model", "tool", "approval", "model", "tool", "model", "validation"])
        create = next(s for s in steps if s["name"] == "create_ticket")
        self.assertEqual(create["status"], "succeeded")
        ticket = self.store.find_ticket_by_request(create["request_id"], self.actor)
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket["status"], "待处理")

    async def test_cancelled_approval_does_not_change_prepared_tool_success(self):
        await self.run_flow(CreationModel(), "登记测试工单", decision=False)
        run, steps = self.latest()
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual([(s["kind"], s["status"]) for s in steps],
                         [("model", "succeeded"), ("tool", "succeeded"), ("approval", "cancelled")])
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0], 0)

    async def test_waiting_for_human_keeps_approval_running_after_tool_finished(self):
        entered = asyncio.Event()
        decision = asyncio.get_running_loop().create_future()
        async def confirm(draft):
            entered.set()
            return await decision
        task = asyncio.create_task(self.run_flow(CreationModel(), "登记测试工单", confirmation=confirm))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            run, steps = self.latest()
            self.assertEqual(run["status"], "running")
            self.assertEqual([(s["kind"], s["status"]) for s in steps],
                             [("model", "succeeded"), ("tool", "succeeded"), ("approval", "running")])
            self.assertLessEqual(steps[1]["finished_at"], steps[2]["started_at"])
            decision.set_result(False)
            await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cancel_during_create_keeps_unknown_and_complete_tool_messages(self):
        entered = asyncio.Event()
        class CommittedThenWaiting(LocalToolClient):
            async def call_tool(self, name, args):
                result = await super().call_tool(name, args)
                if name == "create_ticket":
                    entered.set()
                    await asyncio.Future()
                return result
        history = []
        task = asyncio.create_task(self.run_flow(CreationModel(), "登记测试工单", history=history,
                                                client=CommittedThenWaiting(self.store, self.actor)))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        run, steps = self.latest()
        self.assertEqual(run["status"], "cancelled")
        create = next(s for s in steps if s["name"] == "create_ticket")
        self.assertEqual(create["status"], "unknown")
        self.assertIsNotNone(self.store.find_ticket_by_request(create["request_id"], self.actor))
        self.assertEqual({m.tool_call_id for m in history if isinstance(m, ToolMessage)}, {"prepare", "create"})
        self.assertEqual(self.store.pending_creations(self.actor)[0]["request_id"], create["request_id"])

    async def test_successful_lookup_with_found_false_is_successful_tool_execution(self):
        class NoMatch:
            async def call_tool(self, name, args):
                return SimpleNamespace(data={"ok": True, "found": False})
        await self.run_flow(Responses(request("find_ticket_by_request", {"request_id": "R-original"}, "lookup"),
                                     AIMessage(content="未查到原请求对应的工单。")),
                            "核查创建请求R-original", client=NoMatch())
        run, steps = self.latest()
        tool = next(s for s in steps if s["kind"] == "tool")
        self.assertEqual(tool["status"], "succeeded")
        self.assertEqual(tool["request_id"], "R-original")
        self.assertEqual(run["status"], "completed")

    async def test_creation_timeout_keeps_unknown_and_can_be_checked_in_database(self):
        for committed in (True, False):
            with self.subTest(committed=committed):
                # 每种故障拥有独立真实数据库，避免恢复屏障影响另一子场景。
                business = TicketStore(self.root / f"tickets-{committed}.sqlite3")
                original = self.store
                self.store = business
                try:
                    await self.run_flow(CreationModel(), "登记测试工单",
                                        client=LostCreateReply(business, self.actor, committed))
                finally:
                    self.store = original
                run, steps = self.latest()
                create = next(s for s in steps if s["name"] == "create_ticket")
                self.assertEqual(create["status"], "unknown")
                self.assertEqual(create["error_type"], "TimeoutError")
                self.assertTrue(create["request_id"].startswith("R-"))
                self.assertEqual(business.find_ticket_by_request(create["request_id"], self.actor) is not None, committed)
                self.assertEqual(business.pending_creations(self.actor)[0]["request_id"], create["request_id"])
                self.assertEqual(run["status"], "completed")

    async def test_rejected_answer_is_logged_and_corrected_answer_can_finish(self):
        await self.run_flow(Responses(AIMessage(content="未知结论 [KB-999]"), AIMessage(content="你好。")))
        run, steps = self.latest()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([s["status"] for s in steps if s["kind"] == "validation"], ["failed", "succeeded"])

    async def test_exhausted_round_limit_is_failed_run_not_success(self):
        calls = [request("get_ticket", {"ticket_id": "T-999999"}, f"lookup-{n}") for n in range(5)]
        await self.run_flow(Responses(*calls), "查询工单T-999999当前状态")
        run, steps = self.latest()
        self.assertEqual(run["status"], "failed")
        self.assertEqual(len([s for s in steps if s["kind"] == "model"]), 5)
        self.assertTrue(all(s["finished_at"] for s in steps))

    async def test_task_cancellation_closes_model_step_and_run(self):
        entered = asyncio.Event()
        class WaitingModel:
            model_name = "test-chat-model"
            async def ainvoke(self, messages):
                entered.set()
                await asyncio.Future()
        task = asyncio.create_task(self.run_flow(WaitingModel()))
        waiter = asyncio.create_task(entered.wait())
        try:
            done, _ = await asyncio.wait({task, waiter}, timeout=2, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                await task
            self.assertIn(waiter, done, "模型调用未开始")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            waiter.cancel()
            await asyncio.gather(task, waiter, return_exceptions=True)
        run, steps = self.latest()
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(steps[0]["status"], "cancelled")

    async def test_trace_failure_after_create_does_not_repeat_or_hide_business_result(self):
        class BrokenTrace(TraceStore):
            def finish_step(self, step_id, actor, **values):
                with self.connect() as db:
                    name = db.execute("SELECT name FROM steps WHERE step_id=?", (step_id,)).fetchone()[0]
                if name == "create_ticket":
                    raise sqlite3.OperationalError("PRIVATE-STORAGE-DETAIL")
                return super().finish_step(step_id, actor, **values)
        history = []
        await self.run_flow(CreationModel(), "登记测试工单", trace=BrokenTrace(self.trace.path), history=history)
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0], 1)
        result = next(json.loads(m.content) for m in history if isinstance(m, ToolMessage) and m.name == "create_ticket")
        self.assertTrue(result["ok"])
        self.assertTrue(any(e["type"] == "trace_warning" for e in self.events))
        self.assertNotIn("PRIVATE-STORAGE-DETAIL", json.dumps(self.events))
        self.assertEqual(self.latest()[0]["status"], "failed")

    async def test_finish_write_failure_does_not_report_unpersisted_final_state_as_known(self):
        class CannotFinishTrace(TraceStore):
            def finish_run(self, *args, **kwargs):
                raise sqlite3.OperationalError("PRIVATE-FINISH-DETAIL")
        await self.run_flow(Responses(AIMessage(content="你好")), trace=CannotFinishTrace(self.trace.path))
        run, steps = self.latest()
        self.assertEqual(run["status"], "running")
        finished = next(e for e in self.events if e["type"] == "trace" and e["phase"] == "finished")
        self.assertEqual(finished["status"], "unknown")
        self.assertTrue(finished["recording_incomplete"])

    async def test_cancel_while_policy_log_is_writing_completes_entire_tool_batch(self):
        entered, release = threading.Event(), threading.Event()
        class WaitingPolicyTrace(TraceStore):
            def start_step(self, *args, **kwargs):
                if kwargs["name"] == "tool_policy":
                    entered.set()
                    release.wait(5)
                return super().start_step(*args, **kwargs)
        request_batch = AIMessage(content="", tool_calls=[
            {"name": "forbidden", "args": {}, "id": "blocked"},
            {"name": "get_ticket", "args": {"ticket_id": "T-999999"}, "id": "unexecuted"},
        ])
        history = []
        task = asyncio.create_task(self.run_flow(Responses(request_batch), history=history,
                                                trace=WaitingPolicyTrace(self.trace.path)))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        results = {m.tool_call_id: json.loads(m.content) for m in history if isinstance(m, ToolMessage)}
        self.assertEqual(set(results), {"blocked", "unexecuted"})
        self.assertTrue(results["unexecuted"]["skipped"])
        self.assertEqual(self.latest()[0]["status"], "cancelled")

    async def test_cancel_after_known_create_success_preserves_real_tool_result(self):
        entered, release = threading.Event(), threading.Event()
        class WaitingFinishTrace(TraceStore):
            def finish_step(self, step_id, actor, **kwargs):
                with self.connect() as db:
                    name = db.execute("SELECT name FROM steps WHERE step_id=?", (step_id,)).fetchone()[0]
                if name == "create_ticket":
                    entered.set()
                    release.wait(5)
                return super().finish_step(step_id, actor, **kwargs)
        history = []
        task = asyncio.create_task(self.run_flow(CreationModel(), "登记测试工单", history=history,
                                                trace=WaitingFinishTrace(self.trace.path)))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        result = next(json.loads(m.content) for m in history if isinstance(m, ToolMessage) and m.name == "create_ticket")
        self.assertTrue(result["ok"])
        self.assertTrue(result["ticket"]["ticket_id"].startswith("T-"))
        self.assertNotIn("outcome_unknown", result)
        self.assertEqual(self.store.pending_creations(self.actor), [])
        self.assertEqual(self.latest()[0]["status"], "cancelled")

    async def test_cancel_during_run_start_commit_preserves_id_for_final_cancelled_record(self):
        entered, release = threading.Event(), threading.Event()
        class WaitingStartTrace(TraceStore):
            def start_run(self, *args, **kwargs):
                run_id = super().start_run(*args, **kwargs)
                entered.set()
                release.wait(5)
                return run_id
        task = asyncio.create_task(self.run_flow(Responses(AIMessage(content="不应调用模型")),
                                                trace=WaitingStartTrace(self.trace.path)))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        run, steps = self.latest()
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(steps, [])


class WebTracingTests(unittest.TestCase):
    def test_web_questions_persist_records_for_selected_library_without_raw_session_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            business = TicketStore(root / "tickets.sqlite3")
            actor = Actor("web-trace-user", "web-trace-space")
            catalog = KnowledgeBases(root / "bases.json")
            base = catalog.create("独立资料库")
            app = create_app(
                client_factory=lambda kb: Client(make_server(store=business, actor=actor, searcher=lambda q, k: [])),
                model_factory=lambda url: WebModel(), store_factory=lambda: business,
                actor_factory=lambda: actor, catalog=catalog,
            )
            with TestClient(app) as http:
                response = http.post("/api/sessions", json={"kb_id": base["id"]})
                self.assertEqual(response.status_code, 201)
                sid = response.json()["session_id"]
                def wait_for(predicate):
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        state = http.get(f"/api/sessions/{sid}/events").json()
                        if predicate(state):
                            return state
                        time.sleep(.01)
                    self.fail("网页会话未完成处理")
                wait_for(lambda s: s["ready"])
                for question in ("你好", "继续问候"):
                    http.post(f"/api/sessions/{sid}/messages", json={"message": question})
                    state = wait_for(lambda s: not s["busy"])
                self.assertTrue((root / "traces.sqlite3").exists(), "网页尚未自动采集运行记录")
                trace = TraceStore(root / "traces.sqlite3")
                runs = trace.list_runs(actor)
                self.assertEqual(len(runs), 2)
                self.assertEqual({r["knowledge_base_id"] for r in runs}, {base["id"]})
                self.assertEqual({r["session_id"] for r in runs}, {hashlib.sha256(sid.encode()).hexdigest()})
                self.assertNotIn(sid.encode(), trace.path.read_bytes())
                self.assertEqual({e["run_id"] for e in state["events"] if e["type"] == "trace"},
                                 {r["run_id"] for r in runs})


if __name__ == "__main__":
    unittest.main()

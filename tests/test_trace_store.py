"""执行记录的存储契约：真实临时SQLite，不调用模型或MCP服务。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from ticket_store import Actor

try:
    from trace_store import TraceStore
except ModuleNotFoundError:
    TraceStore = None


class TraceStoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(TraceStore, "执行记录存储尚未实现")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "traces.sqlite3"
        self.store = TraceStore(self.path)
        self.actor = Actor("learner", "workspace-a")

    def start_run(self):
        return self.store.start_run(
            self.actor, session_id="session-a", knowledge_base_id="kb-a",
            model_name="example-model",
        )

    def test_separate_questions_in_one_session_have_distinct_persisted_runs(self):
        first, second = self.start_run(), self.start_run()
        self.assertNotEqual(first, second)
        # 新连接可以立即读到，证明不需要等待整个Agent任务结束再提交。
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                "SELECT session_id, status FROM runs ORDER BY started_at"
            ).fetchall()
        self.assertEqual(rows, [("session-a", "running"), ("session-a", "running")])
        saved = self.store.get_run(first, self.actor)
        self.assertEqual(saved["model_name"], "example-model")
        self.assertEqual(saved["knowledge_base_id"], "kb-a")
        self.assertIsNone(saved["finished_at"])

    def test_unfinished_step_remains_running_after_reopen(self):
        run_id = self.start_run()
        step_id = self.store.start_step(
            run_id, self.actor, kind="tool", name="create_ticket", request_id="R-original"
        )
        reopened = TraceStore(self.path)
        self.assertEqual(reopened.get_run(run_id, self.actor)["status"], "running")
        saved = reopened.list_steps(run_id, self.actor)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["step_id"], step_id)
        self.assertEqual(saved[0]["request_id"], "R-original")
        self.assertEqual(saved[0]["status"], "running")
        self.assertIsNone(saved[0]["finished_at"])

    def test_committed_start_records_survive_process_exit_without_cleanup(self):
        # 子进程不走finally，直接退出；仍必须能读到每步已经提交的开始记录。
        script = """
import os, sys
from pathlib import Path
from ticket_store import Actor
from trace_store import TraceStore
store = TraceStore(Path(sys.argv[1]))
actor = Actor('learner', 'workspace-a')
run_id = store.start_run(actor)
store.start_step(run_id, actor, kind='tool', name='create_ticket', request_id='R-original')
os._exit(0)
"""
        subprocess.run([sys.executable, "-B", "-c", script, str(self.path)],
                       cwd=Path(__file__).resolve().parents[1], check=True,
                       capture_output=True, timeout=20)
        reopened = TraceStore(self.path)
        runs = reopened.list_runs(self.actor)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "running")
        rows = reopened.list_steps(runs[0]["run_id"], self.actor)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "running")
        self.assertEqual(rows[0]["request_id"], "R-original")

    def test_one_run_keeps_three_ordered_model_and_tool_steps(self):
        run_id = self.start_run()
        for kind, name in [("model", "example-model"), ("tool", "get_ticket"),
                           ("model", "example-model")]:
            step_id = self.store.start_step(run_id, self.actor, kind=kind, name=name)
            self.store.finish_step(step_id, self.actor, status="succeeded", duration_ms=12.5)
        self.store.finish_run(run_id, self.actor, status="completed")
        reopened = TraceStore(self.path)
        rows = reopened.list_steps(run_id, self.actor)
        self.assertEqual([row["position"] for row in rows], [1, 2, 3])
        self.assertEqual([row["kind"] for row in rows], ["model", "tool", "model"])
        self.assertEqual({row["run_id"] for row in rows}, {run_id})
        self.assertTrue(all(row["status"] == "succeeded" for row in rows))
        self.assertTrue(all(row["finished_at"] for row in rows))
        self.assertEqual(rows[0]["duration_ms"], 12.5)
        self.assertEqual(reopened.get_run(run_id, self.actor)["status"], "completed")

    def test_write_timeout_stays_unknown_with_original_request(self):
        run_id = self.start_run()
        step_id = self.store.start_step(
            run_id, self.actor, kind="tool", name="create_ticket", request_id="R-original"
        )
        self.store.finish_step(step_id, self.actor, status="unknown", error_type="TimeoutError")
        row = TraceStore(self.path).list_steps(run_id, self.actor)[0]
        self.assertEqual(row["status"], "unknown")
        self.assertEqual(row["request_id"], "R-original")
        self.assertEqual(row["error_type"], "TimeoutError")
        self.assertIsNone(row["duration_ms"])
        self.assertIsNone(row["input_tokens"])
        self.assertIsNone(row["output_tokens"])

    def test_missing_tokens_and_reported_zero_are_not_confused(self):
        run_id = self.start_run()
        first = self.store.start_step(run_id, self.actor, kind="model", name="example-model")
        self.store.finish_step(first, self.actor, status="succeeded")
        second = self.store.start_step(run_id, self.actor, kind="model", name="example-model")
        self.store.finish_step(second, self.actor, status="succeeded", input_tokens=0, output_tokens=7)
        rows = self.store.list_steps(run_id, self.actor)
        self.assertEqual([(row["input_tokens"], row["output_tokens"]) for row in rows],
                         [(None, None), (0, 7)])

    def test_completed_run_cannot_hide_an_unfinished_step(self):
        run_id = self.start_run()
        self.store.start_step(run_id, self.actor, kind="tool", name="create_ticket")
        with self.assertRaises(ValueError):
            self.store.finish_run(run_id, self.actor, status="completed")
        self.assertEqual(self.store.get_run(run_id, self.actor)["status"], "running")

    def test_failed_run_preserves_unfinished_step_without_inventing_outcome(self):
        run_id = self.start_run()
        self.store.start_step(run_id, self.actor, kind="tool", name="create_ticket")
        self.store.finish_run(run_id, self.actor, status="failed")
        reopened = TraceStore(self.path)
        self.assertEqual(reopened.get_run(run_id, self.actor)["status"], "failed")
        self.assertEqual(reopened.list_steps(run_id, self.actor)[0]["status"], "running")

    def test_terminal_records_cannot_be_overwritten_or_extended(self):
        run_id = self.start_run()
        step_id = self.store.start_step(run_id, self.actor, kind="validation", name="answer_guard")
        self.store.finish_step(step_id, self.actor, status="failed", error_type="EvidenceError")
        with self.assertRaises(ValueError):
            self.store.finish_step(step_id, self.actor, status="succeeded")
        self.store.finish_run(run_id, self.actor, status="completed")
        with self.assertRaises(ValueError):
            self.store.finish_run(run_id, self.actor, status="failed")
        with self.assertRaises(ValueError):
            self.store.start_step(run_id, self.actor, kind="model", name="another-model")
        self.assertEqual(self.store.list_steps(run_id, self.actor)[0]["status"], "failed")

    def test_other_users_and_workspaces_cannot_read_or_modify_records(self):
        run_id = self.start_run()
        step_id = self.store.start_step(run_id, self.actor, kind="tool", name="get_ticket")
        outsiders = [Actor("other", "workspace-a"), Actor("learner", "workspace-b"),
                     Actor("admin", "workspace-b", "admin")]
        for outsider in outsiders:
            actions = [lambda: self.store.get_run(run_id, outsider),
                       lambda: self.store.list_steps(run_id, outsider),
                       lambda: self.store.start_step(run_id, outsider, kind="tool", name="get_ticket"),
                       lambda: self.store.finish_step(step_id, outsider, status="succeeded"),
                       lambda: self.store.finish_run(run_id, outsider, status="failed")]
            for action in actions:
                with self.subTest(actor=outsider), self.assertRaises(PermissionError):
                    action()
            self.assertEqual(self.store.list_runs(outsider), [])
        self.assertEqual(self.store.list_steps(run_id, self.actor)[0]["status"], "running")

    def test_missing_run_cannot_create_orphan_step(self):
        with self.assertRaises(PermissionError):
            self.store.start_step("run-missing", self.actor, kind="tool", name="get_ticket")
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM steps").fetchone()[0], 0)

    def test_invalid_metadata_is_rejected_before_writing(self):
        run_id = self.start_run()
        for kind, name in [("unknown-kind", "tool"), ("tool", ""), ("tool", "x" * 201)]:
            with self.subTest(kind=kind, name=name), self.assertRaises(ValueError):
                self.store.start_step(run_id, self.actor, kind=kind, name=name)
        self.assertEqual(self.store.list_steps(run_id, self.actor), [])
        step_id = self.store.start_step(run_id, self.actor, kind="model", name="example-model")
        invalid_fields = [dict(status="typo"), dict(status="running"),
                          dict(duration_ms=-1), dict(duration_ms=float("nan")),
                          dict(duration_ms=float("inf")), dict(input_tokens=-1),
                          dict(input_tokens=True), dict(output_tokens=1.5),
                          dict(error_type="raw exception containing a secret")]
        for fields in invalid_fields:
            args = {"status": "failed", **fields}
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.store.finish_step(step_id, self.actor, **args)
        self.assertEqual(self.store.list_steps(run_id, self.actor)[0]["status"], "running")

    def test_concurrent_steps_keep_unique_contiguous_positions(self):
        run_id = self.start_run()
        def add_step(_):
            return self.store.start_step(run_id, self.actor, kind="tool", name="get_ticket")
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(add_step, range(12)))
        rows = self.store.list_steps(run_id, self.actor)
        self.assertEqual(len(set(ids)), 12)
        self.assertEqual([row["position"] for row in rows], list(range(1, 13)))

    def test_list_runs_filters_owner_and_limits_results_without_deleting(self):
        first, second = self.start_run(), self.start_run()
        self.store.start_run(Actor("other", "workspace-a"))
        recent = self.store.list_runs(self.actor, limit=1)
        self.assertEqual([row["run_id"] for row in recent], [second])
        self.assertEqual({row["run_id"] for row in self.store.list_runs(self.actor)}, {first, second})
        for invalid in [0, -1, 1001, True]:
            with self.subTest(limit=invalid), self.assertRaises(ValueError):
                self.store.list_runs(self.actor, limit=invalid)


if __name__ == "__main__":
    unittest.main()

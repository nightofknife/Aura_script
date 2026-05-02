from __future__ import annotations

import unittest

try:
    from PySide6.QtWidgets import QApplication

    from packages.yihuan_gui.bridge import RunnerBridge
    from packages.yihuan_gui.logic import GuiPreferences
except ModuleNotFoundError as exc:
    if str(getattr(exc, "name", "")).startswith("PySide6"):
        QApplication = None
        RunnerBridge = None
        GuiPreferences = None
    else:
        raise


class _FakeRunner:
    def __init__(self, result=None, exc: Exception | None = None) -> None:
        self.result = result or {"status": "success", "message": "cancel requested"}
        self.exc = exc
        self.cancelled_cids: list[str] = []

    def cancel_task(self, cid: str):
        self.cancelled_cids.append(cid)
        if self.exc is not None:
            raise self.exc
        return self.result


class _FakePollingRunner:
    def __init__(self, batches: list[list[dict]], detail: dict | None = None) -> None:
        self._batches = list(batches)
        self._detail = detail or {"cid": "cid-1", "task_name": "tasks:fishing:auto_loop.yaml", "final_result": {}}
        self.history_calls = 0

    def poll_events(self, **_kwargs):
        if not self._batches:
            return []
        return self._batches.pop(0)

    def get_run(self, _cid: str):
        return dict(self._detail)

    def list_runs(self, **_kwargs):
        self.history_calls += 1
        return []


@unittest.skipIf(QApplication is None, "PySide6 is not installed in this test environment.")
class TestRunnerBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_cancel_task_emits_control_message(self):
        bridge = RunnerBridge(GuiPreferences())
        runner = _FakeRunner()
        bridge._runner = runner
        messages: list[dict] = []
        bridge.control_message.connect(messages.append)

        bridge.cancel_task("cid-123")

        self.assertEqual(runner.cancelled_cids, ["cid-123"])
        self.assertEqual(messages[0]["cid"], "cid-123")
        self.assertIn("停止任务请求已发送", messages[0]["message"])

    def test_cancel_task_without_runner_emits_error(self):
        bridge = RunnerBridge(GuiPreferences())
        errors: list[dict] = []
        bridge.error_occurred.connect(errors.append)

        bridge.cancel_task("cid-123")

        self.assertEqual(errors[0]["kind"], "runtime")
        self.assertIn("取消任务失败", errors[0]["title"])

    def test_cancel_task_exception_emits_error(self):
        bridge = RunnerBridge(GuiPreferences())
        bridge._runner = _FakeRunner(exc=RuntimeError("boom"))
        errors: list[dict] = []
        bridge.error_occurred.connect(errors.append)

        bridge.cancel_task("cid-123")

        self.assertEqual(errors[0]["kind"], "runtime")
        self.assertIn("boom", errors[0]["message"])

    def test_poll_events_caps_per_run_event_cache(self):
        bridge = RunnerBridge(GuiPreferences())
        events = [
            {
                "name": "node.started",
                "payload": {
                    "cid": "cid-1",
                    "game_name": "yihuan",
                    "task_name": "tasks:fishing:auto_loop.yaml",
                    "node_id": f"node-{index}",
                },
            }
            for index in range(bridge._MAX_EVENT_CACHE_EVENTS_PER_RUN + 25)
        ]
        bridge._runner = _FakePollingRunner([events])

        bridge.poll_events()

        self.assertEqual(len(bridge._event_cache["cid-1"]), bridge._MAX_EVENT_CACHE_EVENTS_PER_RUN)

    def test_finished_run_detail_flush_clears_cached_events(self):
        bridge = RunnerBridge(GuiPreferences())
        bridge._runner = _FakePollingRunner(
            [[
                {
                    "name": "task.finished",
                    "payload": {
                        "cid": "cid-1",
                        "game_name": "yihuan",
                        "task_name": "tasks:fishing:auto_loop.yaml",
                        "final_status": "success",
                    },
                }
            ]]
        )
        bridge._event_cache["cid-1"].append({"name": "task.started", "payload": {"cid": "cid-1"}})
        details: list[tuple[str, dict]] = []
        bridge.run_detail_ready.connect(lambda cid, payload: details.append((cid, payload)))

        bridge.poll_events()

        self.assertEqual(details[0][0], "cid-1")
        self.assertNotIn("cid-1", bridge._event_cache)
        self.assertEqual(bridge._runner.history_calls, 1)


if __name__ == "__main__":
    unittest.main()

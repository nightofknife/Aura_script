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


if __name__ == "__main__":
    unittest.main()

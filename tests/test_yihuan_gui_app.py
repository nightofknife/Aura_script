from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from packages.yihuan_gui.app import WORKBENCH_TASKS, YihuanMainWindow
    from packages.yihuan_gui.logic import GuiPreferences, TASK_AUTO_LOOP, TASK_CAFE_AUTO_LOOP
except ModuleNotFoundError as exc:
    if str(getattr(exc, "name", "")).startswith("PySide6"):
        QApplication = None
        Qt = None
        WORKBENCH_TASKS = None
        YihuanMainWindow = None
        GuiPreferences = None
        TASK_AUTO_LOOP = ""
        TASK_CAFE_AUTO_LOOP = ""
    else:
        raise


@unittest.skipIf(QApplication is None, "PySide6 is not installed in this test environment.")
class TestYihuanMainWindowWorkbench(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _make_window(self) -> YihuanMainWindow:
        patches = [
            patch.object(YihuanMainWindow, "_setup_bridge", lambda self: None),
            patch.object(YihuanMainWindow, "_install_quick_stop_hotkey", lambda self, key, show_warning=False: None),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        window = YihuanMainWindow()
        def cleanup_window() -> None:
            window._pending_launch = None
            window._live_state = {"active_runs": {}}
            window.close()

        self.addCleanup(cleanup_window)
        return window

    def test_workbench_left_list_only_shows_business_tasks(self):
        window = self._make_window()

        task_ids = [
            str(window._task_list.item(index).data(Qt.UserRole))
            for index in range(window._task_list.count())
        ]

        self.assertEqual(task_ids, list(WORKBENCH_TASKS.keys()))
        self.assertEqual([WORKBENCH_TASKS[task_id]["label"] for task_id in task_ids], ["钓鱼", "沙威玛"])

    def test_tools_menu_contains_big_map_overlay_action(self):
        window = self._make_window()

        submenu_titles = []
        overlay_action_found = False
        for action in window.menuBar().actions():
            menu = action.menu()
            if menu is None:
                continue
            submenu_titles.append(menu.title())
            overlay_action_found = overlay_action_found or any(
                child_action.text() == "异环大地图点位悬浮层" for child_action in menu.actions()
            )

        self.assertIn("辅助", submenu_titles)
        self.assertTrue(overlay_action_found)

    def test_start_uses_countdown_before_dispatch_and_stop_cancels_it(self):
        window = self._make_window()
        window._runner_ready = True
        window._tasks_ready = True
        window._task_rows = {TASK_AUTO_LOOP: {"task_ref": TASK_AUTO_LOOP}}
        window._ui_preferences = GuiPreferences(task_start_delay_sec=5, quick_stop_hotkey="F8")
        dispatched: list[tuple[str, object]] = []
        window.request_run_task.connect(lambda task_ref, inputs: dispatched.append((task_ref, inputs)))

        window._request_start_selected_task()

        self.assertIsNotNone(window._pending_launch)
        self.assertEqual(dispatched, [])

        window._request_stop_current_task()

        self.assertIsNone(window._pending_launch)
        self.assertEqual(dispatched, [])

    def test_stop_active_task_emits_cancel_request(self):
        window = self._make_window()
        cancelled: list[str] = []
        window.request_cancel_task.connect(cancelled.append)
        window._live_state = {
            "active_runs": {
                "cid-1": {
                    "cid": "cid-1",
                    "task_name": TASK_CAFE_AUTO_LOOP,
                    "status": "running",
                }
            }
        }

        window._request_stop_current_task()

        self.assertEqual(cancelled, ["cid-1"])


if __name__ == "__main__":
    unittest.main()

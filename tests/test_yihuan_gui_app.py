from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from packages.yihuan_gui.app import AUXILIARY_TOOLS, WORKBENCH_TASKS, YihuanMainWindow
    from packages.yihuan_gui.logic import (
        GuiPreferences,
        TASK_AUTO_LOOP,
        TASK_CAFE_AUTO_LOOP,
        TASK_MAHJONG_AUTO_LOOP,
        TASK_ONE_CAFE_REVENUE_RESTOCK,
    )
except ModuleNotFoundError as exc:
    if str(getattr(exc, "name", "")).startswith("PySide6"):
        QApplication = None
        Qt = None
        AUXILIARY_TOOLS = None
        WORKBENCH_TASKS = None
        YihuanMainWindow = None
        GuiPreferences = None
        TASK_AUTO_LOOP = ""
        TASK_CAFE_AUTO_LOOP = ""
        TASK_MAHJONG_AUTO_LOOP = ""
        TASK_ONE_CAFE_REVENUE_RESTOCK = ""
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
        self.assertEqual([WORKBENCH_TASKS[task_id]["label"] for task_id in task_ids], ["钓鱼", "沙威玛", "一咖舍", "麻将"])

    def test_top_menu_contains_task_auxiliary_and_settings_pages(self):
        window = self._make_window()

        action_titles = [action.text() for action in window.menuBar().actions()]

        self.assertIn("任务界面", action_titles)
        self.assertIn("辅助功能", action_titles)
        self.assertIn("设置界面", action_titles)
        self.assertNotIn("辅助", action_titles)

    def test_auxiliary_page_only_shows_big_map_overlay_tool(self):
        window = self._make_window()

        tool_ids = [
            str(window._auxiliary_tool_list.item(index).data(Qt.UserRole))
            for index in range(window._auxiliary_tool_list.count())
        ]

        self.assertEqual(tool_ids, list(AUXILIARY_TOOLS.keys()))
        self.assertEqual(AUXILIARY_TOOLS[tool_ids[0]].label, "大地图点位悬浮层")
        self.assertEqual(window._auxiliary_title_label.text(), "大地图点位悬浮层")

    def test_auxiliary_start_and_stop_use_map_overlay_controller(self):
        class FakeSignal:
            def __init__(self) -> None:
                self.callbacks = []

            def connect(self, callback):
                self.callbacks.append(callback)

        class FakeMapOverlayController:
            last_instance = None

            def __init__(self, *_args, **_kwargs) -> None:
                self.state_changed = FakeSignal()
                self.data_ready = FakeSignal()
                self.log_message = FakeSignal()
                self.started_with = None
                self.stop_called = False
                FakeMapOverlayController.last_instance = self

            def start(self, settings):
                self.started_with = settings

            def stop(self):
                self.stop_called = True

        with patch("packages.yihuan_gui.app.MapOverlayController", FakeMapOverlayController):
            window = self._make_window()
            window._start_selected_auxiliary_tool()
            controller = FakeMapOverlayController.last_instance

            self.assertIsNotNone(controller)
            self.assertIsNotNone(controller.started_with)

            window._map_overlay_running = True
            window._stop_selected_auxiliary_tool()

            self.assertTrue(controller.stop_called)

    def test_map_overlay_settings_widgets_update_qsettings_model(self):
        window = self._make_window()

        window._map_search_edit.setText("Oracle")
        window._map_show_labels_check.setChecked(True)

        self.assertEqual(window._map_overlay_settings.search_text, "Oracle")
        self.assertTrue(window._map_overlay_settings.show_labels)

    def test_start_uses_countdown_before_dispatch_and_stop_cancels_it(self):
        window = self._make_window()
        window._select_task_id("fishing")
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

    def test_one_cafe_inputs_are_collected_from_page_and_settings_defaults(self):
        window = self._make_window()
        window._select_task_id("one_cafe")
        window._one_cafe_defaults = window._one_cafe_defaults.__class__(profile_name="custom_one_cafe")
        window._one_cafe_withdraw_check.setChecked(True)
        window._one_cafe_restock_check.setChecked(False)
        index = window._one_cafe_restock_hours_combo.findData(72)
        self.assertGreaterEqual(index, 0)
        window._one_cafe_restock_hours_combo.setCurrentIndex(index)

        payload = window._collect_selected_task_inputs()

        self.assertEqual(
            payload,
            {
                "profile_name": "custom_one_cafe",
                "withdraw_enabled": True,
                "restock_enabled": False,
                "restock_hours": 72,
            },
        )
        self.assertEqual(WORKBENCH_TASKS["one_cafe"]["task_ref"], TASK_ONE_CAFE_REVENUE_RESTOCK)

    def test_mahjong_inputs_are_collected_from_page_and_settings_defaults(self):
        window = self._make_window()
        window._select_task_id("mahjong")
        window._mahjong_defaults = window._mahjong_defaults.__class__(profile_name="custom_mahjong")
        window._mahjong_max_seconds_spin.setValue(180)
        window._mahjong_start_game_check.setChecked(True)
        window._mahjong_auto_hu_check.setChecked(True)
        window._mahjong_auto_peng_check.setChecked(False)
        window._mahjong_auto_discard_check.setChecked(True)

        payload = window._collect_selected_task_inputs()

        self.assertEqual(
            payload,
            {
                "profile_name": "custom_mahjong",
                "max_seconds": 180,
                "start_game": True,
                "auto_hu": True,
                "auto_peng": False,
                "auto_discard": True,
                "dry_run": False,
                "debug_enabled": False,
            },
        )
        self.assertEqual(WORKBENCH_TASKS["mahjong"]["task_ref"], TASK_MAHJONG_AUTO_LOOP)


if __name__ == "__main__":
    unittest.main()

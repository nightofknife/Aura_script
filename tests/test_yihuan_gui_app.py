from __future__ import annotations

from pathlib import Path
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
        TASK_COMBAT_AUTO_LOOP,
        TASK_MAHJONG_AUTO_LOOP,
        TASK_ONE_CAFE_REVENUE_RESTOCK,
        TASK_PIANO_PLAY_MIDI,
        TASK_TETROMINOES_AUTO_LOOP,
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
        TASK_COMBAT_AUTO_LOOP = ""
        TASK_MAHJONG_AUTO_LOOP = ""
        TASK_ONE_CAFE_REVENUE_RESTOCK = ""
        TASK_PIANO_PLAY_MIDI = ""
        TASK_TETROMINOES_AUTO_LOOP = ""
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
        self.assertEqual(
            [WORKBENCH_TASKS[task_id]["label"] for task_id in task_ids],
            ["钓鱼", "沙威玛", "一咖舍", "麻将", "战斗", "俄罗斯方块", "自动弹钢琴"],
        )

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

    def test_fishing_inputs_include_latest_bait_controls(self):
        window = self._make_window()
        window._select_task_id("fishing")
        window._fishing_defaults = window._fishing_defaults.__class__(profile_name="custom_fishing")
        window._max_rounds_spin.setValue(12)
        window._sell_fish_every_rounds_spin.setValue(4)
        window._bait_buy_repeat_count_spin.setValue(2)
        window._sell_before_buy_bait_check.setChecked(False)

        payload = window._collect_selected_task_inputs()

        self.assertEqual(
            payload,
            {
                "max_rounds": 12,
                "profile_name": "custom_fishing",
                "sell_fish_every_rounds": 4,
                "bait_buy_repeat_count": 2,
                "sell_before_buy_bait": False,
            },
        )

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

    def test_combat_inputs_are_collected_from_page_and_settings_defaults(self):
        window = self._make_window()
        window._select_task_id("combat")
        window._combat_defaults = window._combat_defaults.__class__(profile_name="custom_combat", strategy_name="default")
        window._combat_max_seconds_spin.setValue(240)
        window._combat_max_encounters_spin.setValue(6)
        window._combat_auto_target_check.setChecked(True)
        window._combat_auto_dodge_check.setChecked(False)
        window._combat_debug_enabled_check.setChecked(True)
        window._combat_capture_debug_enabled_check.setChecked(True)
        window._combat_capture_interval_spin.setValue(1.5)
        window._combat_capture_max_images_spin.setValue(80)
        window._combat_capture_raw_enabled_check.setChecked(True)
        window._combat_strategy_combo.addItem("burst", "burst")
        window._combat_strategy_combo.setCurrentIndex(window._combat_strategy_combo.findData("burst"))

        payload = window._collect_selected_task_inputs()

        self.assertEqual(
            payload,
            {
                "profile_name": "custom_combat",
                "strategy_name": "burst",
                "max_seconds": 240,
                "max_encounters": 6,
                "battle_count": 6,
                "auto_target": True,
                "auto_dodge": False,
                "dry_run": False,
                "debug_enabled": True,
                "capture_debug_enabled": True,
                "capture_interval_sec": 1.5,
                "capture_max_images": 80,
                "capture_raw_enabled": True,
            },
        )
        self.assertEqual(WORKBENCH_TASKS["combat"]["task_ref"], TASK_COMBAT_AUTO_LOOP)

    def test_tetrominoes_inputs_are_collected_from_page_and_settings_defaults(self):
        window = self._make_window()
        window._select_task_id("tetrominoes")
        window._tetrominoes_defaults = window._tetrominoes_defaults.__class__(profile_name="custom_tetrominoes")
        window._tetrominoes_max_seconds_spin.setValue(240)
        window._tetrominoes_max_pieces_spin.setValue(80)
        window._tetrominoes_start_game_check.setChecked(False)

        payload = window._collect_selected_task_inputs()

        self.assertEqual(
            payload,
            {
                "profile_name": "custom_tetrominoes",
                "max_seconds": 240,
                "max_pieces": 80,
                "start_game": False,
                "dry_run": False,
                "debug_enabled": False,
            },
        )
        self.assertEqual(WORKBENCH_TASKS["tetrominoes"]["task_ref"], TASK_TETROMINOES_AUTO_LOOP)

    def test_piano_inputs_are_collected_from_page_and_settings_defaults(self):
        window = self._make_window()
        window._select_task_id("piano")
        fixture_path = (
            Path(__file__).resolve().parent / "fixtures" / "yihuan" / "piano" / "no_roll_needed.mid"
        ).resolve()
        window._piano_file_edit.setText(str(fixture_path))
        window._piano_conflict_policy_combo.setCurrentIndex(window._piano_conflict_policy_combo.findData("roll"))
        window._piano_tempo_scale_spin.setValue(1.2)
        window._piano_start_delay_spin.setValue(400)
        window._piano_transpose_spin.setValue(-1)
        window._piano_roll_note_spin.setValue(45)
        window._piano_velocity_threshold_spin.setValue(3)
        window._piano_focus_window_check.setChecked(False)
        window._piano_dry_run_check.setChecked(True)

        payload = window._collect_selected_task_inputs()

        self.assertEqual(
            payload,
            {
                "file_path": str(fixture_path),
                "conflict_policy": "roll",
                "transpose_semitones": -1,
                "tempo_scale": 1.2,
                "start_delay_ms": 400,
                "roll_note_ms": 45,
                "velocity_threshold": 3,
                "focus_window": False,
                "dry_run": True,
            },
        )
        self.assertEqual(WORKBENCH_TASKS["piano"]["task_ref"], TASK_PIANO_PLAY_MIDI)

    def test_save_settings_persists_combat_profile_defaults(self):
        window = self._make_window()
        saved: dict[str, object] = {}
        window._combat_profile_combo.addItem("custom_combat", "custom_combat")
        window._combat_profile_combo.setCurrentIndex(window._combat_profile_combo.findData("custom_combat"))
        window._tetrominoes_profile_combo.addItem("custom_tetrominoes", "custom_tetrominoes")
        window._tetrominoes_profile_combo.setCurrentIndex(
            window._tetrominoes_profile_combo.findData("custom_tetrominoes")
        )

        def _capture_combat_defaults(defaults):
            saved["combat_defaults"] = defaults

        def _capture_tetrominoes_defaults(defaults):
            saved["tetrominoes_defaults"] = defaults

        with (
            patch.object(window._repo, "update_runtime_settings", lambda *_args, **_kwargs: None),
            patch.object(window._repo, "update_fishing_defaults", lambda *_args, **_kwargs: None),
            patch.object(window._repo, "update_cafe_defaults", lambda *_args, **_kwargs: None),
            patch.object(window._repo, "update_one_cafe_defaults", lambda *_args, **_kwargs: None),
            patch.object(window._repo, "update_mahjong_defaults", lambda *_args, **_kwargs: None),
            patch.object(window._repo, "update_combat_defaults", _capture_combat_defaults),
            patch.object(window._repo, "update_tetrominoes_defaults", _capture_tetrominoes_defaults),
            patch.object(window._repo, "save_ui_preferences", lambda *_args, **_kwargs: None),
            patch.object(
                window._repo,
                "get_combat_defaults",
                lambda _task=None: saved.get("combat_defaults", window._combat_defaults),
            ),
            patch.object(
                window._repo,
                "get_tetrominoes_defaults",
                lambda _task=None: saved.get("tetrominoes_defaults", window._tetrominoes_defaults),
            ),
        ):
            window._save_settings()

        self.assertIn("combat_defaults", saved)
        self.assertEqual(saved["combat_defaults"].profile_name, "custom_combat")
        self.assertIn("tetrominoes_defaults", saved)
        self.assertEqual(saved["tetrominoes_defaults"].profile_name, "custom_tetrominoes")


if __name__ == "__main__":
    unittest.main()

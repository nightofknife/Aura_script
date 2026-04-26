from __future__ import annotations

import unittest

from packages.yihuan_gui.logic import (
    FishingRunDefaults,
    GuiPreferences,
    LiveUiState,
    RuntimeSettings,
    TASK_AUTO_LOOP,
    TASK_LIVE_MONITOR,
    TASK_PLAN_READY,
    build_auto_loop_inputs,
    build_settings_sections,
    extract_auto_loop_defaults,
    render_auto_loop_brief_text,
    reduce_live_events,
    render_task_result_html,
    task_is_enabled,
)


class TestYihuanGuiLogic(unittest.TestCase):
    def test_build_auto_loop_inputs_uses_page_value_and_defaults(self):
        payload = build_auto_loop_inputs(12, FishingRunDefaults(profile_name="default_1280x720_cn"))

        self.assertEqual(
            payload,
            {
                "max_rounds": 12,
                "profile_name": "default_1280x720_cn",
            },
        )

    def test_extract_auto_loop_defaults_uses_task_default(self):
        defaults = extract_auto_loop_defaults(
            {
                "inputs": [
                    {"name": "max_rounds", "type": "number", "default": 0},
                    {"name": "profile_name", "type": "string", "default": "custom_profile"},
                ]
            }
        )

        self.assertEqual(defaults.profile_name, "custom_profile")

    def test_reduce_live_events_tracks_auto_loop_lifecycle(self):
        state, finished = reduce_live_events(
            LiveUiState(),
            [
                {
                    "name": "queue.enqueued",
                    "payload": {
                        "cid": "cid-1",
                        "game_name": "yihuan",
                        "task_name": TASK_AUTO_LOOP,
                    },
                },
                {
                    "name": "task.started",
                    "payload": {
                        "cid": "cid-1",
                        "game_name": "yihuan",
                        "task_name": TASK_AUTO_LOOP,
                    },
                },
                {
                    "name": "task.finished",
                    "payload": {
                        "cid": "cid-1",
                        "game_name": "yihuan",
                        "task_name": TASK_AUTO_LOOP,
                        "final_status": "success",
                    },
                },
            ],
        )

        self.assertEqual(finished, ["cid-1"])
        self.assertFalse(state.active_runs)
        self.assertEqual(state.latest_status, "success")

    def test_task_result_renderer_handles_auto_loop(self):
        html = render_task_result_html(
            TASK_AUTO_LOOP,
            {
                "task_name": TASK_AUTO_LOOP,
                "final_result": {
                    "user_data": {
                        "status": "partial",
                        "round_count": 3,
                        "success_count": 2,
                        "failure_count": 1,
                        "stopped_reason": "max_rounds",
                        "results": [
                            {
                                "round_index": 1,
                                "status": "success",
                                "timings": {"total_sec": 2.5, "hook_wait_sec": 1.0, "duel_sec": 1.2},
                                "detection_stats": {
                                    "observation_sec": 1.2,
                                    "samples": 3,
                                    "zone": {"detected_ratio": 1.0},
                                    "indicator": {"detected_ratio": 0.9},
                                    "indicator_raw": {"detected_ratio": 1.0},
                                    "reason_sec": {"ok": 1.2},
                                },
                                "phase_trace": [{"t_ms": 1000, "phase": "duel", "note": "control_inside_tap"}],
                            }
                        ],
                    }
                },
            },
        )

        self.assertIn("部分成功", html)
        self.assertIn("达到最大轮数", html)
        self.assertIn("每轮摘要", html)
        self.assertIn("控制：区间内点按", html)

    def test_auto_loop_brief_text_uses_business_status(self):
        text = render_auto_loop_brief_text(
            {
                "status": "success",
                "final_result": {
                    "user_data": {
                        "status": "failed",
                        "round_count": 2,
                        "success_count": 0,
                        "failure_count": 2,
                        "stopped_reason": "max_rounds",
                    }
                },
            }
        )

        self.assertIn("失败", text)
        self.assertIn("总轮数 2", text)

    def test_runtime_task_guard_disables_auto_loop_when_fishing_task_active(self):
        active_runs = {
            "cid-1": {
                "cid": "cid-1",
                "task_name": TASK_LIVE_MONITOR,
                "status": "running",
            }
        }

        self.assertFalse(task_is_enabled(TASK_AUTO_LOOP, active_runs))
        self.assertTrue(task_is_enabled(TASK_PLAN_READY, active_runs))

    def test_settings_sections_are_grouped_into_runtime_and_ui(self):
        sections = build_settings_sections(
            RuntimeSettings(
                title_regex="(异环|Neverness to Everness)",
                exclude_titles=["Crash Reporter", "Unreal Engine"],
                allow_borderless=True,
                capture_backend="gdi",
                input_backend="sendinput",
                input_profile="default_pc",
            ),
            GuiPreferences(history_limit=40, auto_runtime_probe_on_startup=True, expand_developer_tools=False),
            ["default_pc", "default_1280x720_cn"],
        )

        self.assertEqual([section.title for section in sections], ["运行环境", "界面偏好"])
        self.assertEqual(sections[0].fields[0].key, "runtime.target.title_regex")
        self.assertEqual(sections[1].fields[0].key, "gui.history_limit")


if __name__ == "__main__":
    unittest.main()

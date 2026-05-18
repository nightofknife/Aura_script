from __future__ import annotations

import unittest

from packages.yihuan_gui.logic import (
    CafeRunDefaults,
    CombatRunDefaults,
    FishingRunDefaults,
    GuiPreferences,
    LiveUiState,
    MahjongRunDefaults,
    OneCafeRunDefaults,
    RuntimeSettings,
    RhythmRunDefaults,
    TASK_AUTO_LOOP,
    TASK_CAFE_AUTO_LOOP,
    TASK_COMBAT_AUTO_LOOP,
    TASK_LIVE_MONITOR,
    TASK_MAHJONG_AUTO_LOOP,
    TASK_ONE_CAFE_REVENUE_RESTOCK,
    TASK_PIANO_PLAY_MIDI,
    TASK_RHYTHM_AUTO_LOOP,
    TASK_TETROMINOES_AUTO_LOOP,
    PianoRunDefaults,
    TetrominoesRunDefaults,
    VISIBLE_HISTORY_TASK_REFS,
    TASK_PLAN_READY,
    build_auto_loop_inputs,
    build_cafe_loop_inputs,
    build_combat_loop_inputs,
    build_mahjong_loop_inputs,
    build_one_cafe_inputs,
    build_piano_play_midi_inputs,
    build_rhythm_loop_inputs,
    build_tetrominoes_loop_inputs,
    build_settings_sections,
    cafe_loop_business_status,
    combat_loop_business_status,
    mahjong_loop_business_status,
    one_cafe_business_status,
    piano_play_midi_business_status,
    rhythm_loop_business_status,
    tetrominoes_loop_business_status,
    extract_auto_loop_defaults,
    extract_cafe_loop_defaults,
    extract_combat_loop_defaults,
    extract_mahjong_loop_defaults,
    extract_one_cafe_defaults,
    extract_piano_play_midi_defaults,
    extract_rhythm_loop_defaults,
    extract_tetrominoes_loop_defaults,
    render_auto_loop_brief_text,
    render_cafe_loop_brief_text,
    render_combat_loop_brief_text,
    render_mahjong_loop_brief_text,
    render_one_cafe_brief_text,
    render_piano_play_midi_brief_text,
    render_rhythm_loop_brief_text,
    render_tetrominoes_loop_brief_text,
    reduce_live_events,
    render_task_result_html,
    task_is_enabled,
)


class TestYihuanGuiLogic(unittest.TestCase):
    def test_build_auto_loop_inputs_uses_page_value_and_defaults(self):
        payload = build_auto_loop_inputs(
            12,
            5,
            2,
            False,
            FishingRunDefaults(profile_name="default_1280x720_cn"),
        )

        self.assertEqual(
            payload,
            {
                "max_rounds": 12,
                "profile_name": "default_1280x720_cn",
                "sell_fish_every_rounds": 5,
                "bait_buy_repeat_count": 2,
                "sell_before_buy_bait": False,
            },
        )

    def test_extract_auto_loop_defaults_uses_task_default(self):
        defaults = extract_auto_loop_defaults(
            {
                "inputs": [
                    {"name": "max_rounds", "type": "number", "default": 0},
                    {"name": "profile_name", "type": "string", "default": "custom_profile"},
                    {"name": "sell_fish_every_rounds", "type": "number", "default": 6},
                    {"name": "bait_buy_repeat_count", "type": "number", "default": 3},
                    {"name": "sell_before_buy_bait", "type": "boolean", "default": False},
                ]
            }
        )

        self.assertEqual(defaults.profile_name, "custom_profile")
        self.assertEqual(defaults.sell_fish_every_rounds, 6)
        self.assertEqual(defaults.bait_buy_repeat_count, 3)
        self.assertFalse(defaults.sell_before_buy_bait)

    def test_build_cafe_loop_inputs_uses_page_values_and_defaults(self):
        payload = build_cafe_loop_inputs(
            120,
            8,
            True,
            False,
            False,
            CafeRunDefaults(profile_name="default_1280x720_cn"),
        )

        self.assertEqual(
            payload,
            {
                "profile_name": "default_1280x720_cn",
                "max_seconds": 120,
                "max_orders": 8,
                "start_game": True,
                "wait_level_started": False,
                "full_assist_auto_hammer_mode": False,
                "min_order_interval_sec": 0.3,
                "min_order_duration_sec": 0.0,
            },
        )

    def test_extract_cafe_loop_defaults_uses_task_defaults(self):
        defaults = extract_cafe_loop_defaults(
            {
                "inputs": [
                    {"name": "profile_name", "type": "string", "default": "custom_cafe"},
                    {"name": "max_seconds", "type": "number", "default": 90},
                    {"name": "max_orders", "type": "number", "default": 4},
                    {"name": "start_game", "type": "boolean", "default": False},
                    {"name": "wait_level_started", "type": "boolean", "default": True},
                    {"name": "full_assist_auto_hammer_mode", "type": "boolean", "default": True},
                    {"name": "min_order_interval_sec", "type": "number", "default": 0.75},
                    {"name": "min_order_duration_sec", "type": "number", "default": 0.25},
                ]
            }
        )

        self.assertEqual(defaults.profile_name, "custom_cafe")
        self.assertEqual(defaults.max_seconds, 90)
        self.assertEqual(defaults.max_orders, 4)
        self.assertFalse(defaults.start_game)
        self.assertTrue(defaults.wait_level_started)
        self.assertTrue(defaults.full_assist_auto_hammer_mode)
        self.assertEqual(defaults.min_order_interval_sec, 0.75)
        self.assertEqual(defaults.min_order_duration_sec, 0.25)

    def test_build_one_cafe_inputs_uses_page_values_and_defaults(self):
        payload = build_one_cafe_inputs(
            True,
            False,
            72,
            OneCafeRunDefaults(profile_name="default_1280x720_cn"),
        )

        self.assertEqual(
            payload,
            {
                "profile_name": "default_1280x720_cn",
                "withdraw_enabled": True,
                "restock_enabled": False,
                "restock_hours": 72,
            },
        )

    def test_extract_one_cafe_defaults_uses_task_defaults(self):
        defaults = extract_one_cafe_defaults(
            {
                "inputs": [
                    {"name": "profile_name", "type": "string", "default": "custom_one_cafe"},
                    {"name": "withdraw_enabled", "type": "boolean", "default": False},
                    {"name": "restock_enabled", "type": "boolean", "default": True},
                    {"name": "restock_hours", "type": "number", "default": 4},
                ]
            }
        )

        self.assertEqual(defaults.profile_name, "custom_one_cafe")
        self.assertFalse(defaults.withdraw_enabled)
        self.assertTrue(defaults.restock_enabled)
        self.assertEqual(defaults.restock_hours, 4)

    def test_build_mahjong_loop_inputs_uses_page_values_and_defaults(self):
        payload = build_mahjong_loop_inputs(
            180,
            True,
            True,
            False,
            True,
            MahjongRunDefaults(profile_name="default_1280x720_cn"),
        )

        self.assertEqual(
            payload,
            {
                "profile_name": "default_1280x720_cn",
                "max_seconds": 180,
                "start_game": True,
                "auto_hu": True,
                "auto_peng": False,
                "auto_discard": True,
                "dry_run": False,
                "debug_enabled": False,
            },
        )

    def test_extract_mahjong_loop_defaults_uses_task_defaults(self):
        defaults = extract_mahjong_loop_defaults(
            {
                "inputs": [
                    {"name": "profile_name", "type": "string", "default": "custom_mahjong"},
                    {"name": "max_seconds", "type": "number", "default": 60},
                    {"name": "start_game", "type": "boolean", "default": False},
                    {"name": "auto_hu", "type": "boolean", "default": True},
                    {"name": "auto_peng", "type": "boolean", "default": False},
                    {"name": "auto_discard", "type": "boolean", "default": True},
                ]
            }
        )

        self.assertEqual(defaults.profile_name, "custom_mahjong")
        self.assertEqual(defaults.max_seconds, 60)
        self.assertFalse(defaults.start_game)
        self.assertTrue(defaults.auto_hu)
        self.assertFalse(defaults.auto_peng)
        self.assertTrue(defaults.auto_discard)

    def test_build_combat_loop_inputs_uses_page_values_and_defaults(self):
        payload = build_combat_loop_inputs(
            300,
            5,
            True,
            False,
            "burst",
            True,
            True,
            1.5,
            80,
            True,
            CombatRunDefaults(profile_name="default_1280x720_cn"),
        )

        self.assertEqual(
            payload,
            {
                "profile_name": "default_1280x720_cn",
                "strategy_name": "burst",
                "max_seconds": 300,
                "max_encounters": 5,
                "battle_count": 5,
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

    def test_extract_combat_loop_defaults_uses_task_defaults(self):
        defaults = extract_combat_loop_defaults(
            {
                "inputs": [
                    {"name": "profile_name", "type": "string", "default": "custom_combat"},
                    {"name": "strategy_name", "type": "string", "default": "burst"},
                    {"name": "max_seconds", "type": "number", "default": 180},
                    {"name": "max_encounters", "type": "number", "default": 3},
                    {"name": "battle_count", "type": "number", "default": 4},
                    {"name": "auto_target", "type": "boolean", "default": False},
                    {"name": "auto_dodge", "type": "boolean", "default": True},
                    {"name": "debug_enabled", "type": "boolean", "default": True},
                    {"name": "capture_debug_enabled", "type": "boolean", "default": True},
                    {"name": "capture_interval_sec", "type": "number", "default": 1.5},
                    {"name": "capture_max_images", "type": "number", "default": 80},
                    {"name": "capture_raw_enabled", "type": "boolean", "default": True},
                ]
            }
        )

        self.assertEqual(defaults.profile_name, "custom_combat")
        self.assertEqual(defaults.strategy_name, "burst")
        self.assertEqual(defaults.max_seconds, 180)
        self.assertEqual(defaults.max_encounters, 4)
        self.assertFalse(defaults.auto_target)
        self.assertTrue(defaults.auto_dodge)
        self.assertTrue(defaults.debug_enabled)
        self.assertTrue(defaults.capture_debug_enabled)
        self.assertEqual(defaults.capture_interval_sec, 1.5)
        self.assertEqual(defaults.capture_max_images, 80)
        self.assertTrue(defaults.capture_raw_enabled)

    def test_build_tetrominoes_loop_inputs_uses_page_values_and_defaults(self):
        payload = build_tetrominoes_loop_inputs(
            240,
            80,
            False,
            TetrominoesRunDefaults(profile_name="default_1280x720_cn"),
        )

        self.assertEqual(
            payload,
            {
                "profile_name": "default_1280x720_cn",
                "max_seconds": 240,
                "max_pieces": 80,
                "start_game": False,
                "dry_run": False,
                "debug_enabled": False,
            },
        )

    def test_extract_tetrominoes_loop_defaults_uses_task_defaults(self):
        defaults = extract_tetrominoes_loop_defaults(
            {
                "inputs": [
                    {"name": "profile_name", "type": "string", "default": "custom_tetrominoes"},
                    {"name": "max_seconds", "type": "number", "default": 90},
                    {"name": "max_pieces", "type": "number", "default": 32},
                    {"name": "start_game", "type": "boolean", "default": False},
                ]
            }
        )

        self.assertEqual(defaults.profile_name, "custom_tetrominoes")
        self.assertEqual(defaults.max_seconds, 90)
        self.assertEqual(defaults.max_pieces, 32)
        self.assertFalse(defaults.start_game)

    def test_build_rhythm_loop_inputs_uses_page_values_and_defaults(self):
        payload = build_rhythm_loop_inputs(
            3,
            240,
            True,
            False,
            "a,s,k,l",
            -16,
            True,
            RhythmRunDefaults(profile_name="default_1280x720_cn"),
        )

        self.assertEqual(
            payload,
            {
                "profile_name": "default_1280x720_cn",
                "loop_count": 3,
                "max_seconds": 240,
                "start_game": True,
                "close_result": False,
                "lane_keys": "a,s,k,l",
                "lane_y_offset_px": -16,
                "dry_run": False,
                "debug_enabled": True,
            },
        )

    def test_extract_rhythm_loop_defaults_uses_task_defaults(self):
        defaults = extract_rhythm_loop_defaults(
            {
                "inputs": [
                    {"name": "profile_name", "type": "string", "default": "custom_rhythm"},
                    {"name": "loop_count", "type": "number", "default": 2},
                    {"name": "max_seconds", "type": "number", "default": 90},
                    {"name": "start_game", "type": "boolean", "default": False},
                    {"name": "close_result", "type": "boolean", "default": False},
                    {"name": "lane_keys", "type": "string", "default": "a,s,k,l"},
                    {"name": "lane_y_offset_px", "type": "number", "default": -12},
                    {"name": "debug_enabled", "type": "boolean", "default": True},
                ]
            }
        )

        self.assertEqual(defaults.profile_name, "custom_rhythm")
        self.assertEqual(defaults.loop_count, 2)
        self.assertEqual(defaults.max_seconds, 90)
        self.assertFalse(defaults.start_game)
        self.assertFalse(defaults.close_result)
        self.assertEqual(defaults.lane_keys, "a,s,k,l")
        self.assertEqual(defaults.lane_y_offset_px, -12)
        self.assertTrue(defaults.debug_enabled)

    def test_build_piano_play_midi_inputs_uses_page_values(self):
        payload = build_piano_play_midi_inputs(
            "D:/music/test.mid",
            "roll",
            2,
            1.25,
            500,
            40,
            5,
            False,
            True,
        )

        self.assertEqual(
            payload,
            {
                "file_path": "D:/music/test.mid",
                "conflict_policy": "roll",
                "transpose_semitones": 2,
                "tempo_scale": 1.25,
                "start_delay_ms": 500,
                "roll_note_ms": 40,
                "velocity_threshold": 5,
                "focus_window": False,
                "dry_run": True,
            },
        )

    def test_extract_piano_play_midi_defaults_uses_task_defaults(self):
        defaults = extract_piano_play_midi_defaults(
            {
                "inputs": [
                    {"name": "conflict_policy", "type": "string", "default": "roll"},
                    {"name": "transpose_semitones", "type": "number", "default": -2},
                    {"name": "tempo_scale", "type": "number", "default": 0.8},
                    {"name": "start_delay_ms", "type": "number", "default": 800},
                    {"name": "roll_note_ms", "type": "number", "default": 55},
                    {"name": "velocity_threshold", "type": "number", "default": 4},
                    {"name": "focus_window", "type": "boolean", "default": False},
                    {"name": "dry_run", "type": "boolean", "default": True},
                ]
            }
        )

        self.assertEqual(defaults.conflict_policy, "roll")
        self.assertEqual(defaults.transpose_semitones, -2)
        self.assertEqual(defaults.tempo_scale, 0.8)
        self.assertEqual(defaults.start_delay_ms, 800)
        self.assertEqual(defaults.roll_note_ms, 55)
        self.assertEqual(defaults.velocity_threshold, 4)
        self.assertFalse(defaults.focus_window)
        self.assertTrue(defaults.dry_run)

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

    def test_task_result_renderer_handles_cafe_loop(self):
        detail = {
            "task_name": TASK_CAFE_AUTO_LOOP,
            "final_result": {
                "user_data": {
                    "status": "success",
                    "stopped_reason": "max_orders",
                    "failure_reason": None,
                    "orders_completed": 3,
                    "fake_customers_detected": 2,
                    "fake_customers_driven": 1,
                    "pending_batches": {"coffee": {"batch_size": 3, "ready_in_sec": 0.4}},
                    "min_order_interval_sec": 0.5,
                    "min_order_duration_sec": 0.0,
                    "recognized_counts": {
                        "latte_coffee": 1,
                        "cream_coffee": 1,
                        "bacon_bread": 1,
                    },
                    "batches_made": {"bread": 1, "croissant": 1, "cake": 1, "coffee": 1},
                    "stocks_remaining": {"bread": 2, "croissant": 3, "cake": 6, "coffee": 1},
                    "unknown_scan_count": 0,
                    "level_outcome": "success",
                    "elapsed_sec": 12.5,
                    "profile_name": "default_1280x720_cn",
                    "perf_stats": {
                        "time_sec": {"order_scan": 0.2, "order_guard_sleep": 0.1},
                        "counts": {
                            "order_scan_count": 3,
                            "order_guard_active_block_count": 2,
                            "order_guard_blocked_order_count": 1,
                            "order_guard_deferred_all_count": 1,
                            "order_guard_safe_selection_count": 1,
                        },
                    },
                    "phase_trace": [
                        {"t": 0.5, "event": "fake_customer_hammer_clicked", "candidate_count": 1},
                        {
                            "t": 0.8,
                            "event": "order_guard_blocked_orders",
                            "blocked": [{"order": {"recipe_id": "latte_coffee"}}],
                        },
                        {"t": 0.9, "event": "order_guard_deferred_all", "blocked": []},
                        {"t": 1.0, "event": "order_selected", "recipe_id": "latte_coffee"},
                        {"t": 2.0, "event": "order_completed", "recipe_id": "latte_coffee"},
                    ],
                }
            },
        }

        html = render_task_result_html(TASK_CAFE_AUTO_LOOP, detail)

        self.assertIn("完成订单 3", render_cafe_loop_brief_text(detail))
        self.assertIn("驱赶假顾客 1", render_cafe_loop_brief_text(detail))
        self.assertIn("守卫暂缓 1", render_cafe_loop_brief_text(detail))
        self.assertEqual(cafe_loop_business_status(detail), "success")
        self.assertIn("达到最大订单数", html)
        self.assertIn("拿铁咖啡", html)
        self.assertIn("驱赶假顾客", html)
        self.assertIn("订单安全守卫", html)
        self.assertIn("全部订单暂缓", html)
        self.assertIn("订单守卫阻挡", html)
        self.assertIn("订单守卫暂缓全部订单", html)
        self.assertIn("未完成补货", html)
        self.assertIn("敲走假顾客", html)
        self.assertIn("订单统计", html)
        self.assertIn("事件轨迹", html)

    def test_task_result_renderer_handles_one_cafe(self):
        detail = {
            "task_name": TASK_ONE_CAFE_REVENUE_RESTOCK,
            "final_result": {
                "user_data": {
                    "status": "partial",
                    "stopped_reason": "unknown_restock_state",
                    "failure_reason": "unknown_restock_state",
                    "profile_name": "default_1280x720_cn",
                    "withdraw_attempted": True,
                    "withdraw_result": "claimed",
                    "withdraw_confirmed": True,
                    "restock_attempted": True,
                    "restock_hours": 24,
                    "restock_result": "unknown",
                    "returned_to_world": True,
                    "world_hud_found": False,
                    "phase_trace": [
                        {"phase": "open_city_tycoon", "status": "SUCCESS"},
                        {"phase": "one_cafe_entry", "final_detection_found": True},
                        {"phase": "withdraw", "attempted": True, "result": "claimed"},
                        {"phase": "restock", "attempted": True, "hours": 24, "result": "unknown"},
                        {"phase": "exit", "returned_to_world": True},
                    ],
                }
            },
        }

        html = render_task_result_html(TASK_ONE_CAFE_REVENUE_RESTOCK, detail)

        self.assertEqual(one_cafe_business_status(detail), "partial")
        self.assertIn("收益 已领取", render_one_cafe_brief_text(detail))
        self.assertIn("补货 24 小时 未知", render_one_cafe_brief_text(detail))
        self.assertIn("一咖舍", html)
        self.assertIn("领取收益", html)
        self.assertIn("补货状态未知", html)
        self.assertIn("阶段轨迹", html)

    def test_task_result_renderer_handles_mahjong_loop(self):
        detail = {
            "task_name": TASK_MAHJONG_AUTO_LOOP,
            "final_result": {
                "user_data": {
                    "status": "success",
                    "stopped_reason": "level_end",
                    "failure_reason": None,
                    "profile_name": "default_1280x720_cn",
                    "selected_missing_suit": "tong",
                    "hand_suit_counts": {"wan": 4, "tong": 1, "tiao": 3},
                    "auto_toggles_enabled": {"hu": True, "peng": True, "discard": True},
                    "phase_trace": [
                        {"t": 0.1, "phase": "ready", "note": "initial"},
                        {"t": 0.5, "phase": "dingque", "note": "after_ready", "dingque_button_count": 3},
                        {"t": 1.0, "phase": "playing", "note": "switch_verify", "enabled_switch_count": 3},
                        {"t": 5.0, "phase": "result", "note": "monitor"},
                    ],
                    "elapsed_sec": 5.0,
                }
            },
        }

        html = render_task_result_html(TASK_MAHJONG_AUTO_LOOP, detail)

        self.assertEqual(mahjong_loop_business_status(detail), "success")
        self.assertIn("缺门 筒", render_mahjong_loop_brief_text(detail))
        self.assertIn("自动开关", html)
        self.assertIn("定缺手牌统计", html)
        self.assertIn("验证自动开关", html)

    def test_task_result_renderer_handles_combat_loop(self):
        detail = {
            "task_name": TASK_COMBAT_AUTO_LOOP,
            "final_result": {
                "user_data": {
                    "status": "success",
                    "stopped_reason": "max_encounters",
                    "failure_reason": None,
                    "profile_name": "default_1280x720_cn",
                    "strategy_name": "default",
                    "encounters_completed": 2,
                    "current_phase": "post_combat",
                    "elapsed_sec": 12.5,
                    "screenshot_dir": "D:/tmp/combat_debug/session_001",
                    "capture_stats": {
                        "enabled": True,
                        "capture_interval_sec": 1.5,
                        "capture_max_images": 80,
                        "capture_raw_enabled": True,
                        "capture_failed_count": 1,
                        "skipped_max_images_count": 2,
                    },
                    "last_state": {
                        "enemy_health_found": False,
                        "enemy_health_count": 0,
                        "target_found": False,
                        "target_confidence": 0.0,
                        "skill_available": False,
                        "ultimate_available": True,
                        "current_slot": 2,
                    },
                    "combat_state_trace": [
                        {"t": 0.5, "phase": "enemy_detected", "note": "monitor", "enemy_health_found": True, "enemy_health_count": 2, "target_found": False},
                        {"t": 1.2, "phase": "audio_dodge", "note": "audio_dodge", "enemy_health_found": True, "enemy_health_count": 2, "target_found": True},
                    ],
                    "action_trace": [
                        {"t": 0.8, "action": "auto_target", "binding": "mouse_middle"},
                        {"t": 1.2, "action": "audio_dodge", "score": 0.22},
                        {"t": 1.8, "action": "ultimate", "binding": "q"},
                    ],
                }
            },
        }

        html = render_task_result_html(TASK_COMBAT_AUTO_LOOP, detail)

        self.assertEqual(combat_loop_business_status(detail), "success")
        self.assertIn("战斗 2 场", render_combat_loop_brief_text(detail))
        self.assertIn("最后状态", html)
        self.assertIn("状态轨迹", html)
        self.assertIn("动作轨迹", html)
        self.assertIn("audio_dodge", html)
        self.assertIn("战斗调试", html)
        self.assertIn("截图目录", html)

    def test_task_result_renderer_handles_tetrominoes_loop(self):
        detail = {
            "task_name": TASK_TETROMINOES_AUTO_LOOP,
            "final_result": {
                "user_data": {
                    "status": "failed",
                    "stopped_reason": "recognition_timeout",
                    "failure_reason": "low_confidence",
                    "failure_message": "Board confidence stayed below threshold.",
                    "profile_name": "default_1280x720_cn",
                    "pieces_played": 14,
                    "elapsed_sec": 18.25,
                    "start_game": True,
                    "start_clicked": True,
                    "result_screen_cleared_before_start": False,
                    "final_metrics": {"last_confidence": 0.42, "occupied_cells": 19},
                    "result_screen": {"found": False, "reason": "panel_missing"},
                    "decisions_tail": [{"shape": "T", "target_col": 4}],
                    "operation_log": [{"executed_sequence": ["tetrominoes_left", "tetrominoes_fast_drop"]}],
                    "debug_snapshots": [],
                }
            },
        }

        html = render_task_result_html(TASK_TETROMINOES_AUTO_LOOP, detail)

        self.assertEqual(tetrominoes_loop_business_status(detail), "failed")
        self.assertIn("已放置 14 块", render_tetrominoes_loop_brief_text(detail))
        self.assertIn("识别超时", html)
        self.assertIn("识别置信度不足", html)
        self.assertIn("最终指标", html)
        self.assertIn("最近操作", html)

    def test_task_result_renderer_handles_rhythm_loop(self):
        detail = {
            "task_name": TASK_RHYTHM_AUTO_LOOP,
            "final_result": {
                "user_data": {
                    "status": "success",
                    "stopped_reason": "loop_count",
                    "failure_reason": None,
                    "profile_name": "default_1280x720_cn",
                    "loops_completed": 2,
                    "press_count": 18,
                    "hits_by_lane": {"d": 4, "f": 5, "j": 6, "k": 3},
                    "lane_keys": {"d": "d", "f": "f", "j": "j", "k": "k"},
                    "lane_y_offset_px": -16,
                    "start_game": True,
                    "close_result": True,
                    "result_closed_count": 2,
                    "elapsed_sec": 35.25,
                    "loops": [{"loop_index": 2, "press_count": 8}],
                    "planned_actions": [{"action": "start_song"}],
                }
            },
        }

        html = render_task_result_html(TASK_RHYTHM_AUTO_LOOP, detail)

        self.assertEqual(rhythm_loop_business_status(detail), "success")
        self.assertIn("完成 2 轮", render_rhythm_loop_brief_text(detail))
        self.assertIn("总按键次数", html)
        self.assertIn("各轨触发", html)
        self.assertIn("判定线偏移", html)
        self.assertIn("最近轮次", html)

    def test_task_result_renderer_handles_piano_play_midi(self):
        detail = {
            "task_name": TASK_PIANO_PLAY_MIDI,
            "final_result": {
                "user_data": {
                    "status": "failed",
                    "stopped_reason": "unplayable_score",
                    "failure_reason": "physical_key_conflict",
                    "file_path": "D:/music/test.mid",
                    "conflict_policy": "strict",
                    "dry_run": False,
                    "parsed_summary": {
                        "format_type": 0,
                        "track_count": 1,
                        "division": 480,
                        "note_count": 12,
                        "unsupported_note_count": 0,
                    },
                    "scheduled_note_count": 12,
                    "scheduled_notes": [{"degree": "1", "start_ms": 0}],
                    "action_plan": [{"kind": "key_down", "key": "a", "t_ms": 0}],
                    "performed_actions": [],
                    "conflicts": [{"reason": "physical_key_conflict", "at_ms": 0}],
                    "elapsed_sec": 0.42,
                }
            },
        }

        html = render_task_result_html(TASK_PIANO_PLAY_MIDI, detail)

        self.assertEqual(piano_play_midi_business_status(detail), "failed")
        self.assertIn("strict", build_piano_play_midi_inputs("a.mid", "strict", 0, 1.0, 0, 35, 1, True, False)["conflict_policy"])
        self.assertIn("test.mid", render_piano_play_midi_brief_text(detail))
        self.assertIn("严格", html)
        self.assertIn("存在物理按键冲突", html)
        self.assertIn("动作计划预览", html)

    def test_runtime_task_guard_disables_auto_loop_when_fishing_task_active(self):
        active_runs = {
            "cid-1": {
                "cid": "cid-1",
                "task_name": TASK_LIVE_MONITOR,
                "status": "running",
            }
        }

        self.assertFalse(task_is_enabled(TASK_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_CAFE_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_ONE_CAFE_REVENUE_RESTOCK, active_runs))
        self.assertFalse(task_is_enabled(TASK_MAHJONG_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_COMBAT_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_TETROMINOES_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_RHYTHM_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_PIANO_PLAY_MIDI, active_runs))
        self.assertTrue(task_is_enabled(TASK_PLAN_READY, active_runs))

    def test_runtime_task_guard_disables_fishing_when_cafe_task_active(self):
        active_runs = {
            "cid-1": {
                "cid": "cid-1",
                "task_name": TASK_CAFE_AUTO_LOOP,
                "status": "running",
            }
        }

        self.assertFalse(task_is_enabled(TASK_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_CAFE_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_ONE_CAFE_REVENUE_RESTOCK, active_runs))
        self.assertFalse(task_is_enabled(TASK_MAHJONG_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_COMBAT_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_TETROMINOES_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_RHYTHM_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_PIANO_PLAY_MIDI, active_runs))
        self.assertTrue(task_is_enabled(TASK_PLAN_READY, active_runs))

    def test_runtime_task_guard_disables_other_runtime_tasks_when_one_cafe_active(self):
        active_runs = {
            "cid-1": {
                "cid": "cid-1",
                "task_name": TASK_ONE_CAFE_REVENUE_RESTOCK,
                "status": "running",
            }
        }

        self.assertFalse(task_is_enabled(TASK_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_CAFE_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_ONE_CAFE_REVENUE_RESTOCK, active_runs))
        self.assertFalse(task_is_enabled(TASK_MAHJONG_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_COMBAT_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_TETROMINOES_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_RHYTHM_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_PIANO_PLAY_MIDI, active_runs))
        self.assertTrue(task_is_enabled(TASK_PLAN_READY, active_runs))

    def test_runtime_task_guard_disables_other_runtime_tasks_when_mahjong_active(self):
        active_runs = {
            "cid-1": {
                "cid": "cid-1",
                "task_name": TASK_MAHJONG_AUTO_LOOP,
                "status": "running",
            }
        }

        self.assertFalse(task_is_enabled(TASK_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_CAFE_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_ONE_CAFE_REVENUE_RESTOCK, active_runs))
        self.assertFalse(task_is_enabled(TASK_MAHJONG_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_COMBAT_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_TETROMINOES_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_RHYTHM_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_PIANO_PLAY_MIDI, active_runs))
        self.assertTrue(task_is_enabled(TASK_PLAN_READY, active_runs))

    def test_runtime_task_guard_disables_other_runtime_tasks_when_combat_active(self):
        active_runs = {
            "cid-1": {
                "cid": "cid-1",
                "task_name": TASK_COMBAT_AUTO_LOOP,
                "status": "running",
            }
        }

        self.assertFalse(task_is_enabled(TASK_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_CAFE_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_ONE_CAFE_REVENUE_RESTOCK, active_runs))
        self.assertFalse(task_is_enabled(TASK_MAHJONG_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_COMBAT_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_TETROMINOES_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_RHYTHM_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_PIANO_PLAY_MIDI, active_runs))
        self.assertTrue(task_is_enabled(TASK_PLAN_READY, active_runs))

    def test_runtime_task_guard_disables_other_runtime_tasks_when_tetrominoes_active(self):
        active_runs = {
            "cid-1": {
                "cid": "cid-1",
                "task_name": TASK_TETROMINOES_AUTO_LOOP,
                "status": "running",
            }
        }

        self.assertFalse(task_is_enabled(TASK_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_CAFE_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_ONE_CAFE_REVENUE_RESTOCK, active_runs))
        self.assertFalse(task_is_enabled(TASK_MAHJONG_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_COMBAT_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_TETROMINOES_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_RHYTHM_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_PIANO_PLAY_MIDI, active_runs))
        self.assertTrue(task_is_enabled(TASK_PLAN_READY, active_runs))

    def test_runtime_task_guard_disables_other_runtime_tasks_when_rhythm_active(self):
        active_runs = {
            "cid-1": {
                "cid": "cid-1",
                "task_name": TASK_RHYTHM_AUTO_LOOP,
                "status": "running",
            }
        }

        self.assertFalse(task_is_enabled(TASK_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_CAFE_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_ONE_CAFE_REVENUE_RESTOCK, active_runs))
        self.assertFalse(task_is_enabled(TASK_MAHJONG_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_COMBAT_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_TETROMINOES_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_RHYTHM_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_PIANO_PLAY_MIDI, active_runs))
        self.assertTrue(task_is_enabled(TASK_PLAN_READY, active_runs))

    def test_runtime_task_guard_disables_other_runtime_tasks_when_piano_active(self):
        active_runs = {
            "cid-1": {
                "cid": "cid-1",
                "task_name": TASK_PIANO_PLAY_MIDI,
                "status": "running",
            }
        }

        self.assertFalse(task_is_enabled(TASK_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_CAFE_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_ONE_CAFE_REVENUE_RESTOCK, active_runs))
        self.assertFalse(task_is_enabled(TASK_MAHJONG_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_COMBAT_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_TETROMINOES_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_RHYTHM_AUTO_LOOP, active_runs))
        self.assertFalse(task_is_enabled(TASK_PIANO_PLAY_MIDI, active_runs))
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
        self.assertEqual(sections[1].fields[-2].key, "gui.task_start_delay_sec")
        self.assertEqual(sections[1].fields[-1].key, "gui.quick_stop_hotkey")

    def test_visible_history_tasks_hide_non_workbench_gameplay_tasks(self):
        self.assertIn(TASK_AUTO_LOOP, VISIBLE_HISTORY_TASK_REFS)
        self.assertIn(TASK_CAFE_AUTO_LOOP, VISIBLE_HISTORY_TASK_REFS)
        self.assertIn(TASK_ONE_CAFE_REVENUE_RESTOCK, VISIBLE_HISTORY_TASK_REFS)
        self.assertIn(TASK_MAHJONG_AUTO_LOOP, VISIBLE_HISTORY_TASK_REFS)
        self.assertIn(TASK_COMBAT_AUTO_LOOP, VISIBLE_HISTORY_TASK_REFS)
        self.assertIn(TASK_TETROMINOES_AUTO_LOOP, VISIBLE_HISTORY_TASK_REFS)
        self.assertIn(TASK_RHYTHM_AUTO_LOOP, VISIBLE_HISTORY_TASK_REFS)
        self.assertIn(TASK_PIANO_PLAY_MIDI, VISIBLE_HISTORY_TASK_REFS)
        self.assertNotIn(TASK_LIVE_MONITOR, VISIBLE_HISTORY_TASK_REFS)


if __name__ == "__main__":
    unittest.main()

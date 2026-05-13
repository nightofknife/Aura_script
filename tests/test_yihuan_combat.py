from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy

import cv2
import numpy as np

from plans.aura_base.src.platform.contracts import CaptureResult, TargetRuntimeError
from plans.yihuan.src.actions import combat_actions
from plans.yihuan.src.services.combat_service import YihuanCombatService


class _FakeApp:
    def __init__(
        self,
        *,
        fail_capture: bool = False,
        fail_focus_once: bool = False,
        fail_focus_always: bool = False,
    ) -> None:
        self.fail_capture = fail_capture
        self.fail_focus_once = fail_focus_once
        self.fail_focus_always = fail_focus_always
        self.calls: list[tuple] = []
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.focus_attempts = 0
        self._focus_failed_once_consumed = False

    def capture(self):
        if self.fail_capture:
            return CaptureResult(success=False, image=None)
        return CaptureResult(success=True, image=self.image.copy())

    def click(self, x=None, y=None, button="left", clicks=1, interval=None):
        self._maybe_raise_focus_error()
        self.calls.append(("click", button, x, y, clicks))

    def press_key(self, key, presses=1, interval=0.1):
        self._maybe_raise_focus_error()
        self.calls.append(("press_key", str(key), int(presses)))

    def mouse_down(self, button="left"):
        self._maybe_raise_focus_error()
        self.calls.append(("mouse_down", str(button)))

    def mouse_up(self, button="left"):
        self._maybe_raise_focus_error()
        self.calls.append(("mouse_up", str(button)))

    def look_delta(self, dx, dy):
        self._maybe_raise_focus_error()
        self.calls.append(("look_delta", int(dx), int(dy)))

    def key_down(self, key):
        self._maybe_raise_focus_error()
        self.calls.append(("key_down", str(key)))

    def key_up(self, key):
        self._maybe_raise_focus_error()
        self.calls.append(("key_up", str(key)))

    def focus_with_input(self, click_delay=0.15):
        self.focus_attempts += 1
        self.calls.append(("focus_with_input", round(float(click_delay), 2)))
        return not self.fail_focus_always

    def release_all(self):
        self.calls.append(("release_all",))

    def _maybe_raise_focus_error(self) -> None:
        if self.fail_focus_always:
            raise TargetRuntimeError(
                "window_focus_required",
                "focus_before_input=true but the target window could not be focused.",
                {},
            )
        if self.fail_focus_once and not self._focus_failed_once_consumed:
            self._focus_failed_once_consumed = True
            raise TargetRuntimeError(
                "window_focus_required",
                "focus_before_input=true but the target window could not be focused.",
                {},
            )


class _FakeCombatService:
    def __init__(self, states: list[dict], *, profile_overrides: dict | None = None, clock=None) -> None:
        self._real_service = YihuanCombatService()
        self.profile = self._real_service.load_profile()
        self.profile = _deep_merge_dict(self.profile, {"post_combat_reward": {"enabled": False}})
        if profile_overrides:
            self.profile = _deep_merge_dict(self.profile, profile_overrides)
        self.states = list(states)
        self.last_state = self.states[-1] if self.states else _state(enemy=False)
        self.clock = clock
        self.scan_times: list[float] = []

    def load_profile(self, profile_name=None):
        return dict(self.profile)

    def analyze_frame(self, source_image, *, profile_name=None):
        if self.clock is not None:
            self.scan_times.append(round(float(self.clock.monotonic()), 3))
        if self.states:
            self.last_state = self.states.pop(0)
        return dict(self.last_state)

    def annotate_frame(self, source_image, *, profile_name=None, state=None, overlay=None):
        return self._real_service.annotate_frame(source_image, profile_name=profile_name, state=state, overlay=overlay)


class _FakeInputMapping:
    def execute_binding(self, binding, *, phase: str, app):
        binding_type = str(binding.get("type") or "").strip().lower()
        if binding_type == "key":
            key = str(binding.get("key") or "")
            if phase in {"press", "tap"}:
                app.press_key(key, presses=1)
            elif phase == "hold":
                app.key_down(key)
            elif phase == "release":
                app.key_up(key)
            return {"ok": True, "phase": phase, "binding": dict(binding)}
        if binding_type == "mouse_button":
            button = str(binding.get("button") or "left")
            if phase in {"press", "tap"}:
                app.click(button=button)
            elif phase == "hold":
                app.mouse_down(button=button)
            elif phase == "release":
                app.mouse_up(button=button)
            return {"ok": True, "phase": phase, "binding": dict(binding)}
        raise ValueError(f"Unsupported fake binding type: {binding_type}")


class _FakeAudioDodgeRuntime:
    def __init__(self, triggers: list[dict] | None = None) -> None:
        self.triggers = list(triggers or [])
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def consume_trigger(self) -> dict | None:
        if self.triggers:
            return dict(self.triggers.pop(0))
        return None


class _FakeClock:
    def __init__(self, *, start: float = 0.0) -> None:
        self.now = float(start)

    def monotonic(self) -> float:
        return float(self.now)

    def sleep(self, seconds: float) -> None:
        self.now += max(float(seconds), 0.0)


class TestYihuanCombatService(unittest.TestCase):
    def setUp(self) -> None:
        self.service = YihuanCombatService()
        self.profile = self.service.load_profile()

    def test_analyze_frame_detects_target_enemy_health_and_abilities(self):
        image = _make_supported_image(self.profile)
        _draw_remaining_enemy_marker(image, self.profile)
        _paste_template(image, self.profile["templates"]["target_lock"]["path"], x=625, y=250)
        enemy_red = _rgb_from_cv2_hsv(0, 222, 242)
        _fill(image, (330, 220, 70, 6), enemy_red)
        _fill(image, (640, 245, 54, 5), enemy_red)
        _fill(image, tuple(self.profile["ability_regions"]["skill"]), (220, 180, 35))
        _fill(image, tuple(self.profile["ability_regions"]["ultimate"]), (220, 180, 35))

        state = self.service.analyze_frame(image)

        self.assertTrue(state["in_supported_scene"])
        self.assertTrue(state["in_combat"])
        self.assertTrue(state["remaining_enemy_marker_found"])
        self.assertTrue(state["front_enemy_found"])
        self.assertTrue(state["target_found"])
        self.assertGreaterEqual(state["enemy_health_count"], 2)
        self.assertTrue(state["enemy_health_found"])
        self.assertTrue(state["skill_available"])
        self.assertEqual(state["skill_state"], "ready")
        self.assertTrue(state["ultimate_available"])
        self.assertEqual(state["ultimate_state"], "ready")
        self.assertEqual(state["current_slot"], 1)
        self.assertFalse(state["challenge_success_found"])

    def test_analyze_frame_classifies_cooldown_and_disabled_ability_states(self):
        image = _make_supported_image(self.profile)
        _fill(image, tuple(self.profile["ability_regions"]["skill"]), (36, 36, 36))
        _fill(image, tuple(self.profile["ability_regions"]["ultimate"]), (18, 18, 18))
        _fill(_region_view(image, tuple(self.profile["ability_regions"]["skill"])), (26, 26, 54, 54), (240, 240, 240))

        state = self.service.analyze_frame(image)

        self.assertFalse(state["skill_available"])
        self.assertEqual(state["skill_state"], "cooldown")
        self.assertFalse(state["ultimate_available"])
        self.assertEqual(state["ultimate_state"], "disabled_or_unready")

    def test_analyze_frame_detects_challenge_success_without_enemy_health(self):
        image = _make_supported_image(self.profile)
        _paste_template(image, self.profile["templates"]["challenge_success"]["path"], x=560, y=150)

        state = self.service.analyze_frame(image)

        self.assertTrue(state["in_supported_scene"])
        self.assertFalse(state["in_combat"])
        self.assertTrue(state["challenge_success_found"])
        self.assertFalse(state["enemy_health_found"])

    def test_analyze_frame_uses_remaining_enemy_marker_as_combat_gate(self):
        image = _make_supported_image(self.profile)
        _draw_remaining_enemy_marker(image, self.profile)

        state = self.service.analyze_frame(image)

        self.assertTrue(state["in_combat"])
        self.assertTrue(state["remaining_enemy_marker_found"])
        self.assertFalse(state["front_enemy_found"])
        self.assertFalse(state["enemy_health_found"])

    def test_analyze_frame_detects_post_combat_reward_marker_and_fixed_prompt_roi(self):
        image = np.full((720, 1280, 3), (80, 150, 70), dtype=np.uint8)
        reward_marker_path = self.profile["templates"]["reward_marker"]["path"]
        _paste_template(image, reward_marker_path, x=500, y=340)
        _paste_template(image, self.profile["templates"]["claim_memento_prompt"]["path"], x=777, y=384)

        state = self.service.analyze_frame(image)

        self.assertTrue(state["reward_marker_found"])
        self.assertEqual(state["reward_marker_center_x"], 500 + _template_size(reward_marker_path)[0] // 2)
        self.assertTrue(state["claim_memento_prompt_found"])

        outside_roi = np.full((720, 1280, 3), (80, 150, 70), dtype=np.uint8)
        _paste_template(outside_roi, self.profile["templates"]["claim_memento_prompt"]["path"], x=700, y=384)

        outside_state = self.service.analyze_frame(outside_roi)

        self.assertFalse(outside_state["claim_memento_prompt_found"])

    def test_analyze_frame_enemy_health_without_remaining_marker_is_front_enemy_only(self):
        image = _make_supported_image(self.profile)
        enemy_red = _rgb_from_cv2_hsv(0, 222, 242)
        _fill(image, (330, 220, 70, 6), enemy_red)

        state = self.service.analyze_frame(image)

        self.assertFalse(state["in_combat"])
        self.assertFalse(state["remaining_enemy_marker_found"])
        self.assertTrue(state["front_enemy_found"])
        self.assertTrue(state["enemy_health_found"])

    def test_analyze_frame_does_not_mark_plain_scene_as_combat(self):
        state = self.service.analyze_frame(np.zeros((720, 1280, 3), dtype=np.uint8))

        self.assertFalse(state["in_supported_scene"])
        self.assertFalse(state["in_combat"])
        self.assertIsNone(state["current_slot"])

    def test_analyze_frame_uses_expanded_enemy_health_region(self):
        image = _make_supported_image(self.profile)
        enemy_red = _rgb_from_cv2_hsv(0, 222, 242)
        _fill(image, (182, 198, 72, 6), enemy_red)

        state = self.service.analyze_frame(image)

        self.assertTrue(state["enemy_health_found"])
        self.assertGreaterEqual(state["enemy_health_count"], 1)

    def test_analyze_frame_accepts_short_enemy_health_bar_segments(self):
        image = _make_supported_image(self.profile)
        enemy_red = _rgb_from_cv2_hsv(0, 222, 242)
        _fill(image, (470, 182, 12, 3), enemy_red)

        state = self.service.analyze_frame(image)

        self.assertTrue(state["enemy_health_found"])
        self.assertGreaterEqual(state["enemy_health_count"], 1)

    def test_analyze_frame_detects_enemy_direction_markers(self):
        image = _make_supported_image(self.profile)
        marker_color = _rgb_from_cv2_hsv(176, 70, 242)
        _draw_enemy_direction_marker(image, center=(154, 392), radius=11, color=marker_color)
        _draw_enemy_direction_marker(image, center=(700, 662), radius=11, color=marker_color)

        state = self.service.analyze_frame(image)

        self.assertTrue(state["enemy_direction_found"])
        self.assertGreaterEqual(state["enemy_direction_count"], 2)
        self.assertEqual(state["enemy_direction_primary_side"], "bottom")
        self.assertGreaterEqual(len(state["enemy_direction_markers"]), 2)

    def test_analyze_frame_does_not_report_target_without_front_combat_signal(self):
        image = _make_supported_image(self.profile)
        _paste_template(image, self.profile["templates"]["target_lock"]["path"], x=625, y=250)

        state = self.service.analyze_frame(image)

        self.assertFalse(state["enemy_health_found"])
        self.assertFalse(state["boss_found"])
        self.assertFalse(state["target_found"])

    def test_match_template_in_region_ignores_black_masked_pixels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            template_path = Path(tmp_dir) / "masked_template.png"
            template = np.zeros((12, 12, 3), dtype=np.uint8)
            template[4:8, 4:8] = (240, 230, 70)
            cv2.imwrite(str(template_path), cv2.cvtColor(template, cv2.COLOR_RGB2BGR))

            source_image = np.full((40, 40, 3), 120, dtype=np.uint8)
            source_image[10:22, 10:22] = 255
            source_image[14:18, 14:18] = (240, 230, 70)

            match = self.service._match_template_in_region(
                source_image,
                {
                    "path": str(template_path.resolve()),
                    "search_region": (0, 0, 40, 40),
                    "match_threshold": 0.6,
                    "scales": [1.0],
                },
                profile={"client_size": (40, 40)},
            )

            self.assertTrue(match["found"])
            self.assertGreaterEqual(match["score"], 0.6)


class TestYihuanCombatActions(unittest.TestCase):
    def _run_with_clock(self, app: _FakeApp, service: _FakeCombatService, **kwargs):
        clock = service.clock or _FakeClock()
        input_mapping = _FakeInputMapping()
        with patch("plans.yihuan.src.actions.combat_actions.time.monotonic", side_effect=clock.monotonic), patch(
            "plans.yihuan.src.actions.combat_actions.time.sleep",
            side_effect=clock.sleep,
        ):
            result = combat_actions.yihuan_combat_run_session(app, input_mapping, service, **kwargs)
        return result, clock

    def test_run_session_scans_on_2s_and_3s_intervals_before_and_during_combat(self):
        clock = _FakeClock()
        no_combat = _state(enemy=False, target=False, skill=False, ultimate=False)
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService([no_combat, combat_state, combat_state, no_combat], clock=clock)
        app = _FakeApp()

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["encounters_completed"], 1)
        self.assertEqual(len(service.scan_times[:5]), 5)
        self.assertAlmostEqual(service.scan_times[0], 0.0, places=3)
        self.assertAlmostEqual(service.scan_times[1], 2.0, places=3)
        self.assertAlmostEqual(service.scan_times[2], 5.0, delta=0.3)
        self.assertAlmostEqual(service.scan_times[3], 8.0, delta=0.3)
        self.assertAlmostEqual(service.scan_times[4], 9.0, delta=0.3)
        combat_actions_only = [
            entry for entry in result["action_trace"] if entry["action"] in {"ultimate", "skill", "normal", "auto_target"}
        ]
        self.assertTrue(combat_actions_only)
        self.assertGreaterEqual(combat_actions_only[0]["t"], 2.0)
        self.assertIn("exit_pending_start", [entry["action"] for entry in result["action_trace"]])
        self.assertIn("exit_combat_confirmed", [entry["action"] for entry in result["action_trace"]])

    def test_run_session_normal_attack_uses_high_frequency_click_spam(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService([combat_state, _state(enemy=False, target=False)], clock=clock)
        app = _FakeApp()

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False)

        self.assertEqual(result["status"], "success")
        left_downs = [call for call in app.calls if call == ("mouse_down", "left")]
        left_ups = [call for call in app.calls if call == ("mouse_up", "left")]
        self.assertGreaterEqual(len(left_downs), 4)
        self.assertEqual(len(left_downs), len(left_ups))
        normal_entries = [entry for entry in result["action_trace"] if entry["action"] == "normal"]
        self.assertTrue(normal_entries)
        self.assertGreaterEqual(normal_entries[0].get("tap_count", 0), 4)
        self.assertIsNone(result["screenshot_dir"])
        self.assertEqual(result["screenshots"], [])

    def test_run_session_uses_global_skill_timers_without_visual_ready(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService([combat_state, combat_state, combat_state, combat_state, _state(enemy=False)], clock=clock)
        app = _FakeApp()

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False)

        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(len([call for call in app.calls if call == ("key_down", "e")]), 3)
        self.assertGreaterEqual(len([call for call in app.calls if call == ("key_up", "e")]), 3)
        self.assertGreaterEqual(len([call for call in app.calls if call == ("key_down", "q")]), 2)
        self.assertGreaterEqual(len([call for call in app.calls if call == ("key_up", "q")]), 2)

    def test_run_session_switches_on_fixed_rotation_interval(self):
        clock = _FakeClock()
        slot_one = _state(enemy=True, target=True, skill=False, ultimate=False, current_slot=1)
        slot_two = _state(enemy=True, target=True, skill=False, ultimate=False, current_slot=2)
        service = _FakeCombatService(
            [slot_one, slot_one, slot_one, slot_one, slot_two, slot_two, _state(enemy=False, current_slot=2)],
            clock=clock,
        )
        app = _FakeApp()

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False)

        self.assertEqual(result["status"], "success")
        self.assertIn(("key_down", "2"), app.calls)
        self.assertIn(("key_up", "2"), app.calls)
        switch_entries = [entry for entry in result["action_trace"] if entry["action"] == "switch"]
        self.assertTrue(switch_entries)
        self.assertTrue(switch_entries[-1].get("confirmed"))
        self.assertGreaterEqual(switch_entries[-1]["t"], 10.0)

    def test_run_session_auto_targets_before_skills_when_target_missing(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=False, skill=False, ultimate=False)
        service = _FakeCombatService([combat_state, _state(enemy=False, target=False)], clock=clock)
        app = _FakeApp()

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False, auto_target=True)

        self.assertEqual(result["status"], "success")
        self.assertIn(("click", "middle", None, None, 1), app.calls)
        actions = [entry["action"] for entry in result["action_trace"]]
        self.assertLess(actions.index("auto_target"), actions.index("ultimate"))
        self.assertIn("skill", actions)

    def test_run_session_waits_for_configured_exit_confirmation_scans(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService(
            [combat_state, combat_state, no_combat := _state(enemy=False, target=False), no_combat],
            clock=clock,
            profile_overrides={"combat_exit": {"confirm_required_scans": 2, "confirm_interval_sec": 0.5}},
        )
        app = _FakeApp()

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False)

        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(len(service.scan_times), 4)
        self.assertAlmostEqual(service.scan_times[0], 0.0, places=3)
        self.assertAlmostEqual(service.scan_times[1], 3.0, delta=0.3)
        self.assertAlmostEqual(service.scan_times[2], 6.0, delta=0.3)
        self.assertAlmostEqual(service.scan_times[3], 6.5, delta=0.3)
        actions = [entry["action"] for entry in result["action_trace"]]
        self.assertIn("exit_pending_start", actions)
        self.assertIn("exit_pending_scan", actions)
        self.assertIn("exit_combat_confirmed", actions)

    def test_run_session_holds_combat_and_turns_when_only_rear_enemy_marker_remains(self):
        clock = _FakeClock()
        rear_only_state = _state(
            enemy=False,
            stage_active=True,
            target=False,
            current_slot=1,
            enemy_direction_found=True,
            enemy_direction_count=2,
            enemy_direction_primary_side="left",
            enemy_direction_markers=[{"x": 140, "y": 390, "width": 18, "height": 18, "region": "left"}],
        )
        service = _FakeCombatService(
            [
                _state(enemy=True, target=True, current_slot=1),
                rear_only_state,
                _state(enemy=False, target=False, current_slot=1),
                _state(enemy=False, target=False, current_slot=1),
            ],
            clock=clock,
        )
        app = _FakeApp()

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False, auto_target=True)

        self.assertEqual(result["status"], "success")
        actions = [entry["action"] for entry in result["action_trace"]]
        self.assertIn("turn_to_rear_enemy", actions)
        self.assertIn("auto_target_after_turn", actions)
        self.assertLess(actions.index("turn_to_rear_enemy"), actions.index("exit_pending_start"))
        look_calls = [call for call in app.calls if call[0] == "look_delta"]
        expected_dx = sum(entry.get("turn_dx", 0) for entry in result["action_trace"] if entry["action"] == "turn_to_rear_enemy")
        self.assertGreater(len(look_calls), 1)
        self.assertEqual(sum(call[1] for call in look_calls), expected_dx)
        self.assertEqual(sum(call[2] for call in look_calls), 0)
        self.assertTrue(all(abs(call[1]) <= 80 for call in look_calls))

    def test_collect_post_combat_reward_presses_interact_after_prompt(self):
        clock = _FakeClock()
        service = _FakeCombatService(
            [
                _state(enemy=False, stage_active=False, reward_marker=True, reward_marker_center_x=640),
                _state(
                    enemy=False,
                    stage_active=False,
                    reward_marker=True,
                    reward_marker_center_x=640,
                    claim_prompt=True,
                ),
            ],
            clock=clock,
            profile_overrides={"post_combat_reward": {"enabled": True, "interact_key": "f"}},
        )
        app = _FakeApp()
        action_trace: list[dict] = []
        combat_state_trace: list[dict] = []

        with patch("plans.yihuan.src.actions.combat_actions.time.monotonic", side_effect=clock.monotonic), patch(
            "plans.yihuan.src.actions.combat_actions.time.sleep",
            side_effect=clock.sleep,
        ):
            result = combat_actions._collect_post_combat_reward(
                app,
                service,
                service.profile,
                profile_name="default_1280x720_cn",
                dry_run=False,
                action_trace=action_trace,
                combat_state_trace=combat_state_trace,
                capture_debug=None,
                start_time=0.0,
                trace_limit=50,
            )

        self.assertEqual(result["status"], "success")
        self.assertIn(("key_down", "w"), app.calls)
        self.assertIn(("key_up", "w"), app.calls)
        self.assertIn(("press_key", "f", 1), app.calls)
        actions = [entry["action"] for entry in action_trace]
        self.assertIn("reward_move_forward_start", actions)
        self.assertIn("reward_claim_prompt_found", actions)
        self.assertIn("reward_interact", actions)

    def test_collect_post_combat_reward_aligns_marker_to_configured_x_only(self):
        clock = _FakeClock()
        service = _FakeCombatService(
            [
                _state(enemy=False, stage_active=False, reward_marker=True, reward_marker_center_x=700),
                _state(enemy=False, stage_active=False, reward_marker=True, reward_marker_center_x=638),
                _state(
                    enemy=False,
                    stage_active=False,
                    reward_marker=True,
                    reward_marker_center_x=638,
                    claim_prompt=True,
                ),
            ],
            clock=clock,
            profile_overrides={
                "post_combat_reward": {
                    "enabled": True,
                    "align_target_x": 640,
                    "align_tolerance_px": 20,
                    "look_pixels_per_px": 1.0,
                    "min_turn_pixels": 1,
                    "max_turn_pixels": 200,
                }
            },
        )
        app = _FakeApp()
        action_trace: list[dict] = []
        combat_state_trace: list[dict] = []

        with patch("plans.yihuan.src.actions.combat_actions.time.monotonic", side_effect=clock.monotonic), patch(
            "plans.yihuan.src.actions.combat_actions.time.sleep",
            side_effect=clock.sleep,
        ):
            result = combat_actions._collect_post_combat_reward(
                app,
                service,
                service.profile,
                profile_name="default_1280x720_cn",
                dry_run=False,
                action_trace=action_trace,
                combat_state_trace=combat_state_trace,
                capture_debug=None,
                start_time=0.0,
                trace_limit=50,
            )

        self.assertEqual(result["status"], "success")
        align_entries = [entry for entry in action_trace if entry["action"] == "reward_align_marker"]
        self.assertEqual(len(align_entries), 1)
        self.assertEqual(align_entries[0]["marker_center_x"], 700)
        self.assertEqual(align_entries[0]["target_x"], 640.0)
        self.assertEqual(align_entries[0]["error_x"], 60.0)
        self.assertNotIn("screen_center_x", align_entries[0])
        self.assertEqual(sum(call[1] for call in app.calls if call[0] == "look_delta"), 40)
        self.assertIn(("key_down", "w"), app.calls)
        self.assertIn(("press_key", "f", 1), app.calls)

    def test_reward_turn_pixels_preserves_direction_and_deadzone(self):
        self.assertEqual(
            combat_actions._reward_turn_pixels_for_error(
                -296,
                align_tolerance_px=50,
                look_pixels_per_px=0.55,
                min_turn_pixels=8,
                max_turn_pixels=160,
            ),
            -135,
        )
        self.assertEqual(
            combat_actions._reward_turn_pixels_for_error(
                306,
                align_tolerance_px=50,
                look_pixels_per_px=0.55,
                min_turn_pixels=8,
                max_turn_pixels=160,
            ),
            141,
        )
        self.assertEqual(
            combat_actions._reward_turn_pixels_for_error(
                42,
                align_tolerance_px=50,
                look_pixels_per_px=0.55,
                min_turn_pixels=8,
                max_turn_pixels=160,
            ),
            0,
        )

    def test_run_session_audio_dodge_has_highest_priority(self):
        clock = _FakeClock()
        service = _FakeCombatService([_state(enemy=True, target=True), _state(enemy=False, target=False)], clock=clock)
        app = _FakeApp()
        fake_audio_runtime = _FakeAudioDodgeRuntime(triggers=[{"t": 0.0, "score": 0.31}])

        with patch("plans.yihuan.src.actions.combat_actions.time.monotonic", side_effect=clock.monotonic), patch(
            "plans.yihuan.src.actions.combat_actions.time.sleep",
            side_effect=clock.sleep,
        ), patch(
            "plans.yihuan.src.actions.combat_actions.AudioDodgeRuntime.from_profile",
            return_value=fake_audio_runtime,
        ):
            result = combat_actions.yihuan_combat_run_session(app, _FakeInputMapping(), service, max_encounters=1, auto_dodge=True)

        self.assertEqual(result["status"], "success")
        actions = [entry["action"] for entry in result["action_trace"]]
        self.assertIn("audio_dodge", actions)
        self.assertLess(actions.index("audio_dodge"), actions.index("ultimate"))

    def test_run_session_cancels_after_current_action_chunk(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService([combat_state, combat_state, combat_state], clock=clock)
        app = _FakeApp()

        with patch("plans.yihuan.src.actions.combat_actions.time.monotonic", side_effect=clock.monotonic), patch(
            "plans.yihuan.src.actions.combat_actions.time.sleep",
            side_effect=clock.sleep,
        ), patch(
            "plans.yihuan.src.actions.combat_actions.is_current_task_cancel_requested",
            side_effect=lambda: clock.monotonic() >= 0.35,
        ):
            result = combat_actions.yihuan_combat_run_session(app, _FakeInputMapping(), service, auto_dodge=False)

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["stopped_reason"], "cancelled")
        self.assertIn(("release_all",), app.calls)
        normal_entries = [entry for entry in result["action_trace"] if entry["action"] == "normal"]
        self.assertTrue(normal_entries)
        self.assertGreaterEqual(normal_entries[0].get("tap_count", 0), 4)

    def test_run_session_retries_input_after_focus_activation(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService([combat_state, _state(enemy=False, target=False)], clock=clock)
        app = _FakeApp(fail_focus_once=True)

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False)

        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(app.focus_attempts, 1)
        self.assertIn("focus_with_input", [entry["action"] for entry in result["action_trace"]])

    def test_run_session_returns_window_focus_failed_when_recovery_fails(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService([combat_state], clock=clock)
        app = _FakeApp(fail_focus_always=True)

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "window_focus_failed")

    def test_run_session_challenge_success_shortcuts_post_combat(self):
        clock = _FakeClock()
        service = _FakeCombatService(
            [_state(enemy=True, target=True), _state(enemy=True, target=True, challenge_success=True)],
            clock=clock,
        )
        app = _FakeApp()

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "max_encounters")
        self.assertIn("challenge_success", [entry["action"] for entry in result["action_trace"]])

    def test_run_session_dry_run_records_actions_without_input(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService([combat_state, _state(enemy=False)], clock=clock)
        app = _FakeApp()

        result, _clock = self._run_with_clock(app, service, max_encounters=1, auto_dodge=False, dry_run=True, max_seconds=6)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "max_encounters")
        self.assertEqual(app.calls, [])
        self.assertIn("ultimate", [entry["action"] for entry in result["action_trace"]])
        self.assertIn("skill", [entry["action"] for entry in result["action_trace"]])
        self.assertIn("normal", [entry["action"] for entry in result["action_trace"]])

    def test_run_session_capture_debug_creates_output_and_event_screenshots(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService([combat_state, _state(enemy=False), _state(enemy=False)], clock=clock)
        app = _FakeApp()

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "plans.yihuan.src.actions.combat_actions._combat_capture_root",
            return_value=Path(tmp_dir),
        ):
            result, _clock = self._run_with_clock(
                app,
                service,
                max_encounters=1,
                auto_dodge=False,
                capture_debug_enabled=True,
                capture_interval_sec=0.5,
                capture_max_images=20,
            )
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["screenshot_dir"])
            screenshot_dir = Path(result["screenshot_dir"])
            self.assertTrue(screenshot_dir.is_dir())
            self.assertTrue((screenshot_dir / "index.json").is_file())
            labels = [entry["label"] for entry in result["screenshots"]]
            self.assertIn("initial_monitor", labels)
            self.assertIn("enter_combat", labels)
            self.assertIn("ultimate_due", labels)
            self.assertIn("skill_due", labels)
            self.assertIn("exit_pending_start", labels)
            self.assertIn("exit_combat_confirmed", labels)
            self.assertGreater(result["capture_stats"]["event_count"], 0)
            self.assertGreater(result["capture_stats"]["periodic_count"], 0)
            for entry in result["screenshots"]:
                self.assertTrue(Path(entry["image_path"]).is_file())
                self.assertIsNone(entry["raw_image_path"])

    def test_run_session_capture_debug_respects_max_images(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService([combat_state, combat_state, _state(enemy=False), _state(enemy=False)], clock=clock)
        app = _FakeApp()

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "plans.yihuan.src.actions.combat_actions._combat_capture_root",
            return_value=Path(tmp_dir),
        ):
            result, _clock = self._run_with_clock(
                app,
                service,
                max_encounters=1,
                auto_dodge=False,
                capture_debug_enabled=True,
                capture_interval_sec=0.25,
                capture_max_images=2,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["screenshots"]), 2)
        self.assertGreater(result["capture_stats"]["skipped_max_images_count"], 0)

    def test_run_session_capture_debug_can_save_raw_and_cancelled_frame(self):
        clock = _FakeClock()
        combat_state = _state(enemy=True, target=True, skill=False, ultimate=False)
        service = _FakeCombatService([combat_state, combat_state, combat_state], clock=clock)
        app = _FakeApp()

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "plans.yihuan.src.actions.combat_actions._combat_capture_root",
            return_value=Path(tmp_dir),
        ), patch(
            "plans.yihuan.src.actions.combat_actions.is_current_task_cancel_requested",
            side_effect=lambda: clock.monotonic() >= 0.35,
        ), patch(
            "plans.yihuan.src.actions.combat_actions.time.monotonic",
            side_effect=clock.monotonic,
        ), patch(
            "plans.yihuan.src.actions.combat_actions.time.sleep",
            side_effect=clock.sleep,
        ):
            result = combat_actions.yihuan_combat_run_session(
                app,
                _FakeInputMapping(),
                service,
                auto_dodge=False,
                capture_debug_enabled=True,
                capture_raw_enabled=True,
                capture_interval_sec=0.25,
            )
            self.assertEqual(result["status"], "cancelled")
            labels = [entry["label"] for entry in result["screenshots"]]
            self.assertIn("cancelled", labels)
            raw_entries = [entry for entry in result["screenshots"] if entry["raw_image_path"]]
            self.assertTrue(raw_entries)
            for entry in raw_entries:
                self.assertTrue(Path(entry["raw_image_path"]).is_file())

    def test_run_session_returns_not_in_supported_scene(self):
        clock = _FakeClock()
        service = _FakeCombatService([_state(enemy=False, supported=False)] * 3, clock=clock)
        app = _FakeApp()

        result, _clock = self._run_with_clock(app, service, max_seconds=10, auto_dodge=False)

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["stopped_reason"], "not_in_supported_scene")
        self.assertEqual(service.scan_times[:3], [0.0, 2.0, 4.0])

    def test_run_session_capture_failure_returns_business_failure(self):
        result = combat_actions.yihuan_combat_run_session(
            _FakeApp(fail_capture=True),
            _FakeInputMapping(),
            _FakeCombatService([]),
            max_seconds=1,
            auto_dodge=False,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "capture_failed")

    def test_confirm_switch_slot_requires_consistent_matches(self):
        clock = _FakeClock()
        service = _FakeCombatService(
            [
                _state(enemy=True, current_slot=None),
                _state(enemy=True, current_slot=2),
                _state(enemy=True, current_slot=2),
            ],
            clock=clock,
        )
        app = _FakeApp()

        with patch("plans.yihuan.src.actions.combat_actions.time.monotonic", side_effect=clock.monotonic), patch(
            "plans.yihuan.src.actions.combat_actions.time.sleep",
            side_effect=clock.sleep,
        ):
            confirmed, state, _image = combat_actions._confirm_switch_slot(
                app,
                service,
                profile_name="default_1280x720_cn",
                expected_slot=2,
                confirm_timeout_ms=320,
                required_matches=2,
                poll_ms=80,
            )

        self.assertTrue(confirmed)
        self.assertEqual(state["current_slot"], 2)


def _state(
    *,
    enemy: bool,
    stage_active: bool | None = None,
    target: bool = False,
    supported: bool = True,
    current_slot: int | None = 1,
    skill: bool = False,
    ultimate: bool = False,
    skill_state: str | None = None,
    ultimate_state: str | None = None,
    challenge_success: bool = False,
    enemy_direction_found: bool = False,
    enemy_direction_count: int = 0,
    enemy_direction_primary_side: str | None = None,
    enemy_direction_markers: list[dict] | None = None,
    reward_marker: bool = False,
    reward_marker_center_x: int | None = None,
    claim_prompt: bool = False,
) -> dict:
    resolved_skill_state = skill_state or ("ready" if skill else "disabled_or_unready")
    resolved_ultimate_state = ultimate_state or ("ready" if ultimate else "disabled_or_unready")
    resolved_stage_active = bool(enemy) if stage_active is None else bool(stage_active)
    return {
        "profile_name": "default_1280x720_cn",
        "capture_size": [1280, 720],
        "in_supported_scene": bool(supported),
        "in_combat": resolved_stage_active,
        "remaining_enemy_marker_found": resolved_stage_active,
        "remaining_enemy_marker_count": 1 if resolved_stage_active else 0,
        "remaining_enemy_markers": (
            [{"x": 28, "y": 187, "width": 20, "height": 23, "region": "remaining_enemy_marker"}]
            if resolved_stage_active
            else []
        ),
        "front_enemy_found": bool(enemy),
        "target_found": bool(target),
        "target_confidence": 0.84 if target else 0.0,
        "enemy_level_found": False,
        "enemy_health_found": bool(enemy),
        "enemy_health_count": 2 if enemy else 0,
        "enemy_direction_found": bool(enemy_direction_found),
        "enemy_direction_count": int(enemy_direction_count or 0),
        "enemy_direction_primary_side": enemy_direction_primary_side,
        "enemy_direction_markers": list(enemy_direction_markers or []),
        "boss_found": False,
        "team_found": bool(supported),
        "current_slot": current_slot,
        "team_size": 4,
        "skill_available": bool(skill),
        "skill_state": resolved_skill_state,
        "ultimate_available": bool(ultimate),
        "ultimate_state": resolved_ultimate_state,
        "arc_available": False,
        "challenge_success_found": bool(challenge_success),
        "reward_marker_found": bool(reward_marker),
        "reward_marker_center_x": reward_marker_center_x if reward_marker else None,
        "reward_marker_center_y": 360 if reward_marker else None,
        "reward_marker_box": [630, 350, 20, 20] if reward_marker else [],
        "claim_memento_prompt_found": bool(claim_prompt),
        "claim_memento_prompt_box": [777, 384, 20, 18] if claim_prompt else [],
        "confidence": 0.8 if enemy else 0.1,
        "debug": {},
    }


def _make_supported_image(profile: dict) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    _fill(image, tuple(profile["regions"]["team_panel"]), (180, 180, 180))
    _fill(image, tuple(profile["current_slot_regions"][0]), (60, 190, 210))
    return image


def _paste_template(image: np.ndarray, template_path: str, *, x: int, y: int) -> None:
    template = cv2.imread(str(Path(template_path)), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(template_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    height, width = template.shape[:2]
    image[y : y + height, x : x + width] = template


def _template_size(template_path: str) -> tuple[int, int]:
    template = cv2.imread(str(Path(template_path)), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(template_path)
    height, width = template.shape[:2]
    return width, height


def _region_view(image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = region
    return image[y : y + height, x : x + width]


def _fill(image: np.ndarray, region: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, width, height = region
    image[y : y + height, x : x + width] = color


def _draw_enemy_direction_marker(
    image: np.ndarray,
    *,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    cv2.circle(image, center, int(radius), color, thickness=-1, lineType=cv2.LINE_AA)


def _draw_remaining_enemy_marker(image: np.ndarray, profile: dict) -> None:
    x, y, _width, _height = tuple(profile["regions"]["remaining_enemy_marker"])
    marker_color = _rgb_from_cv2_hsv(94, 180, 230)
    points = np.array(
        [
            [x + 8, y + 24],
            [x + 20, y + 12],
            [x + 30, y + 17],
            [x + 20, y + 24],
            [x + 30, y + 31],
            [x + 20, y + 36],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [points], marker_color, lineType=cv2.LINE_AA)


def _rgb_from_cv2_hsv(h: int, s: int, v: int) -> tuple[int, int, int]:
    hsv = np.array([[[int(h), int(s), int(v)]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _deep_merge_dict(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in dict(overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from plans.aura_base.src.platform.contracts import CaptureResult
from plans.yihuan.src.actions import combat_actions
from plans.yihuan.src.services.combat_service import YihuanCombatService


class _FakeApp:
    def __init__(self, *, fail_capture: bool = False) -> None:
        self.fail_capture = fail_capture
        self.calls: list[tuple] = []
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)

    def capture(self):
        if self.fail_capture:
            return CaptureResult(success=False, image=None)
        return CaptureResult(success=True, image=self.image.copy())

    def click(self, x=None, y=None, button="left", clicks=1, interval=None):
        self.calls.append(("click", button, x, y, clicks))

    def press_key(self, key, presses=1, interval=0.1):
        self.calls.append(("press_key", str(key), int(presses)))

    def mouse_down(self, button="left"):
        self.calls.append(("mouse_down", str(button)))

    def mouse_up(self, button="left"):
        self.calls.append(("mouse_up", str(button)))

    def key_down(self, key):
        self.calls.append(("key_down", str(key)))

    def key_up(self, key):
        self.calls.append(("key_up", str(key)))


class _FakeCombatService:
    def __init__(self, states: list[dict]) -> None:
        self.profile = YihuanCombatService().load_profile()
        self.states = list(states)
        self.last_state = self.states[-1] if self.states else _state(in_combat=False)

    def load_profile(self, profile_name=None):
        return dict(self.profile)

    def analyze_frame(self, source_image, *, profile_name=None):
        if self.states:
            self.last_state = self.states.pop(0)
        return dict(self.last_state)


class TestYihuanCombatService(unittest.TestCase):
    def setUp(self) -> None:
        self.service = YihuanCombatService()

    def test_analyze_frame_detects_supported_combat_and_abilities(self):
        image = _make_supported_image()
        _fill(image, (560, 40, 160, 80), (232, 232, 232))
        _fill(image, (430, 65, 420, 45), (220, 30, 30))
        _fill(image, (1015, 585, 58, 58), (220, 180, 35))
        _fill(image, (1090, 575, 70, 70), (220, 180, 35))
        _fill(image, (935, 585, 58, 58), (220, 180, 35))

        state = self.service.analyze_frame(image)

        self.assertTrue(state["in_supported_scene"])
        self.assertTrue(state["in_combat"])
        self.assertTrue(state["target_found"])
        self.assertTrue(state["enemy_health_found"])
        self.assertTrue(state["skill_available"])
        self.assertTrue(state["ultimate_available"])
        self.assertTrue(state["arc_available"])
        self.assertEqual(state["current_slot"], 1)

    def test_analyze_frame_does_not_mark_plain_scene_as_combat(self):
        state = self.service.analyze_frame(np.zeros((720, 1280, 3), dtype=np.uint8))

        self.assertFalse(state["in_supported_scene"])
        self.assertFalse(state["in_combat"])
        self.assertIsNone(state["current_slot"])


class TestYihuanCombatActions(unittest.TestCase):
    def test_run_session_executes_default_strategy_and_stops_after_combat(self):
        app = _FakeApp()
        combat_state = _state(in_combat=True, current_slot=1, skill=True, ultimate=True, arc=True)
        service = _FakeCombatService([combat_state] * 4 + [_state(in_combat=False)] * 8)

        with patch("plans.yihuan.src.actions.combat_actions.time.sleep"):
            result = combat_actions.yihuan_combat_run_session(app, service)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "out_of_combat")
        self.assertEqual(result["encounters_completed"], 1)
        self.assertEqual([entry["action"] for entry in result["action_trace"][:5]], ["arc", "ultimate", "skill", "normal", "switch"])
        self.assertIn(("press_key", "r", 1), app.calls)
        self.assertIn(("press_key", "q", 1), app.calls)
        self.assertIn(("press_key", "e", 1), app.calls)
        self.assertIn(("mouse_down", "left"), app.calls)
        self.assertIn(("mouse_up", "left"), app.calls)
        self.assertIn(("press_key", "2", 1), app.calls)

    def test_run_session_dry_run_records_actions_without_input(self):
        app = _FakeApp()
        combat_state = _state(in_combat=True, current_slot=1, skill=True, ultimate=False, arc=True)
        service = _FakeCombatService([combat_state] * 3 + [_state(in_combat=False)] * 8)

        with patch("plans.yihuan.src.actions.combat_actions.time.sleep"):
            result = combat_actions.yihuan_combat_run_session(app, service, dry_run=True)

        self.assertEqual(result["status"], "success")
        self.assertEqual(app.calls, [])
        self.assertIn("arc", [entry["action"] for entry in result["action_trace"]])
        self.assertIn("skip_ultimate", [entry["action"] for entry in result["action_trace"]])

    def test_run_session_auto_targets_when_target_is_missing(self):
        app = _FakeApp()
        combat_state = _state(in_combat=True, target=False, skill=False, ultimate=False, arc=False)
        service = _FakeCombatService([combat_state] * 3 + [_state(in_combat=False)] * 8)

        with patch("plans.yihuan.src.actions.combat_actions.time.sleep"):
            result = combat_actions.yihuan_combat_run_session(app, service)

        self.assertEqual(result["status"], "success")
        self.assertIn(("click", "middle", None, None, 1), app.calls)
        self.assertIn("auto_target", [entry["action"] for entry in result["action_trace"]])

    def test_run_session_capture_failure_returns_business_failure(self):
        result = combat_actions.yihuan_combat_run_session(_FakeApp(fail_capture=True), _FakeCombatService([]), max_seconds=1)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "capture_failed")


def _state(
    *,
    in_combat: bool,
    target: bool = True,
    current_slot: int | None = 1,
    skill: bool = False,
    ultimate: bool = False,
    arc: bool = False,
) -> dict:
    return {
        "profile_name": "default_1280x720_cn",
        "capture_size": [1280, 720],
        "in_supported_scene": True,
        "in_combat": bool(in_combat),
        "target_found": bool(target),
        "enemy_level_found": bool(in_combat),
        "enemy_health_found": bool(in_combat),
        "boss_found": False,
        "team_found": True,
        "current_slot": current_slot,
        "team_size": 4,
        "skill_available": bool(skill),
        "ultimate_available": bool(ultimate),
        "arc_available": bool(arc),
        "confidence": 0.8 if in_combat else 0.0,
        "debug": {},
    }


def _make_supported_image() -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    _fill(image, (1020, 120, 240, 330), (180, 180, 180))
    _fill(image, (1150, 135, 88, 58), (60, 190, 210))
    return image


def _fill(image: np.ndarray, region: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, width, height = region
    image[y : y + height, x : x + width] = color

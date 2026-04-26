from __future__ import annotations

import itertools
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from plans.aura_base.src.platform.contracts import CaptureResult
from plans.yihuan.src.actions import fishing_actions
from plans.yihuan.src.services.fishing_service import YihuanFishingService


class _FakeApp:
    def __init__(self):
        self.click_calls: list[tuple[int, int, str]] = []
        self.release_all_calls = 0
        self._image = np.zeros((720, 1280, 3), dtype=np.uint8)

    def click(self, x=None, y=None, button="left", clicks=1, interval=None):
        self.click_calls.append((int(x), int(y), str(button)))

    def release_all(self):
        self.release_all_calls += 1

    def capture(self):
        return CaptureResult(success=True, image=self._image.copy())


class _FakeInputMapping:
    def __init__(self):
        self.calls: list[tuple[str, str, str | None]] = []

    def execute_action(self, action_name: str, *, phase: str, app, profile=None):
        self.calls.append((str(action_name), str(phase), None if profile is None else str(profile)))
        return {"ok": True, "action_name": action_name, "phase": phase}


class _FakeVision:
    def find_template(self, *args, **kwargs):
        raise AssertionError("find_template should not be called in patched action tests.")


class TestYihuanFishingDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_root = Path("tests/fixtures/yihuan_fishing/client")
        cls.service = YihuanFishingService()
        cls.profile = cls.service.load_profile()
        cls.images = {
            name: cls._load_fixture(name)
            for name in ["ready", "bite", "duel", "result"]
        }
        cls.result_texts = {
            "ready": [],
            "bite": [],
            "duel": [],
            "result": ["点击空白区域关闭"],
        }

    @classmethod
    def _load_fixture(cls, name: str):
        image = cv2.imread(str(cls.fixture_root / f"{name}.png"))
        if image is None:
            raise FileNotFoundError(f"Missing fixture image: {name}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def test_fixture_phases_are_detected(self):
        expected = {
            "ready": "ready",
            "bite": "bite",
            "duel": "duel",
            "result": "result",
        }
        for name, phase in expected.items():
            with self.subTest(name=name):
                state = self.service.analyze_state(
                    self.images[name],
                    result_texts=self.result_texts[name],
                    profile_name=self.profile["profile_name"],
                )
                self.assertEqual(state["phase"], phase)

    def test_bite_detector_detects_fixture(self):
        bite = self.service.analyze_bite_marker(
            self.images["bite"],
            profile_name=self.profile["profile_name"],
        )
        self.assertTrue(bite["found"])
        self.assertEqual(bite["reason"], "ok")
        self.assertGreaterEqual(bite["pixel_count"], bite["min_pixels"])
        self.assertGreaterEqual(bite["largest_component_area"], bite["min_component_area"])

    def test_bite_detector_rejects_ready_fixture(self):
        bite = self.service.analyze_bite_marker(
            self.images["ready"],
            profile_name=self.profile["profile_name"],
        )
        self.assertFalse(bite["found"])

    def test_duel_detector_extracts_zone_and_indicator(self):
        duel = self.service.analyze_duel_meter(
            self.images["duel"],
            profile_name=self.profile["profile_name"],
        )
        self.assertTrue(duel["found"])
        self.assertIsNotNone(duel["zone_left"])
        self.assertIsNotNone(duel["zone_right"])
        self.assertIsNotNone(duel["indicator_x"])
        self.assertIn(duel["control_advice"], {"hold_a", "hold_d", "tap_a", "tap_d", "none"})

    @patch("plans.yihuan.src.services.fishing_service.time.monotonic", side_effect=[1.0, 1.08])
    def test_duel_detector_uses_short_memory_for_indicator(self, _monotonic):
        self.service._indicator_memory.clear()
        first = self.service.analyze_duel_meter(
            self.images["duel"],
            profile_name=self.profile["profile_name"],
        )
        self.assertTrue(first["found"])
        self.assertIn(first["indicator_source"], {"primary_component", "dual_component", "dual_projection"})

        missing_indicator = self.images["duel"].copy()
        x, y, width, height = self.profile["duel_meter_region"]
        rect_x, _, rect_w, _ = first["indicator_rect"]
        erase_left = max(int(rect_x) - 6, 0)
        erase_right = min(int(rect_x + rect_w + 6), width)
        erase_width = erase_right - erase_left
        source_left = max(erase_left - 24, 0)
        source_right = source_left + erase_width
        missing_indicator[y:y + height, x + erase_left:x + erase_right] = missing_indicator[y:y + height, x + source_left:x + source_right]

        second = self.service.analyze_duel_meter(
            missing_indicator,
            profile_name=self.profile["profile_name"],
        )
        self.assertTrue(second["found"])
        self.assertEqual(second["indicator_source"], "memory")
        self.assertEqual(second["indicator_x"], first["indicator_x"])

    def test_duel_detector_handles_missing_zone(self):
        self.service._indicator_memory.clear()
        blank = self.images["duel"].copy()
        x, y, width, height = self.profile["duel_meter_region"]
        blank[y:y + height, x:x + width] = 0
        duel = self.service.analyze_duel_meter(blank, profile_name=self.profile["profile_name"])
        self.assertFalse(duel["found"])
        self.assertEqual(duel["reason"], "zone_missing")

    def test_duel_detector_handles_missing_indicator(self):
        self.service._indicator_memory.clear()
        baseline = self.service.analyze_duel_meter(
            self.images["duel"],
            profile_name=self.profile["profile_name"],
        )
        modified = self.images["duel"].copy()
        x, y, width, height = self.profile["duel_meter_region"]
        rect_x, _, rect_w, _ = baseline["indicator_rect"]
        erase_left = max(int(rect_x) - 6, 0)
        erase_right = min(int(rect_x + rect_w + 6), width)
        erase_width = erase_right - erase_left
        source_left = max(erase_left - 24, 0)
        source_right = source_left + erase_width
        modified[y:y + height, x + erase_left:x + erase_right] = modified[y:y + height, x + source_left:x + source_right]
        self.service._indicator_memory.clear()
        duel = self.service.analyze_duel_meter(modified, profile_name=self.profile["profile_name"])
        self.assertFalse(duel["found"])
        self.assertEqual(duel["reason"], "indicator_missing")

    def test_extract_zone_bounds_uses_outermost_large_components(self):
        zone_mask = np.zeros((11, 120), dtype=np.uint8)
        zone_mask[0:2, 4:8] = 255
        zone_mask[:, 18:46] = 255
        zone_mask[:, 58:92] = 255

        bounds = self.service._extract_zone_bounds(zone_mask, min_component_area=20)

        self.assertEqual(bounds, (18, 91))

    def test_duel_control_uses_center_deadband(self):
        result = self.service._build_duel_result(
            region=self.profile["duel_meter_region"],
            zone_left_local=100,
            zone_right_local=200,
            indicator_candidate={
                "center_x": 144.0,
                "rect": [143, 1, 2, 8],
            },
            deadband_px=6,
            indicator_source="primary_component",
            raw_indicator_detected=True,
            indicator_memory_age_ms=None,
        )

        self.assertEqual(result["zone_center"], 555)
        self.assertEqual(result["indicator_x"], 549)
        self.assertEqual(result["center_error_px"], -6)
        self.assertEqual(result["control_advice"], "none")

        result = self.service._build_duel_result(
            region=self.profile["duel_meter_region"],
            zone_left_local=100,
            zone_right_local=200,
            indicator_candidate={
                "center_x": 143.0,
                "rect": [142, 1, 2, 8],
            },
            deadband_px=6,
            indicator_source="primary_component",
            raw_indicator_detected=True,
            indicator_memory_age_ms=None,
        )

        self.assertEqual(result["center_error_px"], -7)
        self.assertEqual(result["control_advice"], "tap_d")

        result = self.service._build_duel_result(
            region=self.profile["duel_meter_region"],
            zone_left_local=100,
            zone_right_local=200,
            indicator_candidate={
                "center_x": 94.0,
                "rect": [93, 1, 2, 8],
            },
            deadband_px=6,
            indicator_source="primary_component",
            raw_indicator_detected=True,
            indicator_memory_age_ms=None,
        )

        self.assertEqual(result["indicator_x"], 499)
        self.assertEqual(result["boundary_error_px"], -6)
        self.assertEqual(result["control_advice"], "hold_d")


class TestYihuanFishingActions(unittest.TestCase):
    def setUp(self):
        self.app = _FakeApp()
        self.input_mapping = _FakeInputMapping()
        self.vision = _FakeVision()
        self.service = YihuanFishingService()
        profile = self.service.load_profile()
        cached = dict(self.service._profile_cache[profile["profile_name"]])
        cached["duel_end_missing_sec"] = 0.0
        cached["bite_timeout_sec"] = 0.1
        cached["duel_timeout_sec"] = 0.1
        cached["hook_timeout_sec"] = 1.0
        cached["post_duel_click_interval_ms"] = 10
        cached["control_min_interval_ms"] = 0
        cached["control_reverse_gap_ms"] = 0
        cached["control_hold_grace_ms"] = 0
        cached["control_zone_jump_px"] = 999
        cached["control_indicator_jump_px"] = 999
        cached["control_zone_width_jump_px"] = 999
        self.service._profile_cache[profile["profile_name"]] = cached

    def _duel_state(self, *, zone_left=620, zone_right=720, indicator_x=670):
        zone_center = int(round((zone_left + zone_right) / 2.0))
        center_error = int(indicator_x - zone_center)
        if indicator_x < zone_left:
            boundary_error = int(indicator_x - zone_left)
        elif indicator_x > zone_right:
            boundary_error = int(indicator_x - zone_right)
        else:
            boundary_error = 0
        return {
            "phase": "duel",
            "control_advice": "none",
            "duel": {
                "found": True,
                "reason": "ok",
                "zone_left": zone_left,
                "zone_right": zone_right,
                "zone_center": zone_center,
                "indicator_x": indicator_x,
                "indicator_detected": True,
                "indicator_raw_detected": True,
                "indicator_source": "primary_component",
                "error_px": center_error,
                "center_error_px": center_error,
                "boundary_error_px": boundary_error,
            },
        }

    def test_hybrid_control_command_uses_hold_tap_and_stability(self):
        profile = self.service.load_profile()
        memory: dict[str, object] = {}

        stable = fishing_actions._compute_duel_control_command(
            self._duel_state(indicator_x=670),
            profile,
            memory,
            now=1.0,
        )
        self.assertIsNone(stable["action_name"])
        self.assertFalse(stable["execute"])
        self.assertTrue(stable["detection_trusted"])

        left_outside = fishing_actions._compute_duel_control_command(
            self._duel_state(indicator_x=610),
            profile,
            memory,
            now=1.1,
        )
        self.assertEqual(left_outside["action_name"], "fish_right")
        self.assertEqual(left_outside["hold_action_name"], "fish_right")
        self.assertEqual(left_outside["control_phase"], "hold")
        self.assertTrue(left_outside["execute"])

        left_edge = fishing_actions._compute_duel_control_command(
            self._duel_state(indicator_x=630),
            profile,
            {},
            now=1.2,
        )
        self.assertEqual(left_edge["action_name"], "fish_right")
        self.assertEqual(left_edge["hold_action_name"], "fish_right")
        self.assertEqual(left_edge["control_phase"], "hold")
        self.assertEqual(left_edge["reason"], "edge_hold")

        inside_left = fishing_actions._compute_duel_control_command(
            self._duel_state(indicator_x=660),
            profile,
            {},
            now=2.1,
        )
        self.assertEqual(inside_left["action_name"], "fish_right")
        self.assertEqual(inside_left["tap_action_name"], "fish_right")
        self.assertEqual(inside_left["control_phase"], "tap")
        self.assertEqual(inside_left["reason"], "inside_tap")

        suspicious_profile = dict(profile)
        suspicious_profile["control_zone_jump_px"] = 40
        suspicious_memory = {
            "last_trusted_at": 3.0,
            "trusted_geometry": {
                "zone_left": 620,
                "zone_right": 720,
                "zone_center": 670,
                "zone_width": 100,
                "indicator_x": 670,
                "center_error_px": 0,
            },
        }
        suspicious = fishing_actions._compute_duel_control_command(
            self._duel_state(zone_left=700, zone_right=795, indicator_x=740),
            suspicious_profile,
            suspicious_memory,
            now=3.05,
        )
        self.assertFalse(suspicious["detection_trusted"])
        self.assertTrue(suspicious["suspicious"])
        self.assertEqual(suspicious["suspicious_reason"], "zone_jump")
        self.assertEqual(suspicious["reason"], "suspicious_release")

    def test_hybrid_control_keeps_hold_briefly_when_detection_is_missing(self):
        profile = dict(self.service.load_profile())
        profile["control_hold_grace_ms"] = 150
        memory = {
            "held_action_name": "fish_right",
            "last_trusted_at": 1.0,
            "trusted_geometry": {
                "zone_left": 620,
                "zone_right": 720,
                "zone_center": 670,
                "zone_width": 100,
                "indicator_x": 610,
                "center_error_px": -60,
            },
        }

        command = fishing_actions._compute_duel_control_command(
            {"phase": "unknown", "control_advice": "none", "duel": {"found": False, "reason": "zone_missing"}},
            profile,
            memory,
            now=1.1,
        )

        self.assertEqual(command["action_name"], "fish_right")
        self.assertEqual(command["hold_action_name"], "fish_right")
        self.assertTrue(command["keep_hold"])
        self.assertEqual(command["reason"], "missing_keep_hold")

    @patch("plans.yihuan.src.actions.fishing_actions.time.sleep", return_value=None)
    def test_run_round_reports_hook_timeout(self, _sleep):
        initial_state = {"phase": "ready", "control_advice": "none", "duel": {}}

        with patch("plans.yihuan.src.actions.fishing_actions._read_state_snapshot", return_value=initial_state):
            with patch(
                "plans.yihuan.src.actions.fishing_actions._wait_for_hook_success",
                return_value={
                    "ok": False,
                    "failure_reason": "hook_timeout",
                    "elapsed_sec": 0.1,
                    "state": initial_state,
                },
            ):
                result = fishing_actions._run_fishing_round_impl(
                    app=self.app,
                    ocr=None,
                    vision=self.vision,
                    input_mapping=self.input_mapping,
                    yihuan_fishing=self.service,
                    round_index=1,
                    profile_name="default_1280x720_cn",
                    bite_timeout_sec=0.0,
                    duel_timeout_sec=0.0,
                )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "hook_timeout")
        self.assertIn(("fish_interact", "press", "default_1280x720_cn"), self.input_mapping.calls)

    @patch("plans.yihuan.src.actions.fishing_actions.time.sleep", return_value=None)
    def test_run_round_reports_duel_timeout(self, _sleep):
        state_sequence = iter([
            {"phase": "ready", "control_advice": "none", "duel": {}},
            {"phase": "bite", "control_advice": "none", "duel": {}},
        ])

        def fake_read_state(*args, **kwargs):
            try:
                return next(state_sequence)
            except StopIteration:
                return {
                    "phase": "duel",
                    "control_advice": "none",
                    "duel": {
                        "found": True,
                        "reason": "ok",
                        "zone_left": 620,
                        "zone_right": 720,
                        "indicator_x": 670,
                        "indicator_detected": True,
                        "indicator_raw_detected": True,
                        "indicator_source": "primary_component",
                        "error_px": 0,
                    },
                }

        with patch("plans.yihuan.src.actions.fishing_actions._read_state_snapshot", side_effect=fake_read_state):
            with patch("plans.yihuan.src.actions.fishing_actions._read_duel_snapshot_fast", side_effect=fake_read_state):
                result = fishing_actions._run_fishing_round_impl(
                    app=self.app,
                    ocr=None,
                    vision=self.vision,
                    input_mapping=self.input_mapping,
                    yihuan_fishing=self.service,
                    round_index=2,
                    profile_name="default_1280x720_cn",
                    bite_timeout_sec=0.1,
                    duel_timeout_sec=0.0,
                )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "duel_timeout")

    @patch("plans.yihuan.src.actions.fishing_actions.time.sleep", return_value=None)
    def test_run_round_successfully_closes_result(self, _sleep):
        state_sequence = iter([
            {"phase": "ready", "control_advice": "none", "duel": {}},
            {"phase": "bite", "control_advice": "none", "duel": {}},
            {
                "phase": "duel",
                "control_advice": "none",
                "duel": {
                    "found": True,
                    "reason": "ok",
                    "zone_left": 620,
                    "zone_right": 720,
                    "indicator_x": 670,
                    "indicator_detected": True,
                    "indicator_raw_detected": True,
                    "indicator_source": "primary_component",
                    "error_px": 0,
                },
            },
            {
                "phase": "duel",
                "control_advice": "hold_d",
                "duel": {
                    "found": True,
                    "reason": "ok",
                    "zone_left": 620,
                    "zone_right": 720,
                    "indicator_x": 600,
                    "indicator_detected": True,
                    "indicator_raw_detected": True,
                    "indicator_source": "primary_component",
                    "error_px": -8,
                },
            },
            {"phase": "result", "control_advice": "none", "duel": {}},
            {"phase": "ready", "control_advice": "none", "duel": {}},
        ])

        def fake_read_state(*args, **kwargs):
            return next(state_sequence)

        with patch("plans.yihuan.src.actions.fishing_actions._read_state_snapshot", side_effect=fake_read_state):
            with patch("plans.yihuan.src.actions.fishing_actions._read_duel_snapshot_fast", side_effect=fake_read_state):
                result = fishing_actions._run_fishing_round_impl(
                    app=self.app,
                    ocr=None,
                    vision=self.vision,
                    input_mapping=self.input_mapping,
                    yihuan_fishing=self.service,
                    round_index=3,
                    profile_name="default_1280x720_cn",
                    bite_timeout_sec=0.1,
                    duel_timeout_sec=0.1,
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(self.app.click_calls[-1], (300, 600, "left"))
        self.assertIn(("fish_right", "hold", "default_1280x720_cn"), self.input_mapping.calls)
        self.assertIn(("fish_right", "release", "default_1280x720_cn"), self.input_mapping.calls)

    @patch("plans.yihuan.src.actions.fishing_actions.time.sleep", return_value=None)
    def test_run_round_switches_held_direction_and_releases_on_ready(self, _sleep):
        state_sequence = iter([
            {"phase": "ready", "control_advice": "none", "duel": {}},
            {"phase": "bite", "control_advice": "none", "duel": {}},
            {
                "phase": "duel",
                "control_advice": "none",
                "duel": {
                    "found": True,
                    "reason": "ok",
                    "zone_left": 620,
                    "zone_right": 720,
                    "indicator_x": 670,
                    "indicator_detected": True,
                    "indicator_raw_detected": True,
                    "indicator_source": "primary_component",
                    "error_px": 0,
                },
            },
            {
                "phase": "duel",
                "control_advice": "hold_d",
                "duel": {
                    "found": True,
                    "reason": "ok",
                    "zone_left": 620,
                    "zone_right": 720,
                    "indicator_x": 600,
                    "indicator_detected": True,
                    "indicator_raw_detected": True,
                    "indicator_source": "primary_component",
                    "error_px": -12,
                },
            },
            {
                "phase": "duel",
                "control_advice": "hold_a",
                "duel": {
                    "found": True,
                    "reason": "ok",
                    "zone_left": 620,
                    "zone_right": 720,
                    "indicator_x": 730,
                    "indicator_detected": True,
                    "indicator_raw_detected": True,
                    "indicator_source": "primary_component",
                    "error_px": 10,
                },
            },
            {"phase": "ready", "control_advice": "none", "duel": {}},
            {"phase": "ready", "control_advice": "none", "duel": {}},
        ])

        def fake_read_state(*args, **kwargs):
            return next(state_sequence)

        with patch("plans.yihuan.src.actions.fishing_actions._read_state_snapshot", side_effect=fake_read_state):
            with patch("plans.yihuan.src.actions.fishing_actions._read_duel_snapshot_fast", side_effect=fake_read_state):
                result = fishing_actions._run_fishing_round_impl(
                    app=self.app,
                    ocr=None,
                    vision=self.vision,
                    input_mapping=self.input_mapping,
                    yihuan_fishing=self.service,
                    round_index=4,
                    profile_name="default_1280x720_cn",
                    bite_timeout_sec=0.1,
                    duel_timeout_sec=0.1,
                )

        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["failure_reason"])
        self.assertEqual(
            self.input_mapping.calls,
            [
                ("fish_interact", "press", "default_1280x720_cn"),
                ("fish_interact", "press", "default_1280x720_cn"),
                ("fish_right", "hold", "default_1280x720_cn"),
                ("fish_right", "release", "default_1280x720_cn"),
                ("fish_left", "hold", "default_1280x720_cn"),
                ("fish_left", "release", "default_1280x720_cn"),
            ],
        )
        self.assertEqual(self.app.click_calls[-1], (300, 600, "left"))

    @patch("plans.yihuan.src.actions.fishing_actions.time.sleep", return_value=None)
    def test_run_round_uses_tap_inside_zone_after_releasing_hold(self, _sleep):
        cached = dict(self.service._profile_cache["default_1280x720_cn"])
        cached["control_lookahead_sec"] = 0.0
        self.service._profile_cache["default_1280x720_cn"] = cached

        initial_state = {"phase": "ready", "control_advice": "none", "duel": {}}
        duel_state = {
            "phase": "duel",
            "control_advice": "none",
            "duel": {
                "found": True,
                "reason": "ok",
                "zone_left": 620,
                "zone_right": 720,
                "indicator_x": 670,
                "indicator_detected": True,
                "indicator_raw_detected": True,
                "indicator_source": "primary_component",
                "error_px": 0,
            },
        }
        duel_sequence = [
            {
                "phase": "duel",
                "control_advice": "hold_d",
                "duel": {
                    "found": True,
                    "reason": "ok",
                    "zone_left": 620,
                    "zone_right": 720,
                    "indicator_x": 610,
                    "indicator_detected": True,
                    "indicator_raw_detected": True,
                    "indicator_source": "primary_component",
                    "error_px": -60,
                },
            },
            {
                "phase": "duel",
                "control_advice": "tap_d",
                "duel": {
                    "found": True,
                    "reason": "ok",
                    "zone_left": 620,
                    "zone_right": 720,
                    "indicator_x": 650,
                    "indicator_detected": True,
                    "indicator_raw_detected": True,
                    "indicator_source": "primary_component",
                    "error_px": -20,
                },
            },
            {"phase": "result", "control_advice": "none", "duel": {}},
        ]
        ready_after_cleanup = {"phase": "ready", "control_advice": "none", "duel": {}}

        with patch(
            "plans.yihuan.src.actions.fishing_actions._read_state_snapshot",
            side_effect=[initial_state, ready_after_cleanup],
        ):
            with patch(
                "plans.yihuan.src.actions.fishing_actions._read_duel_snapshot_fast",
                side_effect=duel_sequence,
            ):
                with patch(
                    "plans.yihuan.src.actions.fishing_actions._wait_for_hook_success",
                    return_value={
                        "ok": True,
                        "failure_reason": None,
                        "elapsed_sec": 0.1,
                        "state": duel_state,
                    },
                ):
                    result = fishing_actions._run_fishing_round_impl(
                        app=self.app,
                        ocr=None,
                        vision=self.vision,
                        input_mapping=self.input_mapping,
                        yihuan_fishing=self.service,
                        round_index=5,
                        profile_name="default_1280x720_cn",
                        bite_timeout_sec=0.0,
                        duel_timeout_sec=0.1,
                    )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            self.input_mapping.calls,
            [
                ("fish_interact", "press", "default_1280x720_cn"),
                ("fish_right", "hold", "default_1280x720_cn"),
                ("fish_right", "release", "default_1280x720_cn"),
                ("fish_right", "tap", "default_1280x720_cn"),
            ],
        )

    @patch("plans.yihuan.src.actions.fishing_actions.time.sleep", return_value=None)
    def test_hook_wait_repeats_interact_until_duel_success(self, _sleep):
        state_sequence = iter([
            {"phase": "ready", "control_advice": "none", "duel": {"found": False, "reason": "zone_missing"}},
            {"phase": "ready", "control_advice": "none", "duel": {"found": False, "reason": "zone_missing"}},
            {
                "phase": "duel",
                "control_advice": "none",
                "duel": {
                    "found": True,
                    "reason": "ok",
                    "zone_left": 500,
                    "zone_right": 600,
                    "indicator_x": 550,
                    "indicator_detected": True,
                    "indicator_raw_detected": True,
                    "indicator_source": "primary_component",
                    "error_px": 0,
                },
            },
        ])

        monotonic_values = itertools.count(step=0.08)

        def fake_read_state(*args, **kwargs):
            return next(state_sequence)

        with patch("plans.yihuan.src.actions.fishing_actions._read_state_snapshot", side_effect=fake_read_state):
            with patch("plans.yihuan.src.actions.fishing_actions.time.monotonic", side_effect=lambda: next(monotonic_values)):
                result = fishing_actions._wait_for_hook_success(
                    self.app,
                    self.vision,
                    self.input_mapping,
                    self.service,
                    profile=self.service.load_profile(),
                    profile_name="default_1280x720_cn",
                    phase_trace=[],
                    start_time=0.0,
                    poll_sec=0.08,
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["failure_reason"], None)
        fish_interact_calls = [call for call in self.input_mapping.calls if call[0] == "fish_interact"]
        self.assertEqual(len(fish_interact_calls), 1)

    def test_run_session_ignores_consecutive_failures_and_stops_at_max_rounds(self):
        round_results = [
            {"status": "failed", "round_index": 1, "phase_trace": [], "failure_reason": "bite_timeout", "timings": {}},
            {"status": "failed", "round_index": 2, "phase_trace": [], "failure_reason": "duel_timeout", "timings": {}},
            {"status": "failed", "round_index": 3, "phase_trace": [], "failure_reason": "hook_timeout", "timings": {}},
            {"status": "success", "round_index": 4, "phase_trace": [], "failure_reason": None, "timings": {}},
        ]

        with patch("plans.yihuan.src.actions.fishing_actions._run_fishing_round_impl", side_effect=round_results) as run_round:
            result = fishing_actions.yihuan_fishing_run_session(
                app=self.app,
                ocr=None,
                vision=self.vision,
                input_mapping=self.input_mapping,
                yihuan_fishing=self.service,
                max_rounds=4,
                profile_name="default_1280x720_cn",
            )

        self.assertEqual(run_round.call_count, 4)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["stopped_reason"], "max_rounds")
        self.assertEqual(result["failure_count"], 3)
        self.assertEqual(result["success_count"], 1)

    @patch("plans.yihuan.src.actions.fishing_actions._close_debug_window")
    @patch("plans.yihuan.src.actions.fishing_actions._save_monitor_debug_frame", return_value="D:/tmp/frame_000.png")
    @patch("plans.yihuan.src.actions.fishing_actions._update_debug_window", return_value=True)
    @patch("plans.yihuan.src.actions.fishing_actions.time.sleep", return_value=None)
    def test_debug_monitor_returns_detection_summary(self, _sleep, _update_window, _save_frame, _close_window):
        state_sequence = iter([
            {
                "phase": "ready",
                "control_advice": "none",
                "duel": {
                    "found": False,
                    "reason": "indicator_missing",
                    "zone_detected": True,
                    "indicator_detected": False,
                    "zone_left": 620,
                    "zone_right": 720,
                },
            },
            {
                "phase": "duel",
                "control_advice": "none",
                "duel": {
                    "found": True,
                    "reason": "ok",
                    "zone_detected": True,
                    "indicator_detected": True,
                    "zone_left": 620,
                    "zone_right": 720,
                    "indicator_x": 670,
                    "indicator_raw_detected": True,
                    "indicator_source": "primary_component",
                    "error_px": 0,
                },
                "_capture_image": np.zeros((720, 1280, 3), dtype=np.uint8),
            },
            {
                "phase": "duel",
                "control_advice": "hold_d",
                "duel": {
                    "found": True,
                    "reason": "ok",
                    "zone_detected": True,
                    "indicator_detected": True,
                    "error_px": -8,
                    "zone_left": 620,
                    "zone_right": 720,
                    "indicator_x": 600,
                    "indicator_raw_detected": True,
                    "indicator_source": "primary_component",
                },
                "_capture_image": np.zeros((720, 1280, 3), dtype=np.uint8),
            },
            {
                "phase": "unknown",
                "control_advice": "none",
                "duel": {
                    "found": False,
                    "reason": "zone_missing",
                    "zone_detected": False,
                    "indicator_detected": False,
                },
                "_capture_image": np.zeros((720, 1280, 3), dtype=np.uint8),
            },
            {
                "phase": "unknown",
                "control_advice": "none",
                "duel": {
                    "found": False,
                    "reason": "zone_missing",
                    "zone_detected": False,
                    "indicator_detected": False,
                },
                "_capture_image": np.zeros((720, 1280, 3), dtype=np.uint8),
            },
        ])

        monotonic_values = itertools.count(step=0.1)

        def fake_read_state(*args, **kwargs):
            return next(state_sequence)

        with patch("plans.yihuan.src.actions.fishing_actions._read_state_snapshot", side_effect=fake_read_state):
            with patch("plans.yihuan.src.actions.fishing_actions.time.monotonic", side_effect=lambda: next(monotonic_values)):
                result = fishing_actions.yihuan_fishing_debug_monitor(
                    app=self.app,
                    ocr=None,
                    vision=self.vision,
                    input_mapping=self.input_mapping,
                    yihuan_fishing=self.service,
                    profile_name="default_1280x720_cn",
                    duration_sec=0.15,
                    poll_ms=10,
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["last_state"]["phase"], "unknown")
        self.assertIsNotNone(result["detection_stats"])
        self.assertGreaterEqual(result["detection_stats"]["observation_sec"], 0.0)
        self.assertEqual(result["detection_stats"]["samples"], 2)
        self.assertGreater(result["detection_stats"]["zone"]["missing_sec"], 0.0)
        self.assertIn("zone_missing", result["detection_stats"]["reason_sec"])
        self.assertEqual(
            self.input_mapping.calls,
            [
                ("fish_interact", "press", "default_1280x720_cn"),
                ("fish_interact", "press", "default_1280x720_cn"),
            ],
        )
        self.assertEqual(result["screenshots"], ["D:/tmp/frame_000.png"])
        _close_window.assert_called_once()


if __name__ == "__main__":
    unittest.main()

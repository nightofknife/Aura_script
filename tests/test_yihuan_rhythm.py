from __future__ import annotations

import unittest

import numpy as np

from plans.aura_base.src.platform.contracts import CaptureResult
from plans.yihuan.src.actions import rhythm_actions
from plans.yihuan.src.services.rhythm_service import YihuanRhythmService


class _FakeApp:
    def __init__(self, images: list[np.ndarray] | None = None) -> None:
        self.images = list(images or [])
        self.image = self.images[0] if self.images else _make_playing_image()
        self.click_calls: list[tuple[int, int, str]] = []
        self.key_down_calls: list[str] = []
        self.key_up_calls: list[str] = []
        self.release_all_calls = 0
        self.focus_calls = 0

    def capture(self):
        if self.images:
            self.image = self.images.pop(0)
        return CaptureResult(success=True, image=self.image.copy())

    def click(self, x=None, y=None, button="left", clicks=1, interval=None):
        self.click_calls.append((int(x), int(y), str(button)))

    def key_down(self, key: str):
        self.key_down_calls.append(str(key))

    def key_up(self, key: str):
        self.key_up_calls.append(str(key))

    def focus_with_input(self, click_delay: float = 0.05):
        self.focus_calls += 1
        return True

    def release_all(self):
        self.release_all_calls += 1


def _make_playing_image(*, notes: tuple[str, ...] = ()) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :] = (80, 160, 220)
    service = YihuanRhythmService()
    profile = service.load_profile()
    for lane_id in profile["lane_order"]:
        x, y = profile["lanes"][lane_id]["point"]
        image[y - 12 : y + 13, x - 7 : x + 8] = (230, 230, 230)
    for lane_id in notes:
        x, y = profile["lanes"][lane_id]["point"]
        image[y - 7 : y + 8, x - 4 : x + 5] = (20, 20, 20)
    return image


def _make_song_select_image() -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :] = (20, 20, 25)
    image[390:645, 0:1280] = (220, 45, 125)
    image[660:682, 1020:1165] = (235, 235, 235)
    return image


def _make_result_image() -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :] = (62, 62, 62)
    image[90:215, 580:705] = (210, 210, 210)
    return image


def _make_vivid_result_image() -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:430, :] = (38, 38, 38)
    image[430:, :] = (190, 110, 70)
    image[90:205, 580:705] = (250, 195, 55)
    return image


class TestYihuanRhythmService(unittest.TestCase):
    def setUp(self) -> None:
        self.service = YihuanRhythmService()

    def test_analyze_notes_detects_dark_icons_at_hit_line(self):
        image = _make_playing_image(notes=("d", "k"))

        state = self.service.analyze_notes(image)

        self.assertTrue(state["lanes"]["d"]["has_note"])
        self.assertFalse(state["lanes"]["f"]["has_note"])
        self.assertFalse(state["lanes"]["j"]["has_note"])
        self.assertTrue(state["lanes"]["k"]["has_note"])
        self.assertGreater(state["lanes"]["d"]["dark_ratio"], state["thresholds"]["dark_ratio"])

    def test_analyze_notes_scales_lane_points_to_capture_size(self):
        image = _make_playing_image(notes=("f",))
        scaled = image[::2, ::2].copy()

        state = self.service.analyze_notes(scaled)

        self.assertTrue(state["lanes"]["f"]["has_note"])
        self.assertFalse(state["lanes"]["d"]["has_note"])
        self.assertEqual(state["capture_size"], [640, 360])

    def test_analyze_notes_accepts_runtime_shifted_profile(self):
        profile = self.service.load_profile()
        lane = dict(profile["lanes"]["f"])
        x, y = lane["point"]
        lane["point"] = (x, y - 16)
        shifted_profile = {**profile, "lanes": {**profile["lanes"], "f": lane}}
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:, :] = (80, 160, 220)
        image[y - 23 : y - 8, x - 4 : x + 5] = (20, 20, 20)

        state = self.service.analyze_notes(image, profile=shifted_profile)

        self.assertTrue(state["lanes"]["f"]["has_note"])
        self.assertEqual(state["lanes"]["f"]["point"], [x, y - 16])

    def test_phase_detection_separates_song_select_result_and_playing(self):
        self.assertEqual(self.service.analyze_phase(_make_song_select_image())["phase"], "song_select")
        self.assertEqual(self.service.analyze_phase(_make_result_image())["phase"], "result")
        self.assertEqual(self.service.analyze_phase(_make_vivid_result_image())["phase"], "result")
        self.assertEqual(self.service.analyze_phase(_make_playing_image())["phase"], "playing")


class TestYihuanRhythmActions(unittest.TestCase):
    def test_lane_y_offset_shifts_runtime_profile_without_mutating_base(self):
        service = YihuanRhythmService()
        profile = service.load_profile()
        base_y = int(profile["lanes"]["d"]["point"][1])

        shifted = rhythm_actions._apply_lane_y_offset(profile, -16)

        self.assertEqual(shifted["lane_y_offset_px"], -16)
        self.assertEqual(shifted["lanes"]["d"]["point"][1], base_y - 16)
        self.assertEqual(profile["lanes"]["d"]["point"][1], base_y)

    def test_run_session_clicks_start_presses_detected_lane_and_closes_result(self):
        service = YihuanRhythmService()
        profile = service.load_profile()
        profile = {
            **profile,
            "post_start_delay_ms": 0,
            "finish_check_interval_ms": 0,
            "frame_interval_ms": 0,
            "result_exit_delay_ms": 0,
            "song_timeout_sec": 1,
        }
        service.load_profile = lambda profile_name=None: dict(profile)  # type: ignore[method-assign]
        app = _FakeApp(
            [
                _make_song_select_image(),
                _make_song_select_image(),
                _make_playing_image(),
                _make_playing_image(notes=("d",)),
                _make_result_image(),
            ]
        )

        result = rhythm_actions.yihuan_rhythm_run_session(
            app,
            service,
            loop_count=1,
            max_seconds=2,
            start_game=True,
            close_result=True,
            lane_keys="d,f,j,k",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "loop_count")
        self.assertEqual(result["loops_completed"], 1)
        self.assertEqual(result["hits_by_lane"]["d"], 1)
        self.assertIn("d", app.key_down_calls)
        self.assertIn("d", app.key_up_calls)
        self.assertGreaterEqual(len(app.click_calls), 2)
        self.assertEqual(app.click_calls[0][:2], (1070, 672))
        self.assertEqual(app.click_calls[-1][:2], (1219, 58))
        self.assertGreaterEqual(app.release_all_calls, 1)


if __name__ == "__main__":
    unittest.main()

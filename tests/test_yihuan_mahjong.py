from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from plans.aura_base.src.platform.contracts import CaptureResult
from plans.yihuan.src.actions import mahjong_actions
from plans.yihuan.src.services.mahjong_service import YihuanMahjongService


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "yihuan" / "mahjong"
TEMPLATE_DIR = (
    Path(__file__).resolve().parents[1]
    / "plans"
    / "yihuan"
    / "data"
    / "mahjong"
    / "default_1280x720_cn"
    / "templates"
)


class _FakeApp:
    def __init__(
        self,
        image: np.ndarray | None = None,
        *,
        images: list[np.ndarray] | None = None,
        fail_capture: bool = False,
    ) -> None:
        self.image = image if image is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
        self.images = list(images or [])
        self.fail_capture = fail_capture
        self.click_calls: list[tuple[int, int, str]] = []

    def capture(self):
        if self.fail_capture:
            return CaptureResult(success=False, image=None)
        if self.images:
            self.image = self.images.pop(0)
        return CaptureResult(success=True, image=self.image.copy())

    def click(self, x=None, y=None, button="left", clicks=1, interval=None):
        self.click_calls.append((int(x), int(y), str(button)))


class TestYihuanMahjongService(unittest.TestCase):
    def setUp(self) -> None:
        self.service = YihuanMahjongService()

    def test_phase_detectors_cover_ready_dingque_playing_and_result(self):
        cases = {
            "ready": _make_ready_image(),
            "dingque": _make_dingque_image({"wan": 4, "tong": 1, "tiao": 3}),
            "playing": _make_playing_image(enabled=True),
            "result": _make_result_image(),
        }

        for expected_phase, image in cases.items():
            with self.subTest(phase=expected_phase):
                state = self.service.analyze_phase(image)
                self.assertEqual(state["phase"], expected_phase)

    def test_real_fixture_phase_detectors_cover_exchange_and_runtime_states(self):
        cases = {
            "ready_1280x720.png": ("ready", None),
            "exchange_one_selected_1280x720.png": ("exchange", False),
            "exchange_three_selected_1280x720.png": ("exchange", True),
            "dingque_1280x720.png": ("dingque", None),
            "playing_1280x720.png": ("playing", None),
            "result_1280x720.png": ("result", None),
        }

        for filename, (expected_phase, expected_confirm) in cases.items():
            with self.subTest(filename=filename):
                state = self.service.analyze_phase(_load_fixture(filename))
                self.assertEqual(state["phase"], expected_phase)
                if expected_confirm is not None:
                    self.assertEqual(state["exchange"]["confirm_enabled"], expected_confirm)

    def test_auto_switch_detection_uses_enabled_and_disabled_templates(self):
        enabled = self.service.analyze_playing(_make_playing_image(enabled=True))
        disabled = self.service.analyze_playing(_make_playing_image(enabled=False))
        real_enabled = self.service.analyze_playing(_load_fixture("playing_1280x720.png"))

        self.assertTrue(enabled["found"])
        self.assertTrue(disabled["found"])
        self.assertTrue(real_enabled["found"])
        self.assertEqual(
            {name: state["enabled"] for name, state in enabled["switches"].items()},
            {"hu": True, "peng": True, "discard": True},
        )
        self.assertEqual(
            {name: state["enabled"] for name, state in disabled["switches"].items()},
            {"hu": False, "peng": False, "discard": False},
        )
        self.assertEqual(
            {name: state["enabled"] for name, state in real_enabled["switches"].items()},
            {"hu": True, "peng": True, "discard": True},
        )
        self.assertEqual(disabled["enabled_switch_count"], 0)

    def test_hand_suit_counts_use_tile_color_rules(self):
        hand = self.service.analyze_hand_suits(_make_dingque_image({"wan": 4, "tong": 1, "tiao": 3}))

        self.assertEqual(hand["counts"], {"wan": 4, "tong": 1, "tiao": 3})
        self.assertEqual(len(hand["candidates"]), 8)

    def test_choose_missing_suit_uses_fewest_count(self):
        selected = self.service.choose_missing_suit({"wan": 4, "tong": 1, "tiao": 3})

        self.assertEqual(selected, "tong")

    def test_choose_missing_suit_uses_fixed_tie_break_when_counts_equal(self):
        detail = self.service.choose_missing_suit_detail({"wan": 3, "tong": 3, "tiao": 3})

        self.assertEqual(detail["selected_suit"], "tong")
        self.assertEqual(detail["reason"], "tie_break")

    def test_choose_exchange_three_uses_fewest_eligible_suit(self):
        detail = self.service.choose_exchange_three_detail(
            [
                {"index": 0, "suit": "wan", "rect": [10, 10, 40, 60]},
                {"index": 1, "suit": "wan", "rect": [60, 10, 40, 60]},
                {"index": 2, "suit": "wan", "rect": [110, 10, 40, 60]},
                {"index": 3, "suit": "wan", "rect": [160, 10, 40, 60]},
                {"index": 4, "suit": "tong", "rect": [210, 10, 40, 60]},
                {"index": 5, "suit": "tong", "rect": [260, 10, 40, 60]},
                {"index": 6, "suit": "tong", "rect": [310, 10, 40, 60]},
                {"index": 7, "suit": "tiao", "rect": [360, 10, 40, 60]},
                {"index": 8, "suit": "tiao", "rect": [410, 10, 40, 60]},
            ]
        )

        self.assertTrue(detail["found"])
        self.assertEqual(detail["selected_suit"], "tong")
        self.assertEqual(detail["tile_indices"], [4, 5, 6])
        self.assertEqual(detail["tile_points"], [[230, 40], [280, 40], [330, 40]])

    def test_choose_exchange_three_reports_insufficient_same_suit(self):
        detail = self.service.choose_exchange_three_detail(
            [
                {"index": 0, "suit": "wan", "rect": [10, 10, 40, 60]},
                {"index": 1, "suit": "tong", "rect": [60, 10, 40, 60]},
                {"index": 2, "suit": "tiao", "rect": [110, 10, 40, 60]},
            ]
        )

        self.assertFalse(detail["found"])
        self.assertEqual(detail["reason"], "insufficient_same_suit")

    def test_pure_future_discard_strategy_clears_missing_suit_first(self):
        result = self.service.recommend_discard(
            [
                {"suit": "wan", "rank": 5, "isolated": False},
                {"suit": "tong", "rank": 1, "isolated": True},
                {"suit": "tiao", "rank": 9, "isolated": True},
            ],
            missing_suit="tong",
        )

        self.assertTrue(result["found"])
        self.assertEqual(result["reason"], "clear_missing_suit")
        self.assertEqual(result["tile"]["suit"], "tong")


class TestYihuanMahjongActions(unittest.TestCase):
    def setUp(self) -> None:
        self.service = YihuanMahjongService()

    def test_run_session_clicks_ready_selects_missing_suit_and_enables_switches(self):
        app = _FakeApp(
            images=[
                _make_ready_image(),
                _make_dingque_image({"wan": 4, "tong": 1, "tiao": 3}),
                _make_playing_image(enabled=False),
                _make_playing_image(enabled=True),
                _make_result_image(),
            ]
        )

        with patch("plans.yihuan.src.actions.mahjong_actions.time.sleep"):
            result = mahjong_actions.yihuan_mahjong_run_session(app, self.service, max_seconds=5)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "level_end")
        self.assertEqual(result["selected_missing_suit"], "tong")
        self.assertEqual(result["hand_suit_counts"], {"wan": 4, "tong": 1, "tiao": 3})
        self.assertEqual(
            app.click_calls,
            [
                (1100, 650, "left"),
                (637, 493, "left"),
                (56, 310, "left"),
                (56, 377, "left"),
                (56, 444, "left"),
            ],
        )
        self.assertEqual(result["auto_toggles_enabled"], {"hu": True, "peng": True, "discard": True})

    def test_run_session_handles_exchange_three_before_dingque(self):
        app = _FakeApp(
            images=[
                _load_fixture("ready_1280x720.png"),
                _load_fixture("exchange_one_selected_1280x720.png"),
                _load_fixture("exchange_three_selected_1280x720.png"),
                _load_fixture("dingque_1280x720.png"),
                _load_fixture("playing_1280x720.png"),
                _make_playing_image(enabled=True),
                _load_fixture("result_1280x720.png"),
            ]
        )

        with patch("plans.yihuan.src.actions.mahjong_actions.time.sleep"):
            result = mahjong_actions.yihuan_mahjong_run_session(app, self.service, max_seconds=5)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "level_end")
        self.assertEqual(result["exchange_selected_suit"], "tong")
        self.assertEqual(result["exchange_result"], "confirmed")
        self.assertEqual(result["exchange_tile_indices"], [4, 5, 6])
        self.assertEqual(result["exchange_tile_points"], [[718, 681], [766, 681], [814, 681]])
        self.assertIn((640, 504, "left"), app.click_calls)
        self.assertIn((637, 493, "left"), app.click_calls)
        self.assertEqual(result["auto_toggles_enabled"], {"hu": True, "peng": True, "discard": True})
        self.assertTrue(any(item["action"] == "exchange_confirm" and item["success"] for item in result["click_verifications"]))

    def test_run_session_can_start_from_exchange_phase(self):
        app = _FakeApp(
            images=[
                _load_fixture("exchange_one_selected_1280x720.png"),
                _load_fixture("exchange_three_selected_1280x720.png"),
                _load_fixture("dingque_1280x720.png"),
                _make_playing_image(enabled=True),
                _load_fixture("result_1280x720.png"),
            ]
        )

        with patch("plans.yihuan.src.actions.mahjong_actions.time.sleep"):
            result = mahjong_actions.yihuan_mahjong_run_session(app, self.service, max_seconds=5)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["exchange_result"], "confirmed")
        self.assertNotIn((1100, 650, "left"), app.click_calls)

    def test_run_session_result_screen_finishes_without_clicking(self):
        app = _FakeApp(_make_result_image())

        result = mahjong_actions.yihuan_mahjong_run_session(app, self.service)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "level_end")
        self.assertEqual(app.click_calls, [])

    def test_run_session_dry_run_records_planned_actions_without_clicking(self):
        app = _FakeApp(
            images=[
                _make_ready_image(),
                _make_dingque_image({"wan": 4, "tong": 1, "tiao": 3}),
                _make_playing_image(enabled=False),
                _make_result_image(),
            ]
        )

        with patch("plans.yihuan.src.actions.mahjong_actions.time.sleep"):
            result = mahjong_actions.yihuan_mahjong_run_session(app, self.service, max_seconds=5, dry_run=True)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "level_end")
        self.assertEqual(app.click_calls, [])
        self.assertEqual(result["executed_actions"], [])
        self.assertGreaterEqual(len(result["planned_actions"]), 3)

    def test_run_session_capture_failure_returns_business_failure(self):
        app = _FakeApp(fail_capture=True)

        result = mahjong_actions.yihuan_mahjong_run_session(app, self.service, max_seconds=1)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stopped_reason"], "failure")
        self.assertEqual(result["failure_reason"], "capture_failed")


def _load_fixture(filename: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURE_DIR / filename), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_template(filename: str) -> np.ndarray:
    image = cv2.imread(str(TEMPLATE_DIR / filename), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _make_base_image() -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :] = (30, 110, 95)
    return image


def _fill(image: np.ndarray, region: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, width, height = region
    image[y : y + height, x : x + width] = color


def _make_ready_image() -> np.ndarray:
    image = _make_base_image()
    _fill(image, (1065, 630, 95, 70), (236, 238, 228))
    _fill(image, (1105, 650, 24, 34), (35, 165, 70))
    return image


def _make_dingque_image(counts: dict[str, int]) -> np.ndarray:
    image = _make_base_image()
    _draw_dingque_button(image, (510, 465, 90, 95), (230, 40, 40))
    _draw_dingque_button(image, (602, 465, 90, 95), (230, 170, 55))
    _draw_dingque_button(image, (695, 465, 90, 95), (105, 210, 35))
    tiles: list[str] = []
    for suit in ("wan", "tong", "tiao"):
        tiles.extend([suit] * int(counts.get(suit, 0)))
    _draw_hand_tiles(image, tiles)
    return image


def _draw_dingque_button(image: np.ndarray, region: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, width, height = region
    _fill(image, region, (18, 18, 18))
    _fill(image, (x + 22, y + 18, width - 44, height - 36), color)


def _draw_hand_tiles(image: np.ndarray, suits: list[str]) -> None:
    x = 330
    y = 630
    for suit in suits:
        _fill(image, (x, y, 42, 70), (232, 230, 220))
        if suit == "wan":
            _fill(image, (x + 13, y + 42, 16, 20), (215, 35, 35))
        elif suit == "tong":
            _fill(image, (x + 9, y + 18, 24, 32), (65, 65, 210))
        else:
            _fill(image, (x + 13, y + 14, 16, 44), (45, 135, 50))
        x += 48


def _make_playing_image(*, enabled: bool) -> np.ndarray:
    image = _make_base_image()
    regions = {
        "hu": (28, 290, 68, 58),
        "peng": (28, 357, 68, 58),
        "discard": (28, 424, 68, 58),
    }
    glyph_offsets = {
        "hu": (17, 14),
        "peng": (17, 13),
        "discard": (18, 9),
    }
    state = "enabled" if enabled else "disabled"
    for name, region in regions.items():
        x, y, width, height = region
        _fill(image, region, (18, 18, 18))
        glyph = _load_template(f"switch_{name}_{state}.png")
        offset_x, offset_y = glyph_offsets[name]
        glyph_h, glyph_w = glyph.shape[:2]
        image[y + offset_y : y + offset_y + glyph_h, x + offset_x : x + offset_x + glyph_w] = glyph
    _draw_hand_tiles(image, ["wan", "tong", "tiao"])
    return image


def _make_result_image() -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :] = (80, 70, 65)
    _fill(image, (565, 80, 690, 575), (24, 24, 24))
    _fill(image, (590, 145, 640, 95), (70, 205, 220))
    _fill(image, (705, 585, 190, 45), (232, 232, 232))
    _fill(image, (970, 585, 190, 45), (232, 232, 232))
    _fill(image, (875, 660, 120, 18), (235, 235, 235))
    return image


if __name__ == "__main__":
    unittest.main()

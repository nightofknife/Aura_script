from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from plans.aura_base.src.platform.contracts import CaptureResult
from plans.yihuan.src.actions import cafe_actions
from plans.yihuan.src.services.cafe_service import YihuanCafeService


def _repeat_clicks(clicks: list[tuple[int, int, str]], repeat_count: int = 1) -> list[tuple[int, int, str]]:
    repeated: list[tuple[int, int, str]] = []
    for click in clicks:
        repeated.extend([click] * repeat_count)
    return repeated


class _FakeApp:
    def __init__(self, events=None):
        self.click_calls: list[tuple[int, int, str]] = []
        self.events = events
        self._image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self._cursor = (0, 0)
        self._mouse_down_button: str | None = None

    def click(self, x=None, y=None, button="left", clicks=1, interval=None):
        self.click_calls.append((int(x), int(y), str(button)))
        if self.events is not None:
            self.events.append(("click", int(x), int(y)))

    def move_to(self, x, y, duration=None):
        self._cursor = (int(x), int(y))

    def mouse_down(self, *, button="left"):
        self._mouse_down_button = str(button)

    def mouse_up(self, *, button="left"):
        x, y = self._cursor
        resolved_button = str(button)
        self.click_calls.append((x, y, resolved_button))
        if self.events is not None:
            self.events.append(("click", x, y))
        self._mouse_down_button = None

    def capture(self):
        return CaptureResult(success=True, image=self._image.copy())


class _FailingCaptureApp(_FakeApp):
    def capture(self):
        return CaptureResult(success=False, image=None)


class _FakeCafeService:
    def __init__(self, orders):
        self.orders = list(orders)
        self.profile = {
            "profile_name": "default_1280x720_cn",
            "start_game_point": (1150, 670),
            "level_started_region": (450, 10, 390, 45),
            "level_started_hsv_lower": (35, 80, 120),
            "level_started_hsv_upper": (65, 255, 255),
            "level_started_min_area": 800,
            "level_started_min_width": 120,
            "level_started_min_aspect_ratio": 6.0,
            "level_started_stable_frames": 2,
            "level_started_timeout_sec": 15.0,
            "level_started_poll_ms": 0,
            "level_end_title_region": (470, 85, 360, 120),
            "level_end_success_hsv_lower": (15, 70, 95),
            "level_end_success_hsv_upper": (45, 255, 255),
            "level_end_success_min_area": 8000,
            "level_end_success_min_width": 120,
            "level_end_success_min_height": 35,
            "level_end_failure_hsv_lower_1": (0, 70, 70),
            "level_end_failure_hsv_upper_1": (10, 255, 255),
            "level_end_failure_hsv_lower_2": (170, 70, 70),
            "level_end_failure_hsv_upper_2": (179, 255, 255),
            "level_end_failure_min_area": 8000,
            "level_end_failure_min_width": 120,
            "level_end_failure_min_height": 35,
            "level_end_buttons_region": (370, 500, 570, 95),
            "level_end_button_hsv_lower": (0, 0, 160),
            "level_end_button_hsv_upper": (179, 85, 255),
            "level_end_button_min_area": 5000,
            "level_end_button_min_width": 160,
            "level_end_button_max_width": 230,
            "level_end_button_min_height": 30,
            "level_end_button_max_height": 55,
            "level_end_button_min_count": 2,
            "level_end_stable_frames": 2,
            "level_end_poll_ms": 0,
            "order_search_region": (220, 60, 720, 250),
            "stock_profiles": {
                "bread": {
                    "station_point": (75, 665),
                    "stock_point": (65, 525),
                    "monitor_region": (20, 500, 120, 80),
                    "monitor_min_pixels": 120,
                    "monitor_white_s_max": 90,
                    "monitor_white_v_min": 170,
                    "monitor_use_warm": False,
                    "batch_size": 3,
                    "make_sec": 1.5,
                    "enabled": True,
                },
                "croissant": {
                    "station_point": (475, 665),
                    "stock_point": (425, 525),
                    "monitor_region": (340, 515, 170, 75),
                    "monitor_min_pixels": 30,
                    "monitor_white_s_max": 90,
                    "monitor_white_v_min": 170,
                    "monitor_use_warm": False,
                    "batch_size": 3,
                    "make_sec": 1.5,
                    "enabled": True,
                },
                "cake": {
                    "station_point": (690, 665),
                    "stock_point": (840, 665),
                    "monitor_region": (770, 635, 190, 80),
                    "monitor_min_pixels": 150,
                    "monitor_white_s_max": 90,
                    "monitor_white_v_min": 170,
                    "monitor_use_warm": True,
                    "monitor_warm_h_max": 35,
                    "monitor_warm_s_min": 60,
                    "monitor_warm_v_min": 120,
                    "batch_size": 6,
                    "make_sec": 1.5,
                    "enabled": True,
                },
                "coffee": {
                    "station_point": (1035, 665),
                    "stock_point": (1190, 665),
                    "monitor_region": (1145, 610, 130, 105),
                    "monitor_min_pixels": 120,
                    "monitor_white_s_max": 90,
                    "monitor_white_v_min": 150,
                    "monitor_use_warm": False,
                    "batch_size": 3,
                    "make_sec": 1.5,
                    "enabled": True,
                },
            },
            "coffee_station_point": (1035, 665),
            "coffee_stock_point": (1190, 665),
            "glass_point": (1200, 535),
            "coffee_cup_point": (825, 525),
            "bacon_point": (130, 430),
            "egg_point": (260, 430),
            "jam_point": (675, 430),
            "cream_point": (925, 430),
            "latte_art_point": (1030, 430),
            "coffee_batch_size": 3,
            "coffee_make_sec": 1.5,
            "start_game_delay_sec": 1.0,
            "click_repeat_count": 1,
            "click_hold_ms": 100,
            "click_repeat_interval_ms": 50,
            "step_delay_ms": 50,
            "craft_delay_ms": 50,
            "poll_ms": 0,
            "min_order_interval_sec": 0.0,
            "min_order_duration_sec": 0.0,
            "fake_customer_enabled": True,
            "fake_customer_region": (360, 220, 570, 190),
            "fake_customer_hammer_point": (70, 325),
            "fake_customer_after_drive_ms": 0,
            "fake_customer_global_cooldown_sec": 1.0,
            "fake_customer_same_spot_cooldown_sec": 3.0,
            "fake_customer_same_spot_distance_px": 85,
            "fake_customer_red_hsv_lower_1": (0, 80, 55),
            "fake_customer_red_hsv_upper_1": (12, 255, 255),
            "fake_customer_red_hsv_lower_2": (170, 80, 55),
            "fake_customer_red_hsv_upper_2": (179, 255, 255),
            "fake_customer_min_component_area": 800,
            "fake_customer_max_component_area": 2000,
            "fake_customer_min_component_width": 25,
            "fake_customer_max_component_width": 75,
            "fake_customer_min_component_height": 38,
            "fake_customer_max_component_height": 90,
            "fake_customer_min_aspect_ratio": 0.7,
            "fake_customer_max_fill_ratio": 0.9,
            "fake_customer_center_x_min": 0,
            "fake_customer_center_x_max": 1280,
            "fake_customer_center_y_min": 235,
            "fake_customer_center_y_max": 315,
            "fake_customer_nms_distance_px": 70,
            "fake_customer_body_region_from_order": (-90, 40, 180, 245),
            "fake_customer_require_order_slot": True,
            "fake_customer_min_red_pixels": 700,
            "fake_customer_min_blue_pixels": 550,
            "fake_customer_min_dark_pixels": 3500,
            "fake_customer_min_skin_pixels": 180,
            "fake_customer_red_check_region_ratio": (0.20, 0.18, 0.80, 0.58),
            "fake_customer_blue_check_region_ratio": (0.25, 0.45, 0.75, 0.75),
            "fake_customer_dark_check_region_ratio": (0.05, 0.15, 0.95, 0.95),
            "fake_customer_skin_check_region_ratio": (0.25, 0.02, 0.75, 0.35),
            "fake_customer_blue_hsv_lower": (95, 35, 30),
            "fake_customer_blue_hsv_upper": (135, 255, 220),
            "fake_customer_dark_hsv_lower": (0, 0, 0),
            "fake_customer_dark_hsv_upper": (179, 150, 115),
            "fake_customer_skin_hsv_lower": (0, 10, 80),
            "fake_customer_skin_hsv_upper": (35, 130, 255),
            "fake_customer_min_red_blue_vertical_gap_px": 18,
            "fake_customer_confirm_frames": 2,
            "fake_customer_clear_frames": 2,
            "fake_customer_after_hammer_suppress_sec": 2.0,
            "max_seconds": 130.0,
        }

    def load_profile(self, profile_name=None):
        return dict(self.profile)

    def detect_level_started(self, image, *, profile_name=None):
        return {
            "started": True,
            "reason": "ok",
            "green_area": 4000,
            "bbox": [476, 21, 326, 14],
            "aspect_ratio": 23.286,
        }

    def detect_level_end(self, image, *, profile_name=None):
        return {
            "ended": False,
            "outcome": None,
            "reason": "buttons_missing",
            "success_title": {"detected": False, "area": 0, "bbox": None, "pixel_count": 0},
            "failure_title": {"detected": False, "area": 0, "bbox": None, "pixel_count": 0},
            "buttons_count": 0,
            "button_bboxes": [],
        }

    def detect_stock_status(self, image, *, profile_name=None):
        return {
            "stocks": {
                stock_id: {
                    "present": True,
                    "reason": "present",
                    "region": list(stock_profile["monitor_region"]),
                    "product_pixels": int(stock_profile["monitor_min_pixels"]),
                    "min_pixels": int(stock_profile["monitor_min_pixels"]),
                    "white_pixels": int(stock_profile["monitor_min_pixels"]),
                    "warm_pixels": 0,
                }
                for stock_id, stock_profile in self.profile["stock_profiles"].items()
            }
        }

    def detect_fake_customers(self, image, *, profile_name=None):
        return {
            "detected": False,
            "reason": "no_candidate",
            "region": list(self.profile["fake_customer_region"]),
            "red_pixels": 0,
            "component_count": 0,
            "kept_count": 0,
            "candidates": [],
        }

    def analyze_orders(self, image, *, profile_name=None):
        if self.orders:
            recipe_id = self.orders.pop(0)
        else:
            recipe_id = "latte_coffee"
        return [
            {
                "recipe_id": recipe_id,
                "score": 0.99,
                "center_x": 640,
                "center_y": 180,
                "bbox": [610, 140, 70, 72],
                "scale": 1.0,
            }
        ]


class _NoOrderThenOrderCafeService(_FakeCafeService):
    def __init__(self):
        super().__init__(["cream_coffee"])
        self.scan_count = 0

    def analyze_orders(self, image, *, profile_name=None):
        self.scan_count += 1
        if self.scan_count == 1:
            return []
        return super().analyze_orders(image, profile_name=profile_name)


class _DelayedLevelStartCafeService(_FakeCafeService):
    def __init__(self, events):
        super().__init__(["cream_coffee"])
        self.events = events
        self.detect_results = [False, True, True]

    def detect_level_started(self, image, *, profile_name=None):
        started = self.detect_results.pop(0) if self.detect_results else True
        self.events.append(("detect_level_started", started))
        return {
            "started": started,
            "reason": "ok" if started else "green_component_missing",
            "green_area": 4000 if started else 0,
            "bbox": [476, 21, 326, 14] if started else None,
            "aspect_ratio": 23.286 if started else 0.0,
        }


class _LevelEndAfterThreeOrdersCafeService(_FakeCafeService):
    def __init__(self):
        super().__init__(["latte_coffee", "latte_coffee", "latte_coffee"])
        self.profile["level_end_stable_frames"] = 1
        self.orders_seen = 0

    def analyze_orders(self, image, *, profile_name=None):
        self.orders_seen += 1
        return super().analyze_orders(image, profile_name=profile_name)

    def detect_level_end(self, image, *, profile_name=None):
        if self.orders_seen < 3:
            return super().detect_level_end(image, profile_name=profile_name)
        return {
            "ended": True,
            "outcome": "success",
            "reason": "ok",
            "success_title": {"detected": True, "area": 12000, "bbox": [470, 107, 360, 76], "pixel_count": 14000},
            "failure_title": {"detected": False, "area": 0, "bbox": None, "pixel_count": 0},
            "buttons_count": 2,
            "button_bboxes": [[415, 539, 184, 39], [682, 539, 184, 39]],
        }


class _VisualBreadEmptyCafeService(_FakeCafeService):
    def __init__(self):
        super().__init__(["bacon_bread", "cream_coffee"])
        self.orders_seen = 0
        self.reported_empty = False

    def analyze_orders(self, image, *, profile_name=None):
        self.orders_seen += 1
        return super().analyze_orders(image, profile_name=profile_name)

    def detect_stock_status(self, image, *, profile_name=None):
        payload = super().detect_stock_status(image, profile_name=profile_name)
        if self.orders_seen >= 1 and not self.reported_empty:
            self.reported_empty = True
            payload["stocks"]["bread"].update(
                {
                    "present": False,
                    "reason": "empty",
                    "product_pixels": 0,
                    "white_pixels": 0,
                }
            )
        return payload


class _FakeCustomerThenOrderCafeService(_FakeCafeService):
    def __init__(self):
        super().__init__(["cream_coffee"])
        self.profile["fake_customer_confirm_frames"] = 1
        self.fake_scan_count = 0

    def detect_fake_customers(self, image, *, profile_name=None):
        self.fake_scan_count += 1
        if self.fake_scan_count == 1:
            return {
                "detected": True,
                "reason": "ok",
                "region": list(self.profile["fake_customer_region"]),
                "red_pixels": 1800,
                "component_count": 1,
                "kept_count": 1,
                "candidates": [
                    {
                        "score": 1000.0,
                        "area": 1000,
                        "aspect_ratio": 1.12,
                        "fill_ratio": 0.45,
                        "center_x": 468,
                        "center_y": 269,
                        "bbox": [443, 240, 50, 56],
                    }
                ],
            }
        return super().detect_fake_customers(image, profile_name=profile_name)


class _PersistentFakeCustomerCafeService(_FakeCafeService):
    def __init__(self):
        super().__init__(["cream_coffee"])

    def detect_fake_customers(self, image, *, profile_name=None):
        return {
            "detected": True,
            "reason": "ok",
            "region": list(self.profile["fake_customer_region"]),
            "red_pixels": 1800,
            "component_count": 1,
            "kept_count": 1,
            "candidates": [
                {
                    "score": 1000.0,
                    "area": 1000,
                    "aspect_ratio": 1.12,
                    "fill_ratio": 0.45,
                    "center_x": 468,
                    "center_y": 269,
                    "bbox": [443, 240, 50, 56],
                }
            ],
        }


class _ConsecutiveDifferentFakeCustomersCafeService(_FakeCafeService):
    def __init__(self):
        super().__init__(["cream_coffee"])
        self.profile["fake_customer_confirm_frames"] = 1
        self.profile["fake_customer_global_cooldown_sec"] = 0
        self.profile["fake_customer_after_hammer_suppress_sec"] = 0
        self.fake_scan_count = 0

    def detect_fake_customers(self, image, *, profile_name=None):
        self.fake_scan_count += 1
        centers = {
            1: (468, 269),
            2: (811, 269),
        }
        if self.fake_scan_count not in centers:
            return super().detect_fake_customers(image, profile_name=profile_name)

        center_x, center_y = centers[self.fake_scan_count]
        return {
            "detected": True,
            "reason": "ok",
            "region": list(self.profile["fake_customer_region"]),
            "red_pixels": 1800,
            "component_count": 1,
            "kept_count": 1,
            "candidates": [
                {
                    "score": 1000.0,
                    "area": 1000,
                    "aspect_ratio": 1.12,
                    "fill_ratio": 0.45,
                    "center_x": center_x,
                    "center_y": center_y,
                    "bbox": [center_x - 25, center_y - 29, 50, 56],
                }
            ],
        }


class TestYihuanCafeDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_root = Path("tests/fixtures/yihuan_cafe/client")
        cls.service = YihuanCafeService()

    def setUp(self):
        self.service.reset_order_scan_state("default_1280x720_cn")

    @classmethod
    def _load_fixture(cls, name: str):
        image = cv2.imread(str(cls.fixture_root / f"{name}.png"))
        if image is None:
            raise FileNotFoundError(f"Missing fixture image: {name}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def test_real_screenshots_detect_expected_cafe_orders(self):
        expected = {
            "cafe_01": {"latte_coffee": 1},
            "cafe_02": {"latte_coffee": 1, "cream_coffee": 1},
            "cafe_03": {"latte_coffee": 1, "cream_coffee": 1},
            "cafe_04": {"latte_coffee": 1, "cream_coffee": 1},
            "cafe_05": {"latte_coffee": 1, "cream_coffee": 1},
            "cafe_06": {"latte_coffee": 1, "cream_coffee": 2},
            "cafe_07": {"latte_coffee": 1, "cream_coffee": 3},
            "cafe_08": {"cream_coffee": 3},
            "cafe_bacon_bread": {"bacon_bread": 1},
            "cafe_egg_croissant": {"egg_croissant": 1},
            "cafe_jam_cake": {"jam_cake": 1},
        }

        for name, minimum_counts in expected.items():
            with self.subTest(name=name):
                orders = self.service.analyze_orders(self._load_fixture(name))
                counts: dict[str, int] = {}
                for order in orders:
                    counts[order["recipe_id"]] = counts.get(order["recipe_id"], 0) + 1
                    self.assertGreaterEqual(float(order["score"]), 0.78)

                for recipe_id, minimum_count in minimum_counts.items():
                    self.assertGreaterEqual(counts.get(recipe_id, 0), minimum_count)

    def test_cafe_orders_are_sorted_left_to_right_and_nms_deduped(self):
        orders = self.service.analyze_orders(self._load_fixture("cafe_07"))
        centers = [int(order["center_x"]) for order in orders]
        self.assertEqual(centers, sorted(centers))

        for index, first in enumerate(orders):
            for second in orders[index + 1:]:
                dx = int(first["center_x"]) - int(second["center_x"])
                dy = int(first["center_y"]) - int(second["center_y"])
                self.assertGreater(dx * dx + dy * dy, 35 * 35)

    def test_empty_order_fallback_is_throttled(self):
        service = YihuanCafeService()
        service.reset_order_scan_state("default_1280x720_cn")
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        fallback_calls = 0
        original_full_scan = service._analyze_orders_full_scan

        def fake_full_scan(source_image, *, profile):
            nonlocal fallback_calls
            fallback_calls += 1
            return []

        service._analyze_orders_full_scan = fake_full_scan
        try:
            for _ in range(9):
                self.assertEqual(service.analyze_orders(blank), [])
            self.assertEqual(fallback_calls, 0)
            self.assertEqual(service.analyze_orders(blank), [])
            self.assertEqual(fallback_calls, 1)
            debug = service.get_last_order_scan_debug()
            self.assertTrue(debug["fallback_used"])
            self.assertEqual(debug["two_stage_miss_count"], 10)
        finally:
            service._analyze_orders_full_scan = original_full_scan
            service.reset_order_scan_state("default_1280x720_cn")

    def test_fifo_order_sort_keeps_older_right_order_before_new_left_order(self):
        service = YihuanCafeService()
        profile = service.load_profile("default_1280x720_cn")
        profile_key = str(profile["profile_name"])
        debug: dict[str, object] = {}

        first_seen = service._sort_orders_for_selection(
            [
                {
                    "recipe_id": "cream_coffee",
                    "score": 0.95,
                    "center_x": 820,
                    "center_y": 180,
                    "bbox": [790, 150, 60, 60],
                    "scale": 1.0,
                }
            ],
            profile_key=profile_key,
            profile=profile,
            debug=debug,
        )
        self.assertEqual(first_seen[0]["center_x"], 820)
        older_track_id = first_seen[0]["track_id"]

        later_seen = service._sort_orders_for_selection(
            [
                {
                    "recipe_id": "latte_coffee",
                    "score": 0.96,
                    "center_x": 460,
                    "center_y": 180,
                    "bbox": [430, 150, 60, 60],
                    "scale": 1.0,
                },
                {
                    "recipe_id": "cream_coffee",
                    "score": 0.95,
                    "center_x": 820,
                    "center_y": 180,
                    "bbox": [790, 150, 60, 60],
                    "scale": 1.0,
                },
            ],
            profile_key=profile_key,
            profile=profile,
            debug={},
        )

        self.assertEqual([order["center_x"] for order in later_seen], [820, 460])
        self.assertEqual(later_seen[0]["track_id"], older_track_id)
        self.assertLess(later_seen[0]["arrival_index"], later_seen[1]["arrival_index"])

        self.assertTrue(service.mark_order_completed(later_seen[0], profile_name="default_1280x720_cn"))
        after_completion = service._sort_orders_for_selection(
            [later_seen[1]],
            profile_key=profile_key,
            profile=profile,
            debug={},
        )
        self.assertEqual([order["center_x"] for order in after_completion], [460])

    def test_level_started_green_bar_detection(self):
        detection = self.service.detect_level_started(self._load_fixture("cafe_started"))

        self.assertTrue(detection["started"])
        self.assertEqual(detection["reason"], "ok")
        self.assertGreaterEqual(int(detection["green_area"]), 800)
        self.assertGreaterEqual(int(detection["bbox"][2]), 120)

        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        blank_detection = self.service.detect_level_started(blank)
        self.assertFalse(blank_detection["started"])

    def test_level_end_detection_distinguishes_success_and_failure(self):
        success_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        success_image[107:183, 470:830] = (255, 230, 0)
        success_image[539:578, 415:599] = (235, 235, 235)
        success_image[539:578, 682:866] = (235, 235, 235)
        success = self.service.detect_level_end(success_image)

        self.assertTrue(success["ended"])
        self.assertEqual(success["outcome"], "success")
        self.assertGreaterEqual(success["buttons_count"], 2)

        failure_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        failure_image[115:178, 470:830] = (255, 0, 0)
        failure_image[539:578, 415:599] = (235, 235, 235)
        failure_image[539:578, 682:866] = (235, 235, 235)
        failure = self.service.detect_level_end(failure_image)

        self.assertTrue(failure["ended"])
        self.assertEqual(failure["outcome"], "failure")

        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        blank_detection = self.service.detect_level_end(blank)
        self.assertFalse(blank_detection["ended"])

    def test_fake_customer_detection_finds_red_scarf_components(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        profile = self.service.load_profile("default_1280x720_cn")
        profile["fake_customer_require_order_slot"] = False

        cv2.rectangle(image, (390, 230), (545, 430), (35, 40, 45), -1)
        cv2.ellipse(image, (468, 269), (25, 28), 0, 0, 360, (220, 30, 30), -1)
        cv2.rectangle(image, (430, 320), (505, 370), (55, 80, 160), -1)
        cv2.ellipse(image, (468, 225), (24, 30), 0, 0, 360, (210, 170, 145), -1)

        cv2.rectangle(image, (735, 230), (890, 430), (35, 40, 45), -1)
        cv2.ellipse(image, (811, 269), (25, 28), 0, 0, 360, (220, 30, 30), -1)
        cv2.rectangle(image, (775, 320), (850, 370), (55, 80, 160), -1)
        cv2.ellipse(image, (811, 225), (24, 30), 0, 0, 360, (210, 170, 145), -1)
        cv2.rectangle(image, (600, 390), (740, 430), (220, 30, 30), -1)

        with patch.object(self.service, "load_profile", return_value=profile):
            detection = self.service.detect_fake_customers(image)

        self.assertTrue(detection["detected"])
        self.assertEqual(len(detection["candidates"]), 2)
        self.assertEqual([candidate["center_x"] for candidate in detection["candidates"]], [468, 811])

    def test_fake_customer_detection_rejects_flat_sofa_like_red_blocks(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(image, (861, 290), (930, 315), (220, 30, 30), -1)

        detection = self.service.detect_fake_customers(image)

        self.assertFalse(detection["detected"], detection)

    def test_fake_customer_detection_with_provided_screenshot(self):
        screenshot_path = Path(r"C:\Users\356\Desktop\QQ20260426-211305.png")
        if not screenshot_path.is_file():
            self.skipTest(f"Provided screenshot is not available: {screenshot_path}")

        raw = cv2.imread(str(screenshot_path))
        if raw is None:
            self.skipTest(f"Failed to load provided screenshot: {screenshot_path}")
        if raw.shape[0] >= 776 and raw.shape[1] >= 1306:
            raw = raw[56:56 + 720, 26:26 + 1280]
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        detection = self.service.detect_fake_customers(image)
        candidates = list(detection["candidates"])

        self.assertTrue(detection["detected"], detection)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(any(440 <= int(candidate["center_x"]) <= 500 for candidate in candidates), candidates)
        self.assertTrue(any(785 <= int(candidate["center_x"]) <= 835 for candidate in candidates), candidates)


class TestYihuanCafeActions(unittest.TestCase):
    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_maintains_coffee_stock_and_recipe_order(self, _sleep):
        app = _FakeApp()
        service = _FakeCafeService([
            "latte_coffee",
            "latte_coffee",
            "latte_coffee",
            "cream_coffee",
        ])

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            profile_name="default_1280x720_cn",
            max_seconds=30,
            max_orders=4,
            start_game=True,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "max_orders")
        self.assertEqual(result["orders_completed"], 4)
        self.assertEqual(result["coffee_batches_made"], 2)
        self.assertEqual(result["coffee_stock_remaining"], 2)
        self.assertEqual(
            result["recognized_counts"],
            {
                "latte_coffee": 3,
                "cream_coffee": 1,
                "bacon_bread": 0,
                "egg_croissant": 0,
                "jam_cake": 0,
            },
        )
        self.assertEqual(result["batches_made"], {"bread": 1, "croissant": 1, "cake": 1, "coffee": 2})
        self.assertEqual(result["stocks_remaining"], {"bread": 3, "croissant": 3, "cake": 6, "coffee": 2})
        trace_events = [entry["event"] for entry in result["phase_trace"]]
        depleted_index = trace_events.index("stock_depleted_after_order")
        replenish_index = trace_events.index("stock_batch_started", depleted_index)
        next_order_index = trace_events.index("order_selected", replenish_index)
        self.assertLess(depleted_index, replenish_index)
        self.assertLess(replenish_index, next_order_index)
        self.assertEqual(
            app.click_calls,
            _repeat_clicks(
                [
                    (1150, 670, "left"),
                    (75, 665, "left"),
                    (475, 665, "left"),
                    (690, 665, "left"),
                    (1035, 665, "left"),
                    (1200, 535, "left"),
                    (1190, 665, "left"),
                    (1030, 430, "left"),
                    (1200, 535, "left"),
                    (1190, 665, "left"),
                    (1030, 430, "left"),
                    (1200, 535, "left"),
                    (1190, 665, "left"),
                    (1030, 430, "left"),
                    (1035, 665, "left"),
                    (825, 525, "left"),
                    (1190, 665, "left"),
                    (925, 430, "left"),
                ]
            ),
        )

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_can_wait_between_orders(self, sleep_mock):
        app = _FakeApp()
        service = _FakeCafeService(["cream_coffee", "cream_coffee"])

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=2,
            start_game=False,
            wait_level_started=False,
            min_order_interval_sec=0.75,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["orders_completed"], 2)
        self.assertEqual(result["min_order_interval_sec"], 0.75)
        interval_sleeps = [
            float(call.args[0])
            for call in sleep_mock.call_args_list
            if call.args and 0.6 <= float(call.args[0]) <= 0.8
        ]
        self.assertEqual(len(interval_sleeps), 1)
        self.assertIn("order_pacing_wait", [entry["event"] for entry in result["phase_trace"]])
        self.assertEqual(result["perf_stats"]["counts"]["order_pacing_wait_count"], 1)
        self.assertEqual(result["perf_stats"]["counts"]["min_order_interval_wait_count"], 1)

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_can_enforce_min_order_duration(self, sleep_mock):
        app = _FakeApp()
        service = _FakeCafeService(["cream_coffee", "cream_coffee"])

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=2,
            start_game=False,
            wait_level_started=False,
            min_order_interval_sec=0.1,
            min_order_duration_sec=1.25,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["orders_completed"], 2)
        self.assertEqual(result["min_order_duration_sec"], 1.25)
        duration_sleeps = [
            float(call.args[0])
            for call in sleep_mock.call_args_list
            if call.args and 1.0 <= float(call.args[0]) <= 1.3
        ]
        self.assertEqual(len(duration_sleeps), 1)
        pacing_events = [
            entry for entry in result["phase_trace"]
            if entry["event"] == "order_pacing_wait"
        ]
        self.assertEqual(len(pacing_events), 1)
        self.assertEqual(pacing_events[0]["rule"], "duration")
        self.assertEqual(result["perf_stats"]["counts"]["min_order_duration_wait_count"], 1)

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_can_skip_start_game_click(self, _sleep):
        app = _FakeApp()
        service = _FakeCafeService(["cream_coffee"])

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=1,
            start_game=False,
        )

        self.assertEqual(result["status"], "success")
        self.assertNotIn((1150, 670, "left"), app.click_calls)
        self.assertEqual(app.click_calls[0], (75, 665, "left"))

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_waits_for_level_start_before_replenishing(self, _sleep):
        events = []
        app = _FakeApp(events=events)
        service = _DelayedLevelStartCafeService(events)

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=1,
            start_game=True,
            wait_level_started=True,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            events[:6],
            [
                ("click", 1150, 670),
                ("detect_level_started", False),
                ("detect_level_started", True),
                ("detect_level_started", True),
                ("click", 75, 665),
                ("click", 475, 665),
            ],
        )

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_replenishes_coffee_before_order_scan(self, _sleep):
        app = _FakeApp()
        service = _NoOrderThenOrderCafeService()

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=1,
            start_game=False,
            wait_level_started=False,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["coffee_batches_made"], 1)
        self.assertEqual(result["batches_made"], {"bread": 1, "croissant": 1, "cake": 1, "coffee": 1})
        self.assertEqual(service.scan_count, 2)
        self.assertEqual(
            app.click_calls,
            _repeat_clicks(
                [
                    (75, 665, "left"),
                    (475, 665, "left"),
                    (690, 665, "left"),
                    (1035, 665, "left"),
                    (825, 525, "left"),
                    (1190, 665, "left"),
                    (925, 430, "left"),
                ]
            ),
        )

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_clicks_hammer_before_order_work_when_fake_customer_seen(self, _sleep):
        app = _FakeApp()
        service = _FakeCustomerThenOrderCafeService()

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=1,
            start_game=False,
            wait_level_started=False,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["orders_completed"], 1)
        self.assertEqual(result["fake_customers_detected"], 1)
        self.assertEqual(result["fake_customers_driven"], 1)
        self.assertEqual(app.click_calls[0], (70, 325, "left"))
        self.assertIn("fake_customer_hammer_clicked", [entry["event"] for entry in result["phase_trace"]])
        self.assertEqual(result["perf_stats"]["counts"]["fake_customer_hammer_click_count"], 1)
        self.assertEqual(
            app.click_calls,
            _repeat_clicks(
                [
                    (70, 325, "left"),
                    (75, 665, "left"),
                    (475, 665, "left"),
                    (690, 665, "left"),
                    (1035, 665, "left"),
                    (825, 525, "left"),
                    (1190, 665, "left"),
                    (925, 430, "left"),
                ]
            ),
        )

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_does_not_repeat_hammer_for_same_persistent_detection(self, _sleep):
        app = _FakeApp()
        service = _PersistentFakeCustomerCafeService()

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=1,
            start_game=False,
            wait_level_started=False,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["orders_completed"], 1)
        hammer_clicks = [click for click in app.click_calls if click == (70, 325, "left")]
        self.assertEqual(len(hammer_clicks), 1)
        self.assertEqual(result["fake_customers_driven"], 1)
        self.assertGreaterEqual(result["perf_stats"]["counts"]["fake_customer_cooldown_skip_count"], 1)

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_hammers_consecutive_fake_customers_at_different_positions(self, _sleep):
        app = _FakeApp()
        service = _ConsecutiveDifferentFakeCustomersCafeService()

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=1,
            start_game=False,
            wait_level_started=False,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["orders_completed"], 1)
        self.assertEqual(result["fake_customers_detected"], 2)
        self.assertEqual(result["fake_customers_driven"], 2)
        hammer_clicks = [click for click in app.click_calls if click == (70, 325, "left")]
        self.assertEqual(len(hammer_clicks), 2)
        hammer_events = [
            entry for entry in result["phase_trace"]
            if entry["event"] == "fake_customer_hammer_clicked"
        ]
        self.assertEqual(len(hammer_events), 2)
        self.assertEqual(
            [event["candidates"][0]["center_x"] for event in hammer_events],
            [468, 811],
        )
        self.assertEqual(result["perf_stats"]["counts"]["fake_customer_hammer_click_count"], 2)

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_stops_on_level_end_before_restocking(self, _sleep):
        app = _FakeApp()
        service = _LevelEndAfterThreeOrdersCafeService()

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=0,
            start_game=False,
            wait_level_started=False,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "level_end")
        self.assertEqual(result["level_outcome"], "success")
        self.assertEqual(result["orders_completed"], 3)
        self.assertEqual(result["coffee_batches_made"], 1)
        self.assertEqual(result["coffee_stock_remaining"], 0)
        self.assertEqual(result["batches_made"], {"bread": 1, "croissant": 1, "cake": 1, "coffee": 1})
        self.assertEqual(
            app.click_calls,
            _repeat_clicks(
                [
                    (75, 665, "left"),
                    (475, 665, "left"),
                    (690, 665, "left"),
                    (1035, 665, "left"),
                    (1200, 535, "left"),
                    (1190, 665, "left"),
                    (1030, 430, "left"),
                    (1200, 535, "left"),
                    (1190, 665, "left"),
                    (1030, 430, "left"),
                    (1200, 535, "left"),
                    (1190, 665, "left"),
                    (1030, 430, "left"),
                ]
            ),
        )

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_executes_non_coffee_recipes(self, _sleep):
        app = _FakeApp()
        service = _FakeCafeService(["bacon_bread", "egg_croissant", "jam_cake"])

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=3,
            start_game=False,
            wait_level_started=False,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "max_orders")
        self.assertEqual(result["orders_completed"], 3)
        self.assertEqual(result["recognized_counts"]["bacon_bread"], 1)
        self.assertEqual(result["recognized_counts"]["egg_croissant"], 1)
        self.assertEqual(result["recognized_counts"]["jam_cake"], 1)
        self.assertEqual(result["batches_made"], {"bread": 1, "croissant": 1, "cake": 1, "coffee": 1})
        self.assertEqual(result["stocks_remaining"]["bread"], 2)
        self.assertEqual(result["stocks_remaining"]["croissant"], 2)
        self.assertEqual(result["stocks_remaining"]["cake"], 5)
        self.assertTrue(
            result["stocks_remaining"]["coffee"] == 3 or "coffee" in result["pending_batches"],
            result,
        )
        self.assertEqual(
            app.click_calls,
            _repeat_clicks(
                [
                    (75, 665, "left"),
                    (475, 665, "left"),
                    (690, 665, "left"),
                    (1035, 665, "left"),
                    (65, 525, "left"),
                    (130, 430, "left"),
                    (425, 525, "left"),
                    (260, 430, "left"),
                    (840, 665, "left"),
                    (675, 430, "left"),
                ]
            ),
        )

    @patch("plans.yihuan.src.actions.cafe_actions.time.sleep", return_value=None)
    def test_run_session_replenishes_when_visual_stock_is_empty(self, _sleep):
        app = _FakeApp()
        service = _VisualBreadEmptyCafeService()

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=2,
            start_game=False,
            wait_level_started=False,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["batches_made"]["bread"], 2)
        self.assertTrue(
            result["stocks_remaining"]["bread"] == 3 or "bread" in result["pending_batches"],
            result,
        )
        trace_events = [entry["event"] for entry in result["phase_trace"]]
        self.assertIn("stock_visual_empty_correction", trace_events)
        self.assertEqual(
            app.click_calls,
            _repeat_clicks(
                [
                    (75, 665, "left"),
                    (475, 665, "left"),
                    (690, 665, "left"),
                    (1035, 665, "left"),
                    (65, 525, "left"),
                    (130, 430, "left"),
                    (75, 665, "left"),
                    (825, 525, "left"),
                    (1190, 665, "left"),
                    (925, 430, "left"),
                ]
            ),
        )

    def test_run_session_reports_capture_failure(self):
        app = _FailingCaptureApp()
        service = _FakeCafeService(["latte_coffee"])

        result = cafe_actions.yihuan_cafe_run_session(
            app=app,
            yihuan_cafe=service,
            max_seconds=30,
            max_orders=1,
            start_game=False,
            wait_level_started=False,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "capture_failed")


if __name__ == "__main__":
    unittest.main()

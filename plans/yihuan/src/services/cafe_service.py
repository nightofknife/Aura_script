"""Cafe order detection for the Yihuan plan."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import yaml

from packages.aura_core.api import service_info


Region = tuple[int, int, int, int]
Point = tuple[int, int]


@service_info(
    alias="yihuan_cafe",
    public=True,
    singleton=True,
    description="Load cafe calibration and detect Yihuan cafe mini-game orders.",
)
class YihuanCafeService:
    _DEFAULT_PROFILE = "default_1280x720_cn"

    def __init__(self) -> None:
        self._plan_root = Path(__file__).resolve().parents[2]
        self._profile_dir = self._plan_root / "data" / "cafe"
        self._template_dir = self._profile_dir / "templates"
        self._profile_cache: dict[str, dict[str, Any]] = {}
        self._template_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._template_gray_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._template_scaled_gray_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
        self._order_scan_state: dict[str, dict[str, Any]] = {}
        self._order_arrival_state: dict[str, dict[str, Any]] = {}
        self._last_order_scan_debug: dict[str, Any] = {}

    def load_profile(self, profile_name: str | None = None) -> dict[str, Any]:
        resolved_name = str(profile_name or self._DEFAULT_PROFILE).strip() or self._DEFAULT_PROFILE
        cached = self._profile_cache.get(resolved_name)
        if cached is not None:
            return dict(cached)

        profile_path = self._profile_dir / f"{resolved_name}.yaml"
        if not profile_path.is_file():
            raise FileNotFoundError(f"Cafe profile not found: {profile_path}")

        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        order_templates = {
            str(recipe_id): str(template_name)
            for recipe_id, template_name in dict(payload.get("order_templates") or {}).items()
            if str(recipe_id).strip() and str(template_name).strip()
        }
        if not order_templates:
            raise ValueError(f"Cafe profile has no order_templates: {profile_path}")

        stock_profiles = self._coerce_stock_profiles(payload)

        normalized = {
            "profile_name": str(payload.get("profile_name") or resolved_name),
            "client_size": self._coerce_size(payload.get("client_size"), default=(1280, 720)),
            "start_game_point": self._coerce_point(payload.get("start_game_point"), default=(1150, 670)),
            "level_started_region": self._coerce_region(
                payload.get("level_started_region"),
                default=(450, 10, 390, 45),
            ),
            "level_started_hsv_lower": self._coerce_hsv_triplet(
                payload.get("level_started_hsv_lower"),
                default=(35, 80, 120),
            ),
            "level_started_hsv_upper": self._coerce_hsv_triplet(
                payload.get("level_started_hsv_upper"),
                default=(65, 255, 255),
            ),
            "level_started_min_area": max(int(payload.get("level_started_min_area", 800) or 800), 1),
            "level_started_min_width": max(int(payload.get("level_started_min_width", 120) or 120), 1),
            "level_started_min_aspect_ratio": max(
                float(payload.get("level_started_min_aspect_ratio", 6.0) or 6.0),
                1.0,
            ),
            "level_started_stable_frames": max(int(payload.get("level_started_stable_frames", 2) or 2), 1),
            "level_started_timeout_sec": max(float(payload.get("level_started_timeout_sec", 15.0) or 15.0), 0.1),
            "level_started_poll_ms": max(int(payload.get("level_started_poll_ms", 150) or 150), 10),
            "level_end_title_region": self._coerce_region(
                payload.get("level_end_title_region"),
                default=(470, 85, 360, 120),
            ),
            "level_end_success_hsv_lower": self._coerce_hsv_triplet(
                payload.get("level_end_success_hsv_lower"),
                default=(15, 70, 95),
            ),
            "level_end_success_hsv_upper": self._coerce_hsv_triplet(
                payload.get("level_end_success_hsv_upper"),
                default=(45, 255, 255),
            ),
            "level_end_success_min_area": max(int(payload.get("level_end_success_min_area", 8000) or 8000), 1),
            "level_end_success_min_width": max(int(payload.get("level_end_success_min_width", 120) or 120), 1),
            "level_end_success_min_height": max(int(payload.get("level_end_success_min_height", 35) or 35), 1),
            "level_end_failure_hsv_lower_1": self._coerce_hsv_triplet(
                payload.get("level_end_failure_hsv_lower_1"),
                default=(0, 70, 70),
            ),
            "level_end_failure_hsv_upper_1": self._coerce_hsv_triplet(
                payload.get("level_end_failure_hsv_upper_1"),
                default=(10, 255, 255),
            ),
            "level_end_failure_hsv_lower_2": self._coerce_hsv_triplet(
                payload.get("level_end_failure_hsv_lower_2"),
                default=(170, 70, 70),
            ),
            "level_end_failure_hsv_upper_2": self._coerce_hsv_triplet(
                payload.get("level_end_failure_hsv_upper_2"),
                default=(179, 255, 255),
            ),
            "level_end_failure_min_area": max(int(payload.get("level_end_failure_min_area", 8000) or 8000), 1),
            "level_end_failure_min_width": max(int(payload.get("level_end_failure_min_width", 120) or 120), 1),
            "level_end_failure_min_height": max(int(payload.get("level_end_failure_min_height", 35) or 35), 1),
            "level_end_buttons_region": self._coerce_region(
                payload.get("level_end_buttons_region"),
                default=(370, 500, 570, 95),
            ),
            "level_end_button_hsv_lower": self._coerce_hsv_triplet(
                payload.get("level_end_button_hsv_lower"),
                default=(0, 0, 160),
            ),
            "level_end_button_hsv_upper": self._coerce_hsv_triplet(
                payload.get("level_end_button_hsv_upper"),
                default=(179, 85, 255),
            ),
            "level_end_button_min_area": max(int(payload.get("level_end_button_min_area", 5000) or 5000), 1),
            "level_end_button_min_width": max(int(payload.get("level_end_button_min_width", 160) or 160), 1),
            "level_end_button_max_width": max(int(payload.get("level_end_button_max_width", 230) or 230), 1),
            "level_end_button_min_height": max(int(payload.get("level_end_button_min_height", 30) or 30), 1),
            "level_end_button_max_height": max(int(payload.get("level_end_button_max_height", 55) or 55), 1),
            "level_end_button_min_count": max(int(payload.get("level_end_button_min_count", 2) or 2), 1),
            "level_end_stable_frames": max(int(payload.get("level_end_stable_frames", 2) or 2), 1),
            "level_end_poll_ms": max(int(payload.get("level_end_poll_ms", 120) or 120), 10),
            "fake_customer_enabled": self._coerce_bool(payload.get("fake_customer_enabled", True)),
            "fake_customer_region": self._coerce_region(
                payload.get("fake_customer_region"),
                default=(360, 220, 570, 70),
            ),
            "fake_customer_hammer_point": self._coerce_point(
                payload.get("fake_customer_hammer_point"),
                default=(70, 325),
            ),
            "fake_customer_hammer_debug_enabled": self._coerce_bool(
                payload.get("fake_customer_hammer_debug_enabled", True)
            ),
            "fake_customer_hammer_debug_dir": str(
                payload.get("fake_customer_hammer_debug_dir") or "tmp/cafe_fake_customer_hammers"
            ),
            "fake_customer_max_hammers_per_scan": max(
                self._coerce_int(payload.get("fake_customer_max_hammers_per_scan", 3), 3),
                1,
            ),
            "fake_customer_multi_hammer_interval_ms": max(
                self._coerce_int(payload.get("fake_customer_multi_hammer_interval_ms", 120), 120),
                0,
            ),
            "full_assist_hammer_interval_sec": max(
                self._coerce_float(payload.get("full_assist_hammer_interval_sec", 1.0), 1.0),
                0.05,
            ),
            "order_decision_debug_enabled": self._coerce_bool(
                payload.get("order_decision_debug_enabled", False)
            ),
            "order_decision_debug_dir": str(payload.get("order_decision_debug_dir") or "tmp/cafe_order_decisions"),
            "fake_customer_order_guard_enabled": self._coerce_bool(
                payload.get("fake_customer_order_guard_enabled", True)
            ),
            "fake_customer_order_guard_distance_px": max(
                int(payload.get("fake_customer_order_guard_distance_px", 85) or 85),
                1,
            ),
            "fake_customer_order_guard_confirm_ttl_sec": max(
                self._coerce_float(payload.get("fake_customer_order_guard_confirm_ttl_sec", 0.6), 0.6),
                0.0,
            ),
            "fake_customer_order_guard_cooldown_ttl_sec": max(
                self._coerce_float(payload.get("fake_customer_order_guard_cooldown_ttl_sec", 0.6), 0.6),
                0.0,
            ),
            "fake_customer_order_guard_after_hammer_ttl_sec": max(
                self._coerce_float(payload.get("fake_customer_order_guard_after_hammer_ttl_sec", 0.8), 0.8),
                0.0,
            ),
            "fake_customer_order_guard_poll_ms": max(
                self._coerce_int(payload.get("fake_customer_order_guard_poll_ms", 40), 40),
                0,
            ),
            "fake_customer_after_drive_ms": max(
                self._coerce_int(payload.get("fake_customer_after_drive_ms", 200), 200),
                0,
            ),
            "fake_customer_global_cooldown_sec": max(
                self._coerce_float(payload.get("fake_customer_global_cooldown_sec", 1.0), 1.0),
                0.0,
            ),
            "fake_customer_same_spot_cooldown_sec": max(
                self._coerce_float(payload.get("fake_customer_same_spot_cooldown_sec", 3.0), 3.0),
                0.0,
            ),
            "fake_customer_same_spot_distance_px": max(
                int(payload.get("fake_customer_same_spot_distance_px", 85) or 85),
                1,
            ),
            "fake_customer_red_hsv_lower_1": self._coerce_hsv_triplet(
                payload.get("fake_customer_red_hsv_lower_1"),
                default=(0, 145, 90),
            ),
            "fake_customer_red_hsv_upper_1": self._coerce_hsv_triplet(
                payload.get("fake_customer_red_hsv_upper_1"),
                default=(5, 255, 255),
            ),
            "fake_customer_red_hsv_lower_2": self._coerce_hsv_triplet(
                payload.get("fake_customer_red_hsv_lower_2"),
                default=(176, 145, 90),
            ),
            "fake_customer_red_hsv_upper_2": self._coerce_hsv_triplet(
                payload.get("fake_customer_red_hsv_upper_2"),
                default=(179, 255, 255),
            ),
            "fake_customer_min_component_area": max(
                int(payload.get("fake_customer_min_component_area", 650) or 650),
                1,
            ),
            "fake_customer_max_component_area": max(
                int(payload.get("fake_customer_max_component_area", 2000) or 2000),
                1,
            ),
            "fake_customer_min_component_width": max(
                int(payload.get("fake_customer_min_component_width", 25) or 25),
                1,
            ),
            "fake_customer_max_component_width": max(
                int(payload.get("fake_customer_max_component_width", 90) or 90),
                1,
            ),
            "fake_customer_min_component_height": max(
                int(payload.get("fake_customer_min_component_height", 30) or 30),
                1,
            ),
            "fake_customer_max_component_height": max(
                int(payload.get("fake_customer_max_component_height", 45) or 45),
                1,
            ),
            "fake_customer_hammer_clip_top_margin_px": max(
                int(payload.get("fake_customer_hammer_clip_top_margin_px", 18) or 18),
                0,
            ),
            "fake_customer_hammer_clip_min_height_px": max(
                int(payload.get("fake_customer_hammer_clip_min_height_px", 48) or 48),
                1,
            ),
            "fake_customer_min_aspect_ratio": max(
                float(payload.get("fake_customer_min_aspect_ratio", 0.6) or 0.6),
                0.0,
            ),
            "fake_customer_max_fill_ratio": max(
                float(payload.get("fake_customer_max_fill_ratio", 0.9) or 0.9),
                0.0,
            ),
            "fake_customer_center_x_min": max(
                int(payload.get("fake_customer_center_x_min", 0) or 0),
                0,
            ),
            "fake_customer_center_x_max": max(
                int(payload.get("fake_customer_center_x_max", 1280) or 1280),
                0,
            ),
            "fake_customer_center_y_min": max(
                int(payload.get("fake_customer_center_y_min", 235) or 235),
                0,
            ),
            "fake_customer_center_y_max": max(
                int(payload.get("fake_customer_center_y_max", 315) or 315),
                0,
            ),
            "fake_customer_nms_distance_px": max(
                int(payload.get("fake_customer_nms_distance_px", 70) or 70),
                1,
            ),
            "fake_customer_body_region_from_order": self._coerce_offset_region(
                payload.get("fake_customer_body_region_from_order"),
                default=(-90, 40, 180, 245),
            ),
            "fake_customer_require_order_slot": self._coerce_bool(
                payload.get("fake_customer_require_order_slot", True)
            ),
            "fake_customer_min_red_pixels": max(
                int(payload.get("fake_customer_min_red_pixels", 700) or 700),
                1,
            ),
            "fake_customer_min_blue_pixels": max(
                int(payload.get("fake_customer_min_blue_pixels", 550) or 550),
                1,
            ),
            "fake_customer_min_dark_pixels": max(
                int(payload.get("fake_customer_min_dark_pixels", 3500) or 3500),
                0,
            ),
            "fake_customer_min_skin_pixels": max(
                int(payload.get("fake_customer_min_skin_pixels", 180) or 180),
                0,
            ),
            "fake_customer_red_check_region_ratio": self._coerce_ratio_region(
                payload.get("fake_customer_red_check_region_ratio"),
                default=(0.20, 0.18, 0.80, 0.58),
            ),
            "fake_customer_blue_check_region_ratio": self._coerce_ratio_region(
                payload.get("fake_customer_blue_check_region_ratio"),
                default=(0.25, 0.45, 0.75, 0.75),
            ),
            "fake_customer_dark_check_region_ratio": self._coerce_ratio_region(
                payload.get("fake_customer_dark_check_region_ratio"),
                default=(0.05, 0.15, 0.95, 0.95),
            ),
            "fake_customer_skin_check_region_ratio": self._coerce_ratio_region(
                payload.get("fake_customer_skin_check_region_ratio"),
                default=(0.25, 0.02, 0.75, 0.35),
            ),
            "fake_customer_blue_hsv_lower": self._coerce_hsv_triplet(
                payload.get("fake_customer_blue_hsv_lower"),
                default=(95, 35, 30),
            ),
            "fake_customer_blue_hsv_upper": self._coerce_hsv_triplet(
                payload.get("fake_customer_blue_hsv_upper"),
                default=(135, 255, 220),
            ),
            "fake_customer_dark_hsv_lower": self._coerce_hsv_triplet(
                payload.get("fake_customer_dark_hsv_lower"),
                default=(0, 0, 0),
            ),
            "fake_customer_dark_hsv_upper": self._coerce_hsv_triplet(
                payload.get("fake_customer_dark_hsv_upper"),
                default=(179, 150, 115),
            ),
            "fake_customer_skin_hsv_lower": self._coerce_hsv_triplet(
                payload.get("fake_customer_skin_hsv_lower"),
                default=(0, 10, 80),
            ),
            "fake_customer_skin_hsv_upper": self._coerce_hsv_triplet(
                payload.get("fake_customer_skin_hsv_upper"),
                default=(35, 130, 255),
            ),
            "fake_customer_min_red_blue_vertical_gap_px": max(
                int(payload.get("fake_customer_min_red_blue_vertical_gap_px", 18) or 18),
                0,
            ),
            "fake_customer_confirm_frames": max(
                int(payload.get("fake_customer_confirm_frames", 2) or 2),
                1,
            ),
            "fake_customer_clear_frames": max(
                int(payload.get("fake_customer_clear_frames", 2) or 2),
                1,
            ),
            "fake_customer_after_hammer_suppress_sec": max(
                self._coerce_float(payload.get("fake_customer_after_hammer_suppress_sec", 2.0), 2.0),
                0.0,
            ),
            "order_search_region": self._coerce_region(payload.get("order_search_region"), default=(120, 60, 980, 255)),
            "order_locator_enabled": self._coerce_bool(payload.get("order_locator_enabled", True)),
            "order_locator_fallback_full_scan": self._coerce_bool(
                payload.get("order_locator_fallback_full_scan", False)
            ),
            "order_locator_fallback_every_no_order_scans": max(
                int(payload.get("order_locator_fallback_every_no_order_scans", 10) or 10),
                1,
            ),
            "order_locator_region": self._coerce_region(
                payload.get("order_locator_region"),
                default=(120, 75, 980, 210),
            ),
            "order_locator_hough_dp": max(float(payload.get("order_locator_hough_dp", 1.0) or 1.0), 0.1),
            "order_locator_hough_min_dist": max(
                int(payload.get("order_locator_hough_min_dist", 45) or 45),
                1,
            ),
            "order_locator_hough_param1": max(
                int(payload.get("order_locator_hough_param1", 80) or 80),
                1,
            ),
            "order_locator_hough_param2": max(
                int(payload.get("order_locator_hough_param2", 30) or 30),
                1,
            ),
            "order_locator_min_radius": max(int(payload.get("order_locator_min_radius", 18) or 18), 1),
            "order_locator_max_radius": max(int(payload.get("order_locator_max_radius", 45) or 45), 1),
            "order_locator_yellow_hsv_lower": self._coerce_hsv_triplet(
                payload.get("order_locator_yellow_hsv_lower"),
                default=(8, 70, 80),
            ),
            "order_locator_yellow_hsv_upper": self._coerce_hsv_triplet(
                payload.get("order_locator_yellow_hsv_upper"),
                default=(45, 255, 255),
            ),
            "order_locator_min_yellow_pixels": max(
                int(payload.get("order_locator_min_yellow_pixels", 80) or 80),
                1,
            ),
            "order_locator_yellow_inner_pad": max(
                int(payload.get("order_locator_yellow_inner_pad", 8) or 8),
                0,
            ),
            "order_locator_yellow_outer_pad": max(
                int(payload.get("order_locator_yellow_outer_pad", 10) or 10),
                0,
            ),
            "order_locator_nms_distance_px": max(
                int(payload.get("order_locator_nms_distance_px", 45) or 45),
                1,
            ),
            "order_classify_region_size": self._coerce_size(
                payload.get("order_classify_region_size"),
                default=(130, 120),
            ),
            "order_templates": order_templates,
            "template_threshold": float(payload.get("template_threshold", 0.78) or 0.78),
            "template_scales": self._coerce_float_list(
                payload.get("template_scales"),
                default=(0.65, 0.85, 1.0, 1.15),
            ),
            "match_nms_distance_px": max(int(payload.get("match_nms_distance_px", 35) or 35), 1),
            "match_peak_window_px": max(int(payload.get("match_peak_window_px", 9) or 9), 1),
            "max_candidates_per_template_scale": max(int(payload.get("max_candidates_per_template_scale", 80) or 80), 1),
            "order_sort_strategy": self._coerce_choice(
                payload.get("order_sort_strategy"),
                default="fifo",
                allowed={"fifo", "left_to_right"},
            ),
            "order_track_distance_px": max(int(payload.get("order_track_distance_px", 80) or 80), 1),
            "order_track_forget_after_scans": max(
                int(payload.get("order_track_forget_after_scans", 40) or 40),
                1,
            ),
            "stock_profiles": stock_profiles,
            "coffee_station_point": stock_profiles["coffee"]["station_point"],
            "coffee_stock_point": stock_profiles["coffee"]["stock_point"],
            "coffee_stock_points": self._coerce_point_list(
                payload.get("coffee_stock_points"),
                default=((1145, 665), (1190, 665), (1235, 665)),
            ),
            "glass_point": self._coerce_point(payload.get("glass_point"), default=(1200, 535)),
            "coffee_cup_point": self._coerce_point(payload.get("coffee_cup_point"), default=(825, 525)),
            "bacon_point": self._coerce_point(payload.get("bacon_point"), default=(130, 430)),
            "egg_point": self._coerce_point(payload.get("egg_point"), default=(260, 430)),
            "jam_point": self._coerce_point(payload.get("jam_point"), default=(675, 430)),
            "cream_point": self._coerce_point(payload.get("cream_point"), default=(925, 430)),
            "latte_art_point": self._coerce_point(payload.get("latte_art_point"), default=(1030, 430)),
            "coffee_batch_size": stock_profiles["coffee"]["batch_size"],
            "coffee_make_sec": stock_profiles["coffee"]["make_sec"],
            "start_game_delay_sec": max(float(payload.get("start_game_delay_sec", 1.0) or 1.0), 0.0),
            "click_repeat_count": max(int(payload.get("click_repeat_count", 2) or 2), 1),
            "click_hold_ms": max(int(payload.get("click_hold_ms", 100) or 100), 0),
            "click_repeat_interval_ms": max(int(payload.get("click_repeat_interval_ms", 100) or 100), 0),
            "single_click_notes": self._coerce_string_list(
                payload.get("single_click_notes"),
                default=("start_game", "fake_customer_hammer"),
            ),
            "step_delay_ms": max(int(payload.get("step_delay_ms", 100) or 100), 0),
            "craft_delay_ms": max(int(payload.get("craft_delay_ms", 100) or 100), 0),
            "poll_ms": max(int(payload.get("poll_ms", 120) or 120), 0),
            "min_order_interval_sec": max(float(payload.get("min_order_interval_sec", 0.5) or 0.0), 0.0),
            "min_order_duration_sec": max(float(payload.get("min_order_duration_sec", 0.0) or 0.0), 0.0),
            "max_seconds": max(float(payload.get("max_seconds", 130.0) or 130.0), 0.1),
        }
        self._profile_cache[resolved_name] = normalized
        return dict(normalized)

    def analyze_orders(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
    ) -> list[dict[str, Any]]:
        profile = self.load_profile(profile_name)
        profile_key = str(profile["profile_name"])
        scan_start = time.perf_counter()
        debug: dict[str, Any] = {
            "strategy": "two_stage",
            "locator_count": 0,
            "classified_count": 0,
            "fallback_used": False,
            "fallback_allowed": False,
            "fallback_reason": None,
            "fallback_skipped_reason": None,
            "two_stage_miss_count": int(self._order_scan_state.get(profile_key, {}).get("two_stage_miss_count", 0)),
            "locator_sec": 0.0,
            "classification_sec": 0.0,
            "fallback_sec": 0.0,
            "total_sec": 0.0,
            "order_locator_region": list(profile["order_locator_region"]),
            "order_search_region": list(profile["order_search_region"]),
            "locator_centers": [],
        }
        if bool(profile["order_locator_enabled"]):
            locator_start = time.perf_counter()
            locators = self.detect_order_candidates(source_image, profile_name=profile_name)
            debug["locator_sec"] = time.perf_counter() - locator_start
            debug["locator_count"] = len(locators)
            debug["locator_centers"] = [
                {
                    "x": int(locator["center_x"]),
                    "y": int(locator["center_y"]),
                    "r": int(locator.get("radius", 0) or 0),
                    "yellow": int(locator.get("yellow_pixels", 0) or 0),
                }
                for locator in locators
            ]
            if locators:
                candidates: list[dict[str, Any]] = []
                classify_start = time.perf_counter()
                for locator in locators:
                    classified = self._classify_order_candidate(source_image, locator, profile=profile)
                    if classified is not None:
                        candidates.append(classified)
                debug["classification_sec"] = time.perf_counter() - classify_start
                debug["classified_count"] = len(candidates)

                if candidates:
                    kept = self._nms_by_center_distance(candidates, int(profile["match_nms_distance_px"]))
                    kept = self._sort_orders_for_selection(kept, profile_key=profile_key, profile=profile, debug=debug)
                    self._set_order_scan_debug(
                        profile_key,
                        debug,
                        scan_start=scan_start,
                        orders_returned=len(kept),
                        reset_miss_count=True,
                    )
                    return kept

            if not bool(profile["order_locator_fallback_full_scan"]):
                debug["fallback_skipped_reason"] = "disabled"
                self._increment_order_scan_miss(profile_key, debug)
                self._age_order_tracks(profile_key, profile=profile, debug=debug)
                self._set_order_scan_debug(profile_key, debug, scan_start=scan_start, orders_returned=0)
                return []

            miss_count = self._increment_order_scan_miss(profile_key, debug)
            fallback_every = int(profile["order_locator_fallback_every_no_order_scans"])
            debug["fallback_allowed"] = miss_count % fallback_every == 0
            debug["two_stage_miss_count"] = miss_count
            if not bool(debug["fallback_allowed"]):
                debug["fallback_skipped_reason"] = f"throttled_{miss_count}_of_{fallback_every}"
                self._age_order_tracks(profile_key, profile=profile, debug=debug)
                self._set_order_scan_debug(profile_key, debug, scan_start=scan_start, orders_returned=0)
                return []

            debug["strategy"] = "full_scan_fallback"
            debug["fallback_used"] = True
            debug["fallback_reason"] = "two_stage_no_match"
            fallback_start = time.perf_counter()
            orders = self._analyze_orders_full_scan(source_image, profile=profile)
            debug["fallback_sec"] = time.perf_counter() - fallback_start
            if orders:
                self._reset_order_scan_miss(profile_key)
                debug["two_stage_miss_count"] = 0
                orders = self._sort_orders_for_selection(
                    orders,
                    profile_key=profile_key,
                    profile=profile,
                    debug=debug,
                )
            else:
                self._age_order_tracks(profile_key, profile=profile, debug=debug)
            self._set_order_scan_debug(profile_key, debug, scan_start=scan_start, orders_returned=len(orders))
            return orders

        debug["strategy"] = "full_scan"
        debug["fallback_used"] = True
        debug["fallback_allowed"] = True
        debug["fallback_reason"] = "locator_disabled"
        fallback_start = time.perf_counter()
        orders = self._analyze_orders_full_scan(source_image, profile=profile)
        debug["fallback_sec"] = time.perf_counter() - fallback_start
        if orders:
            orders = self._sort_orders_for_selection(orders, profile_key=profile_key, profile=profile, debug=debug)
        else:
            self._age_order_tracks(profile_key, profile=profile, debug=debug)
        self._set_order_scan_debug(profile_key, debug, scan_start=scan_start, orders_returned=len(orders))
        return orders

    def get_last_order_scan_debug(self) -> dict[str, Any]:
        return dict(self._last_order_scan_debug)

    def reset_order_scan_state(self, profile_name: str | None = None) -> None:
        if profile_name is None:
            self._order_scan_state.clear()
            self._order_arrival_state.clear()
            self._last_order_scan_debug = {}
            return

        profile = self.load_profile(profile_name)
        profile_key = str(profile["profile_name"])
        self._order_scan_state.pop(profile_key, None)
        self._order_arrival_state.pop(profile_key, None)
        if self._last_order_scan_debug.get("profile_name") == profile_key:
            self._last_order_scan_debug = {}

    def mark_order_completed(self, order: dict[str, Any] | None = None, *, profile_name: str | None = None) -> bool:
        profile = self.load_profile(profile_name)
        profile_key = str(profile["profile_name"])
        state = self._order_arrival_state.get(profile_key)
        if not state:
            return False

        raw_track_id = None if order is None else order.get("track_id")
        if raw_track_id is None:
            return False

        tracks = state.setdefault("tracks", {})
        track_id = int(raw_track_id)
        removed = tracks.pop(track_id, None) is not None
        state["tracks"] = tracks
        return removed

    def _increment_order_scan_miss(self, profile_key: str, debug: dict[str, Any]) -> int:
        state = self._order_scan_state.setdefault(profile_key, {"two_stage_miss_count": 0})
        state["two_stage_miss_count"] = int(state.get("two_stage_miss_count", 0)) + 1
        debug["two_stage_miss_count"] = int(state["two_stage_miss_count"])
        return int(state["two_stage_miss_count"])

    def _reset_order_scan_miss(self, profile_key: str) -> None:
        state = self._order_scan_state.setdefault(profile_key, {"two_stage_miss_count": 0})
        state["two_stage_miss_count"] = 0

    def _set_order_scan_debug(
        self,
        profile_key: str,
        debug: dict[str, Any],
        *,
        scan_start: float,
        orders_returned: int,
        reset_miss_count: bool = False,
    ) -> None:
        if reset_miss_count:
            self._reset_order_scan_miss(profile_key)
            debug["two_stage_miss_count"] = 0

        debug["profile_name"] = profile_key
        debug["orders_returned"] = int(orders_returned)
        debug["total_sec"] = time.perf_counter() - scan_start
        self._last_order_scan_debug = dict(debug)

    def _sort_orders_for_selection(
        self,
        orders: list[dict[str, Any]],
        *,
        profile_key: str,
        profile: dict[str, Any],
        debug: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if str(profile.get("order_sort_strategy") or "fifo") == "left_to_right":
            debug["order_sort_strategy"] = "left_to_right"
            kept = sorted(orders, key=lambda item: (float(item["center_x"]), float(item["center_y"])))
            debug["order_track_count"] = len(self._order_arrival_state.get(profile_key, {}).get("tracks", {}))
            return kept

        debug["order_sort_strategy"] = "fifo"
        return self._sort_orders_by_arrival(orders, profile_key=profile_key, profile=profile, debug=debug)

    def _sort_orders_by_arrival(
        self,
        orders: list[dict[str, Any]],
        *,
        profile_key: str,
        profile: dict[str, Any],
        debug: dict[str, Any],
    ) -> list[dict[str, Any]]:
        state = self._order_arrival_state.setdefault(
            profile_key,
            {
                "scan_index": 0,
                "next_track_id": 1,
                "next_arrival_index": 1,
                "tracks": {},
            },
        )
        state["scan_index"] = int(state.get("scan_index", 0)) + 1
        scan_index = int(state["scan_index"])
        tracks: dict[int, dict[str, Any]] = {
            int(track_id): dict(track)
            for track_id, track in dict(state.get("tracks") or {}).items()
        }
        self._prune_order_tracks(tracks, scan_index=scan_index, profile=profile)

        track_distance_sq = float(int(profile["order_track_distance_px"]) ** 2)
        unmatched_track_ids = set(tracks.keys())
        decorated: list[dict[str, Any]] = []
        for order in sorted(orders, key=lambda item: (float(item["center_x"]), float(item["center_y"]))):
            center_x = float(order["center_x"])
            center_y = float(order["center_y"])
            best_track_id: int | None = None
            best_distance_sq: float | None = None
            for track_id in list(unmatched_track_ids):
                track = tracks[track_id]
                dx = center_x - float(track["center_x"])
                dy = center_y - float(track["center_y"])
                distance_sq = dx * dx + dy * dy
                if distance_sq > track_distance_sq:
                    continue
                if best_distance_sq is None or distance_sq < best_distance_sq:
                    best_track_id = track_id
                    best_distance_sq = distance_sq

            if best_track_id is None:
                best_track_id = int(state.get("next_track_id", 1))
                state["next_track_id"] = best_track_id + 1
                tracks[best_track_id] = {
                    "track_id": best_track_id,
                    "first_seen_scan": scan_index,
                    "first_seen_order": int(state.get("next_arrival_index", 1)),
                    "last_seen_scan": scan_index,
                    "center_x": center_x,
                    "center_y": center_y,
                    "recipe_id": str(order.get("recipe_id") or ""),
                }
                state["next_arrival_index"] = int(state.get("next_arrival_index", 1)) + 1
            else:
                unmatched_track_ids.discard(best_track_id)

            track = tracks[best_track_id]
            track["last_seen_scan"] = scan_index
            track["center_x"] = center_x
            track["center_y"] = center_y
            track["recipe_id"] = str(order.get("recipe_id") or "")
            order = dict(order)
            order["track_id"] = int(best_track_id)
            order["arrival_index"] = int(track["first_seen_order"])
            order["first_seen_scan"] = int(track["first_seen_scan"])
            order["last_seen_scan"] = int(track["last_seen_scan"])
            decorated.append(order)

        state["tracks"] = tracks
        self._order_arrival_state[profile_key] = state
        decorated.sort(
            key=lambda item: (
                int(item.get("arrival_index", 0)),
                float(item["center_x"]),
                float(item["center_y"]),
            )
        )
        debug["order_track_count"] = len(tracks)
        debug["returned_track_ids"] = [int(item["track_id"]) for item in decorated]
        debug["returned_arrival_indices"] = [int(item["arrival_index"]) for item in decorated]
        return decorated

    def _age_order_tracks(self, profile_key: str, *, profile: dict[str, Any], debug: dict[str, Any]) -> None:
        state = self._order_arrival_state.get(profile_key)
        if not state:
            debug["order_sort_strategy"] = str(profile.get("order_sort_strategy") or "fifo")
            debug["order_track_count"] = 0
            return

        state["scan_index"] = int(state.get("scan_index", 0)) + 1
        tracks: dict[int, dict[str, Any]] = {
            int(track_id): dict(track)
            for track_id, track in dict(state.get("tracks") or {}).items()
        }
        self._prune_order_tracks(tracks, scan_index=int(state["scan_index"]), profile=profile)
        state["tracks"] = tracks
        self._order_arrival_state[profile_key] = state
        debug["order_sort_strategy"] = str(profile.get("order_sort_strategy") or "fifo")
        debug["order_track_count"] = len(tracks)

    @staticmethod
    def _prune_order_tracks(
        tracks: dict[int, dict[str, Any]],
        *,
        scan_index: int,
        profile: dict[str, Any],
    ) -> None:
        forget_after = int(profile["order_track_forget_after_scans"])
        stale_track_ids = [
            track_id
            for track_id, track in tracks.items()
            if int(scan_index) - int(track.get("last_seen_scan", 0)) > forget_after
        ]
        for track_id in stale_track_ids:
            tracks.pop(track_id, None)

    def detect_order_candidates(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
    ) -> list[dict[str, Any]]:
        profile = self.load_profile(profile_name)
        region = profile["order_locator_region"]
        crop = self._crop_region(source_image, region)
        if crop.size == 0:
            return []

        crop_rgb = self._ensure_rgb(crop)
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=float(profile["order_locator_hough_dp"]),
            minDist=float(profile["order_locator_hough_min_dist"]),
            param1=float(profile["order_locator_hough_param1"]),
            param2=float(profile["order_locator_hough_param2"]),
            minRadius=int(profile["order_locator_min_radius"]),
            maxRadius=int(profile["order_locator_max_radius"]),
        )
        if circles is None:
            return []

        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        yellow_mask = cv2.inRange(
            hsv,
            np.array(profile["order_locator_yellow_hsv_lower"], dtype=np.uint8),
            np.array(profile["order_locator_yellow_hsv_upper"], dtype=np.uint8),
        )

        region_x, region_y, _, _ = region
        candidates: list[dict[str, Any]] = []
        for raw_x, raw_y, raw_radius in np.round(circles[0, :]).astype(int).tolist():
            radius = int(raw_radius)
            yellow_pixels = self._count_annulus_pixels(
                yellow_mask,
                center_x=int(raw_x),
                center_y=int(raw_y),
                radius=radius,
                inner_pad=int(profile["order_locator_yellow_inner_pad"]),
                outer_pad=int(profile["order_locator_yellow_outer_pad"]),
            )
            if yellow_pixels < int(profile["order_locator_min_yellow_pixels"]):
                continue

            center_x = int(region_x + raw_x)
            center_y = int(region_y + raw_y)
            candidates.append(
                {
                    "score": float(yellow_pixels),
                    "center_x": center_x,
                    "center_y": center_y,
                    "radius": radius,
                    "yellow_pixels": yellow_pixels,
                    "bbox": [center_x - radius, center_y - radius, radius * 2, radius * 2],
                }
            )

        kept = self._nms_by_center_distance(candidates, int(profile["order_locator_nms_distance_px"]))
        kept.sort(key=lambda item: (float(item["center_x"]), float(item["center_y"])))
        return kept

    def _analyze_orders_full_scan(
        self,
        source_image: np.ndarray,
        *,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        region = profile["order_search_region"]
        crop = self._crop_region(source_image, region)
        if crop.size == 0:
            return []

        crop_gray = cv2.cvtColor(self._ensure_rgb(crop), cv2.COLOR_RGB2GRAY)
        candidates: list[dict[str, Any]] = []
        for recipe_id, template_name in profile["order_templates"].items():
            template, mask = self._load_template_gray(template_name)
            candidates.extend(
                self._match_template_multiscale_gray(
                    crop_gray,
                    template,
                    mask,
                    recipe_id=recipe_id,
                    template_name=template_name,
                    region=region,
                    profile=profile,
                )
            )

        kept = self._nms_by_center_distance(candidates, int(profile["match_nms_distance_px"]))
        kept.sort(key=lambda item: (float(item["center_x"]), float(item["center_y"])))
        return kept

    def _classify_order_candidate(
        self,
        source_image: np.ndarray,
        locator: dict[str, Any],
        *,
        profile: dict[str, Any],
    ) -> dict[str, Any] | None:
        region_width, region_height = profile["order_classify_region_size"]
        center_x = int(locator["center_x"])
        center_y = int(locator["center_y"])
        region = (
            int(round(center_x - region_width / 2.0)),
            int(round(center_y - region_height / 2.0)),
            int(region_width),
            int(region_height),
        )
        crop = self._crop_region(source_image, region)
        if crop.size == 0:
            return None

        best: dict[str, Any] | None = None
        crop_gray = cv2.cvtColor(self._ensure_rgb(crop), cv2.COLOR_RGB2GRAY)
        for recipe_id, template_name in profile["order_templates"].items():
            template, mask = self._load_template_gray(template_name)
            match = self._best_template_match_multiscale(
                crop_gray,
                template,
                mask,
                recipe_id=recipe_id,
                template_name=template_name,
                region=region,
                profile=profile,
            )
            if match is None:
                continue
            if best is None or float(match["score"]) > float(best["score"]):
                best = match

        if best is None:
            return None

        best.update(
            {
                "center_x": int(locator["center_x"]),
                "center_y": int(locator["center_y"]),
                "locator_radius": int(locator["radius"]),
                "locator_yellow_pixels": int(locator["yellow_pixels"]),
                "locator_bbox": list(locator["bbox"]),
            }
        )
        return best

    def detect_level_started(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        region = profile["level_started_region"]
        crop = self._crop_region(source_image, region)
        if crop.size == 0:
            return {
                "started": False,
                "reason": "empty_region",
                "green_area": 0,
                "bbox": None,
                "aspect_ratio": 0.0,
            }

        if crop.ndim == 2:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        else:
            crop_rgb = crop[:, :, :3]
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        lower = np.array(profile["level_started_hsv_lower"], dtype=np.uint8)
        upper = np.array(profile["level_started_hsv_upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if component_count <= 1:
            return {
                "started": False,
                "reason": "green_component_missing",
                "green_area": 0,
                "bbox": None,
                "aspect_ratio": 0.0,
            }

        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        left = int(stats[best_label, cv2.CC_STAT_LEFT])
        top = int(stats[best_label, cv2.CC_STAT_TOP])
        width = int(stats[best_label, cv2.CC_STAT_WIDTH])
        height = int(stats[best_label, cv2.CC_STAT_HEIGHT])
        area = int(stats[best_label, cv2.CC_STAT_AREA])
        aspect_ratio = float(width) / max(float(height), 1.0)
        region_x, region_y, _, _ = region
        bbox = [region_x + left, region_y + top, width, height]

        min_area = int(profile["level_started_min_area"])
        min_width = int(profile["level_started_min_width"])
        min_aspect_ratio = float(profile["level_started_min_aspect_ratio"])
        if area < min_area:
            reason = "area_too_small"
        elif width < min_width:
            reason = "width_too_small"
        elif aspect_ratio < min_aspect_ratio:
            reason = "aspect_ratio_too_small"
        else:
            reason = "ok"

        return {
            "started": reason == "ok",
            "reason": reason,
            "green_area": area,
            "bbox": bbox,
            "aspect_ratio": round(aspect_ratio, 3),
        }

    def detect_level_end(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        title_crop = self._crop_region(source_image, profile["level_end_title_region"])
        buttons_crop = self._crop_region(source_image, profile["level_end_buttons_region"])
        if title_crop.size == 0 or buttons_crop.size == 0:
            return {
                "ended": False,
                "outcome": None,
                "reason": "empty_region",
                "success_title": self._empty_component_detection(),
                "failure_title": self._empty_component_detection(),
                "buttons_count": 0,
                "button_bboxes": [],
            }

        success_title = self._detect_colored_component(
            title_crop,
            profile["level_end_title_region"],
            lower=profile["level_end_success_hsv_lower"],
            upper=profile["level_end_success_hsv_upper"],
            min_area=int(profile["level_end_success_min_area"]),
            min_width=int(profile["level_end_success_min_width"]),
            min_height=int(profile["level_end_success_min_height"]),
        )
        failure_title = self._detect_red_component(
            title_crop,
            profile["level_end_title_region"],
            lower_1=profile["level_end_failure_hsv_lower_1"],
            upper_1=profile["level_end_failure_hsv_upper_1"],
            lower_2=profile["level_end_failure_hsv_lower_2"],
            upper_2=profile["level_end_failure_hsv_upper_2"],
            min_area=int(profile["level_end_failure_min_area"]),
            min_width=int(profile["level_end_failure_min_width"]),
            min_height=int(profile["level_end_failure_min_height"]),
        )
        buttons = self._detect_button_components(buttons_crop, profile)
        buttons_ok = len(buttons) >= int(profile["level_end_button_min_count"])

        outcome: str | None = None
        if buttons_ok and bool(success_title["detected"]):
            outcome = "success"
        elif buttons_ok and bool(failure_title["detected"]):
            outcome = "failure"
        elif buttons_ok:
            outcome = "unknown"

        if not buttons_ok:
            reason = "buttons_missing"
        elif outcome == "unknown":
            reason = "title_missing"
        else:
            reason = "ok"

        return {
            "ended": outcome is not None,
            "outcome": outcome,
            "reason": reason,
            "success_title": success_title,
            "failure_title": failure_title,
            "buttons_count": len(buttons),
            "button_bboxes": [button["bbox"] for button in buttons],
        }

    def detect_fake_customers(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        if not bool(profile.get("fake_customer_enabled", True)):
            return {
                "detected": False,
                "reason": "disabled",
                "region": list(profile.get("fake_customer_region") or (360, 220, 570, 70)),
                "red_pixels": 0,
                "component_count": 0,
                "kept_count": 0,
                "candidates": [],
            }

        region = self._coerce_region(
            profile.get("fake_customer_region"),
            default=(360, 220, 570, 70),
        )
        crop = self._crop_region(source_image, region)
        if crop.size == 0:
            return {
                "detected": False,
                "reason": "empty_region",
                "region": list(region),
                "red_pixels": 0,
                "component_count": 0,
                "kept_count": 0,
                "candidates": [],
            }

        crop_rgb = self._ensure_rgb(crop)
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        mask = self._red_mask_for_hsv(hsv, profile)
        red_pixels = int(np.count_nonzero(mask))
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        region_x, region_y, _, _ = region
        min_area = int(profile.get("fake_customer_min_component_area", 650) or 650)
        max_area = int(profile.get("fake_customer_max_component_area", 2000) or 2000)
        min_width = int(profile.get("fake_customer_min_component_width", 25) or 25)
        max_width = int(profile.get("fake_customer_max_component_width", 75) or 75)
        min_height = int(profile.get("fake_customer_min_component_height", 30) or 30)
        max_height = int(profile.get("fake_customer_max_component_height", 45) or 45)
        min_aspect_ratio = float(profile.get("fake_customer_min_aspect_ratio", 0.6) or 0.6)
        max_fill_ratio = float(profile.get("fake_customer_max_fill_ratio", 0.9) or 0.9)
        min_center_x, max_center_x = sorted(
            (
                int(profile.get("fake_customer_center_x_min", 0) or 0),
                int(profile.get("fake_customer_center_x_max", 1280) or 1280),
            )
        )
        min_center_y, max_center_y = sorted(
            (
                int(profile.get("fake_customer_center_y_min", 235) or 235),
                int(profile.get("fake_customer_center_y_max", 315) or 315),
            )
        )

        candidates: list[dict[str, Any]] = []
        reject_counts: dict[str, int] = {}

        def count_reject(reason: str) -> None:
            reject_counts[reason] = int(reject_counts.get(reason, 0)) + 1

        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            center_x = float(region_x + centroids[label][0])
            center_y = float(region_y + centroids[label][1])

            if area < min_area or area > max_area:
                count_reject("area_out_of_range")
                continue
            if width < min_width or width > max_width:
                count_reject("width_out_of_range")
                continue
            if self._is_fake_customer_hammer_animation_component(
                top=top,
                height=height,
                profile=profile,
            ):
                count_reject("hammer_animation_top_clipped")
                continue
            if height < min_height or height > max_height:
                count_reject("height_out_of_range")
                continue
            if center_x < min_center_x or center_x > max_center_x:
                count_reject("center_x_out_of_range")
                continue
            if center_y < min_center_y or center_y > max_center_y:
                count_reject("center_y_out_of_range")
                continue

            aspect_ratio = float(height) / max(float(width), 1.0)
            fill_ratio = float(area) / max(float(width * height), 1.0)
            if aspect_ratio < min_aspect_ratio:
                count_reject("aspect_ratio_too_small")
                continue
            if max_fill_ratio > 0 and fill_ratio > max_fill_ratio:
                count_reject("fill_ratio_too_large")
                continue

            candidates.append(
                {
                    "score": round(float(area), 3),
                    "area": area,
                    "aspect_ratio": round(aspect_ratio, 4),
                    "fill_ratio": round(fill_ratio, 4),
                    "center_x": int(round(center_x)),
                    "center_y": int(round(center_y)),
                    "bbox": [region_x + left, region_y + top, width, height],
                    "source": "color_component",
                }
            )

        candidates = self._nms_by_center_distance(
            candidates,
            int(profile.get("fake_customer_nms_distance_px", 70) or 70),
        )
        candidates.sort(key=lambda item: (int(item["center_x"]), int(item["center_y"])))
        reason = "ok" if candidates else "no_candidate"
        return {
            "detected": bool(candidates),
            "reason": reason,
            "region": list(region),
            "red_pixels": red_pixels,
            "component_count": max(component_count - 1, 0),
            "kept_count": len(candidates),
            "reject_counts": reject_counts,
            "candidates": candidates,
        }

    def save_fake_customer_hammer_debug_image(
        self,
        source_image: np.ndarray,
        detection: dict[str, Any],
        *,
        profile_name: str | None = None,
        sequence: int | None = None,
        output_dir: str | Path | None = None,
    ) -> str | None:
        """Save the exact frame that triggered a hammer click plus its red HSV mask."""
        if source_image is None or source_image.size == 0:
            return None

        profile = self.load_profile(profile_name)
        region = self._coerce_region(
            profile.get("fake_customer_region"),
            default=(360, 220, 570, 70),
        )
        image_rgb = np.ascontiguousarray(self._ensure_rgb(source_image).copy())
        crop = self._crop_region(image_rgb, region)
        if crop.size == 0:
            return None

        crop_rgb = self._ensure_rgb(crop)
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        mask = self._red_mask_for_hsv(hsv, profile)

        original_panel = image_rgb.copy()
        mask_panel = np.zeros_like(image_rgb)
        x, y, width, height = region
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_panel[y : y + crop_rgb.shape[0], x : x + crop_rgb.shape[1]] = mask_rgb

        roi_color = (180, 0, 255)
        candidate_color = (255, 0, 0)
        cv2.rectangle(original_panel, (x, y), (x + width, y + height), roi_color, 2)
        cv2.rectangle(mask_panel, (x, y), (x + width, y + height), roi_color, 2)

        for candidate in list(detection.get("candidates") or []):
            bbox = candidate.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            left, top, box_width, box_height = [int(value) for value in bbox[:4]]
            right = left + box_width
            bottom = top + box_height
            center_x = int(candidate.get("center_x", left + box_width // 2) or (left + box_width // 2))
            center_y = int(candidate.get("center_y", top + box_height // 2) or (top + box_height // 2))
            for panel in (original_panel, mask_panel):
                cv2.rectangle(panel, (left, top), (right, bottom), candidate_color, 2)
                cv2.line(panel, (center_x - 8, center_y), (center_x + 8, center_y), candidate_color, 1)
                cv2.line(panel, (center_x, center_y - 8), (center_x, center_y + 8), candidate_color, 1)

        cv2.putText(original_panel, "original", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(mask_panel, "hsv red mask", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        combined = np.concatenate([original_panel, mask_panel], axis=1)

        raw_output_dir = Path(output_dir or profile.get("fake_customer_hammer_debug_dir") or "tmp/cafe_fake_customer_hammers")
        if not raw_output_dir.is_absolute():
            repo_root = self._plan_root.parents[1]
            raw_output_dir = repo_root / raw_output_dir
        raw_output_dir.mkdir(parents=True, exist_ok=True)

        index = 0 if sequence is None else int(sequence)
        filename = f"hammer_{time.strftime('%Y%m%d-%H%M%S')}_{index:03d}.png"
        output_path = raw_output_dir / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        return str(output_path)

    def save_order_decision_debug_image(
        self,
        source_image: np.ndarray,
        orders: list[dict[str, Any]],
        fake_detection: dict[str, Any],
        *,
        selected_order: dict[str, Any] | None = None,
        profile_name: str | None = None,
        sequence: int | None = None,
        output_dir: str | Path | None = None,
    ) -> str | None:
        """Save the exact frame used for order selection plus the fake-customer HSV mask."""
        if source_image is None or source_image.size == 0:
            return None

        profile = self.load_profile(profile_name)
        fake_region = self._coerce_region(
            profile.get("fake_customer_region"),
            default=(360, 220, 570, 70),
        )
        order_region = self._coerce_region(
            profile.get("order_search_region"),
            default=(220, 60, 720, 250),
        )
        image_rgb = np.ascontiguousarray(self._ensure_rgb(source_image).copy())
        fake_crop = self._crop_region(image_rgb, fake_region)
        if fake_crop.size == 0:
            return None

        hsv = cv2.cvtColor(self._ensure_rgb(fake_crop), cv2.COLOR_RGB2HSV)
        mask = self._red_mask_for_hsv(hsv, profile)

        original_panel = image_rgb.copy()
        mask_panel = np.zeros_like(image_rgb)
        fake_x, fake_y, fake_width, fake_height = fake_region
        order_x, order_y, order_width, order_height = order_region
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_panel[fake_y : fake_y + fake_crop.shape[0], fake_x : fake_x + fake_crop.shape[1]] = mask_rgb

        fake_roi_color = (180, 0, 255)
        order_roi_color = (0, 200, 255)
        fake_color = (255, 0, 0)
        order_color = (0, 255, 255)
        selected_color = (255, 255, 0)
        for panel in (original_panel, mask_panel):
            cv2.rectangle(
                panel,
                (fake_x, fake_y),
                (fake_x + fake_width, fake_y + fake_height),
                fake_roi_color,
                2,
            )
            cv2.rectangle(
                panel,
                (order_x, order_y),
                (order_x + order_width, order_y + order_height),
                order_roi_color,
                2,
            )

        def draw_center(panel: np.ndarray, center_x: int, center_y: int, color: tuple[int, int, int]) -> None:
            cv2.line(panel, (center_x - 9, center_y), (center_x + 9, center_y), color, 2)
            cv2.line(panel, (center_x, center_y - 9), (center_x, center_y + 9), color, 2)

        def draw_bbox_or_center(panel: np.ndarray, item: dict[str, Any], color: tuple[int, int, int]) -> None:
            bbox = item.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                left, top, width, height = [int(value) for value in bbox[:4]]
                cv2.rectangle(panel, (left, top), (left + width, top + height), color, 2)
            center_x = int(item.get("center_x", 0) or 0)
            center_y = int(item.get("center_y", 0) or 0)
            if center_x > 0 and center_y > 0:
                draw_center(panel, center_x, center_y, color)

        for candidate in list(fake_detection.get("all_candidates") or fake_detection.get("candidates") or []):
            for panel in (original_panel, mask_panel):
                draw_bbox_or_center(panel, candidate, fake_color)

        selected_track_id = None if selected_order is None else selected_order.get("track_id")
        selected_center = None
        if selected_order is not None:
            selected_center = (
                int(selected_order.get("center_x", 0) or 0),
                int(selected_order.get("center_y", 0) or 0),
            )

        for order in orders:
            is_selected = False
            if selected_order is not None:
                is_selected = order is selected_order
                if not is_selected and selected_track_id is not None:
                    is_selected = order.get("track_id") == selected_track_id
                if not is_selected and selected_center is not None:
                    is_selected = (
                        int(order.get("center_x", 0) or 0),
                        int(order.get("center_y", 0) or 0),
                    ) == selected_center
            color = selected_color if is_selected else order_color
            for panel in (original_panel, mask_panel):
                draw_bbox_or_center(panel, order, color)

        reason = str(fake_detection.get("reason") or ("detected" if fake_detection.get("detected") else "none"))
        order_label = "none" if selected_order is None else str(selected_order.get("recipe_id") or "unknown")
        cv2.putText(
            original_panel,
            f"original fake={reason} selected={order_label}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            mask_panel,
            "fake red mask + order markers",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        combined = np.concatenate([original_panel, mask_panel], axis=1)

        raw_output_dir = Path(output_dir or profile.get("order_decision_debug_dir") or "tmp/cafe_order_decisions")
        if not raw_output_dir.is_absolute():
            repo_root = self._plan_root.parents[1]
            raw_output_dir = repo_root / raw_output_dir
        raw_output_dir.mkdir(parents=True, exist_ok=True)

        index = 0 if sequence is None else int(sequence)
        filename = f"order_decision_{time.strftime('%Y%m%d-%H%M%S')}_{index:03d}.png"
        output_path = raw_output_dir / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        return str(output_path)

    def _fake_customer_body_region_from_order(
        self,
        locator: dict[str, Any],
        profile: dict[str, Any],
    ) -> Region:
        offset_x, offset_y, width, height = profile["fake_customer_body_region_from_order"]
        center_x = int(locator.get("center_x", 0) or 0)
        center_y = int(locator.get("center_y", 0) or 0)
        return (
            max(center_x + int(offset_x), 0),
            max(center_y + int(offset_y), 0),
            int(width),
            int(height),
        )

    def _is_fake_customer_hammer_animation_component(
        self,
        *,
        top: int,
        height: int,
        profile: dict[str, Any],
    ) -> bool:
        top_margin_px = int(profile.get("fake_customer_hammer_clip_top_margin_px", 18) or 18)
        min_height_px = int(profile.get("fake_customer_hammer_clip_min_height_px", 48) or 48)
        return int(top) <= top_margin_px and int(height) >= min_height_px

    def _fake_customer_fallback_slots_from_red(
        self,
        source_image: np.ndarray,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        region = self._coerce_region(profile.get("fake_customer_region"), default=(360, 220, 570, 70))
        crop = self._crop_region(source_image, region)
        if crop.size == 0:
            return []

        crop_rgb = self._ensure_rgb(crop)
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        mask = self._red_mask_for_hsv(hsv, profile)
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if component_count <= 1:
            return []

        region_x, region_y, _, _ = region
        min_area = int(profile.get("fake_customer_min_component_area", 650) or 650)
        max_area = int(profile.get("fake_customer_max_component_area", 2000) or 2000)
        min_width = int(profile.get("fake_customer_min_component_width", 25) or 25)
        max_width = int(profile.get("fake_customer_max_component_width", 75) or 75)
        min_height = int(profile.get("fake_customer_min_component_height", 30) or 30)
        max_height = int(profile.get("fake_customer_max_component_height", 45) or 45)
        min_aspect_ratio = float(profile.get("fake_customer_min_aspect_ratio", 0.6) or 0.6)
        max_fill_ratio = float(profile.get("fake_customer_max_fill_ratio", 0.9) or 0.9)
        _, _, slot_width, slot_height = profile["fake_customer_body_region_from_order"]
        slots: list[dict[str, Any]] = []
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < min_area or area > max_area:
                continue
            if width < min_width or width > max_width:
                continue
            if self._is_fake_customer_hammer_animation_component(
                top=top,
                height=height,
                profile=profile,
            ):
                continue
            if height < min_height or height > max_height:
                continue
            aspect_ratio = float(height) / max(float(width), 1.0)
            fill_ratio = float(area) / max(float(width * height), 1.0)
            if aspect_ratio < min_aspect_ratio:
                continue
            if max_fill_ratio > 0 and fill_ratio > max_fill_ratio:
                continue

            center_x = int(round(region_x + float(centroids[label][0])))
            body_left = max(int(round(center_x - int(slot_width) / 2.0)), 0)
            body_top = max(region_y + top - int(round(int(slot_height) * 0.26)), 0)
            slots.append(
                {
                    "source": "red_fallback",
                    "region": (body_left, body_top, int(slot_width), int(slot_height)),
                }
            )

        return slots

    def _detect_fake_customer_features_in_region(
        self,
        source_image: np.ndarray,
        region: Region,
        profile: dict[str, Any],
        *,
        source: str,
        order_locator: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        crop = self._crop_region(source_image, region)
        debug: dict[str, Any] = {
            "source": source,
            "region": list(region),
            "reason": "empty_region",
            "red_pixels": 0,
            "blue_pixels": 0,
            "dark_pixels": 0,
            "skin_pixels": 0,
        }
        if order_locator is not None:
            debug["order_center_x"] = int(order_locator.get("center_x", 0) or 0)
            debug["order_center_y"] = int(order_locator.get("center_y", 0) or 0)
        if crop.size == 0:
            return None, debug

        crop_rgb = self._ensure_rgb(crop)
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        red_mask = self._red_mask_for_hsv(hsv, profile)
        blue_mask = cv2.inRange(
            hsv,
            np.array(profile["fake_customer_blue_hsv_lower"], dtype=np.uint8),
            np.array(profile["fake_customer_blue_hsv_upper"], dtype=np.uint8),
        )
        dark_mask = cv2.inRange(
            hsv,
            np.array(profile["fake_customer_dark_hsv_lower"], dtype=np.uint8),
            np.array(profile["fake_customer_dark_hsv_upper"], dtype=np.uint8),
        )
        skin_mask = cv2.inRange(
            hsv,
            np.array(profile["fake_customer_skin_hsv_lower"], dtype=np.uint8),
            np.array(profile["fake_customer_skin_hsv_upper"], dtype=np.uint8),
        )

        red_bounds = self._ratio_bounds(red_mask.shape[1], red_mask.shape[0], profile["fake_customer_red_check_region_ratio"])
        blue_bounds = self._ratio_bounds(
            blue_mask.shape[1],
            blue_mask.shape[0],
            profile["fake_customer_blue_check_region_ratio"],
        )
        dark_bounds = self._ratio_bounds(
            dark_mask.shape[1],
            dark_mask.shape[0],
            profile["fake_customer_dark_check_region_ratio"],
        )
        skin_bounds = self._ratio_bounds(
            skin_mask.shape[1],
            skin_mask.shape[0],
            profile["fake_customer_skin_check_region_ratio"],
        )

        red_crop = self._slice_mask(red_mask, red_bounds)
        blue_crop = self._slice_mask(blue_mask, blue_bounds)
        dark_crop = self._slice_mask(dark_mask, dark_bounds)
        skin_crop = self._slice_mask(skin_mask, skin_bounds)
        red_pixels = int(np.count_nonzero(red_crop))
        blue_pixels = int(np.count_nonzero(blue_crop))
        dark_pixels = int(np.count_nonzero(dark_crop))
        skin_pixels = int(np.count_nonzero(skin_crop))
        debug.update(
            {
                "reason": "checked",
                "red_pixels": red_pixels,
                "blue_pixels": blue_pixels,
                "dark_pixels": dark_pixels,
                "skin_pixels": skin_pixels,
                "red_region": list(red_bounds),
                "blue_region": list(blue_bounds),
            }
        )

        red_component = self._largest_component(red_crop, offset_x=red_bounds[0], offset_y=red_bounds[1])
        if red_component is None:
            debug["reason"] = "red_component_missing"
            return None, debug
        debug["red_component"] = red_component

        if red_pixels < int(profile["fake_customer_min_red_pixels"]):
            debug["reason"] = "red_pixels_too_few"
            return None, debug
        if int(red_component["area"]) < int(profile.get("fake_customer_min_component_area", 650) or 650):
            debug["reason"] = "red_component_too_small"
            return None, debug
        if blue_pixels < int(profile["fake_customer_min_blue_pixels"]):
            debug["reason"] = "blue_pixels_too_few"
            return None, debug
        if dark_pixels < int(profile["fake_customer_min_dark_pixels"]):
            debug["reason"] = "dark_pixels_too_few"
            return None, debug
        if skin_pixels < int(profile["fake_customer_min_skin_pixels"]):
            debug["reason"] = "skin_pixels_too_few"
            return None, debug

        blue_center = self._mask_center(blue_crop, offset_x=blue_bounds[0], offset_y=blue_bounds[1])
        if blue_center is None:
            debug["reason"] = "blue_component_missing"
            return None, debug
        red_center_y = float(red_component["center_y"])
        blue_center_y = float(blue_center[1])
        vertical_gap = blue_center_y - red_center_y
        debug["red_blue_vertical_gap_px"] = round(vertical_gap, 2)
        if vertical_gap < int(profile["fake_customer_min_red_blue_vertical_gap_px"]):
            debug["reason"] = "red_blue_layout_invalid"
            return None, debug

        region_x, region_y, _, _ = region
        red_bbox = list(red_component["bbox"])
        abs_red_bbox = [region_x + red_bbox[0], region_y + red_bbox[1], red_bbox[2], red_bbox[3]]
        center_x = int(round(region_x + float(red_component["center_x"])))
        center_y = int(round(region_y + float(red_component["center_y"])))
        score = (
            float(red_component["area"])
            + min(float(blue_pixels), 2500.0) * 0.35
            + min(float(dark_pixels), 12000.0) * 0.04
            + min(float(skin_pixels), 3500.0) * 0.04
        )
        candidate = {
            "score": round(score, 3),
            "area": int(red_component["area"]),
            "center_x": center_x,
            "center_y": center_y,
            "bbox": abs_red_bbox,
            "body_bbox": list(region),
            "source": source,
            "red_pixels": red_pixels,
            "blue_pixels": blue_pixels,
            "dark_pixels": dark_pixels,
            "skin_pixels": skin_pixels,
            "red_blue_vertical_gap_px": round(vertical_gap, 2),
        }
        if order_locator is not None:
            candidate["order_center_x"] = int(order_locator.get("center_x", 0) or 0)
            candidate["order_center_y"] = int(order_locator.get("center_y", 0) or 0)
        return candidate, {**debug, "reason": "ok"}

    def _red_mask_for_hsv(self, hsv: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
        mask_1 = cv2.inRange(
            hsv,
            np.array(profile.get("fake_customer_red_hsv_lower_1", (0, 145, 90)), dtype=np.uint8),
            np.array(profile.get("fake_customer_red_hsv_upper_1", (5, 255, 255)), dtype=np.uint8),
        )
        mask_2 = cv2.inRange(
            hsv,
            np.array(profile.get("fake_customer_red_hsv_lower_2", (176, 145, 90)), dtype=np.uint8),
            np.array(profile.get("fake_customer_red_hsv_upper_2", (179, 255, 255)), dtype=np.uint8),
        )
        return cv2.bitwise_or(mask_1, mask_2)

    @staticmethod
    def _largest_component(mask: np.ndarray, *, offset_x: int = 0, offset_y: int = 0) -> dict[str, Any] | None:
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if component_count <= 1:
            return None
        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        left = int(stats[best_label, cv2.CC_STAT_LEFT])
        top = int(stats[best_label, cv2.CC_STAT_TOP])
        width = int(stats[best_label, cv2.CC_STAT_WIDTH])
        height = int(stats[best_label, cv2.CC_STAT_HEIGHT])
        area = int(stats[best_label, cv2.CC_STAT_AREA])
        center_x = float(centroids[best_label][0]) + float(offset_x)
        center_y = float(centroids[best_label][1]) + float(offset_y)
        return {
            "area": area,
            "bbox": [int(offset_x) + left, int(offset_y) + top, width, height],
            "center_x": round(center_x, 3),
            "center_y": round(center_y, 3),
            "fill_ratio": round(float(area) / max(float(width * height), 1.0), 4),
        }

    @staticmethod
    def _mask_center(mask: np.ndarray, *, offset_x: int = 0, offset_y: int = 0) -> tuple[float, float] | None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return float(np.mean(xs)) + float(offset_x), float(np.mean(ys)) + float(offset_y)

    @staticmethod
    def _ratio_bounds(width: int, height: int, ratio_region: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = ratio_region
        left = max(min(int(round(float(width) * x1)), width), 0)
        top = max(min(int(round(float(height) * y1)), height), 0)
        right = max(min(int(round(float(width) * x2)), width), left + 1)
        bottom = max(min(int(round(float(height) * y2)), height), top + 1)
        return left, top, right, bottom

    @staticmethod
    def _slice_mask(mask: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = bounds
        return mask[top:bottom, left:right]

    def _match_template_multiscale_gray(
        self,
        crop_gray: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray,
        *,
        recipe_id: str,
        template_name: str,
        region: Region,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        threshold = float(profile["template_threshold"])
        peak_window = int(profile["match_peak_window_px"])
        max_candidates = int(profile["max_candidates_per_template_scale"])
        region_x, region_y, _, _ = region
        candidates: list[dict[str, Any]] = []

        for scale in profile["template_scales"]:
            template_width = max(int(round(template.shape[1] * float(scale))), 4)
            template_height = max(int(round(template.shape[0] * float(scale))), 4)
            if template_width >= crop_gray.shape[1] or template_height >= crop_gray.shape[0]:
                continue

            scaled_template, scaled_mask = self._load_scaled_template_gray(
                template_name,
                scale=float(scale),
            )
            if int(np.count_nonzero(scaled_mask)) < 20:
                continue

            result = cv2.matchTemplate(
                crop_gray,
                scaled_template,
                cv2.TM_CCOEFF_NORMED,
                mask=scaled_mask,
            )
            result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
            peak_mask = result >= threshold
            if peak_window > 1:
                kernel = np.ones((peak_window, peak_window), dtype=np.uint8)
                peak_mask &= result == cv2.dilate(result, kernel)

            ys, xs = np.where(peak_mask)
            scale_candidates: list[dict[str, Any]] = []
            for y, x in zip(ys.tolist(), xs.tolist()):
                score = float(result[int(y), int(x)])
                bbox = [int(region_x + x), int(region_y + y), int(template_width), int(template_height)]
                scale_candidates.append(
                    {
                        "recipe_id": recipe_id,
                        "score": round(score, 4),
                        "center_x": int(round(bbox[0] + bbox[2] / 2.0)),
                        "center_y": int(round(bbox[1] + bbox[3] / 2.0)),
                        "bbox": bbox,
                        "scale": round(float(scale), 3),
                    }
                )

            scale_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
            candidates.extend(scale_candidates[:max_candidates])

        return candidates

    def _match_template_multiscale(
        self,
        crop: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray,
        *,
        recipe_id: str,
        region: Region,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        threshold = float(profile["template_threshold"])
        peak_window = int(profile["match_peak_window_px"])
        max_candidates = int(profile["max_candidates_per_template_scale"])
        region_x, region_y, _, _ = region
        candidates: list[dict[str, Any]] = []

        for scale in profile["template_scales"]:
            template_width = max(int(round(template.shape[1] * float(scale))), 4)
            template_height = max(int(round(template.shape[0] * float(scale))), 4)
            if template_width >= crop.shape[1] or template_height >= crop.shape[0]:
                continue

            interpolation = cv2.INTER_AREA if float(scale) < 1.0 else cv2.INTER_CUBIC
            scaled_template = cv2.resize(template, (template_width, template_height), interpolation=interpolation)
            scaled_mask = cv2.resize(mask, (template_width, template_height), interpolation=cv2.INTER_NEAREST)
            if int(np.count_nonzero(scaled_mask)) < 20:
                continue

            result = cv2.matchTemplate(
                crop,
                scaled_template,
                cv2.TM_CCOEFF_NORMED,
                mask=scaled_mask,
            )
            result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
            peak_mask = result >= threshold
            if peak_window > 1:
                kernel = np.ones((peak_window, peak_window), dtype=np.uint8)
                peak_mask &= result == cv2.dilate(result, kernel)

            ys, xs = np.where(peak_mask)
            scale_candidates: list[dict[str, Any]] = []
            for y, x in zip(ys.tolist(), xs.tolist()):
                score = float(result[int(y), int(x)])
                bbox = [int(region_x + x), int(region_y + y), int(template_width), int(template_height)]
                scale_candidates.append(
                    {
                        "recipe_id": recipe_id,
                        "score": round(score, 4),
                        "center_x": int(round(bbox[0] + bbox[2] / 2.0)),
                        "center_y": int(round(bbox[1] + bbox[3] / 2.0)),
                        "bbox": bbox,
                        "scale": round(float(scale), 3),
                    }
                )

            scale_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
            candidates.extend(scale_candidates[:max_candidates])

        return candidates

    def _best_template_match_multiscale(
        self,
        crop: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray,
        *,
        recipe_id: str,
        template_name: str | None = None,
        region: Region,
        profile: dict[str, Any],
    ) -> dict[str, Any] | None:
        threshold = float(profile["template_threshold"])
        region_x, region_y, _, _ = region
        best: dict[str, Any] | None = None

        for scale in profile["template_scales"]:
            template_width = max(int(round(template.shape[1] * float(scale))), 4)
            template_height = max(int(round(template.shape[0] * float(scale))), 4)
            if template_width >= crop.shape[1] or template_height >= crop.shape[0]:
                continue

            if template_name:
                scaled_template, scaled_mask = self._load_scaled_template_gray(
                    template_name,
                    scale=float(scale),
                )
            else:
                interpolation = cv2.INTER_AREA if float(scale) < 1.0 else cv2.INTER_CUBIC
                scaled_template = cv2.resize(template, (template_width, template_height), interpolation=interpolation)
                scaled_mask = cv2.resize(mask, (template_width, template_height), interpolation=cv2.INTER_NEAREST)
            if int(np.count_nonzero(scaled_mask)) < 20:
                continue

            result = cv2.matchTemplate(
                crop,
                scaled_template,
                cv2.TM_CCOEFF_NORMED,
                mask=scaled_mask,
            )
            result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, max_value, _, max_location = cv2.minMaxLoc(result)
            score = float(max_value)
            if score < threshold:
                continue

            left, top = int(max_location[0]), int(max_location[1])
            bbox = [int(region_x + left), int(region_y + top), int(template_width), int(template_height)]
            candidate = {
                "recipe_id": recipe_id,
                "score": round(score, 4),
                "center_x": int(round(bbox[0] + bbox[2] / 2.0)),
                "center_y": int(round(bbox[1] + bbox[3] / 2.0)),
                "bbox": bbox,
                "scale": round(float(scale), 3),
            }
            if best is None or float(candidate["score"]) > float(best["score"]):
                best = candidate

        return best

    def _load_template(self, template_name: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._template_cache.get(template_name)
        if cached is not None:
            return cached

        template_path = self._template_dir / template_name
        if not template_path.is_file():
            raise FileNotFoundError(f"Cafe order template not found: {template_path}")

        rgba = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise ValueError(f"Failed to load cafe order template: {template_path}")
        if rgba.ndim == 2:
            image = cv2.cvtColor(rgba, cv2.COLOR_GRAY2RGB)
            mask = np.full(rgba.shape, 255, dtype=np.uint8)
        elif rgba.shape[2] == 4:
            rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
            image = rgba[:, :, :3]
            mask = rgba[:, :, 3]
        else:
            image = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGB)
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)

        loaded = (np.ascontiguousarray(image), np.ascontiguousarray(mask))
        self._template_cache[template_name] = loaded
        return loaded

    def _load_template_gray(self, template_name: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._template_gray_cache.get(template_name)
        if cached is not None:
            return cached

        image, mask = self._load_template(template_name)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        loaded = (np.ascontiguousarray(gray), mask)
        self._template_gray_cache[template_name] = loaded
        return loaded

    def _load_scaled_template_gray(self, template_name: str, *, scale: float) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (template_name, round(float(scale), 4))
        cached = self._template_scaled_gray_cache.get(cache_key)
        if cached is not None:
            return cached

        template, mask = self._load_template_gray(template_name)
        template_width = max(int(round(template.shape[1] * float(scale))), 4)
        template_height = max(int(round(template.shape[0] * float(scale))), 4)
        interpolation = cv2.INTER_AREA if float(scale) < 1.0 else cv2.INTER_CUBIC
        scaled_template = cv2.resize(template, (template_width, template_height), interpolation=interpolation)
        scaled_mask = cv2.resize(mask, (template_width, template_height), interpolation=cv2.INTER_NEAREST)
        loaded = (np.ascontiguousarray(scaled_template), np.ascontiguousarray(scaled_mask))
        self._template_scaled_gray_cache[cache_key] = loaded
        return loaded

    @staticmethod
    def _empty_component_detection() -> dict[str, Any]:
        return {
            "detected": False,
            "area": 0,
            "bbox": None,
            "pixel_count": 0,
        }

    def _detect_colored_component(
        self,
        crop: np.ndarray,
        region: Region,
        *,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
        min_area: int,
        min_width: int,
        min_height: int,
    ) -> dict[str, Any]:
        if crop.size == 0:
            return self._empty_component_detection()
        crop_rgb = self._ensure_rgb(crop)
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        return self._best_component_detection(
            mask,
            region,
            min_area=min_area,
            min_width=min_width,
            min_height=min_height,
        )

    def _detect_red_component(
        self,
        crop: np.ndarray,
        region: Region,
        *,
        lower_1: tuple[int, int, int],
        upper_1: tuple[int, int, int],
        lower_2: tuple[int, int, int],
        upper_2: tuple[int, int, int],
        min_area: int,
        min_width: int,
        min_height: int,
    ) -> dict[str, Any]:
        if crop.size == 0:
            return self._empty_component_detection()
        crop_rgb = self._ensure_rgb(crop)
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        mask_1 = cv2.inRange(hsv, np.array(lower_1, dtype=np.uint8), np.array(upper_1, dtype=np.uint8))
        mask_2 = cv2.inRange(hsv, np.array(lower_2, dtype=np.uint8), np.array(upper_2, dtype=np.uint8))
        return self._best_component_detection(
            cv2.bitwise_or(mask_1, mask_2),
            region,
            min_area=min_area,
            min_width=min_width,
            min_height=min_height,
        )

    def _detect_button_components(self, crop: np.ndarray, profile: dict[str, Any]) -> list[dict[str, Any]]:
        if crop.size == 0:
            return []
        crop_rgb = self._ensure_rgb(crop)
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(profile["level_end_button_hsv_lower"], dtype=np.uint8),
            np.array(profile["level_end_button_hsv_upper"], dtype=np.uint8),
        )
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        region_x, region_y, _, _ = profile["level_end_buttons_region"]
        buttons: list[dict[str, Any]] = []
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < int(profile["level_end_button_min_area"]):
                continue
            if width < int(profile["level_end_button_min_width"]):
                continue
            if width > int(profile["level_end_button_max_width"]):
                continue
            if height < int(profile["level_end_button_min_height"]):
                continue
            if height > int(profile["level_end_button_max_height"]):
                continue
            buttons.append(
                {
                    "area": area,
                    "bbox": [region_x + left, region_y + top, width, height],
                }
            )
        buttons.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
        return buttons

    @staticmethod
    def _best_component_detection(
        mask: np.ndarray,
        region: Region,
        *,
        min_area: int,
        min_width: int,
        min_height: int,
    ) -> dict[str, Any]:
        pixel_count = int(np.count_nonzero(mask))
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if component_count <= 1:
            return {
                "detected": False,
                "area": 0,
                "bbox": None,
                "pixel_count": pixel_count,
            }

        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        left = int(stats[best_label, cv2.CC_STAT_LEFT])
        top = int(stats[best_label, cv2.CC_STAT_TOP])
        width = int(stats[best_label, cv2.CC_STAT_WIDTH])
        height = int(stats[best_label, cv2.CC_STAT_HEIGHT])
        area = int(stats[best_label, cv2.CC_STAT_AREA])
        region_x, region_y, _, _ = region
        bbox = [region_x + left, region_y + top, width, height]
        return {
            "detected": area >= min_area and width >= min_width and height >= min_height,
            "area": area,
            "bbox": bbox,
            "pixel_count": pixel_count,
        }

    @staticmethod
    def _ensure_rgb(crop: np.ndarray) -> np.ndarray:
        if crop.ndim == 2:
            return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        return crop[:, :, :3]

    @staticmethod
    def _nms_by_center_distance(candidates: list[dict[str, Any]], distance_px: int) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        distance_sq = float(distance_px * distance_px)
        for candidate in sorted(candidates, key=lambda item: float(item["score"]), reverse=True):
            candidate_x = float(candidate["center_x"])
            candidate_y = float(candidate["center_y"])
            duplicate = False
            for existing in kept:
                dx = candidate_x - float(existing["center_x"])
                dy = candidate_y - float(existing["center_y"])
                if dx * dx + dy * dy <= distance_sq:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    @staticmethod
    def _crop_region(source_image: np.ndarray, region: Region) -> np.ndarray:
        x, y, width, height = [int(value) for value in region]
        if source_image is None or source_image.size == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        image_height, image_width = source_image.shape[:2]
        left = max(min(x, image_width), 0)
        top = max(min(y, image_height), 0)
        right = max(min(x + width, image_width), left)
        bottom = max(min(y + height, image_height), top)
        return source_image[top:bottom, left:right]

    @staticmethod
    def _count_annulus_pixels(
        mask: np.ndarray,
        *,
        center_x: int,
        center_y: int,
        radius: int,
        inner_pad: int,
        outer_pad: int,
    ) -> int:
        if mask.size == 0:
            return 0

        outer_radius = max(int(radius) + int(outer_pad), 1)
        inner_radius = max(int(radius) - int(inner_pad), 0)
        left = max(int(center_x) - outer_radius, 0)
        top = max(int(center_y) - outer_radius, 0)
        right = min(int(center_x) + outer_radius + 1, mask.shape[1])
        bottom = min(int(center_y) + outer_radius + 1, mask.shape[0])
        if right <= left or bottom <= top:
            return 0

        local_mask = mask[top:bottom, left:right]
        yy, xx = np.ogrid[top:bottom, left:right]
        distance_sq = (xx - int(center_x)) ** 2 + (yy - int(center_y)) ** 2
        annulus = (distance_sq <= outer_radius * outer_radius) & (
            distance_sq >= inner_radius * inner_radius
        )
        return int(np.count_nonzero(local_mask[annulus]))

    @staticmethod
    def _coerce_size(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return max(int(value[0]), 1), max(int(value[1]), 1)
        return default

    @staticmethod
    def _coerce_region(value: Any, *, default: Region) -> Region:
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return (
                max(int(value[0]), 0),
                max(int(value[1]), 0),
                max(int(value[2]), 1),
                max(int(value[3]), 1),
            )
        return default

    @staticmethod
    def _coerce_offset_region(value: Any, *, default: Region) -> Region:
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return (
                int(value[0]),
                int(value[1]),
                max(int(value[2]), 1),
                max(int(value[3]), 1),
            )
        return default

    @staticmethod
    def _coerce_ratio_region(value: Any, *, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return default

        left = max(min(float(value[0]), 1.0), 0.0)
        top = max(min(float(value[1]), 1.0), 0.0)
        right = max(min(float(value[2]), 1.0), 0.0)
        bottom = max(min(float(value[3]), 1.0), 0.0)
        if right <= left or bottom <= top:
            return default
        return left, top, right, bottom

    @staticmethod
    def _coerce_point(value: Any, *, default: Point) -> Point:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
        return default

    @staticmethod
    def _coerce_hsv_triplet(value: Any, *, default: tuple[int, int, int]) -> tuple[int, int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return (
                max(min(int(value[0]), 179), 0),
                max(min(int(value[1]), 255), 0),
                max(min(int(value[2]), 255), 0),
            )
        return default

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_choice(value: Any, *, default: str, allowed: set[str]) -> str:
        normalized = str(value or default).strip().lower()
        return normalized if normalized in allowed else default

    @staticmethod
    def _coerce_string_list(value: Any, *, default: tuple[str, ...]) -> list[str]:
        if not isinstance(value, list):
            return list(default)
        result = [str(item).strip() for item in value if str(item).strip()]
        return result or list(default)

    def _coerce_point_list(self, value: Any, *, default: tuple[Point, ...]) -> list[Point]:
        if not isinstance(value, list):
            return list(default)
        points: list[Point] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                points.append((int(item[0]), int(item[1])))
        return points or list(default)

    @staticmethod
    def _coerce_float_list(value: Any, *, default: tuple[float, ...]) -> list[float]:
        if not isinstance(value, list):
            return list(default)
        result: list[float] = []
        for item in value:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if number > 0:
                result.append(number)
        return result or list(default)

    def _coerce_stock_profiles(self, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw_profiles = dict(payload.get("stock_profiles") or {})
        defaults: dict[str, dict[str, Any]] = {
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
                "batch_size": 6,
                "make_sec": 1.5,
                "enabled": True,
            },
            "coffee": {
                "station_point": self._coerce_point(payload.get("coffee_station_point"), default=(1035, 665)),
                "stock_point": self._coerce_point(payload.get("coffee_stock_point"), default=(1190, 665)),
                "monitor_region": (1145, 610, 130, 105),
                "monitor_min_pixels": 120,
                "monitor_white_s_max": 90,
                "monitor_white_v_min": 150,
                "monitor_use_warm": False,
                "batch_size": max(int(payload.get("coffee_batch_size", 3) or 3), 1),
                "make_sec": max(float(payload.get("coffee_make_sec", 1.5) or 1.5), 0.0),
                "enabled": True,
            },
        }

        normalized: dict[str, dict[str, Any]] = {}
        for stock_id, fallback in defaults.items():
            raw = dict(raw_profiles.get(stock_id) or {})
            enabled_value = raw.get("enabled", fallback["enabled"])
            if isinstance(enabled_value, str):
                enabled = enabled_value.strip().lower() not in {"0", "false", "no", "off", ""}
            else:
                enabled = bool(enabled_value)
            normalized[stock_id] = {
                "station_point": self._coerce_point(raw.get("station_point"), default=fallback["station_point"]),
                "stock_point": self._coerce_point(raw.get("stock_point"), default=fallback["stock_point"]),
                "monitor_region": self._coerce_region(
                    raw.get("monitor_region"),
                    default=fallback["monitor_region"],
                ),
                "monitor_min_pixels": max(
                    int(raw.get("monitor_min_pixels", fallback["monitor_min_pixels"]) or fallback["monitor_min_pixels"]),
                    1,
                ),
                "monitor_white_s_max": max(
                    min(int(raw.get("monitor_white_s_max", fallback["monitor_white_s_max"]) or fallback["monitor_white_s_max"]), 255),
                    0,
                ),
                "monitor_white_v_min": max(
                    min(int(raw.get("monitor_white_v_min", fallback["monitor_white_v_min"]) or fallback["monitor_white_v_min"]), 255),
                    0,
                ),
                "monitor_use_warm": self._coerce_bool(raw.get("monitor_use_warm", fallback["monitor_use_warm"])),
                "monitor_warm_h_max": max(min(int(raw.get("monitor_warm_h_max", 35) or 35), 179), 0),
                "monitor_warm_s_min": max(min(int(raw.get("monitor_warm_s_min", 60) or 60), 255), 0),
                "monitor_warm_v_min": max(min(int(raw.get("monitor_warm_v_min", 120) or 120), 255), 0),
                "batch_size": max(int(raw.get("batch_size", fallback["batch_size"]) or fallback["batch_size"]), 1),
                "make_sec": max(float(raw.get("make_sec", fallback["make_sec"]) or fallback["make_sec"]), 0.0),
                "enabled": enabled,
            }

        return normalized

    def detect_stock_status(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        stocks: dict[str, Any] = {}
        for stock_id, stock_profile in profile["stock_profiles"].items():
            if not bool(stock_profile.get("enabled", True)):
                continue
            region = stock_profile["monitor_region"]
            crop = self._crop_region(source_image, region)
            if crop.size == 0:
                stocks[stock_id] = {
                    "present": None,
                    "reason": "empty_region",
                    "region": list(region),
                    "product_pixels": 0,
                    "min_pixels": int(stock_profile["monitor_min_pixels"]),
                    "white_pixels": 0,
                    "warm_pixels": 0,
                }
                continue

            crop_rgb = self._ensure_rgb(crop)
            hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
            hue, sat, val = cv2.split(hsv)
            white_mask = (
                (sat <= int(stock_profile["monitor_white_s_max"]))
                & (val >= int(stock_profile["monitor_white_v_min"]))
            )
            warm_mask = np.zeros_like(white_mask, dtype=bool)
            if bool(stock_profile.get("monitor_use_warm", False)):
                warm_mask = (
                    (hue <= int(stock_profile["monitor_warm_h_max"]))
                    & (sat >= int(stock_profile["monitor_warm_s_min"]))
                    & (val >= int(stock_profile["monitor_warm_v_min"]))
                )
            product_mask = white_mask | warm_mask
            white_pixels = int(np.count_nonzero(white_mask))
            warm_pixels = int(np.count_nonzero(warm_mask))
            product_pixels = int(np.count_nonzero(product_mask))
            min_pixels = int(stock_profile["monitor_min_pixels"])
            present = product_pixels >= min_pixels
            stocks[stock_id] = {
                "present": present,
                "reason": "present" if present else "empty",
                "region": list(region),
                "product_pixels": product_pixels,
                "min_pixels": min_pixels,
                "white_pixels": white_pixels,
                "warm_pixels": warm_pixels,
            }

        return {"stocks": stocks}

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)

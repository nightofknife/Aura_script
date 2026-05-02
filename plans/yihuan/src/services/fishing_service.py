"""Fishing calibration and state analysis for the Yihuan plan."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import yaml

from packages.aura_core.api import service_info


Region = tuple[int, int, int, int]


@service_info(
    alias="yihuan_fishing",
    public=True,
    singleton=True,
    description="Provide fishing calibration loading and state detection for the Yihuan plan.",
)
class YihuanFishingService:
    _DEFAULT_PROFILE = "default_1280x720_cn"
    _READY_TEMPLATE_NAME = "ready_hook_anchor.png"
    _READY_TEMPLATE_MASK_NAME = "ready_hook_anchor_mask.png"
    _READY_TEMPLATE_THRESHOLD = 0.8
    _READY_PRIORITY_THRESHOLD = 0.95

    def __init__(self) -> None:
        self._plan_root = Path(__file__).resolve().parents[2]
        self._profile_dir = self._plan_root / "data" / "fishing"
        self._template_dir = self._profile_dir / "templates"
        self._profile_cache: dict[str, dict[str, Any]] = {}
        self._template_cache: dict[str, np.ndarray] = {}
        self._indicator_memory: dict[str, dict[str, Any]] = {}

    def load_profile(self, profile_name: str | None = None) -> dict[str, Any]:
        resolved_name = str(profile_name or self._DEFAULT_PROFILE).strip() or self._DEFAULT_PROFILE
        cached = self._profile_cache.get(resolved_name)
        if cached is not None:
            return dict(cached)

        profile_path = self._profile_dir / f"{resolved_name}.yaml"
        if not profile_path.is_file():
            raise FileNotFoundError(f"Fishing profile not found: {profile_path}")

        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        client_size = self._coerce_size(payload.get("client_size"), default=(1280, 720))
        duel_meter_region = self._coerce_region(payload.get("duel_meter_region"))
        zone_meter_region = self._coerce_region(payload.get("zone_meter_region") or duel_meter_region)
        indicator_meter_region = self._coerce_region(payload.get("indicator_meter_region") or duel_meter_region)
        normalized = {
            "profile_name": str(payload.get("profile_name") or resolved_name),
            "client_size": client_size,
            "bite_text_region": self._coerce_region(payload.get("bite_text_region")),
            "bite_marker_region": self._coerce_region(payload.get("bite_marker_region") or (1130, 580, 130, 120)),
            "bite_hsv_bounds_standard": self._coerce_hsv_pair(
                payload.get("bite_hsv_bounds_standard"),
                default=((150, 30, 45), (220, 100, 100)),
            ),
            "bite_min_pixels": max(int(payload.get("bite_min_pixels", 1200) or 1200), 1),
            "bite_min_component_area": max(int(payload.get("bite_min_component_area", 1500) or 1500), 1),
            "duel_meter_region": duel_meter_region,
            "zone_meter_region": zone_meter_region,
            "indicator_meter_region": indicator_meter_region,
            "zone_hsv_bounds_standard": self._coerce_hsv_pair(
                payload.get("zone_hsv_bounds_standard"),
                default=((164, 74, 94), (172, 82, 70)),
            ),
            "zone_min_component_area": max(int(payload.get("zone_min_component_area", 20) or 20), 1),
            "indicator_primary_hsv_bounds_standard": self._coerce_hsv_pair(
                payload.get("indicator_primary_hsv_bounds_standard"),
                default=((30, 31, 71), (90, 100, 100)),
            ),
            "indicator_secondary_hsv_bounds_standard": self._coerce_hsv_pair(
                payload.get("indicator_secondary_hsv_bounds_standard"),
                default=((0, 0, 82), (110, 55, 100)),
            ),
            "indicator_memory_ms": max(int(payload.get("indicator_memory_ms", 150) or 150), 0),
            "result_text_region": self._coerce_region(payload.get("result_text_region")),
            "ready_anchor_region": self._coerce_region(payload.get("ready_anchor_region")),
            "result_close_point": self._coerce_point(payload.get("result_close_point")),
            "result_close_action": self._profile_str(payload, "result_close_action", "menu_back"),
            "duel_end_missing_sec": max(float(payload.get("duel_end_missing_sec", 1.0) or 0.0), 0.0),
            "post_duel_click_interval_ms": max(int(payload.get("post_duel_click_interval_ms", 1000) or 1000), 10),
            "post_duel_close_interval_ms": max(
                self._profile_int(
                    payload,
                    "post_duel_close_interval_ms",
                    int(payload.get("post_duel_click_interval_ms", 1000) or 1000),
                ),
                10,
            ),
            "post_duel_ready_stable_sec": max(
                self._profile_float(payload, "post_duel_ready_stable_sec", 1.0),
                0.0,
            ),
            "ocr_texts": {
                "bite_required": self._coerce_text_list((payload.get("ocr_texts") or {}).get("bite_required")),
                "result_required": self._coerce_text_list((payload.get("ocr_texts") or {}).get("result_required")),
            },
            "poll_ms": max(self._profile_int(payload, "poll_ms", 80), 0),
            "bite_timeout_sec": max(float(payload.get("bite_timeout_sec", 15.0) or 15.0), 0.1),
            "hook_timeout_sec": max(float(payload.get("hook_timeout_sec", 3.0) or 3.0), 0.1),
            "hook_press_interval_ms": max(int(payload.get("hook_press_interval_ms", 70) or 70), 10),
            "hook_success_min_zone_width": max(int(payload.get("hook_success_min_zone_width", 60) or 60), 1),
            "hook_success_max_zone_width": max(int(payload.get("hook_success_max_zone_width", 120) or 120), 1),
            "hook_success_confirm_frames": max(int(payload.get("hook_success_confirm_frames", 1) or 1), 1),
            "duel_timeout_sec": max(float(payload.get("duel_timeout_sec", 20.0) or 20.0), 0.1),
            "deadband_px": max(int(payload.get("deadband_px", 6) or 6), 0),
            "tap_press_ms": max(int(payload.get("tap_press_ms", 35) or 35), 0),
            "tap_cooldown_ms": max(int(payload.get("tap_cooldown_ms", 70) or 70), 0),
            "control_mode": str(payload.get("control_mode") or "pulse_predictive"),
            "control_lookahead_sec": max(self._profile_float(payload, "control_lookahead_sec", 0.08), 0.0),
            "control_deadband_px": max(self._profile_int(payload, "control_deadband_px", 6), 0),
            "control_inside_min_pulse_ms": max(self._profile_int(payload, "control_inside_min_pulse_ms", 25), 0),
            "control_inside_max_pulse_ms": max(self._profile_int(payload, "control_inside_max_pulse_ms", 55), 0),
            "control_outside_min_pulse_ms": max(self._profile_int(payload, "control_outside_min_pulse_ms", 60), 0),
            "control_outside_max_pulse_ms": max(self._profile_int(payload, "control_outside_max_pulse_ms", 130), 0),
            "control_pulse_gain_ms_per_px": max(
                self._profile_float(payload, "control_pulse_gain_ms_per_px", 0.8),
                0.0,
            ),
            "control_min_interval_ms": max(self._profile_int(payload, "control_min_interval_ms", 20), 0),
            "control_reverse_gap_ms": max(self._profile_int(payload, "control_reverse_gap_ms", 15), 0),
            "control_hold_margin_px": max(self._profile_int(payload, "control_hold_margin_px", 0), 0),
            "control_hold_grace_ms": max(self._profile_int(payload, "control_hold_grace_ms", 150), 0),
            "control_detection_jump_window_ms": max(
                self._profile_int(payload, "control_detection_jump_window_ms", 250),
                0,
            ),
            "control_zone_jump_px": max(self._profile_int(payload, "control_zone_jump_px", 55), 0),
            "control_indicator_jump_px": max(self._profile_int(payload, "control_indicator_jump_px", 55), 0),
            "control_zone_width_jump_px": max(self._profile_int(payload, "control_zone_width_jump_px", 30), 0),
            "control_zone_width_min_px": max(self._profile_int(payload, "control_zone_width_min_px", 70), 0),
            "control_zone_width_max_px": max(self._profile_int(payload, "control_zone_width_max_px", 120), 1),
            "bait_recovery_enabled": self._profile_bool(payload, "bait_recovery_enabled", True),
            "sell_before_buy_bait": self._profile_bool(payload, "sell_before_buy_bait", True),
            "bait_recovery_max_attempts_per_round": max(
                self._profile_int(payload, "bait_recovery_max_attempts_per_round", 1),
                0,
            ),
            "bait_recovery_step_interval_ms": max(
                self._profile_int(payload, "bait_recovery_step_interval_ms", 500),
                0,
            ),
            "bait_recovery_ready_timeout_sec": max(
                self._profile_float(payload, "bait_recovery_ready_timeout_sec", 8.0),
                0.1,
            ),
            "bait_shortage_region": self._coerce_region(
                payload.get("bait_shortage_region") or (350, 300, 600, 120)
            ),
            "bait_shortage_template": self._profile_str(
                payload,
                "bait_shortage_template",
                "bait_shortage_bar_center.png",
            ),
            "bait_shortage_mask": self._profile_optional_str(payload, "bait_shortage_mask"),
            "bait_shortage_match_threshold": self._profile_threshold(
                payload,
                "bait_shortage_match_threshold",
                0.78,
            ),
            "bait_shortage_dark_bar_region": self._coerce_region(
                payload.get("bait_shortage_dark_bar_region") or (0, 330, 1280, 65)
            ),
            "bait_shortage_dark_pixel_threshold": max(
                self._profile_int(payload, "bait_shortage_dark_pixel_threshold", 35),
                0,
            ),
            "bait_shortage_dark_row_ratio": self._profile_threshold(
                payload,
                "bait_shortage_dark_row_ratio",
                0.80,
            ),
            "bait_shortage_dark_min_height_px": max(
                self._profile_int(payload, "bait_shortage_dark_min_height_px", 35),
                1,
            ),
            "sell_open_action": self._profile_str(payload, "sell_open_action", "fish_sell_shop"),
            "sell_step_interval_ms": max(self._profile_int(payload, "sell_step_interval_ms", 1000), 0),
            "sell_click_hold_ms": max(self._profile_int(payload, "sell_click_hold_ms", 100), 0),
            "sell_tab_point": self._coerce_point(payload.get("sell_tab_point") or (110, 280)),
            "sell_one_click_point": self._coerce_point(payload.get("sell_one_click_point") or (710, 640)),
            "sell_confirm_point": self._coerce_point(payload.get("sell_confirm_point") or (780, 470)),
            "sell_success_close_point": self._coerce_point(
                payload.get("sell_success_close_point") or (300, 400)
            ),
            "sell_success_close_clicks": max(self._profile_int(payload, "sell_success_close_clicks", 3), 0),
            "sell_success_close_interval_ms": max(
                self._profile_int(payload, "sell_success_close_interval_ms", 1000),
                10,
            ),
            "sell_ui_timeout_sec": max(self._profile_float(payload, "sell_ui_timeout_sec", 5.0), 0.1),
            "sell_confirm_template": self._profile_str(payload, "sell_confirm_template", "sell_confirm_title.png"),
            "sell_confirm_region": self._coerce_region(
                payload.get("sell_confirm_region") or (260, 190, 760, 160)
            ),
            "sell_confirm_match_threshold": self._profile_threshold(
                payload,
                "sell_confirm_match_threshold",
                0.78,
            ),
            "sell_confirm_wait_timeout_sec": max(
                self._profile_float(payload, "sell_confirm_wait_timeout_sec", 3.0),
                0.1,
            ),
            "sell_success_template": self._profile_str(payload, "sell_success_template", "sell_success_title.png"),
            "sell_success_region": self._coerce_region(
                payload.get("sell_success_region") or (330, 330, 620, 190)
            ),
            "sell_success_match_threshold": self._profile_threshold(
                payload,
                "sell_success_match_threshold",
                0.78,
            ),
            "sell_success_wait_timeout_sec": max(
                self._profile_float(payload, "sell_success_wait_timeout_sec", 3.0),
                0.1,
            ),
            "bait_shop_open_action": self._profile_str(payload, "bait_shop_open_action", "fish_bait_shop"),
            "bait_step_interval_ms": max(self._profile_int(payload, "bait_step_interval_ms", 800), 0),
            "bait_click_hold_ms": max(self._profile_int(payload, "bait_click_hold_ms", 100), 0),
            "bait_item_kind": self._profile_str(payload, "bait_item_kind", "universal_bait"),
            "bait_item_point": self._coerce_point(payload.get("bait_item_point") or (374, 165)),
            "bait_item_after_wait_ms": max(self._profile_int(payload, "bait_item_after_wait_ms", 500), 0),
            "bait_max_point": self._coerce_point(payload.get("bait_max_point") or (1220, 636)),
            "bait_max_clicks": max(self._profile_int(payload, "bait_max_clicks", 5), 1),
            "bait_max_click_interval_ms": max(self._profile_int(payload, "bait_max_click_interval_ms", 500), 0),
            "bait_max_after_wait_ms": max(self._profile_int(payload, "bait_max_after_wait_ms", 1200), 0),
            "bait_buy_point": self._coerce_point(payload.get("bait_buy_point") or (1074, 686)),
            "bait_confirm_point": self._coerce_point(payload.get("bait_confirm_point") or (780, 470)),
            "bait_buy_confirm_template": self._profile_str(
                payload,
                "bait_buy_confirm_template",
                "bait_buy_confirm_title.png",
            ),
            "bait_buy_confirm_region": self._coerce_region(
                payload.get("bait_buy_confirm_region") or (260, 190, 760, 160)
            ),
            "bait_buy_confirm_match_threshold": self._profile_threshold(
                payload,
                "bait_buy_confirm_match_threshold",
                0.78,
            ),
            "bait_buy_confirm_timeout_sec": max(
                self._profile_float(payload, "bait_buy_confirm_timeout_sec", 3.0),
                0.1,
            ),
            "bait_success_close_point": self._coerce_point(
                payload.get("bait_success_close_point") or (300, 400)
            ),
            "bait_success_close_clicks": max(self._profile_int(payload, "bait_success_close_clicks", 3), 0),
            "bait_success_close_interval_ms": max(
                self._profile_int(payload, "bait_success_close_interval_ms", 1000),
                10,
            ),
            "bait_change_open_action": self._profile_str(payload, "bait_change_open_action", "fish_bait_change"),
            "bait_change_open_wait_sec": max(
                self._profile_float(payload, "bait_change_open_wait_sec", 2.0),
                0.0,
            ),
            "bait_change_confirm_point": self._coerce_point(
                payload.get("bait_change_confirm_point") or (780, 470)
            ),
            "bait_change_after_click_wait_sec": max(
                self._profile_float(payload, "bait_change_after_click_wait_sec", 1.0),
                0.0,
            ),
            "bait_change_template": self._profile_str(payload, "bait_change_template", "bait_change_title.png"),
            "bait_change_region": self._coerce_region(
                payload.get("bait_change_region") or (260, 190, 760, 160)
            ),
            "bait_change_match_threshold": self._profile_threshold(
                payload,
                "bait_change_match_threshold",
                0.78,
            ),
            "trace_limit": max(self._profile_int(payload, "trace_limit", 240), 1),
            "session_results_limit": max(self._profile_int(payload, "session_results_limit", 60), 1),
            "active_sell_results_limit": max(self._profile_int(payload, "active_sell_results_limit", 20), 1),
        }
        self._profile_cache[resolved_name] = normalized
        return dict(normalized)

    def analyze_state(
        self,
        source_image: np.ndarray,
        *,
        bite_texts: list[str] | None = None,
        result_texts: list[str] | None = None,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        bite_texts = [text for text in (bite_texts or []) if str(text).strip()]
        result_texts = [text for text in (result_texts or []) if str(text).strip()]
        bite_marker = self.analyze_bite_marker(source_image, profile_name=profile["profile_name"])
        duel = self.analyze_duel_meter(source_image, profile_name=profile["profile_name"])
        ready_anchor = self.analyze_ready_anchor(source_image, profile_name=profile["profile_name"])
        result_match = self._match_required_texts(result_texts, profile["ocr_texts"]["result_required"])

        phase = "unknown"
        ready_confidence = float(ready_anchor.get("confidence") or 0.0)
        if result_match["matched"]:
            phase = "result"
        elif ready_anchor["found"] and ready_confidence >= self._READY_PRIORITY_THRESHOLD:
            phase = "ready"
        elif bite_marker["found"]:
            phase = "bite"
        elif duel["found"]:
            phase = "duel"
        elif ready_anchor["found"]:
            phase = "ready"

        return {
            "phase": phase,
            "profile_name": profile["profile_name"],
            "bite": {
                "matched": bite_marker["found"],
                "detector": "marker_color",
                "texts": bite_texts,
                "required": [],
                "missing": [],
                "reason": bite_marker["reason"],
                "region": list(bite_marker["region"]),
                "pixel_count": bite_marker["pixel_count"],
                "min_pixels": bite_marker["min_pixels"],
                "largest_component_area": bite_marker["largest_component_area"],
                "largest_component_rect": bite_marker["largest_component_rect"],
                "min_component_area": bite_marker["min_component_area"],
                "mask_ratio": bite_marker["mask_ratio"],
                "hsv_cv2_lower": list(bite_marker["hsv_cv2_lower"]),
                "hsv_cv2_upper": list(bite_marker["hsv_cv2_upper"]),
            },
            "result": {
                "matched": result_match["matched"],
                "texts": result_texts,
                "required": list(profile["ocr_texts"]["result_required"]),
                "missing": result_match["missing"],
            },
            "ready_anchor": ready_anchor,
            "duel": duel,
            "control_advice": duel["control_advice"] if duel["found"] else "none",
        }

    def analyze_bite_marker(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        region = profile["bite_marker_region"]
        crop = self._crop_region(source_image, region)
        lower, upper = self._standard_hsv_pair_to_cv2(profile["bite_hsv_bounds_standard"])
        if crop.size == 0:
            return {
                "found": False,
                "reason": "empty_region",
                "region": list(region),
                "pixel_count": 0,
                "min_pixels": int(profile["bite_min_pixels"]),
                "largest_component_area": 0,
                "largest_component_rect": None,
                "min_component_area": int(profile["bite_min_component_area"]),
                "mask_ratio": 0.0,
                "hsv_cv2_lower": list(lower),
                "hsv_cv2_upper": list(upper),
            }

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        pixel_count = int(np.count_nonzero(mask))
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        largest_component_area = 0
        largest_component_rect: list[int] | None = None
        for label in range(1, num_labels):
            x, y, width, height, area = [int(value) for value in stats[label]]
            if area <= largest_component_area:
                continue
            largest_component_area = area
            largest_component_rect = [x, y, width, height]

        min_pixels = int(profile["bite_min_pixels"])
        min_component_area = int(profile["bite_min_component_area"])
        found = pixel_count >= min_pixels and largest_component_area >= min_component_area
        reason = "ok"
        if not found:
            if pixel_count < min_pixels:
                reason = "pixel_count_below_threshold"
            else:
                reason = "component_below_threshold"

        total_pixels = max(int(region[2] * region[3]), 1)
        return {
            "found": bool(found),
            "reason": reason,
            "region": list(region),
            "pixel_count": pixel_count,
            "min_pixels": min_pixels,
            "largest_component_area": int(largest_component_area),
            "largest_component_rect": largest_component_rect,
            "min_component_area": min_component_area,
            "mask_ratio": round(float(pixel_count) / float(total_pixels), 4),
            "hsv_cv2_lower": list(lower),
            "hsv_cv2_upper": list(upper),
        }

    def analyze_ready_anchor(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        region = profile["ready_anchor_region"]
        crop = self._crop_region(source_image, region)
        template = self._load_template(self._READY_TEMPLATE_NAME)
        if crop.size == 0 or template.size == 0:
            return {
                "found": False,
                "confidence": 0.0,
                "threshold": self._READY_TEMPLATE_THRESHOLD,
                "region": list(region),
                "match_rect": None,
            }

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        result = cv2.matchTemplate(crop_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, top_left = cv2.minMaxLoc(result)
        match_rect = [
            int(region[0] + top_left[0]),
            int(region[1] + top_left[1]),
            int(template.shape[1]),
            int(template.shape[0]),
        ]
        return {
            "found": bool(confidence >= self._READY_TEMPLATE_THRESHOLD),
            "confidence": round(float(confidence), 4),
            "threshold": self._READY_TEMPLATE_THRESHOLD,
            "region": list(region),
            "match_rect": match_rect,
        }

    def analyze_ready_anchor_with_vision(
        self,
        source_image: np.ndarray,
        vision: Any,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        region = profile["ready_anchor_region"]
        crop = self._crop_region(source_image, region)
        template_path = self._template_dir / self._READY_TEMPLATE_NAME
        mask_path = self._template_dir / self._READY_TEMPLATE_MASK_NAME
        if crop.size == 0 or not template_path.is_file():
            return {
                "found": False,
                "confidence": 0.0,
                "threshold": self._READY_TEMPLATE_THRESHOLD,
                "region": list(region),
                "match_rect": None,
                "template_path": str(template_path),
                "mask_path": str(mask_path) if mask_path.is_file() else None,
                "backend": "vision",
            }

        match_result = vision.find_template(
            source_image=crop,
            template_image=str(template_path),
            mask_image=str(mask_path) if mask_path.is_file() else None,
            threshold=self._READY_TEMPLATE_THRESHOLD,
            use_grayscale=True,
            match_method=cv2.TM_CCOEFF_NORMED,
            preprocess="none",
        )
        top_left = match_result.top_left or (0, 0)
        rect = match_result.rect
        match_rect = None
        if rect is not None:
            match_rect = [
                int(region[0] + rect[0]),
                int(region[1] + rect[1]),
                int(rect[2]),
                int(rect[3]),
            ]
        elif top_left is not None:
            template = self._load_template(self._READY_TEMPLATE_NAME)
            match_rect = [
                int(region[0] + top_left[0]),
                int(region[1] + top_left[1]),
                int(template.shape[1]),
                int(template.shape[0]),
            ]
        return {
            "found": bool(match_result.found),
            "confidence": round(float(match_result.confidence), 4),
            "threshold": self._READY_TEMPLATE_THRESHOLD,
            "region": list(region),
            "match_rect": match_rect,
            "template_path": str(template_path),
            "mask_path": str(mask_path) if mask_path.is_file() else None,
            "backend": "vision",
            "debug_info": dict(match_result.debug_info or {}),
        }

    def match_template_with_vision(
        self,
        source_image: np.ndarray,
        vision: Any,
        *,
        template_name: str,
        region: Region,
        threshold: float,
        mask_name: str | None = None,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        _ = self.load_profile(profile_name)
        crop = self._crop_region(source_image, region)
        template_path = self._resolve_template_path(template_name)
        mask_path = self._resolve_template_path(mask_name) if mask_name else None
        response = {
            "found": False,
            "confidence": 0.0,
            "threshold": float(threshold),
            "region": list(region),
            "match_rect": None,
            "template_path": str(template_path),
            "mask_path": str(mask_path) if mask_path is not None and mask_path.is_file() else None,
            "backend": "vision",
            "debug_info": {},
        }
        if crop.size == 0:
            response["debug_info"] = {"error": "empty_region"}
            return response
        if not template_path.is_file():
            response["debug_info"] = {"error": "template_not_found"}
            return response

        match_result = vision.find_template(
            source_image=crop,
            template_image=str(template_path),
            mask_image=str(mask_path) if mask_path is not None and mask_path.is_file() else None,
            threshold=float(threshold),
            use_grayscale=True,
            match_method=cv2.TM_CCOEFF_NORMED,
            preprocess="none",
        )
        rect = match_result.rect
        top_left = match_result.top_left
        match_rect = None
        if rect is not None:
            match_rect = [
                int(region[0] + rect[0]),
                int(region[1] + rect[1]),
                int(rect[2]),
                int(rect[3]),
            ]
        elif top_left is not None:
            template = self._load_template(str(template_path.name))
            match_rect = [
                int(region[0] + top_left[0]),
                int(region[1] + top_left[1]),
                int(template.shape[1]),
                int(template.shape[0]),
            ]

        response.update(
            {
                "found": bool(match_result.found),
                "confidence": round(float(match_result.confidence), 4),
                "match_rect": match_rect,
                "debug_info": dict(match_result.debug_info or {}),
            }
        )
        return response

    def analyze_bait_shortage(
        self,
        source_image: np.ndarray,
        vision: Any,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        if not bool(profile["bait_recovery_enabled"]):
            return {
                "found": False,
                "reason": "disabled",
                "template_match": None,
                "dark_bar": None,
                "confidence": 0.0,
                "match_rect": None,
            }

        template_match = self.match_template_with_vision(
            source_image,
            vision,
            template_name=str(profile["bait_shortage_template"]),
            region=profile["bait_shortage_region"],
            threshold=float(profile["bait_shortage_match_threshold"]),
            mask_name=profile.get("bait_shortage_mask"),
            profile_name=profile["profile_name"],
        )
        dark_bar = self._analyze_dark_bar(
            source_image,
            region=profile["bait_shortage_dark_bar_region"],
            pixel_threshold=int(profile["bait_shortage_dark_pixel_threshold"]),
            row_ratio=float(profile["bait_shortage_dark_row_ratio"]),
            min_height_px=int(profile["bait_shortage_dark_min_height_px"]),
        )

        template_found = bool(template_match.get("found"))
        dark_found = bool(dark_bar.get("found"))
        found = template_found and dark_found
        reason = "ok"
        if not template_found:
            reason = "template_missing"
        elif not dark_found:
            reason = "dark_bar_missing"

        return {
            "found": bool(found),
            "reason": reason,
            "template_match": template_match,
            "dark_bar": dark_bar,
            "confidence": float(template_match.get("confidence") or 0.0),
            "match_rect": template_match.get("match_rect"),
        }

    def analyze_duel_meter(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
        deadband_px: int | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        zone_region, indicator_region, merged_region = self._resolve_duel_regions(profile)
        resolved_deadband = deadband_px if deadband_px is not None else profile["deadband_px"]
        zone_crop = self._crop_region(source_image, zone_region)
        indicator_crop = self._crop_region(source_image, indicator_region)
        if zone_crop.size == 0 or indicator_crop.size == 0:
            return self._empty_duel_result(
                merged_region,
                resolved_deadband,
                reason="empty_region",
                zone_region=zone_region,
                indicator_region=indicator_region,
            )

        zone_hsv = cv2.cvtColor(zone_crop, cv2.COLOR_RGB2HSV)
        indicator_hsv = cv2.cvtColor(indicator_crop, cv2.COLOR_RGB2HSV)
        zone_lower, zone_upper = self._standard_hsv_pair_to_cv2(profile["zone_hsv_bounds_standard"])
        zone_mask = cv2.inRange(zone_hsv, zone_lower, zone_upper)
        zone_bounds = self._extract_zone_bounds(
            zone_mask,
            min_component_area=int(profile["zone_min_component_area"]),
        )
        indicator_masks = self._build_indicator_masks(indicator_hsv, profile)
        if zone_bounds is None:
            raw_indicator_candidate = self._extract_indicator_candidate(
                indicator_masks["primary_mask"],
                indicator_masks["combined_mask"],
            )
            raw_indicator_detected = raw_indicator_candidate is not None
            result = self._empty_duel_result(
                merged_region,
                resolved_deadband,
                reason="zone_missing",
                zone_region=zone_region,
                indicator_region=indicator_region,
            )
            result["indicator_raw_detected"] = raw_indicator_detected
            result["indicator_source"] = str(raw_indicator_candidate.get("source") or "none") if raw_indicator_candidate is not None else "none"
            return result

        zone_left_local, zone_right_local = zone_bounds
        zone_left_abs = int(zone_region[0] + zone_left_local)
        zone_right_abs = int(zone_region[0] + zone_right_local)
        indicator_zone_left_local = int(zone_left_abs - indicator_region[0])
        indicator_zone_right_local = int(zone_right_abs - indicator_region[0])
        indicator_candidate = self._extract_indicator_candidate(
            indicator_masks["primary_mask"],
            indicator_masks["combined_mask"],
            zone_left_local=indicator_zone_left_local,
            zone_right_local=indicator_zone_right_local,
        )
        raw_indicator_candidate = indicator_candidate
        raw_indicator_detected = indicator_candidate is not None
        if indicator_candidate is None:
            remembered_result = self._recall_indicator_memory(
                profile_name=profile["profile_name"],
                profile=profile,
                zone_region=zone_region,
                indicator_region=indicator_region,
                zone_left_local=zone_left_local,
                zone_right_local=zone_right_local,
                deadband_px=resolved_deadband,
                raw_indicator_detected=raw_indicator_detected,
            )
            if remembered_result is not None:
                return remembered_result

            result = self._empty_duel_result(
                merged_region,
                resolved_deadband,
                reason="indicator_missing",
                zone_region=zone_region,
                indicator_region=indicator_region,
            )
            result["zone_detected"] = True
            result["indicator_raw_detected"] = raw_indicator_detected
            result["indicator_source"] = str(raw_indicator_candidate.get("source") or "none") if raw_indicator_candidate is not None else "none"
            result["zone_left"] = zone_left_abs
            result["zone_right"] = zone_right_abs
            result["zone_center"] = int(round((zone_left_abs + zone_right_abs) / 2.0))
            return result

        self._remember_indicator(
            profile_name=profile["profile_name"],
            profile=profile,
            indicator_region=indicator_region,
            indicator_candidate=indicator_candidate,
        )
        return self._build_duel_result(
            zone_region=zone_region,
            indicator_region=indicator_region,
            zone_left_local=zone_left_local,
            zone_right_local=zone_right_local,
            indicator_candidate=indicator_candidate,
            deadband_px=resolved_deadband,
            hold_margin_px=int(profile.get("control_hold_margin_px") or 0),
            indicator_source=str(indicator_candidate.get("source") or "raw"),
            raw_indicator_detected=raw_indicator_detected,
            indicator_memory_age_ms=None,
        )

    def save_duel_debug_artifact(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
        tag: str = "duel",
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        zone_region, indicator_region, merged_region = self._resolve_duel_regions(profile)
        zone_crop = self._crop_region(source_image, zone_region)
        indicator_crop = self._crop_region(source_image, indicator_region)
        if zone_crop.size == 0 or indicator_crop.size == 0:
            return {
                "ok": False,
                "reason": "empty_region",
                "path": None,
                "region": list(merged_region),
                "zone_region": list(zone_region),
                "indicator_region": list(indicator_region),
            }

        zone_hsv = cv2.cvtColor(zone_crop, cv2.COLOR_RGB2HSV)
        indicator_hsv = cv2.cvtColor(indicator_crop, cv2.COLOR_RGB2HSV)
        zone_lower, zone_upper = self._standard_hsv_pair_to_cv2(profile["zone_hsv_bounds_standard"])
        zone_mask = cv2.inRange(zone_hsv, zone_lower, zone_upper)
        indicator_masks = self._build_indicator_masks(indicator_hsv, profile)
        duel = self.analyze_duel_meter(source_image, profile_name=profile["profile_name"])

        zone_annotated = zone_crop.copy()
        indicator_annotated = indicator_crop.copy()
        if duel.get("zone_left") is not None and duel.get("zone_right") is not None:
            local_zone_left = int(duel["zone_left"]) - zone_region[0]
            local_zone_right = int(duel["zone_right"]) - zone_region[0]
            cv2.rectangle(
                zone_annotated,
                (local_zone_left, 0),
                (local_zone_right, max(zone_crop.shape[0] - 1, 0)),
                (0, 255, 0),
                1,
            )
        if duel.get("indicator_x") is not None:
            local_indicator_x = int(duel["indicator_x"]) - indicator_region[0]
            cv2.line(
                indicator_annotated,
                (local_indicator_x, 0),
                (local_indicator_x, max(indicator_crop.shape[0] - 1, 0)),
                (255, 255, 0),
                1,
            )

        zone_vis = cv2.cvtColor(zone_mask, cv2.COLOR_GRAY2RGB)
        indicator_vis = cv2.cvtColor(indicator_masks["combined_mask"], cv2.COLOR_GRAY2RGB)
        panel_height = max(zone_crop.shape[0], indicator_crop.shape[0], 1)
        zone_raw_panel = self._pad_image_to_height(zone_crop, panel_height)
        zone_annotated_panel = self._pad_image_to_height(zone_annotated, panel_height)
        indicator_raw_panel = self._pad_image_to_height(indicator_crop, panel_height)
        indicator_annotated_panel = self._pad_image_to_height(indicator_annotated, panel_height)
        zone_vis_panel = self._pad_image_to_height(zone_vis, panel_height)
        indicator_vis_panel = self._pad_image_to_height(indicator_vis, panel_height)
        blank_panel = np.zeros_like(zone_raw_panel)
        top = np.hstack((zone_raw_panel, zone_annotated_panel, indicator_raw_panel, indicator_annotated_panel))
        bottom = np.hstack((zone_vis_panel, blank_panel, indicator_vis_panel, blank_panel))
        composite = np.vstack((top, bottom))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        out_dir = self._plan_root.parent.parent / "logs" / "yihuan_fishing_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = out_dir / f"{tag}_{timestamp}.png"
        cv2.imwrite(str(artifact_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        return {
            "ok": True,
            "path": str(artifact_path),
            "region": list(merged_region),
            "zone_region": list(zone_region),
            "indicator_region": list(indicator_region),
            "zone_hsv_cv2_lower": list(zone_lower),
            "zone_hsv_cv2_upper": list(zone_upper),
            "indicator_primary_hsv_cv2_lower": list(indicator_masks["primary_lower"]),
            "indicator_primary_hsv_cv2_upper": list(indicator_masks["primary_upper"]),
            "indicator_secondary_hsv_cv2_lower": list(indicator_masks["secondary_lower"]),
            "indicator_secondary_hsv_cv2_upper": list(indicator_masks["secondary_upper"]),
            "duel": duel,
        }

    def build_duel_debug_view(
        self,
        source_image: np.ndarray,
        *,
        state: dict[str, Any] | None = None,
        profile_name: str | None = None,
        held_action_name: str | None = None,
    ) -> np.ndarray:
        profile = self.load_profile(profile_name)
        zone_region, indicator_region, merged_region = self._resolve_duel_regions(profile)
        zone_crop = self._crop_region(source_image, zone_region)
        indicator_crop = self._crop_region(source_image, indicator_region)
        if zone_crop.size == 0:
            zone_crop = np.zeros((max(zone_region[3], 1), max(zone_region[2], 1), 3), dtype=np.uint8)
        if indicator_crop.size == 0:
            indicator_crop = np.zeros((max(indicator_region[3], 1), max(indicator_region[2], 1), 3), dtype=np.uint8)

        duel = dict(((state or {}).get("duel") or {}))
        if not duel:
            duel = self.analyze_duel_meter(source_image, profile_name=profile["profile_name"])
        phase = str((state or {}).get("phase") or "unknown")
        advice = str((state or {}).get("control_advice") or duel.get("control_advice") or "none")

        zone_hsv = cv2.cvtColor(zone_crop, cv2.COLOR_RGB2HSV)
        indicator_hsv = cv2.cvtColor(indicator_crop, cv2.COLOR_RGB2HSV)
        zone_lower, zone_upper = self._standard_hsv_pair_to_cv2(profile["zone_hsv_bounds_standard"])
        zone_mask = cv2.inRange(zone_hsv, zone_lower, zone_upper)
        indicator_masks = self._build_indicator_masks(indicator_hsv, profile)

        zone_annotated = zone_crop.copy()
        indicator_annotated = indicator_crop.copy()
        if duel.get("zone_left") is not None and duel.get("zone_right") is not None:
            zone_left_local = int(duel["zone_left"]) - zone_region[0]
            zone_right_local = int(duel["zone_right"]) - zone_region[0]
            cv2.rectangle(
                zone_annotated,
                (zone_left_local, 0),
                (zone_right_local, max(zone_crop.shape[0] - 1, 0)),
                (0, 255, 0),
                1,
            )
        if duel.get("indicator_x") is not None:
            indicator_x_local = int(duel["indicator_x"]) - indicator_region[0]
            cv2.line(
                indicator_annotated,
                (indicator_x_local, 0),
                (indicator_x_local, max(indicator_crop.shape[0] - 1, 0)),
                (255, 255, 0),
                1,
            )

        zone_vis = cv2.cvtColor(zone_mask, cv2.COLOR_GRAY2RGB)
        indicator_vis = cv2.cvtColor(indicator_masks["combined_mask"], cv2.COLOR_GRAY2RGB)
        panel_height = max(zone_crop.shape[0], indicator_crop.shape[0], 1)
        zone_raw_panel = self._pad_image_to_height(zone_crop, panel_height)
        zone_annotated_panel = self._pad_image_to_height(zone_annotated, panel_height)
        indicator_raw_panel = self._pad_image_to_height(indicator_crop, panel_height)
        indicator_annotated_panel = self._pad_image_to_height(indicator_annotated, panel_height)
        zone_vis_panel = self._pad_image_to_height(zone_vis, panel_height)
        indicator_vis_panel = self._pad_image_to_height(indicator_vis, panel_height)

        scale = max(int(np.ceil(220 / max(panel_height, 1))), 1)

        def _scale_view(image: np.ndarray) -> np.ndarray:
            return cv2.resize(
                image,
                (max(image.shape[1] * scale, 1), max(image.shape[0] * scale, 1)),
                interpolation=cv2.INTER_NEAREST,
            )

        zone_raw_scaled = _scale_view(zone_raw_panel)
        zone_annotated_scaled = _scale_view(zone_annotated_panel)
        indicator_raw_scaled = _scale_view(indicator_raw_panel)
        indicator_annotated_scaled = _scale_view(indicator_annotated_panel)
        zone_scaled = _scale_view(zone_vis_panel)
        indicator_scaled = _scale_view(indicator_vis_panel)

        info_width = max(zone_raw_scaled.shape[1], indicator_raw_scaled.shape[1], 420)
        info_height = zone_raw_scaled.shape[0] + zone_scaled.shape[0]
        info_panel = np.zeros((info_height, info_width, 3), dtype=np.uint8)
        info_lines = [
            f"phase: {phase}",
            f"duel_found: {bool(duel.get('found'))}",
            f"reason: {duel.get('reason')}",
            f"zone_roi: {zone_region}",
            f"indicator_roi: {indicator_region}",
            f"zone: {duel.get('zone_left')} .. {duel.get('zone_right')}",
            f"indicator: {duel.get('indicator_x')}",
            f"indicator_source: {duel.get('indicator_source')}",
            f"error_px: {duel.get('error_px')}",
            f"advice: {advice}",
            f"held: {self._held_action_label(held_action_name)}",
        ]
        for index, line in enumerate(info_lines):
            y = 28 + (index * 26)
            cv2.putText(
                info_panel,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        blank_top = np.zeros_like(zone_raw_scaled)
        blank_bottom = np.zeros_like(zone_scaled)
        top = np.hstack(
            (
                zone_raw_scaled,
                zone_annotated_scaled,
                indicator_raw_scaled,
                indicator_annotated_scaled,
                info_panel[: zone_raw_scaled.shape[0], :, :],
            )
        )
        bottom = np.hstack(
            (
                zone_scaled,
                blank_bottom,
                indicator_scaled,
                blank_bottom,
                info_panel[zone_raw_scaled.shape[0] :, :, :],
            )
        )
        composite = np.vstack((top, bottom))
        return cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    def build_duel_live_monitor_view(
        self,
        source_image: np.ndarray,
        *,
        state: dict[str, Any] | None = None,
        profile_name: str | None = None,
    ) -> np.ndarray:
        profile = self.load_profile(profile_name)
        zone_region, indicator_region, merged_region = self._resolve_duel_regions(profile)
        merged_crop = self._crop_region(source_image, merged_region)
        if merged_crop.size == 0:
            merged_crop = np.zeros((max(merged_region[3], 1), max(merged_region[2], 1), 3), dtype=np.uint8)

        zone_crop = self._crop_region(source_image, zone_region)
        indicator_crop = self._crop_region(source_image, indicator_region)
        if zone_crop.size == 0:
            zone_crop = np.zeros((max(zone_region[3], 1), max(zone_region[2], 1), 3), dtype=np.uint8)
        if indicator_crop.size == 0:
            indicator_crop = np.zeros((max(indicator_region[3], 1), max(indicator_region[2], 1), 3), dtype=np.uint8)

        duel = dict(((state or {}).get("duel") or {}))
        if not duel:
            duel = self.analyze_duel_meter(source_image, profile_name=profile["profile_name"])
        phase = str((state or {}).get("phase") or ("duel" if duel.get("found") else "unknown"))
        advice = str((state or {}).get("control_advice") or duel.get("control_advice") or "none")

        zone_hsv = cv2.cvtColor(zone_crop, cv2.COLOR_RGB2HSV)
        indicator_hsv = cv2.cvtColor(indicator_crop, cv2.COLOR_RGB2HSV)
        zone_lower, zone_upper = self._standard_hsv_pair_to_cv2(profile["zone_hsv_bounds_standard"])
        zone_mask = cv2.inRange(zone_hsv, zone_lower, zone_upper)
        indicator_masks = self._build_indicator_masks(indicator_hsv, profile)

        annotated = merged_crop.copy()
        zone_color = (255, 0, 255)
        indicator_color = (255, 0, 0)
        if duel.get("zone_left") is not None and duel.get("zone_right") is not None:
            zone_left_local = int(duel["zone_left"]) - merged_region[0]
            zone_right_local = int(duel["zone_right"]) - merged_region[0]
            zone_top_local = int(zone_region[1] - merged_region[1])
            zone_bottom_local = int(zone_top_local + zone_region[3] - 1)
            cv2.rectangle(
                annotated,
                (zone_left_local, zone_top_local),
                (zone_right_local, zone_bottom_local),
                zone_color,
                1,
            )
        if duel.get("indicator_x") is not None:
            indicator_x_local = int(duel["indicator_x"]) - merged_region[0]
            indicator_top_local = int(indicator_region[1] - merged_region[1])
            indicator_bottom_local = int(indicator_top_local + indicator_region[3] - 1)
            cv2.line(
                annotated,
                (indicator_x_local, indicator_top_local),
                (indicator_x_local, indicator_bottom_local),
                indicator_color,
                1,
            )

        zone_mask_panel = np.zeros((merged_crop.shape[0], merged_crop.shape[1], 3), dtype=np.uint8)
        indicator_mask_panel = np.zeros((merged_crop.shape[0], merged_crop.shape[1], 3), dtype=np.uint8)
        zone_gray = cv2.cvtColor(zone_mask, cv2.COLOR_GRAY2RGB)
        indicator_gray = cv2.cvtColor(indicator_masks["combined_mask"], cv2.COLOR_GRAY2RGB)
        zone_top = int(zone_region[1] - merged_region[1])
        zone_left = int(zone_region[0] - merged_region[0])
        indicator_top = int(indicator_region[1] - merged_region[1])
        indicator_left = int(indicator_region[0] - merged_region[0])
        zone_mask_panel[zone_top:zone_top + zone_gray.shape[0], zone_left:zone_left + zone_gray.shape[1]] = zone_gray
        indicator_mask_panel[
            indicator_top:indicator_top + indicator_gray.shape[0],
            indicator_left:indicator_left + indicator_gray.shape[1],
        ] = indicator_gray

        scale = max(int(np.ceil(220 / max(merged_crop.shape[0], 1))), 1)

        def _scale_view(image: np.ndarray) -> np.ndarray:
            return cv2.resize(
                image,
                (max(image.shape[1] * scale, 1), max(image.shape[0] * scale, 1)),
                interpolation=cv2.INTER_NEAREST,
            )

        raw_scaled = _scale_view(merged_crop)
        annotated_scaled = _scale_view(annotated)
        zone_mask_scaled = _scale_view(zone_mask_panel)
        indicator_mask_scaled = _scale_view(indicator_mask_panel)

        info_width = max(raw_scaled.shape[1], 420)
        info_height = raw_scaled.shape[0] + zone_mask_scaled.shape[0]
        info_panel = np.zeros((info_height, info_width, 3), dtype=np.uint8)
        info_lines = [
            f"phase: {phase}",
            f"reason: {duel.get('reason')}",
            f"zone_roi: {zone_region}",
            f"indicator_roi: {indicator_region}",
            f"zone: {duel.get('zone_left')} .. {duel.get('zone_right')}",
            f"indicator: {duel.get('indicator_x')}",
            f"indicator_source: {duel.get('indicator_source')}",
            f"advice: {advice}",
            "legend: purple=zone, red=indicator",
        ]
        for index, line in enumerate(info_lines):
            y = 28 + (index * 26)
            cv2.putText(
                info_panel,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        top = np.hstack((raw_scaled, annotated_scaled, info_panel[: raw_scaled.shape[0], :, :]))
        bottom = np.hstack((zone_mask_scaled, indicator_mask_scaled, info_panel[raw_scaled.shape[0] :, :, :]))
        composite = np.vstack((top, bottom))
        return cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    def _build_indicator_masks(self, hsv: np.ndarray, profile: dict[str, Any]) -> dict[str, Any]:
        primary_lower, primary_upper = self._standard_hsv_pair_to_cv2(profile["indicator_primary_hsv_bounds_standard"])
        primary_mask = cv2.inRange(hsv, primary_lower, primary_upper)
        secondary_lower, secondary_upper = self._standard_hsv_pair_to_cv2(profile["indicator_secondary_hsv_bounds_standard"])
        secondary_mask = cv2.inRange(hsv, secondary_lower, secondary_upper)
        combined_mask = cv2.bitwise_or(primary_mask, secondary_mask)
        combined_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 1), dtype=np.uint8),
        )
        return {
            "primary_lower": primary_lower,
            "primary_upper": primary_upper,
            "secondary_lower": secondary_lower,
            "secondary_upper": secondary_upper,
            "primary_mask": primary_mask,
            "secondary_mask": secondary_mask,
            "combined_mask": combined_mask,
        }

    def _build_duel_result(
        self,
        *,
        region: Region | None = None,
        zone_region: Region | None = None,
        indicator_region: Region | None = None,
        zone_left_local: int,
        zone_right_local: int,
        indicator_candidate: dict[str, Any],
        deadband_px: int,
        hold_margin_px: int = 0,
        indicator_source: str,
        raw_indicator_detected: bool,
        indicator_memory_age_ms: float | None,
    ) -> dict[str, Any]:
        if region is not None:
            if zone_region is None:
                zone_region = region
            if indicator_region is None:
                indicator_region = region
        if zone_region is None or indicator_region is None:
            raise ValueError("zone_region and indicator_region must be provided")
        merged_region = self._merge_regions(zone_region, indicator_region)
        indicator_x_local = float(indicator_candidate["center_x"])
        indicator_x = int(indicator_region[0] + indicator_x_local)
        zone_left = int(zone_region[0] + zone_left_local)
        zone_right = int(zone_region[0] + zone_right_local)
        zone_center = int(round((zone_left + zone_right) / 2.0))
        center_error_px = int(indicator_x - zone_center)
        boundary_error_px = self._signed_error(indicator_x, zone_left, zone_right)
        hold_margin = max(int(hold_margin_px), 0)
        left_hold_threshold = zone_left + hold_margin
        right_hold_threshold = zone_right - hold_margin

        control_advice = "none"
        if indicator_x <= left_hold_threshold:
            control_advice = "hold_d"
        elif indicator_x >= right_hold_threshold:
            control_advice = "hold_a"
        elif center_error_px < -deadband_px:
            control_advice = "tap_d"
        elif center_error_px > deadband_px:
            control_advice = "tap_a"

        return {
            "found": True,
            "reason": "ok",
            "region": list(merged_region),
            "zone_region": list(zone_region),
            "indicator_region": list(indicator_region),
            "zone_detected": True,
            "indicator_detected": True,
            "indicator_raw_detected": bool(raw_indicator_detected),
            "indicator_source": str(indicator_source),
            "indicator_memory_age_ms": None if indicator_memory_age_ms is None else round(float(indicator_memory_age_ms), 1),
            "zone_left": zone_left,
            "zone_right": zone_right,
            "zone_center": zone_center,
            "indicator_x": indicator_x,
            "indicator_rect": indicator_candidate.get("rect"),
            "error_px": int(center_error_px),
            "center_error_px": int(center_error_px),
            "boundary_error_px": int(boundary_error_px),
            "deadband_px": int(deadband_px),
            "hold_margin_px": int(hold_margin),
            "control_advice": control_advice,
        }

    def _remember_indicator(
        self,
        *,
        profile_name: str,
        profile: dict[str, Any],
        indicator_region: Region,
        indicator_candidate: dict[str, Any],
    ) -> None:
        memory_window_ms = int(profile.get("indicator_memory_ms") or 0)
        if memory_window_ms <= 0:
            return
        self._indicator_memory[profile_name] = {
            "saved_at": float(time.monotonic()),
            "max_age_ms": memory_window_ms,
            "indicator_x": int(indicator_region[0] + float(indicator_candidate["center_x"])),
            "indicator_rect": list(indicator_candidate.get("rect")) if indicator_candidate.get("rect") is not None else None,
            "source": str(indicator_candidate.get("source") or "raw"),
        }

    def _recall_indicator_memory(
        self,
        *,
        profile_name: str,
        profile: dict[str, Any],
        zone_region: Region,
        indicator_region: Region,
        zone_left_local: int,
        zone_right_local: int,
        deadband_px: int,
        raw_indicator_detected: bool,
    ) -> dict[str, Any] | None:
        memory_window_ms = int(profile.get("indicator_memory_ms") or 0)
        if memory_window_ms <= 0:
            return None
        memory = self._indicator_memory.get(profile_name)
        if not memory:
            return None
        age_ms = (time.monotonic() - float(memory["saved_at"])) * 1000.0
        max_age_ms = float(memory.get("max_age_ms") or memory_window_ms)
        if age_ms > max_age_ms:
            return None

        indicator_x = int(memory["indicator_x"])
        zone_left = int(zone_region[0] + zone_left_local)
        zone_right = int(zone_region[0] + zone_right_local)
        if indicator_x < zone_left - 32 or indicator_x > zone_right + 32:
            return None

        local_x = float(indicator_x - indicator_region[0])
        if local_x < 0 or local_x > indicator_region[2]:
            return None

        return self._build_duel_result(
            zone_region=zone_region,
            indicator_region=indicator_region,
            zone_left_local=zone_left_local,
            zone_right_local=zone_right_local,
            indicator_candidate={
                "center_x": local_x,
                "rect": memory.get("indicator_rect"),
            },
            deadband_px=deadband_px,
            hold_margin_px=int(profile.get("control_hold_margin_px") or 0),
            indicator_source="memory",
            raw_indicator_detected=raw_indicator_detected,
            indicator_memory_age_ms=age_ms,
        )

    def _extract_zone_bounds(self, zone_mask: np.ndarray, *, min_component_area: int = 20) -> tuple[int, int] | None:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(zone_mask, 8)
        kept_components: list[tuple[int, int]] = []
        for label in range(1, num_labels):
            x, _, width, _, area = [int(value) for value in stats[label]]
            if area < int(min_component_area):
                continue
            kept_components.append((x, x + width - 1))

        if not kept_components:
            return None

        zone_left = min(component[0] for component in kept_components)
        zone_right = max(component[1] for component in kept_components)
        if zone_right - zone_left < 24:
            return None
        return zone_left, zone_right

    def _extract_indicator_candidate(
        self,
        primary_mask: np.ndarray,
        combined_mask: np.ndarray,
        *,
        zone_left_local: int | None = None,
        zone_right_local: int | None = None,
    ) -> dict[str, Any] | None:
        primary_component = self._extract_indicator_component(
            primary_mask,
            zone_left_local=zone_left_local,
            zone_right_local=zone_right_local,
        )
        if primary_component is not None:
            primary_component["source"] = "primary_component"
            return primary_component

        dual_component = self._extract_indicator_component(
            combined_mask,
            zone_left_local=zone_left_local,
            zone_right_local=zone_right_local,
        )
        if dual_component is not None:
            dual_component["source"] = "dual_component"
            return dual_component

        projected = self._extract_indicator_projection(
            combined_mask,
            zone_left_local=zone_left_local,
            zone_right_local=zone_right_local,
        )
        if projected is not None:
            projected["source"] = "dual_projection"
            return projected
        return None

    def _extract_indicator_projection(
        self,
        indicator_mask: np.ndarray,
        *,
        zone_left_local: int | None = None,
        zone_right_local: int | None = None,
    ) -> dict[str, Any] | None:
        if indicator_mask.size == 0:
            return None

        width = int(indicator_mask.shape[1])
        search_left = 0
        search_right = width - 1
        if search_right < search_left:
            return None

        window = indicator_mask[:, search_left:search_right + 1]
        if window.size == 0:
            return None

        column_strength = np.sum(window > 0, axis=0).astype(np.int32)
        if column_strength.size == 0:
            return None

        baseline = int(np.median(column_strength))
        peak = int(column_strength.max())
        peak_threshold = max(baseline + 2, 5)
        if peak < peak_threshold:
            return None

        peak_index = int(np.argmax(column_strength))
        expansion_threshold = max(baseline + 1, peak - 2, 4)
        left = peak_index
        while left > 0 and int(column_strength[left - 1]) >= expansion_threshold:
            left -= 1
        right = peak_index
        while right + 1 < column_strength.size and int(column_strength[right + 1]) >= expansion_threshold:
            right += 1

        local_left = int(search_left + left)
        local_right = int(search_left + right)
        candidate_slice = indicator_mask[:, local_left:local_right + 1]
        if candidate_slice.size == 0:
            return None

        row_strength = np.sum(candidate_slice > 0, axis=1)
        active_rows = np.where(row_strength > 0)[0]
        if active_rows.size == 0:
            return None

        y0 = int(active_rows[0])
        y1 = int(active_rows[-1])
        width_px = int(local_right - local_left + 1)
        height_px = int(y1 - y0 + 1)
        area = int(np.count_nonzero(candidate_slice))
        if height_px < 5 or area < 6:
            return None

        return {
            "center_x": float(local_left + (width_px / 2.0)),
            "rect": [local_left, y0, width_px, height_px],
            "area": area,
        }

    def _extract_indicator_component(
        self,
        indicator_mask: np.ndarray,
        *,
        zone_left_local: int | None = None,
        zone_right_local: int | None = None,
    ) -> dict[str, Any] | None:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(indicator_mask, 8)
        best_component: dict[str, Any] | None = None
        best_score = -1.0
        for label in range(1, num_labels):
            x, y, width, height, area = [int(value) for value in stats[label]]
            if area < 6 or height < 5:
                continue
            center_x = x + (width / 2.0)
            score = float(height * 10 + area)
            if score <= best_score:
                continue
            best_score = score
            best_component = {
                "center_x": float(center_x),
                "rect": [x, y, width, height],
                "area": area,
            }
        return best_component

    def _empty_duel_result(
        self,
        region: Region,
        deadband_px: int,
        *,
        reason: str,
        zone_region: Region | None = None,
        indicator_region: Region | None = None,
    ) -> dict[str, Any]:
        return {
            "found": False,
            "reason": reason,
            "region": list(region),
            "zone_region": list(zone_region) if zone_region is not None else None,
            "indicator_region": list(indicator_region) if indicator_region is not None else None,
            "zone_detected": False,
            "indicator_detected": False,
            "indicator_raw_detected": False,
            "indicator_source": "none",
            "indicator_memory_age_ms": None,
            "zone_left": None,
            "zone_right": None,
            "zone_center": None,
            "indicator_x": None,
            "indicator_rect": None,
            "error_px": None,
            "center_error_px": None,
            "boundary_error_px": None,
            "deadband_px": int(deadband_px),
            "hold_margin_px": 0,
            "control_advice": "none",
        }

    def _match_required_texts(self, texts: list[str], required: list[str]) -> dict[str, Any]:
        normalized_texts = [str(text).strip() for text in texts if str(text).strip()]
        missing = []
        joined = "\n".join(normalized_texts)
        for token in required:
            if token not in joined:
                missing.append(token)
        return {"matched": not missing, "missing": missing}

    def _analyze_dark_bar(
        self,
        source_image: np.ndarray,
        *,
        region: Region,
        pixel_threshold: int,
        row_ratio: float,
        min_height_px: int,
    ) -> dict[str, Any]:
        crop = self._crop_region(source_image, region)
        if crop.size == 0:
            return {
                "found": False,
                "reason": "empty_region",
                "region": list(region),
                "pixel_threshold": int(pixel_threshold),
                "row_ratio": float(row_ratio),
                "min_height_px": int(min_height_px),
                "max_run_height_px": 0,
                "max_run_rect": None,
            }

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        dark_mask = gray < int(pixel_threshold)
        per_row_ratio = np.mean(dark_mask, axis=1)
        matching_rows = [int(row) for row in np.where(per_row_ratio >= float(row_ratio))[0]]
        runs = self._contiguous_runs(matching_rows)
        best_run: tuple[int, int] | None = None
        best_height = 0
        for start, end in runs:
            height = int(end - start + 1)
            if height <= best_height:
                continue
            best_height = height
            best_run = (int(start), int(end))

        found = best_height >= int(min_height_px)
        max_run_rect = None
        if best_run is not None:
            max_run_rect = [int(region[0]), int(region[1] + best_run[0]), int(region[2]), int(best_height)]
        return {
            "found": bool(found),
            "reason": "ok" if found else "height_below_threshold",
            "region": list(region),
            "pixel_threshold": int(pixel_threshold),
            "row_ratio": float(row_ratio),
            "min_height_px": int(min_height_px),
            "max_run_height_px": int(best_height),
            "max_run_rect": max_run_rect,
            "matching_row_count": len(matching_rows),
        }

    def _resolve_template_path(self, name: str | None) -> Path:
        raw_name = str(name or "").strip()
        if not raw_name:
            return self._template_dir / ""
        path = Path(raw_name)
        if path.is_absolute():
            return path
        return self._template_dir / path

    def _load_template(self, name: str) -> np.ndarray:
        cached = self._template_cache.get(name)
        if cached is not None:
            return cached.copy()
        path = self._template_dir / name
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Fishing template not found: {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._template_cache[name] = rgb
        return rgb.copy()

    def _crop_region(self, source_image: np.ndarray, region: Region) -> np.ndarray:
        x, y, width, height = region
        return source_image[int(y): int(y + height), int(x): int(x + width)].copy()

    def _contiguous_runs(self, values: list[int]) -> list[tuple[int, int]]:
        if not values:
            return []
        runs: list[tuple[int, int]] = []
        start = values[0]
        previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue
            runs.append((start, previous))
            start = value
            previous = value
        runs.append((start, previous))
        return runs

    def _signed_error(self, indicator_x: int, zone_left: int, zone_right: int) -> int:
        if indicator_x < zone_left:
            return indicator_x - zone_left
        if indicator_x > zone_right:
            return indicator_x - zone_right
        return 0

    def _resolve_duel_regions(self, profile: dict[str, Any]) -> tuple[Region, Region, Region]:
        zone_region = tuple(int(value) for value in profile.get("zone_meter_region") or profile["duel_meter_region"])
        indicator_region = tuple(
            int(value) for value in profile.get("indicator_meter_region") or profile["duel_meter_region"]
        )
        return zone_region, indicator_region, self._merge_regions(zone_region, indicator_region)

    def _merge_regions(self, first: Region, second: Region) -> Region:
        left = min(int(first[0]), int(second[0]))
        top = min(int(first[1]), int(second[1]))
        right = max(int(first[0] + first[2]), int(second[0] + second[2]))
        bottom = max(int(first[1] + first[3]), int(second[1] + second[3]))
        return left, top, max(right - left, 1), max(bottom - top, 1)

    def _pad_image_to_height(self, image: np.ndarray, target_height: int) -> np.ndarray:
        if image.shape[0] >= target_height:
            return image
        pad_bottom = max(int(target_height - image.shape[0]), 0)
        return cv2.copyMakeBorder(
            image,
            0,
            pad_bottom,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    def _held_action_label(self, held_action_name: str | None) -> str:
        if held_action_name == "fish_left":
            return "holding_a"
        if held_action_name == "fish_right":
            return "holding_d"
        return "released"

    def _coerce_region(self, value: Any) -> Region:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"Expected a [x, y, width, height] region, got: {value!r}")
        x, y, width, height = [int(item) for item in value]
        if width <= 0 or height <= 0:
            raise ValueError(f"Region width/height must be > 0, got: {value!r}")
        return x, y, width, height

    def _coerce_point(self, value: Any) -> tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Expected a [x, y] point, got: {value!r}")
        return int(value[0]), int(value[1])

    def _coerce_size(self, value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
        if value is None:
            return default
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Expected a [width, height] size, got: {value!r}")
        width, height = [int(item) for item in value]
        if width <= 0 or height <= 0:
            raise ValueError(f"Size must be > 0, got: {value!r}")
        return width, height

    def _coerce_text_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if not isinstance(value, (list, tuple)):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _profile_int(self, payload: dict[str, Any], key: str, default: int) -> int:
        value = payload.get(key, default)
        if value is None:
            value = default
        return int(value)

    def _profile_float(self, payload: dict[str, Any], key: str, default: float) -> float:
        value = payload.get(key, default)
        if value is None:
            value = default
        return float(value)

    def _profile_bool(self, payload: dict[str, Any], key: str, default: bool) -> bool:
        value = payload.get(key, default)
        if value is None:
            return bool(default)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def _profile_str(self, payload: dict[str, Any], key: str, default: str) -> str:
        value = payload.get(key, default)
        if value is None:
            value = default
        return str(value).strip() or str(default)

    def _profile_optional_str(self, payload: dict[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _profile_threshold(self, payload: dict[str, Any], key: str, default: float) -> float:
        value = self._profile_float(payload, key, default)
        return max(min(float(value), 1.0), 0.0)


    def _coerce_hsv_pair(
        self,
        value: Any,
        *,
        default: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return default
        first = self._coerce_triplet(value[0])
        second = self._coerce_triplet(value[1])
        return first, second

    def _coerce_triplet(self, value: Any) -> tuple[int, int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"Expected a [v1, v2, v3] triplet, got: {value!r}")
        return int(value[0]), int(value[1]), int(value[2])

    def _standard_hsv_pair_to_cv2(
        self,
        bounds: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        lower_raw, upper_raw = bounds
        h_low, h_high = sorted((int(lower_raw[0]), int(upper_raw[0])))
        s_low, s_high = sorted((int(lower_raw[1]), int(upper_raw[1])))
        v_low, v_high = sorted((int(lower_raw[2]), int(upper_raw[2])))
        lower = (
            max(min(int(round(h_low / 2.0)), 179), 0),
            max(min(int(round(s_low * 255.0 / 100.0)), 255), 0),
            max(min(int(round(v_low * 255.0 / 100.0)), 255), 0),
        )
        upper = (
            max(min(int(round(h_high / 2.0)), 179), 0),
            max(min(int(round(s_high * 255.0 / 100.0)), 255), 0),
            max(min(int(round(v_high * 255.0 / 100.0)), 255), 0),
        )
        return lower, upper


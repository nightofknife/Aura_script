"""Cafe order detection for the Yihuan plan."""

from __future__ import annotations

from pathlib import Path
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
            "order_search_region": self._coerce_region(payload.get("order_search_region"), default=(220, 60, 720, 250)),
            "order_templates": order_templates,
            "template_threshold": float(payload.get("template_threshold", 0.78) or 0.78),
            "template_scales": self._coerce_float_list(
                payload.get("template_scales"),
                default=(0.65, 0.85, 1.0, 1.15),
            ),
            "match_nms_distance_px": max(int(payload.get("match_nms_distance_px", 35) or 35), 1),
            "match_peak_window_px": max(int(payload.get("match_peak_window_px", 9) or 9), 1),
            "max_candidates_per_template_scale": max(int(payload.get("max_candidates_per_template_scale", 80) or 80), 1),
            "coffee_station_point": self._coerce_point(payload.get("coffee_station_point"), default=(1035, 665)),
            "coffee_stock_point": self._coerce_point(payload.get("coffee_stock_point"), default=(1190, 665)),
            "coffee_stock_points": self._coerce_point_list(
                payload.get("coffee_stock_points"),
                default=((1145, 665), (1190, 665), (1235, 665)),
            ),
            "glass_point": self._coerce_point(payload.get("glass_point"), default=(1200, 535)),
            "coffee_cup_point": self._coerce_point(payload.get("coffee_cup_point"), default=(825, 525)),
            "cream_point": self._coerce_point(payload.get("cream_point"), default=(925, 430)),
            "latte_art_point": self._coerce_point(payload.get("latte_art_point"), default=(1030, 430)),
            "coffee_batch_size": max(int(payload.get("coffee_batch_size", 3) or 3), 1),
            "coffee_make_sec": max(float(payload.get("coffee_make_sec", 1.5) or 1.5), 0.0),
            "start_game_delay_sec": max(float(payload.get("start_game_delay_sec", 1.0) or 1.0), 0.0),
            "click_repeat_count": max(int(payload.get("click_repeat_count", 3) or 3), 1),
            "click_hold_ms": max(int(payload.get("click_hold_ms", 100) or 100), 0),
            "click_repeat_interval_ms": max(int(payload.get("click_repeat_interval_ms", 100) or 100), 0),
            "step_delay_ms": max(int(payload.get("step_delay_ms", 100) or 100), 0),
            "craft_delay_ms": max(int(payload.get("craft_delay_ms", 100) or 100), 0),
            "poll_ms": max(int(payload.get("poll_ms", 120) or 120), 0),
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
        region = profile["order_search_region"]
        crop = self._crop_region(source_image, region)
        if crop.size == 0:
            return []

        candidates: list[dict[str, Any]] = []
        for recipe_id, template_name in profile["order_templates"].items():
            template, mask = self._load_template(template_name)
            candidates.extend(
                self._match_template_multiscale(
                    crop,
                    template,
                    mask,
                    recipe_id=recipe_id,
                    region=region,
                    profile=profile,
                )
            )

        kept = self._nms_by_center_distance(candidates, int(profile["match_nms_distance_px"]))
        kept.sort(key=lambda item: (float(item["center_x"]), float(item["center_y"])))
        return kept

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

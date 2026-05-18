"""Visual helpers for Yihuan's four-lane rhythm mini-game."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from packages.aura_core.api import service_info


Point = tuple[int, int]
Region = tuple[int, int, int, int]


@service_info(
    alias="yihuan_rhythm",
    public=True,
    singleton=True,
    description="Detect Yihuan four-lane rhythm notes and screen phases.",
)
class YihuanRhythmService:
    """Small-ROI detectors for the Yihuan drum rhythm mini-game."""

    _DEFAULT_PROFILE = "default_1280x720_cn"
    _DEFAULT_LANES = {
        "d": {"key": "d", "point": (295, 531)},
        "f": {"key": "f", "point": (519, 531)},
        "j": {"key": "j", "point": (760, 531)},
        "k": {"key": "k", "point": (985, 531)},
    }

    def __init__(self) -> None:
        self._plan_root = Path(__file__).resolve().parents[2]
        self._profile_dir = self._plan_root / "data" / "rhythm"
        self._profile_cache: dict[str, dict[str, Any]] = {}

    def load_profile(self, profile_name: str | None = None) -> dict[str, Any]:
        resolved_name = str(profile_name or self._DEFAULT_PROFILE).strip() or self._DEFAULT_PROFILE
        cached = self._profile_cache.get(resolved_name)
        if cached is not None:
            return dict(cached)

        profile_path = self._profile_dir / f"{resolved_name}.yaml"
        if not profile_path.is_file():
            raise FileNotFoundError(f"Rhythm profile not found: {profile_path}")

        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            payload = {}

        lane_order = self._coerce_lane_order(payload.get("lane_order"))
        lanes = self._coerce_lanes(payload.get("lanes"), lane_order=lane_order)
        normalized = {
            "profile_name": str(payload.get("profile_name") or resolved_name),
            "client_size": self._coerce_size(payload.get("client_size"), default=(1280, 720)),
            "lane_order": lane_order,
            "lanes": lanes,
            "detect_radius": self._coerce_point(payload.get("detect_radius"), default=(5, 10)),
            "brightness_threshold": max(self._coerce_int(payload.get("brightness_threshold"), 100), 0),
            "dark_ratio_threshold": min(max(self._coerce_float(payload.get("dark_ratio_threshold"), 0.06), 0.0), 1.0),
            "retrigger_interval_ms": max(self._coerce_int(payload.get("retrigger_interval_ms"), 85), 0),
            "key_down_ms": max(self._coerce_int(payload.get("key_down_ms"), 20), 1),
            "frame_interval_ms": max(self._coerce_int(payload.get("frame_interval_ms"), 8), 0),
            "finish_check_interval_ms": max(self._coerce_int(payload.get("finish_check_interval_ms"), 250), 0),
            "start_song_point": self._coerce_point(payload.get("start_song_point"), default=(1070, 672)),
            "start_button_region": self._coerce_region(payload.get("start_button_region"), default=(930, 640, 270, 65)),
            "song_select_pink_region": self._coerce_region(
                payload.get("song_select_pink_region"),
                default=(0, 390, 1280, 255),
            ),
            "song_select_min_pink_ratio": min(
                max(self._coerce_float(payload.get("song_select_min_pink_ratio"), 0.12), 0.0),
                1.0,
            ),
            "song_select_min_start_white_ratio": min(
                max(self._coerce_float(payload.get("song_select_min_start_white_ratio"), 0.04), 0.0),
                1.0,
            ),
            "start_timeout_sec": max(self._coerce_float(payload.get("start_timeout_sec"), 15.0), 0.1),
            "post_start_delay_ms": max(self._coerce_int(payload.get("post_start_delay_ms"), 1000), 0),
            "result_low_s_region": self._coerce_region(payload.get("result_low_s_region"), default=(0, 0, 1280, 720)),
            "result_top_low_s_region": self._coerce_region(
                payload.get("result_top_low_s_region"),
                default=(0, 0, 1280, 430),
            ),
            "result_grade_region": self._coerce_region(payload.get("result_grade_region"), default=(540, 60, 220, 210)),
            "result_min_low_s_ratio": min(max(self._coerce_float(payload.get("result_min_low_s_ratio"), 0.90), 0.0), 1.0),
            "result_min_top_low_s_ratio": min(
                max(self._coerce_float(payload.get("result_min_top_low_s_ratio"), 0.75), 0.0),
                1.0,
            ),
            "result_min_grade_white_ratio": min(
                max(self._coerce_float(payload.get("result_min_grade_white_ratio"), 0.08), 0.0),
                1.0,
            ),
            "result_min_grade_vivid_ratio": min(
                max(self._coerce_float(payload.get("result_min_grade_vivid_ratio"), 0.12), 0.0),
                1.0,
            ),
            "result_close_point": self._coerce_point(payload.get("result_close_point"), default=(1219, 58)),
            "result_exit_delay_ms": max(self._coerce_int(payload.get("result_exit_delay_ms"), 1000), 0),
            "result_return_timeout_sec": max(self._coerce_float(payload.get("result_return_timeout_sec"), 10.0), 0.1),
            "song_timeout_sec": max(self._coerce_float(payload.get("song_timeout_sec"), 240.0), 0.1),
            "song_select_completion_min_sec": max(
                self._coerce_float(payload.get("song_select_completion_min_sec"), 8.0),
                0.0,
            ),
            "debug_snapshot_dir": str(payload.get("debug_snapshot_dir") or "tmp/rhythm_debug"),
            "debug_snapshot_interval_sec": max(
                self._coerce_float(payload.get("debug_snapshot_interval_sec"), 1.0),
                0.0,
            ),
            "debug_snapshot_max_count": max(self._coerce_int(payload.get("debug_snapshot_max_count"), 40), 0),
        }
        self._profile_cache[resolved_name] = normalized
        return dict(normalized)

    def _resolve_profile(
        self,
        profile_name: str | None = None,
        profile: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if profile is not None:
            return dict(profile)
        return self.load_profile(profile_name)

    def analyze_notes(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
        profile: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self._resolve_profile(profile_name, profile)
        image = self._as_rgb_image(source_image)
        radius_x, radius_y = profile["detect_radius"]
        threshold = int(profile["brightness_threshold"])
        lanes: dict[str, dict[str, Any]] = {}

        for lane_id in profile["lane_order"]:
            lane = dict(profile["lanes"][lane_id])
            point = self.scale_point(image, lane["point"], profile=profile)
            crop = self._crop_center(image, point, radius_x=radius_x, radius_y=radius_y)
            if crop.size == 0:
                dark_ratio = 0.0
                mean_brightness = 0.0
                min_brightness = 0.0
                max_brightness = 0.0
            else:
                brightness = crop[:, :, :3].mean(axis=2)
                dark_ratio = float((brightness < threshold).mean())
                mean_brightness = float(brightness.mean())
                min_brightness = float(brightness.min())
                max_brightness = float(brightness.max())
            has_note = dark_ratio >= float(profile["dark_ratio_threshold"])
            lanes[lane_id] = {
                "lane": lane_id,
                "key": str(lane["key"]),
                "point": [int(point[0]), int(point[1])],
                "dark_ratio": dark_ratio,
                "mean_brightness": mean_brightness,
                "min_brightness": min_brightness,
                "max_brightness": max_brightness,
                "has_note": bool(has_note),
            }

        return {
            "profile_name": profile["profile_name"],
            "capture_size": [int(image.shape[1]), int(image.shape[0])],
            "lanes": lanes,
            "lane_order": list(profile["lane_order"]),
            "thresholds": {
                "brightness": threshold,
                "dark_ratio": float(profile["dark_ratio_threshold"]),
            },
        }

    def analyze_song_select(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
        profile: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self._resolve_profile(profile_name, profile)
        image = self._as_rgb_image(source_image)
        pink_region = self.scale_region(image, profile["song_select_pink_region"], profile=profile)
        start_region = self.scale_region(image, profile["start_button_region"], profile=profile)
        pink_crop = self._crop_region(image, pink_region)
        start_crop = self._crop_region(image, start_region)
        pink_ratio = self._pink_ratio(pink_crop)
        start_white_ratio = self._white_ratio(start_crop)
        found = (
            pink_ratio >= float(profile["song_select_min_pink_ratio"])
            and start_white_ratio >= float(profile["song_select_min_start_white_ratio"])
        )
        return {
            "found": bool(found),
            "pink_ratio": pink_ratio,
            "pink_threshold": float(profile["song_select_min_pink_ratio"]),
            "start_white_ratio": start_white_ratio,
            "start_white_threshold": float(profile["song_select_min_start_white_ratio"]),
            "pink_region": list(pink_region),
            "start_button_region": list(start_region),
        }

    def analyze_result_screen(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
        profile: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self._resolve_profile(profile_name, profile)
        image = self._as_rgb_image(source_image)
        low_s_region = self.scale_region(image, profile["result_low_s_region"], profile=profile)
        top_low_s_region = self.scale_region(image, profile["result_top_low_s_region"], profile=profile)
        grade_region = self.scale_region(image, profile["result_grade_region"], profile=profile)
        low_s_crop = self._crop_region(image, low_s_region)
        top_low_s_crop = self._crop_region(image, top_low_s_region)
        grade_crop = self._crop_region(image, grade_region)
        low_s_ratio = self._low_saturation_ratio(low_s_crop)
        top_low_s_ratio = self._low_saturation_ratio(top_low_s_crop)
        grade_white_ratio = self._white_ratio(grade_crop)
        grade_vivid_ratio = self._vivid_ratio(grade_crop)
        legacy_gray_result = (
            low_s_ratio >= float(profile["result_min_low_s_ratio"])
            and grade_white_ratio >= float(profile["result_min_grade_white_ratio"])
        )
        vivid_grade_result = (
            top_low_s_ratio >= float(profile["result_min_top_low_s_ratio"])
            and grade_vivid_ratio >= float(profile["result_min_grade_vivid_ratio"])
        )
        found = legacy_gray_result or vivid_grade_result
        return {
            "found": bool(found),
            "low_s_ratio": low_s_ratio,
            "low_s_threshold": float(profile["result_min_low_s_ratio"]),
            "top_low_s_ratio": top_low_s_ratio,
            "top_low_s_threshold": float(profile["result_min_top_low_s_ratio"]),
            "grade_white_ratio": grade_white_ratio,
            "grade_white_threshold": float(profile["result_min_grade_white_ratio"]),
            "grade_vivid_ratio": grade_vivid_ratio,
            "grade_vivid_threshold": float(profile["result_min_grade_vivid_ratio"]),
            "low_s_region": list(low_s_region),
            "top_low_s_region": list(top_low_s_region),
            "grade_region": list(grade_region),
        }

    def analyze_phase(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
        profile: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self._resolve_profile(profile_name, profile)
        result = self.analyze_result_screen(source_image, profile=profile)
        song_select = self.analyze_song_select(source_image, profile=profile)
        if result["found"]:
            phase = "result"
        elif song_select["found"]:
            phase = "song_select"
        else:
            phase = "playing"
        return {
            "phase": phase,
            "result": result,
            "song_select": song_select,
        }

    def draw_debug_overlay(
        self,
        source_image: np.ndarray,
        note_state: Mapping[str, Any],
        *,
        profile_name: str | None = None,
        profile: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        profile = self._resolve_profile(profile_name, profile)
        image = self._as_rgb_image(source_image).copy()
        radius_x, radius_y = profile["detect_radius"]
        lanes = dict(note_state.get("lanes") or {})
        for lane_id in profile["lane_order"]:
            lane_state = dict(lanes.get(lane_id) or {})
            point = lane_state.get("point") or self.scale_point(image, profile["lanes"][lane_id]["point"], profile=profile)
            x, y = int(point[0]), int(point[1])
            color = (255, 80, 80) if lane_state.get("has_note") else (80, 220, 255)
            cv2.rectangle(image, (x - radius_x, y - radius_y), (x + radius_x, y + radius_y), color, 1)
            label = f"{lane_id}:{float(lane_state.get('dark_ratio') or 0.0):.2f}"
            cv2.putText(image, label, (x - 24, y - radius_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        return image

    def scale_point(
        self,
        source_image: np.ndarray | None,
        point: Point,
        *,
        profile: Mapping[str, Any] | None = None,
    ) -> Point:
        profile_map = dict(profile or self.load_profile())
        scale_x, scale_y = self._scale_factors(source_image, profile_map)
        return int(round(int(point[0]) * scale_x)), int(round(int(point[1]) * scale_y))

    def scale_region(
        self,
        source_image: np.ndarray,
        region: Region,
        *,
        profile: Mapping[str, Any] | None = None,
    ) -> Region:
        profile_map = dict(profile or self.load_profile())
        scale_x, scale_y = self._scale_factors(source_image, profile_map)
        x, y, w, h = region
        return (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            max(int(round(w * scale_x)), 1),
            max(int(round(h * scale_y)), 1),
        )

    @staticmethod
    def _as_rgb_image(source_image: np.ndarray) -> np.ndarray:
        image = np.asarray(source_image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Rhythm analysis expects an RGB image with at least 3 channels.")
        return image[:, :, :3]

    @staticmethod
    def _scale_factors(source_image: np.ndarray | None, profile: Mapping[str, Any]) -> tuple[float, float]:
        client_w, client_h = profile.get("client_size") or (1280, 720)
        if source_image is None:
            return 1.0, 1.0
        height, width = int(source_image.shape[0]), int(source_image.shape[1])
        return width / max(float(client_w), 1.0), height / max(float(client_h), 1.0)

    @staticmethod
    def _crop_region(source_image: np.ndarray, region: Region) -> np.ndarray:
        x, y, w, h = region
        height, width = source_image.shape[:2]
        x0 = max(int(x), 0)
        y0 = max(int(y), 0)
        x1 = min(max(int(x + w), x0), width)
        y1 = min(max(int(y + h), y0), height)
        return source_image[y0:y1, x0:x1]

    def _crop_center(
        self,
        source_image: np.ndarray,
        point: Point,
        *,
        radius_x: int,
        radius_y: int,
    ) -> np.ndarray:
        x, y = int(point[0]), int(point[1])
        return self._crop_region(
            source_image,
            (x - int(radius_x), y - int(radius_y), int(radius_x) * 2 + 1, int(radius_y) * 2 + 1),
        )

    @staticmethod
    def _white_ratio(image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2HSV)
        return float(((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 150)).mean())

    @staticmethod
    def _low_saturation_ratio(image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2HSV)
        return float((hsv[:, :, 1] < 45).mean())

    @staticmethod
    def _vivid_ratio(image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2HSV)
        return float(((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 140)).mean())

    @staticmethod
    def _pink_ratio(image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        rgb = image[:, :, :3]
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]
        mask = (r >= 160) & (g <= 100) & (b >= 80) & (b <= 210) & ((r - g) >= 60) & ((b - g) >= 20)
        return float(mask.mean())

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _coerce_size(self, value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
        point = self._coerce_point(value, default=default)
        return max(int(point[0]), 1), max(int(point[1]), 1)

    def _coerce_point(self, value: Any, *, default: Point) -> Point:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return self._coerce_int(value[0], default[0]), self._coerce_int(value[1], default[1])
        return int(default[0]), int(default[1])

    def _coerce_region(self, value: Any, *, default: Region) -> Region:
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return (
                self._coerce_int(value[0], default[0]),
                self._coerce_int(value[1], default[1]),
                max(self._coerce_int(value[2], default[2]), 1),
                max(self._coerce_int(value[3], default[3]), 1),
            )
        return int(default[0]), int(default[1]), int(default[2]), int(default[3])

    def _coerce_lane_order(self, value: Any) -> list[str]:
        if isinstance(value, (list, tuple)):
            lanes = [str(item).strip().lower() for item in value if str(item).strip()]
            if lanes:
                return lanes
        return ["d", "f", "j", "k"]

    def _coerce_lanes(self, value: Any, *, lane_order: list[str]) -> dict[str, dict[str, Any]]:
        payload = dict(value or {}) if isinstance(value, Mapping) else {}
        lanes: dict[str, dict[str, Any]] = {}
        for lane_id in lane_order:
            default_lane = dict(self._DEFAULT_LANES.get(lane_id) or {"key": lane_id, "point": (0, 0)})
            raw_lane = dict(payload.get(lane_id) or {}) if isinstance(payload.get(lane_id), Mapping) else {}
            lanes[lane_id] = {
                "key": str(raw_lane.get("key") or default_lane["key"]).strip() or str(default_lane["key"]),
                "point": self._coerce_point(raw_lane.get("point"), default=default_lane["point"]),
            }
        return lanes

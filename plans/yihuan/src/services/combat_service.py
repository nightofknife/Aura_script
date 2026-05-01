"""Yihuan-specific combat detection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from packages.aura_core.api import service_info


Region = tuple[int, int, int, int]
Point = tuple[int, int]
HsvTriplet = tuple[int, int, int]


@service_info(
    alias="yihuan_combat",
    public=True,
    singleton=True,
    description="Recognize Yihuan combat HUD state and load combat automation profiles.",
)
class YihuanCombatService:
    """Load combat calibration and convert screenshots into Yihuan combat state."""

    _DEFAULT_PROFILE = "default_1280x720_cn"
    ABILITY_NAMES = ("skill", "ultimate", "arc")

    def __init__(self) -> None:
        self._plan_root = Path(__file__).resolve().parents[2]
        self._profile_dir = self._plan_root / "data" / "combat"
        self._profile_cache: dict[str, dict[str, Any]] = {}

    def load_profile(self, profile_name: str | None = None) -> dict[str, Any]:
        resolved_name = str(profile_name or self._DEFAULT_PROFILE).strip() or self._DEFAULT_PROFILE
        cached = self._profile_cache.get(resolved_name)
        if cached is not None:
            return dict(cached)

        profile_path = self._profile_dir / f"{resolved_name}.yaml"
        if not profile_path.is_file():
            raise FileNotFoundError(f"Yihuan combat profile not found: {profile_path}")

        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            payload = {}

        runtime_payload = dict(payload.get("runtime") or {})
        keys_payload = dict(payload.get("keys") or {})
        regions_payload = dict(payload.get("regions") or {})
        thresholds_payload = dict(payload.get("thresholds") or {})
        hsv_payload = dict(payload.get("hsv_rules") or {})
        strategies_payload = dict(payload.get("strategies") or {})

        normalized = {
            "profile_name": str(payload.get("profile_name") or resolved_name),
            "client_size": self._coerce_size(payload.get("client_size"), default=(1280, 720)),
            "runtime": {
                "poll_ms": max(self._coerce_int(runtime_payload.get("poll_ms"), 100), 10),
                "combat_enter_stable_frames": max(
                    self._coerce_int(runtime_payload.get("combat_enter_stable_frames"), 3),
                    1,
                ),
                "combat_exit_stable_frames": max(
                    self._coerce_int(runtime_payload.get("combat_exit_stable_frames"), 8),
                    1,
                ),
                "unsupported_scene_stable_frames": max(
                    self._coerce_int(runtime_payload.get("unsupported_scene_stable_frames"), 3),
                    1,
                ),
                "input_cooldown_ms": max(self._coerce_int(runtime_payload.get("input_cooldown_ms"), 120), 0),
                "target_retry_interval_ms": max(
                    self._coerce_int(runtime_payload.get("target_retry_interval_ms"), 1000),
                    0,
                ),
                "post_switch_delay_ms": max(self._coerce_int(runtime_payload.get("post_switch_delay_ms"), 350), 0),
                "idle_timeout_sec": max(self._coerce_float(runtime_payload.get("idle_timeout_sec"), 10.0), 0.0),
                "dry_run_max_seconds": max(
                    self._coerce_float(runtime_payload.get("dry_run_max_seconds"), 2.0),
                    0.1,
                ),
                "trace_limit": max(self._coerce_int(runtime_payload.get("trace_limit"), 240), 1),
            },
            "keys": {
                "normal_attack": str(keys_payload.get("normal_attack") or "mouse_left"),
                "skill": str(keys_payload.get("skill") or "e"),
                "ultimate": str(keys_payload.get("ultimate") or "q"),
                "arc": str(keys_payload.get("arc") or "r"),
                "target": str(keys_payload.get("target") or "mouse_middle"),
                "switch_1": str(keys_payload.get("switch_1") or "1"),
                "switch_2": str(keys_payload.get("switch_2") or "2"),
                "switch_3": str(keys_payload.get("switch_3") or "3"),
                "switch_4": str(keys_payload.get("switch_4") or "4"),
            },
            "release_keys": [
                str(item)
                for item in (payload.get("release_keys") or ["w", "a", "s", "d", "e", "q", "r", "1", "2", "3", "4"])
            ],
            "regions": {
                "target_icon": self._coerce_region(regions_payload.get("target_icon"), default=(560, 40, 160, 80)),
                "enemy_level_text": self._coerce_region(
                    regions_payload.get("enemy_level_text"),
                    default=(520, 20, 240, 80),
                ),
                "enemy_health_bar": self._coerce_region(
                    regions_payload.get("enemy_health_bar"),
                    default=(430, 65, 420, 45),
                ),
                "boss_health_bar": self._coerce_region(
                    regions_payload.get("boss_health_bar"),
                    default=(260, 620, 760, 65),
                ),
                "team_panel": self._coerce_region(regions_payload.get("team_panel"), default=(1020, 120, 240, 330)),
                "skill": self._coerce_region(regions_payload.get("skill"), default=(1015, 585, 58, 58)),
                "ultimate": self._coerce_region(regions_payload.get("ultimate"), default=(1090, 575, 70, 70)),
                "arc": self._coerce_region(regions_payload.get("arc"), default=(935, 585, 58, 58)),
            },
            "current_slot_regions": [
                self._coerce_region(item, default=(0, 0, 1, 1))
                for item in (
                    payload.get("current_slot_regions")
                    or [(1150, 135, 88, 58), (1150, 205, 88, 58), (1150, 275, 88, 58), (1150, 345, 88, 58)]
                )
            ],
            "hsv_rules": {
                "target_present": self._coerce_hsv_rule(hsv_payload.get("target_present"), default_min_ratio=0.035),
                "level_text": self._coerce_hsv_rule(hsv_payload.get("level_text"), default_min_ratio=0.025),
                "enemy_health_red": self._coerce_hsv_rule(
                    hsv_payload.get("enemy_health_red"),
                    default_min_ratio=0.02,
                ),
                "boss_health_red": self._coerce_hsv_rule(
                    hsv_payload.get("boss_health_red"),
                    default_min_ratio=0.02,
                ),
                "team_hud": self._coerce_hsv_rule(hsv_payload.get("team_hud"), default_min_ratio=0.025),
                "slot_active": self._coerce_hsv_rule(hsv_payload.get("slot_active"), default_min_ratio=0.018),
                "ability_ready": self._coerce_hsv_rule(hsv_payload.get("ability_ready"), default_min_ratio=0.03),
            },
            "thresholds": {
                "supported_scene_min_features": max(
                    self._coerce_int(thresholds_payload.get("supported_scene_min_features"), 1),
                    1,
                ),
                "combat_min_features": max(self._coerce_int(thresholds_payload.get("combat_min_features"), 2), 1),
                "confidence_feature_count": max(
                    self._coerce_int(thresholds_payload.get("confidence_feature_count"), 5),
                    1,
                ),
            },
            "strategies": self._coerce_strategies(strategies_payload),
            "debug_snapshot_dir": str(payload.get("debug_snapshot_dir") or "tmp/combat_debug"),
        }
        self._profile_cache[resolved_name] = normalized
        return dict(normalized)

    def analyze_frame(self, source_image: np.ndarray, *, profile_name: str | None = None) -> dict[str, Any]:
        if source_image is None or not isinstance(source_image, np.ndarray) or source_image.size == 0:
            raise ValueError("Combat source image is empty.")

        profile = self.load_profile(profile_name)
        regions = dict(profile["regions"])
        hsv_rules = dict(profile["hsv_rules"])

        target_ratio = self._hsv_ratio_for_region(source_image, regions["target_icon"], hsv_rules["target_present"], profile)
        level_ratio = self._hsv_ratio_for_region(source_image, regions["enemy_level_text"], hsv_rules["level_text"], profile)
        enemy_health_ratio = self._hsv_ratio_for_region(
            source_image,
            regions["enemy_health_bar"],
            hsv_rules["enemy_health_red"],
            profile,
        )
        boss_health_ratio = self._hsv_ratio_for_region(
            source_image,
            regions["boss_health_bar"],
            hsv_rules["boss_health_red"],
            profile,
        )
        team_ratio = self._hsv_ratio_for_region(source_image, regions["team_panel"], hsv_rules["team_hud"], profile)
        ability_ratios = {
            name: self._hsv_ratio_for_region(source_image, regions[name], hsv_rules["ability_ready"], profile)
            for name in self.ABILITY_NAMES
        }
        slot_ratios = [
            self._hsv_ratio_for_region(source_image, region, hsv_rules["slot_active"], profile)
            for region in profile["current_slot_regions"]
        ]

        target_found = target_ratio >= hsv_rules["target_present"]["min_ratio"]
        enemy_level_found = level_ratio >= hsv_rules["level_text"]["min_ratio"]
        enemy_health_found = enemy_health_ratio >= hsv_rules["enemy_health_red"]["min_ratio"]
        boss_found = boss_health_ratio >= hsv_rules["boss_health_red"]["min_ratio"]
        team_found = team_ratio >= hsv_rules["team_hud"]["min_ratio"]
        ability_available = {
            name: ability_ratios[name] >= hsv_rules["ability_ready"]["min_ratio"]
            for name in self.ABILITY_NAMES
        }

        current_slot = self._detect_current_slot(slot_ratios, hsv_rules["slot_active"]["min_ratio"])
        supported_features = [
            team_found,
            any(ability_available.values()),
            current_slot is not None,
        ]
        combat_features = [
            target_found,
            enemy_level_found,
            enemy_health_found,
            boss_found,
        ]
        supported_feature_count = sum(1 for value in supported_features if value)
        combat_feature_count = sum(1 for value in combat_features if value)

        in_supported_scene = supported_feature_count >= int(profile["thresholds"]["supported_scene_min_features"])
        in_combat = combat_feature_count >= int(profile["thresholds"]["combat_min_features"])
        confidence = min(
            1.0,
            combat_feature_count / float(profile["thresholds"]["confidence_feature_count"]),
        )

        return {
            "profile_name": profile["profile_name"],
            "capture_size": [int(source_image.shape[1]), int(source_image.shape[0])],
            "in_supported_scene": bool(in_supported_scene),
            "in_combat": bool(in_combat),
            "target_found": bool(target_found),
            "enemy_level_found": bool(enemy_level_found),
            "enemy_health_found": bool(enemy_health_found),
            "boss_found": bool(boss_found),
            "team_found": bool(team_found),
            "current_slot": current_slot,
            "team_size": int(len(profile["current_slot_regions"])),
            "skill_available": bool(ability_available["skill"]),
            "ultimate_available": bool(ability_available["ultimate"]),
            "arc_available": bool(ability_available["arc"]),
            "confidence": round(float(confidence), 3),
            "debug": {
                "ratios": {
                    "target": round(float(target_ratio), 4),
                    "enemy_level": round(float(level_ratio), 4),
                    "enemy_health": round(float(enemy_health_ratio), 4),
                    "boss_health": round(float(boss_health_ratio), 4),
                    "team": round(float(team_ratio), 4),
                    "skill": round(float(ability_ratios["skill"]), 4),
                    "ultimate": round(float(ability_ratios["ultimate"]), 4),
                    "arc": round(float(ability_ratios["arc"]), 4),
                    "slots": [round(float(value), 4) for value in slot_ratios],
                },
                "feature_counts": {
                    "supported": int(supported_feature_count),
                    "combat": int(combat_feature_count),
                },
            },
        }

    def scale_region(self, source_image: np.ndarray, region: Region, *, profile: Mapping[str, Any]) -> Region:
        base_w, base_h = profile["client_size"]
        height, width = source_image.shape[:2]
        x, y, region_w, region_h = region
        sx = width / float(base_w)
        sy = height / float(base_h)
        scaled = (
            int(round(x * sx)),
            int(round(y * sy)),
            max(int(round(region_w * sx)), 1),
            max(int(round(region_h * sy)), 1),
        )
        return self._clamp_region(scaled, width=width, height=height)

    def scale_point(self, source_image: np.ndarray, point: Point, *, profile: Mapping[str, Any]) -> Point:
        base_w, base_h = profile["client_size"]
        height, width = source_image.shape[:2]
        x, y = point
        return (
            int(round(x * width / float(base_w))),
            int(round(y * height / float(base_h))),
        )

    def _hsv_ratio_for_region(
        self,
        source_image: np.ndarray,
        region: Region,
        rule: Mapping[str, Any],
        profile: Mapping[str, Any],
    ) -> float:
        scaled_region = self.scale_region(source_image, region, profile=profile)
        crop = self._crop_region(source_image, scaled_region)
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        mask = self._hsv_mask(hsv, rule["lower"], rule["upper"])
        if rule.get("lower_2") is not None and rule.get("upper_2") is not None:
            mask = cv2.bitwise_or(mask, self._hsv_mask(hsv, rule["lower_2"], rule["upper_2"]))
        return float(np.count_nonzero(mask)) / float(mask.size)

    def _detect_current_slot(self, slot_ratios: list[float], min_ratio: float) -> int | None:
        if not slot_ratios:
            return None
        best_index = max(range(len(slot_ratios)), key=lambda idx: slot_ratios[idx])
        if slot_ratios[best_index] < min_ratio:
            return None
        return int(best_index + 1)

    def _crop_region(self, source_image: np.ndarray, region: Region) -> np.ndarray:
        x, y, width, height = region
        return source_image[y : y + height, x : x + width]

    def _hsv_mask(self, hsv: np.ndarray, lower: HsvTriplet, upper: HsvTriplet) -> np.ndarray:
        return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

    def _clamp_region(self, region: Region, *, width: int, height: int) -> Region:
        x, y, region_w, region_h = region
        x = min(max(int(x), 0), max(width - 1, 0))
        y = min(max(int(y), 0), max(height - 1, 0))
        max_w = max(width - x, 1)
        max_h = max(height - y, 1)
        return (x, y, min(max(int(region_w), 1), max_w), min(max(int(region_h), 1), max_h))

    def _coerce_hsv_rule(self, value: Any, *, default_min_ratio: float) -> dict[str, Any]:
        payload = dict(value or {}) if isinstance(value, Mapping) else {}
        return {
            "lower": self._coerce_hsv_triplet(payload.get("lower"), default=(0, 0, 0)),
            "upper": self._coerce_hsv_triplet(payload.get("upper"), default=(179, 255, 255)),
            "lower_2": self._coerce_optional_hsv_triplet(payload.get("lower_2")),
            "upper_2": self._coerce_optional_hsv_triplet(payload.get("upper_2")),
            "min_ratio": min(max(self._coerce_float(payload.get("min_ratio"), default_min_ratio), 0.0), 1.0),
        }

    def _coerce_optional_hsv_triplet(self, value: Any) -> HsvTriplet | None:
        if value is None:
            return None
        return self._coerce_hsv_triplet(value, default=(0, 0, 0))

    def _coerce_hsv_triplet(self, value: Any, *, default: HsvTriplet) -> HsvTriplet:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return (
                min(max(self._coerce_int(value[0], default[0]), 0), 179),
                min(max(self._coerce_int(value[1], default[1]), 0), 255),
                min(max(self._coerce_int(value[2], default[2]), 0), 255),
            )
        return default

    def _coerce_size(self, value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return (
                max(self._coerce_int(value[0], default[0]), 1),
                max(self._coerce_int(value[1], default[1]), 1),
            )
        return default

    def _coerce_region(self, value: Any, *, default: Region) -> Region:
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return (
                self._coerce_int(value[0], default[0]),
                self._coerce_int(value[1], default[1]),
                max(self._coerce_int(value[2], default[2]), 1),
                max(self._coerce_int(value[3], default[3]), 1),
            )
        return default

    def _coerce_strategies(self, value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping) and value:
            return {str(name): dict(config or {}) for name, config in value.items()}
        return {
            "default": {
                "loop": [
                    {"if_available": "arc"},
                    {"if_available": "ultimate"},
                    {"if_available": "skill"},
                    {"normal": {"duration_ms": 900}},
                    {"switch": "next"},
                ]
            }
        }

    def _coerce_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

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
    ABILITY_NAMES = ("skill", "ultimate")

    def __init__(self) -> None:
        self._plan_root = Path(__file__).resolve().parents[2]
        self._profile_dir = self._plan_root / "data" / "combat"
        self._template_dir = self._profile_dir / "templates"
        self._profile_cache: dict[str, dict[str, Any]] = {}
        self._template_cache: dict[Path, np.ndarray | None] = {}
        self._template_mask_cache: dict[Path, np.ndarray | None] = {}

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
        ability_regions_payload = dict(payload.get("ability_regions") or {})
        thresholds_payload = dict(payload.get("thresholds") or {})
        ability_thresholds_payload = dict(payload.get("ability_thresholds") or {})
        hsv_payload = dict(payload.get("hsv_rules") or {})
        templates_payload = dict(payload.get("templates") or {})
        strategies_payload = dict(payload.get("strategies") or {})
        health_rect_payload = dict(payload.get("enemy_health_rect") or {})
        boss_health_rect_payload = dict(payload.get("boss_health_rect") or {})
        enemy_direction_rect_payload = dict(payload.get("enemy_direction_rect") or {})
        remaining_enemy_marker_rect_payload = dict(payload.get("remaining_enemy_marker_rect") or {})
        audio_dodge_payload = dict(payload.get("audio_dodge") or {})
        switch_policy_payload = dict(payload.get("switch_policy") or {})
        ability_schedule_payload = dict(payload.get("ability_schedule") or {})
        combat_exit_payload = dict(payload.get("combat_exit") or {})
        enemy_direction_reacquire_payload = dict(payload.get("enemy_direction_reacquire") or {})
        post_combat_reward_payload = dict(payload.get("post_combat_reward") or {})

        ability_regions = {
            "skill": self._coerce_region(
                ability_regions_payload.get("skill") or regions_payload.get("skill"),
                default=(1068, 586, 106, 106),
            ),
            "ultimate": self._coerce_region(
                ability_regions_payload.get("ultimate") or regions_payload.get("ultimate"),
                default=(1160, 582, 112, 112),
            ),
        }

        normalized = {
            "profile_name": str(payload.get("profile_name") or resolved_name),
            "client_size": self._coerce_size(payload.get("client_size"), default=(1280, 720)),
            "runtime": {
                "poll_ms": max(self._coerce_int(runtime_payload.get("poll_ms"), 80), 10),
                "monitor_scan_interval_sec": max(
                    self._coerce_float(runtime_payload.get("monitor_scan_interval_sec"), 2.0),
                    0.1,
                ),
                "combat_scan_interval_sec": max(
                    self._coerce_float(runtime_payload.get("combat_scan_interval_sec"), 3.0),
                    0.1,
                ),
                "action_loop_sleep_ms": max(self._coerce_int(runtime_payload.get("action_loop_sleep_ms"), 80), 10),
                "combat_enter_stable_frames": max(
                    self._coerce_int(runtime_payload.get("combat_enter_stable_frames"), 2),
                    1,
                ),
                "combat_exit_stable_frames": max(
                    self._coerce_int(runtime_payload.get("combat_exit_stable_frames"), 10),
                    1,
                ),
                "unsupported_scene_stable_frames": max(
                    self._coerce_int(runtime_payload.get("unsupported_scene_stable_frames"), 3),
                    1,
                ),
                "input_cooldown_ms": max(self._coerce_int(runtime_payload.get("input_cooldown_ms"), 120), 0),
                "target_retry_interval_ms": max(
                    self._coerce_int(runtime_payload.get("target_retry_interval_ms"), 800),
                    0,
                ),
                "skill_lockout_ms": max(self._coerce_int(runtime_payload.get("skill_lockout_ms"), 420), 0),
                "ultimate_lockout_ms": max(self._coerce_int(runtime_payload.get("ultimate_lockout_ms"), 720), 0),
                "post_combat_cooldown_ms": max(
                    self._coerce_int(runtime_payload.get("post_combat_cooldown_ms"), 700),
                    0,
                ),
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
                "target": str(keys_payload.get("target") or "mouse_middle"),
                "switch_1": str(keys_payload.get("switch_1") or "1"),
                "switch_2": str(keys_payload.get("switch_2") or "2"),
                "switch_3": str(keys_payload.get("switch_3") or "3"),
                "switch_4": str(keys_payload.get("switch_4") or "4"),
            },
            "release_keys": [
                str(item)
                for item in (payload.get("release_keys") or ["w", "a", "s", "d", "shift", "e", "q", "1", "2", "3", "4"])
            ],
            "regions": {
                "target_search_region": self._coerce_region(
                    regions_payload.get("target_search_region"),
                    default=(200, 120, 880, 320),
                ),
                "enemy_health_search_region": self._coerce_region(
                    regions_payload.get("enemy_health_search_region"),
                    default=(0, 0, 1280, 720),
                ),
                "enemy_direction_left": self._coerce_region(
                    regions_payload.get("enemy_direction_left"),
                    default=(48, 310, 210, 240),
                ),
                "enemy_direction_right": self._coerce_region(
                    regions_payload.get("enemy_direction_right"),
                    default=(1030, 280, 210, 280),
                ),
                "enemy_direction_bottom": self._coerce_region(
                    regions_payload.get("enemy_direction_bottom"),
                    default=(500, 610, 300, 90),
                ),
                "remaining_enemy_marker": self._coerce_region(
                    regions_payload.get("remaining_enemy_marker"),
                    default=(20, 170, 70, 80),
                ),
                "boss_health_bar": self._coerce_region(
                    regions_payload.get("boss_health_bar"),
                    default=(250, 612, 780, 72),
                ),
                "team_panel": self._coerce_region(regions_payload.get("team_panel"), default=(1110, 110, 170, 420)),
                "skill": ability_regions["skill"],
                "ultimate": ability_regions["ultimate"],
                "challenge_success_region": self._coerce_region(
                    regions_payload.get("challenge_success_region"),
                    default=(450, 130, 380, 150),
                ),
                "reward_marker_search_region": self._coerce_region(
                    regions_payload.get("reward_marker_search_region"),
                    default=(120, 110, 900, 520),
                ),
                "claim_memento_prompt_region": self._coerce_region(
                    regions_payload.get("claim_memento_prompt_region"),
                    default=(760, 372, 50, 50),
                ),
            },
            "ability_regions": ability_regions,
            "current_slot_regions": [
                self._coerce_region(item, default=(0, 0, 1, 1))
                for item in (
                    payload.get("current_slot_regions")
                    or [
                        (1138, 150, 110, 84),
                        (1138, 238, 110, 84),
                        (1138, 326, 110, 84),
                        (1138, 414, 110, 84),
                    ]
                )
            ],
            "switch_policy": {
                "mode": str(switch_policy_payload.get("mode") or "fixed_rotation"),
                "empty_skill_loops_before_switch": max(
                    self._coerce_int(switch_policy_payload.get("empty_skill_loops_before_switch"), 2),
                    1,
                ),
                "switch_interval_sec": max(
                    self._coerce_float(switch_policy_payload.get("switch_interval_sec"), 10.0),
                    0.1,
                ),
                "confirm_timeout_ms": max(
                    self._coerce_int(switch_policy_payload.get("confirm_timeout_ms"), 260),
                    0,
                ),
                "confirm_required_matches": max(
                    self._coerce_int(switch_policy_payload.get("confirm_required_matches"), 2),
                    1,
                ),
                "post_switch_delay_ms": max(
                    self._coerce_int(switch_policy_payload.get("post_switch_delay_ms"), 0),
                    0,
                ),
                "failed_switch_cooldown_ms": max(
                    self._coerce_int(switch_policy_payload.get("failed_switch_cooldown_ms"), 480),
                    0,
                ),
            },
            "combat_exit": {
                "confirm_required_scans": max(
                    self._coerce_int(combat_exit_payload.get("confirm_required_scans"), 2),
                    1,
                ),
                "confirm_interval_sec": max(
                    self._coerce_float(combat_exit_payload.get("confirm_interval_sec"), 1.0),
                    0.1,
                ),
                "post_exit_cooldown_ms": max(
                    self._coerce_int(
                        combat_exit_payload.get("post_exit_cooldown_ms"),
                        runtime_payload.get("post_combat_cooldown_ms"),
                    ),
                    0,
                ),
                "challenge_success_immediate": bool(
                    combat_exit_payload.get("challenge_success_immediate", True)
                ),
                "reset_on_reacquire": bool(combat_exit_payload.get("reset_on_reacquire", True)),
            },
            "post_combat_reward": {
                "enabled": bool(post_combat_reward_payload.get("enabled", True)),
                "search_timeout_sec": max(
                    self._coerce_float(post_combat_reward_payload.get("search_timeout_sec"), 14.0),
                    0.1,
                ),
                "walk_timeout_sec": max(
                    self._coerce_float(post_combat_reward_payload.get("walk_timeout_sec"), 26.0),
                    0.1,
                ),
                "scan_interval_sec": max(
                    self._coerce_float(post_combat_reward_payload.get("scan_interval_sec"), 0.18),
                    0.05,
                ),
                "align_target_x": max(
                    self._coerce_int(post_combat_reward_payload.get("align_target_x"), 640),
                    1,
                ),
                "align_tolerance_px": max(
                    self._coerce_int(post_combat_reward_payload.get("align_tolerance_px"), 50),
                    1,
                ),
                "look_pixels_per_px": max(
                    self._coerce_float(post_combat_reward_payload.get("look_pixels_per_px"), 0.55),
                    0.01,
                ),
                "min_turn_pixels": max(
                    self._coerce_int(post_combat_reward_payload.get("min_turn_pixels"), 8),
                    1,
                ),
                "max_turn_pixels": max(
                    self._coerce_int(post_combat_reward_payload.get("max_turn_pixels"), 160),
                    1,
                ),
                "post_turn_delay_ms": max(
                    self._coerce_int(post_combat_reward_payload.get("post_turn_delay_ms"), 120),
                    0,
                ),
                "walk_key": str(post_combat_reward_payload.get("walk_key") or "w"),
                "walk_step_sec": max(
                    self._coerce_float(post_combat_reward_payload.get("walk_step_sec"), 0.24),
                    0.05,
                ),
                "search_turn_pixels": max(
                    self._coerce_int(post_combat_reward_payload.get("search_turn_pixels"), 520),
                    1,
                ),
                "search_turn_interval_sec": max(
                    self._coerce_float(post_combat_reward_payload.get("search_turn_interval_sec"), 0.65),
                    0.1,
                ),
                "prompt_required_scans": max(
                    self._coerce_int(post_combat_reward_payload.get("prompt_required_scans"), 1),
                    1,
                ),
                "interact_key": str(post_combat_reward_payload.get("interact_key") or "f"),
                "post_interact_delay_ms": max(
                    self._coerce_int(post_combat_reward_payload.get("post_interact_delay_ms"), 120),
                    0,
                ),
            },
            "ability_schedule": {
                "skill_interval_sec": max(
                    self._coerce_float(ability_schedule_payload.get("skill_interval_sec"), 3.0),
                    0.1,
                ),
                "ultimate_interval_sec": max(
                    self._coerce_float(ability_schedule_payload.get("ultimate_interval_sec"), 5.0),
                    0.1,
                ),
                "immediate_on_combat_enter": bool(ability_schedule_payload.get("immediate_on_combat_enter", True)),
                "immediate_after_switch": bool(ability_schedule_payload.get("immediate_after_switch", True)),
            },
            "enemy_direction_reacquire": {
                "turn_interval_sec": max(
                    self._coerce_float(enemy_direction_reacquire_payload.get("turn_interval_sec"), 0.9),
                    0.1,
                ),
                "turn_pixels_side": max(
                    self._coerce_int(enemy_direction_reacquire_payload.get("turn_pixels_side"), 220),
                    1,
                ),
                "turn_pixels_bottom": max(
                    self._coerce_int(enemy_direction_reacquire_payload.get("turn_pixels_bottom"), 340),
                    1,
                ),
                "vertical_delta": self._coerce_int(enemy_direction_reacquire_payload.get("vertical_delta"), 0),
                "post_turn_delay_ms": max(
                    self._coerce_int(enemy_direction_reacquire_payload.get("post_turn_delay_ms"), 80),
                    0,
                ),
                "auto_target_after_turn": bool(
                    enemy_direction_reacquire_payload.get("auto_target_after_turn", True)
                ),
            },
            "hsv_rules": {
                "enemy_health_red": self._coerce_hsv_rule(
                    hsv_payload.get("enemy_health_red"),
                    default_lower=(0, 95, 100),
                    default_upper=(12, 255, 255),
                    default_min_ratio=0.02,
                    default_lower_2=(165, 95, 100),
                    default_upper_2=(179, 255, 255),
                ),
                "boss_health_red": self._coerce_hsv_rule(
                    hsv_payload.get("boss_health_red"),
                    default_lower=(0, 95, 100),
                    default_upper=(12, 255, 255),
                    default_min_ratio=0.02,
                    default_lower_2=(165, 95, 100),
                    default_upper_2=(179, 255, 255),
                ),
                "team_hud": self._coerce_hsv_rule(
                    hsv_payload.get("team_hud"),
                    default_lower=(0, 0, 140),
                    default_upper=(179, 100, 255),
                    default_min_ratio=0.02,
                ),
                "slot_active": self._coerce_hsv_rule(
                    hsv_payload.get("slot_active"),
                    default_lower=(60, 25, 90),
                    default_upper=(110, 255, 255),
                    default_min_ratio=0.016,
                ),
                "ability_ready": self._coerce_hsv_rule(
                    hsv_payload.get("ability_ready"),
                    default_lower=(12, 35, 120),
                    default_upper=(55, 255, 255),
                    default_min_ratio=0.028,
                ),
                "enemy_direction_pink": self._coerce_hsv_rule(
                    hsv_payload.get("enemy_direction_pink"),
                    default_lower=(170, 50, 150),
                    default_upper=(179, 150, 255),
                    default_min_ratio=0.003,
                    default_lower_2=(0, 50, 150),
                    default_upper_2=(4, 150, 255),
                ),
                "remaining_enemy_marker_cyan": self._coerce_hsv_rule(
                    hsv_payload.get("remaining_enemy_marker_cyan"),
                    default_lower=(85, 80, 80),
                    default_upper=(105, 255, 255),
                    default_min_ratio=0.01,
                ),
            },
            "ability_thresholds": self._coerce_ability_thresholds(ability_thresholds_payload),
            "enemy_health_rect": {
                "min_width": max(self._coerce_int(health_rect_payload.get("min_width"), 24), 1),
                "short_min_width": max(self._coerce_int(health_rect_payload.get("short_min_width"), 12), 1),
                "max_width": max(self._coerce_int(health_rect_payload.get("max_width"), 220), 1),
                "min_height": max(self._coerce_int(health_rect_payload.get("min_height"), 2), 1),
                "max_height": max(self._coerce_int(health_rect_payload.get("max_height"), 8), 1),
                "min_aspect_ratio": max(self._coerce_float(health_rect_payload.get("min_aspect_ratio"), 4.0), 0.1),
                "max_aspect_ratio": max(self._coerce_float(health_rect_payload.get("max_aspect_ratio"), 90.0), 0.1),
                "min_fill_ratio": min(
                    max(self._coerce_float(health_rect_payload.get("min_fill_ratio"), 0.42), 0.0),
                    1.0,
                ),
                "short_max_y_ratio": min(
                    max(self._coerce_float(health_rect_payload.get("short_max_y_ratio"), 0.58), 0.0),
                    1.0,
                ),
            },
            "boss_health_rect": {
                "min_width": max(self._coerce_int(boss_health_rect_payload.get("min_width"), 220), 1),
                "max_width": max(self._coerce_int(boss_health_rect_payload.get("max_width"), 900), 1),
                "min_height": max(self._coerce_int(boss_health_rect_payload.get("min_height"), 4), 1),
                "max_height": max(self._coerce_int(boss_health_rect_payload.get("max_height"), 24), 1),
                "min_aspect_ratio": max(self._coerce_float(boss_health_rect_payload.get("min_aspect_ratio"), 8.0), 0.1),
                "max_aspect_ratio": max(self._coerce_float(boss_health_rect_payload.get("max_aspect_ratio"), 120.0), 0.1),
                "min_fill_ratio": min(
                    max(self._coerce_float(boss_health_rect_payload.get("min_fill_ratio"), 0.18), 0.0),
                    1.0,
                ),
            },
            "enemy_direction_rect": {
                "min_width": max(self._coerce_int(enemy_direction_rect_payload.get("min_width"), 10), 1),
                "max_width": max(self._coerce_int(enemy_direction_rect_payload.get("max_width"), 40), 1),
                "min_height": max(self._coerce_int(enemy_direction_rect_payload.get("min_height"), 10), 1),
                "max_height": max(self._coerce_int(enemy_direction_rect_payload.get("max_height"), 40), 1),
                "min_area": max(self._coerce_int(enemy_direction_rect_payload.get("min_area"), 80), 1),
                "max_area": max(self._coerce_int(enemy_direction_rect_payload.get("max_area"), 420), 1),
                "min_aspect_ratio": max(
                    self._coerce_float(enemy_direction_rect_payload.get("min_aspect_ratio"), 0.55),
                    0.1,
                ),
                "max_aspect_ratio": max(
                    self._coerce_float(enemy_direction_rect_payload.get("max_aspect_ratio"), 1.6),
                    0.1,
                ),
                "min_fill_ratio": min(
                    max(self._coerce_float(enemy_direction_rect_payload.get("min_fill_ratio"), 0.16), 0.0),
                    1.0,
                ),
                "max_fill_ratio": min(
                    max(self._coerce_float(enemy_direction_rect_payload.get("max_fill_ratio"), 0.88), 0.0),
                    1.0,
                ),
            },
            "remaining_enemy_marker_rect": {
                "min_width": max(self._coerce_int(remaining_enemy_marker_rect_payload.get("min_width"), 12), 1),
                "max_width": max(self._coerce_int(remaining_enemy_marker_rect_payload.get("max_width"), 36), 1),
                "min_height": max(self._coerce_int(remaining_enemy_marker_rect_payload.get("min_height"), 14), 1),
                "max_height": max(self._coerce_int(remaining_enemy_marker_rect_payload.get("max_height"), 34), 1),
                "min_area": max(self._coerce_int(remaining_enemy_marker_rect_payload.get("min_area"), 120), 1),
                "max_area": max(self._coerce_int(remaining_enemy_marker_rect_payload.get("max_area"), 850), 1),
                "min_aspect_ratio": max(
                    self._coerce_float(remaining_enemy_marker_rect_payload.get("min_aspect_ratio"), 0.45),
                    0.1,
                ),
                "max_aspect_ratio": max(
                    self._coerce_float(remaining_enemy_marker_rect_payload.get("max_aspect_ratio"), 1.65),
                    0.1,
                ),
                "min_fill_ratio": min(
                    max(self._coerce_float(remaining_enemy_marker_rect_payload.get("min_fill_ratio"), 0.22), 0.0),
                    1.0,
                ),
                "max_fill_ratio": min(
                    max(self._coerce_float(remaining_enemy_marker_rect_payload.get("max_fill_ratio"), 0.9), 0.0),
                    1.0,
                ),
            },
            "thresholds": {
                "supported_scene_min_features": max(
                    self._coerce_int(thresholds_payload.get("supported_scene_min_features"), 1),
                    1,
                ),
            },
            "templates": {
                "target_lock": self._coerce_template_spec(
                    templates_payload.get("target_lock"),
                    default_path=f"{resolved_name}/target_lock_diamond.png",
                    default_region=regions_payload.get("target_search_region") or (200, 120, 880, 320),
                    default_threshold=0.68,
                    default_scales=(0.75, 0.9, 1.0, 1.15, 1.3),
                ),
                "remaining_enemy_marker": self._coerce_template_spec(
                    templates_payload.get("remaining_enemy_marker"),
                    default_path=f"{resolved_name}/remaining_enemy_marker.png",
                    default_region=regions_payload.get("remaining_enemy_marker") or (20, 170, 70, 80),
                    default_threshold=0.78,
                    default_scales=(0.9, 1.0, 1.1),
                ),
                "challenge_success": self._coerce_template_spec(
                    templates_payload.get("challenge_success"),
                    default_path=f"{resolved_name}/challenge_success.png",
                    default_region=regions_payload.get("challenge_success_region") or (450, 130, 380, 150),
                    default_threshold=0.74,
                    default_scales=(1.0,),
                ),
                "reward_marker": self._coerce_template_spec(
                    templates_payload.get("reward_marker"),
                    default_path=f"{resolved_name}/reward_marker.png",
                    default_region=regions_payload.get("reward_marker_search_region") or (80, 300, 1040, 330),
                    default_threshold=0.72,
                    default_scales=(0.8, 0.9, 1.0, 1.12, 1.25),
                ),
                "claim_memento_prompt": self._coerce_template_spec(
                    templates_payload.get("claim_memento_prompt"),
                    default_path=f"{resolved_name}/claim_memento_prompt.png",
                    default_region=regions_payload.get("claim_memento_prompt_region") or (760, 372, 50, 50),
                    default_threshold=0.94,
                    default_scales=(0.9, 1.0, 1.08),
                ),
            },
            "audio_dodge": {
                "enabled": bool(audio_dodge_payload.get("enabled", False)),
                "sample_path": str(
                    (self._profile_dir / str(audio_dodge_payload.get("sample_path") or "audio/dodge_sample.wav")).resolve()
                ),
                "sample_rate": max(self._coerce_int(audio_dodge_payload.get("sample_rate"), 32000), 8000),
                "channels": max(self._coerce_int(audio_dodge_payload.get("channels"), 2), 1),
                "chunk_size": max(self._coerce_int(audio_dodge_payload.get("chunk_size"), 768), 128),
                "threshold": max(self._coerce_float(audio_dodge_payload.get("threshold"), 0.13), 0.0),
                "ratio": max(self._coerce_float(audio_dodge_payload.get("ratio"), 1.0), 0.01),
                "allow_repeat": bool(audio_dodge_payload.get("allow_repeat", False)),
                "cooldown_sec": max(self._coerce_float(audio_dodge_payload.get("cooldown_sec"), 0.42), 0.0),
                "window_sec": max(self._coerce_float(audio_dodge_payload.get("window_sec"), 0.22), 0.05),
                "trigger_max_age_sec": max(
                    self._coerce_float(audio_dodge_payload.get("trigger_max_age_sec"), 0.22),
                    0.05,
                ),
                "reconnect_delay_sec": max(
                    self._coerce_float(audio_dodge_payload.get("reconnect_delay_sec"), 2.0),
                    0.1,
                ),
                "pre_emphasis": min(
                    max(self._coerce_float(audio_dodge_payload.get("pre_emphasis"), 0.97), 0.0),
                    0.999,
                ),
                "dodge_pause_ms": max(self._coerce_int(audio_dodge_payload.get("dodge_pause_ms"), 120), 0),
                "right_hold_ms": max(self._coerce_int(audio_dodge_payload.get("right_hold_ms"), 18), 0),
                "post_right_delay_ms": max(
                    self._coerce_int(audio_dodge_payload.get("post_right_delay_ms"), 10),
                    0,
                ),
                "shift_hold_ms": max(self._coerce_int(audio_dodge_payload.get("shift_hold_ms"), 24), 0),
                "dodge_key": str(audio_dodge_payload.get("dodge_key") or "shift"),
                "dodge_mouse_button": str(audio_dodge_payload.get("dodge_mouse_button") or "right"),
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
        ability_regions = dict(profile["ability_regions"])

        enemy_health_boxes = self._find_enemy_health_boxes(
            source_image,
            regions["enemy_health_search_region"],
            hsv_rules["enemy_health_red"],
            dict(profile["enemy_health_rect"]),
            profile,
        )
        enemy_direction_markers = self._find_enemy_direction_markers(
            source_image,
            {
                "left": regions["enemy_direction_left"],
                "right": regions["enemy_direction_right"],
                "bottom": regions["enemy_direction_bottom"],
            },
            hsv_rules["enemy_direction_pink"],
            dict(profile["enemy_direction_rect"]),
            profile,
        )
        remaining_enemy_markers = self._find_remaining_enemy_markers(source_image, profile)
        boss_health_boxes = self._find_enemy_health_boxes(
            source_image,
            regions["boss_health_bar"],
            hsv_rules["boss_health_red"],
            dict(profile["boss_health_rect"]),
            profile,
        )
        team_ratio = self._hsv_ratio_for_region(source_image, regions["team_panel"], hsv_rules["team_hud"], profile)
        ability_analysis = {
            name: self._analyze_ability_region(
                source_image,
                ability_regions[name],
                ready_rule=hsv_rules["ability_ready"],
                thresholds=dict(profile["ability_thresholds"]),
                profile=profile,
            )
            for name in self.ABILITY_NAMES
        }
        slot_ratios = [
            self._hsv_ratio_for_region(source_image, region, hsv_rules["slot_active"], profile)
            for region in profile["current_slot_regions"]
        ]
        target_match = self._match_template_in_region(
            source_image,
            dict(profile["templates"])["target_lock"],
            profile=profile,
        )
        remaining_enemy_marker_match = self._match_template_in_region(
            source_image,
            dict(profile["templates"])["remaining_enemy_marker"],
            profile=profile,
        )
        success_match = self._match_template_in_region(
            source_image,
            dict(profile["templates"])["challenge_success"],
            profile=profile,
        )
        reward_marker_match = {"found": False, "score": 0.0, "box": None}
        claim_prompt_match = {"found": False, "score": 0.0, "box": None}

        enemy_health_found = bool(enemy_health_boxes)
        enemy_direction_found = bool(enemy_direction_markers)
        remaining_enemy_marker_found = bool(remaining_enemy_markers or remaining_enemy_marker_match["found"])
        if (
            not remaining_enemy_markers
            and bool(remaining_enemy_marker_match["found"])
            and remaining_enemy_marker_match.get("box") is not None
        ):
            mx, my, mw, mh = [int(item) for item in remaining_enemy_marker_match["box"]]
            remaining_enemy_markers = [
                {
                    "x": mx,
                    "y": my,
                    "width": mw,
                    "height": mh,
                    "region": "remaining_enemy_marker_template",
                }
            ]
        enemy_direction_primary_side = self._resolve_primary_enemy_direction(enemy_direction_markers)
        boss_found = bool(boss_health_boxes)
        front_enemy_found = bool(enemy_health_found or boss_found)
        team_found = team_ratio >= hsv_rules["team_hud"]["min_ratio"]
        ability_available = {
            name: ability_analysis[name]["state"] == "ready"
            for name in self.ABILITY_NAMES
        }
        current_slot = self._detect_current_slot(slot_ratios, hsv_rules["slot_active"]["min_ratio"])
        target_found = bool(target_match["found"] and (enemy_health_found or boss_found))
        challenge_success_found = bool(success_match["found"])
        reward_scan_allowed = bool(not remaining_enemy_marker_found)
        if reward_scan_allowed:
            reward_marker_match = self._match_template_in_region(
                source_image,
                dict(profile["templates"])["reward_marker"],
                profile=profile,
            )
            claim_prompt_match = self._match_template_in_region(
                source_image,
                dict(profile["templates"])["claim_memento_prompt"],
                profile=profile,
            )
        reward_marker_found = bool(reward_marker_match["found"])
        reward_marker_box = reward_marker_match.get("box")
        reward_marker_center_x = None
        reward_marker_center_y = None
        if reward_marker_box is not None:
            rx, ry, rw, rh = [int(item) for item in reward_marker_box]
            reward_marker_center_x = int(round(rx + rw / 2.0))
            reward_marker_center_y = int(round(ry + rh / 2.0))
        claim_memento_prompt_found = bool(claim_prompt_match["found"])

        supported_features = [
            team_found,
            any(ability_available.values()),
            current_slot is not None,
        ]
        supported_feature_count = sum(1 for value in supported_features if value)
        in_supported_scene = supported_feature_count >= int(profile["thresholds"]["supported_scene_min_features"])
        in_combat = bool(remaining_enemy_marker_found)
        combat_gate_reason = "remaining_enemy_marker" if remaining_enemy_marker_found else "none"
        confidence = min(
            1.0,
            (
                int(remaining_enemy_marker_found)
                + int(front_enemy_found)
                + int(target_found)
                + (1 if any(ability_available.values()) else 0)
            )
            / 4.0,
        )

        return {
            "profile_name": profile["profile_name"],
            "capture_size": [int(source_image.shape[1]), int(source_image.shape[0])],
            "in_supported_scene": bool(in_supported_scene),
            "in_combat": bool(in_combat),
            "target_found": bool(target_found),
            "target_confidence": round(float(target_match["score"]), 3),
            "enemy_level_found": False,
            "front_enemy_found": bool(front_enemy_found),
            "remaining_enemy_marker_found": bool(remaining_enemy_marker_found),
            "remaining_enemy_marker_count": int(len(remaining_enemy_markers)),
            "remaining_enemy_markers": [
                {
                    "x": int(box["x"]),
                    "y": int(box["y"]),
                    "width": int(box["width"]),
                    "height": int(box["height"]),
                    "region": str(box["region"]),
                }
                for box in remaining_enemy_markers[:5]
            ],
            "enemy_health_found": bool(enemy_health_found),
            "enemy_health_count": int(len(enemy_health_boxes)),
            "enemy_direction_found": bool(enemy_direction_found),
            "enemy_direction_count": int(len(enemy_direction_markers)),
            "enemy_direction_primary_side": enemy_direction_primary_side,
            "enemy_direction_markers": [
                {
                    "x": int(box["x"]),
                    "y": int(box["y"]),
                    "width": int(box["width"]),
                    "height": int(box["height"]),
                    "region": str(box["region"]),
                }
                for box in enemy_direction_markers[:10]
            ],
            "boss_found": bool(boss_found),
            "team_found": bool(team_found),
            "current_slot": current_slot,
            "team_size": int(len(profile["current_slot_regions"])),
            "skill_available": bool(ability_available["skill"]),
            "skill_state": str(ability_analysis["skill"]["state"]),
            "ultimate_available": bool(ability_available["ultimate"]),
            "ultimate_state": str(ability_analysis["ultimate"]["state"]),
            "arc_available": False,
            "challenge_success_found": bool(challenge_success_found),
            "reward_marker_found": bool(reward_marker_found),
            "reward_marker_confidence": round(float(reward_marker_match["score"]), 3),
            "reward_marker_box": [
                int(item)
                for item in (list(reward_marker_box) if reward_marker_box is not None else [])
            ],
            "reward_marker_center_x": reward_marker_center_x,
            "reward_marker_center_y": reward_marker_center_y,
            "claim_memento_prompt_found": bool(claim_memento_prompt_found),
            "claim_memento_prompt_confidence": round(float(claim_prompt_match["score"]), 3),
            "claim_memento_prompt_box": [
                int(item)
                for item in (
                    list(claim_prompt_match["box"]) if claim_prompt_match.get("box") is not None else []
                )
            ],
            "confidence": round(float(confidence), 3),
            "debug": {
                "combat_gate_reason": combat_gate_reason,
                "remaining_enemy_marker_search_region": list(
                    self.scale_region(source_image, regions["remaining_enemy_marker"], profile=profile)
                ),
                "remaining_enemy_marker_boxes": [
                    {
                        "x": int(box["x"]),
                        "y": int(box["y"]),
                        "width": int(box["width"]),
                        "height": int(box["height"]),
                        "region": str(box["region"]),
                    }
                    for box in remaining_enemy_markers[:5]
                ],
                "enemy_signal_count": int(len(enemy_health_boxes)),
                "front_enemy_signal_count": int(len(enemy_health_boxes) + len(boss_health_boxes)),
                "boss_signal": bool(boss_found),
                "enemy_health_search_region": list(
                    self.scale_region(source_image, regions["enemy_health_search_region"], profile=profile)
                ),
                "enemy_direction_search_regions": {
                    "left": list(self.scale_region(source_image, regions["enemy_direction_left"], profile=profile)),
                    "right": list(self.scale_region(source_image, regions["enemy_direction_right"], profile=profile)),
                    "bottom": list(self.scale_region(source_image, regions["enemy_direction_bottom"], profile=profile)),
                },
                "boss_health_search_region": list(
                    self.scale_region(source_image, regions["boss_health_bar"], profile=profile)
                ),
                "reward_marker_search_region": list(
                    self.scale_region(source_image, regions["reward_marker_search_region"], profile=profile)
                ),
                "claim_memento_prompt_region": list(
                    self.scale_region(source_image, regions["claim_memento_prompt_region"], profile=profile)
                ),
                "ratios": {
                    "boss_health": round(float(len(boss_health_boxes)), 4),
                    "team": round(float(team_ratio), 4),
                    "skill": {
                        "ready": round(float(ability_analysis["skill"]["ready_ratio"]), 4),
                        "ring_ready": round(float(ability_analysis["skill"]["ring_ready_ratio"]), 4),
                        "bright": round(float(ability_analysis["skill"]["bright_ratio"]), 4),
                        "dark": round(float(ability_analysis["skill"]["dark_ratio"]), 4),
                        "white": round(float(ability_analysis["skill"]["white_ratio"]), 4),
                    },
                    "ultimate": {
                        "ready": round(float(ability_analysis["ultimate"]["ready_ratio"]), 4),
                        "ring_ready": round(float(ability_analysis["ultimate"]["ring_ready_ratio"]), 4),
                        "bright": round(float(ability_analysis["ultimate"]["bright_ratio"]), 4),
                        "dark": round(float(ability_analysis["ultimate"]["dark_ratio"]), 4),
                        "white": round(float(ability_analysis["ultimate"]["white_ratio"]), 4),
                    },
                    "slots": [round(float(value), 4) for value in slot_ratios],
                },
                "matches": {
                    "target_lock": round(float(target_match["score"]), 4),
                    "remaining_enemy_marker": round(float(remaining_enemy_marker_match["score"]), 4),
                    "challenge_success": round(float(success_match["score"]), 4),
                    "reward_marker": round(float(reward_marker_match["score"]), 4),
                    "claim_memento_prompt": round(float(claim_prompt_match["score"]), 4),
                },
                "target_lock_box": [
                    int(item)
                    for item in (list(target_match["box"]) if target_match.get("box") is not None else [])
                ],
                "enemy_health_boxes": [
                    [int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])]
                    for box in enemy_health_boxes[:10]
                ],
                "enemy_direction_markers": [
                    {
                        "x": int(box["x"]),
                        "y": int(box["y"]),
                        "width": int(box["width"]),
                        "height": int(box["height"]),
                        "region": str(box["region"]),
                    }
                    for box in enemy_direction_markers[:10]
                ],
                "enemy_direction_primary_side": enemy_direction_primary_side,
                "boss_health_boxes": [
                    [int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])]
                    for box in boss_health_boxes[:5]
                ],
                "reward_marker_box": [
                    int(item)
                    for item in (list(reward_marker_box) if reward_marker_box is not None else [])
                ],
                "claim_memento_prompt_box": [
                    int(item)
                    for item in (
                        list(claim_prompt_match["box"]) if claim_prompt_match.get("box") is not None else []
                    )
                ],
                "feature_counts": {
                    "supported": int(supported_feature_count),
                    "remaining_enemy_marker": int(len(remaining_enemy_markers)),
                    "enemy_health": int(len(enemy_health_boxes)),
                    "enemy_direction": int(len(enemy_direction_markers)),
                },
            },
        }

    def annotate_frame(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
        state: Mapping[str, Any] | None = None,
        overlay: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        resolved_state = dict(state or self.analyze_frame(source_image, profile_name=profile_name))
        canvas = source_image.copy()
        search_region = list(dict(resolved_state.get("debug") or {}).get("enemy_health_search_region") or [])
        if len(search_region) >= 4:
            sx, sy, sw, sh = [int(item) for item in search_region[:4]]
            cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), (80, 220, 120), 2)
        direction_regions = dict(dict(resolved_state.get("debug") or {}).get("enemy_direction_search_regions") or {})
        for region_name, region in direction_regions.items():
            if not isinstance(region, (list, tuple)) or len(region) < 4:
                continue
            rx, ry, rw, rh = [int(item) for item in region[:4]]
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (255, 120, 210), 1)
            cv2.putText(
                canvas,
                str(region_name),
                (rx + 4, max(ry - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 120, 210),
                1,
                cv2.LINE_AA,
            )
        remaining_region = list(dict(resolved_state.get("debug") or {}).get("remaining_enemy_marker_search_region") or [])
        if len(remaining_region) >= 4:
            rx, ry, rw, rh = [int(item) for item in remaining_region[:4]]
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (80, 235, 235), 2)
        target_lock_box = list(dict(resolved_state.get("debug") or {}).get("target_lock_box") or [])
        if len(target_lock_box) >= 4:
            tx, ty, tw, th = [int(item) for item in target_lock_box[:4]]
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (255, 220, 80), 2)
        for box in list(dict(resolved_state.get("debug") or {}).get("enemy_health_boxes") or []):
            x, y, width, height = [int(item) for item in box[:4]]
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (255, 80, 80), 2)
        for box in list(dict(resolved_state.get("debug") or {}).get("enemy_direction_markers") or []):
            if not isinstance(box, Mapping):
                continue
            x = int(box.get("x") or 0)
            y = int(box.get("y") or 0)
            width = int(box.get("width") or 0)
            height = int(box.get("height") or 0)
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (255, 120, 210), 2)
        for box in list(dict(resolved_state.get("debug") or {}).get("remaining_enemy_marker_boxes") or []):
            if not isinstance(box, Mapping):
                continue
            x = int(box.get("x") or 0)
            y = int(box.get("y") or 0)
            width = int(box.get("width") or 0)
            height = int(box.get("height") or 0)
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (80, 235, 235), 2)
        boss_boxes = list(dict(resolved_state.get("debug") or {}).get("boss_health_boxes") or [])
        if boss_boxes:
            x, y, width, height = [int(item) for item in boss_boxes[0][:4]]
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (80, 180, 255), 2)
        reward_region = list(dict(resolved_state.get("debug") or {}).get("reward_marker_search_region") or [])
        if len(reward_region) >= 4:
            rx, ry, rw, rh = [int(item) for item in reward_region[:4]]
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (120, 255, 160), 1)
        reward_box = list(dict(resolved_state.get("debug") or {}).get("reward_marker_box") or [])
        if len(reward_box) >= 4:
            rx, ry, rw, rh = [int(item) for item in reward_box[:4]]
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (120, 255, 160), 2)
            cv2.line(canvas, (rx + rw // 2, 0), (rx + rw // 2, canvas.shape[0]), (120, 255, 160), 1)
        claim_region = list(dict(resolved_state.get("debug") or {}).get("claim_memento_prompt_region") or [])
        if len(claim_region) >= 4:
            px, py, pw, ph = [int(item) for item in claim_region[:4]]
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (255, 120, 210), 1)
        claim_box = list(dict(resolved_state.get("debug") or {}).get("claim_memento_prompt_box") or [])
        if len(claim_box) >= 4:
            px, py, pw, ph = [int(item) for item in claim_box[:4]]
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (255, 120, 210), 2)

        extra = dict(overlay or {})
        summary_lines = [
            f"phase={str(extra.get('phase') or 'unknown')}",
            f"note={str(extra.get('note') or '')}",
            f"t={float(extra.get('t') or 0.0):.3f}s",
            f"remaining_marker={bool(resolved_state.get('remaining_enemy_marker_found'))}",
            f"front_enemy={bool(resolved_state.get('front_enemy_found'))}",
            f"enemy_health_count={int(resolved_state.get('enemy_health_count') or 0)}",
            f"enemy_health_found={bool(resolved_state.get('enemy_health_found'))}",
            f"enemy_direction_count={int(resolved_state.get('enemy_direction_count') or 0)}",
            f"enemy_direction_found={bool(resolved_state.get('enemy_direction_found'))}",
            f"enemy_direction_side={resolved_state.get('enemy_direction_primary_side')}",
            f"boss_found={bool(resolved_state.get('boss_found'))}",
            f"in_combat={bool(resolved_state.get('in_combat'))}",
            f"combat_active={bool(extra.get('combat_active'))}",
            f"target_found={bool(resolved_state.get('target_found'))}",
            f"reward_marker={bool(resolved_state.get('reward_marker_found'))}",
            f"reward_x={resolved_state.get('reward_marker_center_x')}",
            f"claim_prompt={bool(resolved_state.get('claim_memento_prompt_found'))}",
            f"current_slot={resolved_state.get('current_slot')}",
            f"skill_state={resolved_state.get('skill_state')}",
            f"ultimate_state={resolved_state.get('ultimate_state')}",
        ]
        for index, text in enumerate(summary_lines):
            y = 28 + index * 26
            cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas, resolved_state

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

    def _find_enemy_health_boxes(
        self,
        source_image: np.ndarray,
        region: Region,
        rule: Mapping[str, Any],
        rect_config: Mapping[str, Any],
        profile: Mapping[str, Any],
    ) -> list[dict[str, int]]:
        scaled_region = self.scale_region(source_image, region, profile=profile)
        crop = self._crop_region(source_image, scaled_region)
        if crop.size == 0:
            return []

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        mask = self._hsv_mask(hsv, rule["lower"], rule["upper"])
        if rule.get("lower_2") is not None and rule.get("upper_2") is not None:
            mask = cv2.bitwise_or(mask, self._hsv_mask(hsv, rule["lower_2"], rule["upper_2"]))
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale_x = source_image.shape[1] / float(profile["client_size"][0])
        scale_y = source_image.shape[0] / float(profile["client_size"][1])
        min_width = max(int(round(float(rect_config["min_width"]) * scale_x)), 1)
        short_min_width = max(int(round(float(rect_config.get("short_min_width") or rect_config["min_width"]) * scale_x)), 1)
        max_width = max(int(round(float(rect_config["max_width"]) * scale_x)), min_width)
        min_height = max(int(round(float(rect_config["min_height"]) * scale_y)), 1)
        max_height = max(int(round(float(rect_config["max_height"]) * scale_y)), min_height)
        min_fill_ratio = float(rect_config["min_fill_ratio"])
        min_aspect_ratio = float(rect_config["min_aspect_ratio"])
        max_aspect_ratio = float(rect_config["max_aspect_ratio"])
        short_max_y_ratio = float(rect_config.get("short_max_y_ratio") or 1.0)

        boxes: list[dict[str, int]] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width > max_width:
                continue
            if height < min_height or height > max_height:
                continue
            aspect_ratio = width / float(max(height, 1))
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            candidate_mask = mask[y : y + height, x : x + width]
            if candidate_mask.size == 0:
                continue
            fill_ratio = float(np.count_nonzero(candidate_mask)) / float(candidate_mask.size)
            if fill_ratio < min_fill_ratio:
                continue
            is_normal_width = width >= min_width
            is_short_width = width >= short_min_width
            if not is_normal_width:
                if not is_short_width:
                    continue
                candidate_center_y = y + (height / 2.0)
                if candidate_center_y > crop.shape[0] * short_max_y_ratio:
                    continue
            boxes.append(
                {
                    "x": int(scaled_region[0] + x),
                    "y": int(scaled_region[1] + y),
                    "width": int(width),
                    "height": int(height),
                }
            )
        boxes.sort(key=lambda item: (item["y"], item["x"]))
        return boxes

    def _find_remaining_enemy_markers(
        self,
        source_image: np.ndarray,
        profile: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        return self._find_enemy_direction_markers(
            source_image,
            {"remaining_enemy_marker": dict(profile["regions"])["remaining_enemy_marker"]},
            dict(profile["hsv_rules"])["remaining_enemy_marker_cyan"],
            dict(profile["remaining_enemy_marker_rect"]),
            profile,
        )

    def _find_enemy_direction_markers(
        self,
        source_image: np.ndarray,
        regions: Mapping[str, Region],
        rule: Mapping[str, Any],
        rect_config: Mapping[str, Any],
        profile: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        scale_x = source_image.shape[1] / float(profile["client_size"][0])
        scale_y = source_image.shape[0] / float(profile["client_size"][1])
        min_width = max(int(round(float(rect_config["min_width"]) * scale_x)), 1)
        max_width = max(int(round(float(rect_config["max_width"]) * scale_x)), min_width)
        min_height = max(int(round(float(rect_config["min_height"]) * scale_y)), 1)
        max_height = max(int(round(float(rect_config["max_height"]) * scale_y)), min_height)
        min_area = max(int(round(float(rect_config["min_area"]) * scale_x * scale_y)), 1)
        max_area = max(int(round(float(rect_config["max_area"]) * scale_x * scale_y)), min_area)
        min_aspect_ratio = float(rect_config["min_aspect_ratio"])
        max_aspect_ratio = float(rect_config["max_aspect_ratio"])
        min_fill_ratio = float(rect_config["min_fill_ratio"])
        max_fill_ratio = float(rect_config["max_fill_ratio"])

        boxes: list[dict[str, Any]] = []
        kernel = np.ones((3, 3), dtype=np.uint8)
        for region_name, region in regions.items():
            scaled_region = self.scale_region(source_image, region, profile=profile)
            crop = self._crop_region(source_image, scaled_region)
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            mask = self._hsv_mask(hsv, rule["lower"], rule["upper"])
            if rule.get("lower_2") is not None and rule.get("upper_2") is not None:
                mask = cv2.bitwise_or(mask, self._hsv_mask(hsv, rule["lower_2"], rule["upper_2"]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if width < min_width or width > max_width:
                    continue
                if height < min_height or height > max_height:
                    continue
                aspect_ratio = width / float(max(height, 1))
                if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                    continue
                candidate_mask = mask[y : y + height, x : x + width]
                if candidate_mask.size == 0:
                    continue
                area = int(np.count_nonzero(candidate_mask))
                if area < min_area or area > max_area:
                    continue
                fill_ratio = float(area) / float(candidate_mask.size)
                if fill_ratio < min_fill_ratio or fill_ratio > max_fill_ratio:
                    continue
                boxes.append(
                    {
                        "x": int(scaled_region[0] + x),
                        "y": int(scaled_region[1] + y),
                        "width": int(width),
                        "height": int(height),
                        "region": str(region_name),
                    }
                )
        boxes.sort(key=lambda item: (item["region"], item["y"], item["x"]))
        return boxes

    def _resolve_primary_enemy_direction(self, markers: list[Mapping[str, Any]]) -> str | None:
        if not markers:
            return None
        side_counts: dict[str, int] = {}
        for marker in markers:
            side = str(marker.get("region") or "").strip().lower()
            if not side:
                continue
            side_counts[side] = side_counts.get(side, 0) + 1
        if not side_counts:
            return None
        priority = {"bottom": 3, "right": 2, "left": 1}
        return max(side_counts, key=lambda side: (side_counts[side], priority.get(side, 0)))

    def _match_template_in_region(
        self,
        source_image: np.ndarray,
        template_spec: Mapping[str, Any],
        *,
        profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        template_path = template_spec.get("path")
        template = self._load_template_image(template_path)
        if template is None:
            return {"found": False, "score": 0.0, "box": None}
        template_mask = (
            self._load_template_mask(template_path, mask_path=template_spec.get("mask_path"))
            if bool(template_spec.get("use_mask", True))
            else None
        )

        search_region = self.scale_region(source_image, tuple(template_spec["search_region"]), profile=profile)
        crop = self._crop_region(source_image, search_region)
        if crop.size == 0:
            return {"found": False, "score": 0.0, "box": None}

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        scale_x = source_image.shape[1] / float(profile["client_size"][0])
        scale_y = source_image.shape[0] / float(profile["client_size"][1])
        base_scale = max((scale_x + scale_y) / 2.0, 0.01)
        best_score = 0.0
        best_box: tuple[int, int, int, int] | None = None

        for relative_scale in template_spec["scales"]:
            scaled_w = max(int(round(template_gray.shape[1] * base_scale * float(relative_scale))), 1)
            scaled_h = max(int(round(template_gray.shape[0] * base_scale * float(relative_scale))), 1)
            if scaled_w > crop_gray.shape[1] or scaled_h > crop_gray.shape[0]:
                continue
            interpolation = cv2.INTER_AREA if scaled_w < template_gray.shape[1] else cv2.INTER_LINEAR
            resized = cv2.resize(template_gray, (scaled_w, scaled_h), interpolation=interpolation)
            if template_mask is not None:
                resized_mask = cv2.resize(template_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                if not np.any(resized_mask):
                    continue
                result = cv2.matchTemplate(crop_gray, resized, cv2.TM_CCORR_NORMED, mask=resized_mask)
            else:
                result = cv2.matchTemplate(crop_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, max_loc = cv2.minMaxLoc(result)
            if max_value > best_score:
                best_score = float(max_value)
                best_box = (
                    int(search_region[0] + max_loc[0]),
                    int(search_region[1] + max_loc[1]),
                    int(scaled_w),
                    int(scaled_h),
                )

        return {
            "found": bool(best_score >= float(template_spec["match_threshold"])),
            "score": float(best_score),
            "box": best_box,
        }

    def _load_template_image(self, template_path: Any) -> np.ndarray | None:
        if template_path is None:
            return None
        path = Path(str(template_path)).resolve()
        cached = self._template_cache.get(path)
        if cached is not None or path in self._template_cache:
            return cached
        if not path.is_file():
            self._template_cache[path] = None
            return None
        raw_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw_image is None:
            self._template_cache[path] = None
            self._template_mask_cache[path] = None
            return None
        alpha_mask = None
        if raw_image.ndim == 3 and raw_image.shape[2] == 4:
            alpha_mask = raw_image[:, :, 3]
            image = cv2.cvtColor(raw_image[:, :, :3], cv2.COLOR_BGR2RGB)
        elif raw_image.ndim == 3:
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        self._template_cache[path] = image
        self._template_mask_cache[path] = self._build_template_mask(image, alpha_mask=alpha_mask)
        return image

    def _load_template_mask(self, template_path: Any, *, mask_path: Any | None = None) -> np.ndarray | None:
        if mask_path is not None:
            path = Path(str(mask_path)).resolve()
            cached = self._template_mask_cache.get(path)
            if cached is not None or path in self._template_mask_cache:
                return cached
            if not path.is_file():
                self._template_mask_cache[path] = None
                return None
            raw_mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw_mask is None:
                self._template_mask_cache[path] = None
                return None
            if raw_mask.ndim == 3 and raw_mask.shape[2] == 4:
                mask_source = raw_mask[:, :, 3]
            elif raw_mask.ndim == 3:
                mask_source = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_source = raw_mask
            mask = (mask_source > 8).astype(np.uint8) * 255
            self._template_mask_cache[path] = mask if np.any(mask) else None
            return self._template_mask_cache[path]
        if template_path is None:
            return None
        path = Path(str(template_path)).resolve()
        if path not in self._template_cache:
            self._load_template_image(template_path)
        return self._template_mask_cache.get(path)

    def _build_template_mask(self, image: np.ndarray, *, alpha_mask: np.ndarray | None = None) -> np.ndarray | None:
        if image.size == 0:
            return None
        if alpha_mask is not None:
            visible = (alpha_mask > 8).astype(np.uint8) * 255
            if np.all(visible):
                return None
            if not np.any(visible):
                return None
            return visible
        non_black = np.any(image > 8, axis=2).astype(np.uint8) * 255
        if np.all(non_black):
            return None
        if not np.any(non_black):
            return None
        return non_black

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

    def _analyze_ability_region(
        self,
        source_image: np.ndarray,
        region: Region,
        *,
        ready_rule: Mapping[str, Any],
        thresholds: Mapping[str, Any],
        profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        scaled_region = self.scale_region(source_image, region, profile=profile)
        crop = self._crop_region(source_image, scaled_region)
        if crop.size == 0:
            return {
                "state": "disabled_or_unready",
                "ready_ratio": 0.0,
                "ring_ready_ratio": 0.0,
                "bright_ratio": 0.0,
                "dark_ratio": 1.0,
                "white_ratio": 0.0,
            }

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        ready_mask = self._hsv_mask(hsv, ready_rule["lower"], ready_rule["upper"])
        if ready_rule.get("lower_2") is not None and ready_rule.get("upper_2") is not None:
            ready_mask = cv2.bitwise_or(
                ready_mask,
                self._hsv_mask(hsv, ready_rule["lower_2"], ready_rule["upper_2"]),
            )

        total_pixels = float(max(gray.size, 1))
        ready_ratio = float(np.count_nonzero(ready_mask)) / total_pixels
        ring_mask = self._build_ring_mask(gray.shape[0], gray.shape[1])
        if ring_mask is not None and np.any(ring_mask):
            ring_ready_ratio = float(np.count_nonzero((ready_mask > 0) & ring_mask)) / float(np.count_nonzero(ring_mask))
        else:
            ring_ready_ratio = ready_ratio

        shared = dict(thresholds.get("shared") or {})
        sat_max = int(shared.get("white_sat_max", 45))
        white_value_min = int(shared.get("white_value_min", 170))
        bright_value_min = int(shared.get("bright_value_min", 140))
        dark_value_max = int(shared.get("dark_value_max", 95))

        white_mask = (hsv[:, :, 1] <= sat_max) & (hsv[:, :, 2] >= white_value_min)
        bright_mask = gray >= bright_value_min
        dark_mask = gray <= dark_value_max
        white_ratio = float(np.count_nonzero(white_mask)) / total_pixels
        bright_ratio = float(np.count_nonzero(bright_mask)) / total_pixels
        dark_ratio = float(np.count_nonzero(dark_mask)) / total_pixels

        ready_cfg = dict(thresholds.get("ready") or {})
        cooldown_cfg = dict(thresholds.get("cooldown") or {})
        disabled_cfg = dict(thresholds.get("disabled") or {})

        ready_detected = bool(
            ready_ratio >= float(ready_cfg.get("highlight_min_ratio", 0.028))
            or (
                ring_ready_ratio >= float(ready_cfg.get("ring_highlight_min_ratio", 0.04))
                and dark_ratio <= float(ready_cfg.get("max_dark_ratio", 0.72))
            )
            or (
                ready_ratio >= float(ready_cfg.get("secondary_highlight_min_ratio", 0.018))
                and bright_ratio >= float(ready_cfg.get("bright_min_ratio", 0.12))
                and dark_ratio <= float(ready_cfg.get("max_dark_ratio", 0.72))
            )
        )
        cooldown_detected = bool(
            white_ratio >= float(cooldown_cfg.get("white_min_ratio", 0.01))
            or (
                dark_ratio >= float(cooldown_cfg.get("dark_min_ratio", 0.24))
                and bright_ratio >= float(cooldown_cfg.get("bright_min_ratio", 0.08))
            )
        )
        disabled_detected = bool(
            dark_ratio >= float(disabled_cfg.get("dark_min_ratio", 0.48))
            and bright_ratio <= float(disabled_cfg.get("bright_max_ratio", 0.16))
            and white_ratio <= float(disabled_cfg.get("white_max_ratio", 0.012))
        )

        if ready_detected:
            state = "ready"
        elif cooldown_detected:
            state = "cooldown"
        elif disabled_detected:
            state = "disabled_or_unready"
        else:
            state = "cooldown" if bright_ratio >= 0.06 or white_ratio >= 0.006 else "disabled_or_unready"

        return {
            "state": state,
            "ready_ratio": ready_ratio,
            "ring_ready_ratio": ring_ready_ratio,
            "bright_ratio": bright_ratio,
            "dark_ratio": dark_ratio,
            "white_ratio": white_ratio,
        }

    def _build_ring_mask(self, height: int, width: int) -> np.ndarray | None:
        if height <= 2 or width <= 2:
            return None
        yy, xx = np.ogrid[:height, :width]
        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
        radius_y = max(height / 2.0, 1.0)
        radius_x = max(width / 2.0, 1.0)
        normalized = ((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2
        outer = normalized <= 1.0
        inner = normalized <= 0.42
        ring = outer & ~inner
        return ring if np.any(ring) else None

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

    def _coerce_hsv_rule(
        self,
        value: Any,
        *,
        default_lower: HsvTriplet,
        default_upper: HsvTriplet,
        default_min_ratio: float,
        default_lower_2: HsvTriplet | None = None,
        default_upper_2: HsvTriplet | None = None,
    ) -> dict[str, Any]:
        payload = dict(value or {}) if isinstance(value, Mapping) else {}
        return {
            "lower": self._coerce_hsv_triplet(payload.get("lower"), default=default_lower),
            "upper": self._coerce_hsv_triplet(payload.get("upper"), default=default_upper),
            "lower_2": self._coerce_optional_hsv_triplet(payload.get("lower_2"), default=default_lower_2),
            "upper_2": self._coerce_optional_hsv_triplet(payload.get("upper_2"), default=default_upper_2),
            "min_ratio": min(max(self._coerce_float(payload.get("min_ratio"), default_min_ratio), 0.0), 1.0),
        }

    def _coerce_ability_thresholds(self, value: Any) -> dict[str, Any]:
        payload = dict(value or {}) if isinstance(value, Mapping) else {}
        ready_payload = dict(payload.get("ready") or {})
        cooldown_payload = dict(payload.get("cooldown") or {})
        disabled_payload = dict(payload.get("disabled") or {})
        shared_payload = dict(payload.get("shared") or {})
        return {
            "ready": {
                "highlight_min_ratio": min(
                    max(self._coerce_float(ready_payload.get("highlight_min_ratio"), 0.028), 0.0),
                    1.0,
                ),
                "ring_highlight_min_ratio": min(
                    max(self._coerce_float(ready_payload.get("ring_highlight_min_ratio"), 0.04), 0.0),
                    1.0,
                ),
                "secondary_highlight_min_ratio": min(
                    max(self._coerce_float(ready_payload.get("secondary_highlight_min_ratio"), 0.018), 0.0),
                    1.0,
                ),
                "bright_min_ratio": min(
                    max(self._coerce_float(ready_payload.get("bright_min_ratio"), 0.12), 0.0),
                    1.0,
                ),
                "max_dark_ratio": min(
                    max(self._coerce_float(ready_payload.get("max_dark_ratio"), 0.72), 0.0),
                    1.0,
                ),
            },
            "cooldown": {
                "white_min_ratio": min(
                    max(self._coerce_float(cooldown_payload.get("white_min_ratio"), 0.01), 0.0),
                    1.0,
                ),
                "dark_min_ratio": min(
                    max(self._coerce_float(cooldown_payload.get("dark_min_ratio"), 0.24), 0.0),
                    1.0,
                ),
                "bright_min_ratio": min(
                    max(self._coerce_float(cooldown_payload.get("bright_min_ratio"), 0.08), 0.0),
                    1.0,
                ),
            },
            "disabled": {
                "dark_min_ratio": min(
                    max(self._coerce_float(disabled_payload.get("dark_min_ratio"), 0.48), 0.0),
                    1.0,
                ),
                "bright_max_ratio": min(
                    max(self._coerce_float(disabled_payload.get("bright_max_ratio"), 0.16), 0.0),
                    1.0,
                ),
                "white_max_ratio": min(
                    max(self._coerce_float(disabled_payload.get("white_max_ratio"), 0.012), 0.0),
                    1.0,
                ),
            },
            "shared": {
                "white_sat_max": min(max(self._coerce_int(shared_payload.get("white_sat_max"), 45), 0), 255),
                "white_value_min": min(max(self._coerce_int(shared_payload.get("white_value_min"), 170), 0), 255),
                "bright_value_min": min(max(self._coerce_int(shared_payload.get("bright_value_min"), 140), 0), 255),
                "dark_value_max": min(max(self._coerce_int(shared_payload.get("dark_value_max"), 95), 0), 255),
            },
        }

    def _coerce_template_spec(
        self,
        value: Any,
        *,
        default_path: str,
        default_region: Any,
        default_threshold: float,
        default_scales: tuple[float, ...],
    ) -> dict[str, Any]:
        payload = dict(value or {}) if isinstance(value, Mapping) else {}
        return {
            "path": str(self._resolve_template_path(payload.get("path") or default_path)),
            "search_region": self._coerce_region(payload.get("search_region"), default=self._coerce_region(default_region, default=(0, 0, 1, 1))),
            "match_threshold": min(max(self._coerce_float(payload.get("match_threshold"), default_threshold), 0.0), 1.0),
            "scales": self._coerce_scale_list(payload.get("scales"), default=default_scales),
            "use_mask": bool(payload.get("use_mask", True)),
            "mask_path": (
                str(self._resolve_template_path(payload.get("mask_path")))
                if payload.get("mask_path") is not None
                else None
            ),
        }

    def _resolve_template_path(self, value: Any) -> Path:
        raw = Path(str(value))
        if raw.is_absolute():
            return raw
        return (self._template_dir / raw).resolve()

    def _coerce_scale_list(self, value: Any, *, default: tuple[float, ...]) -> list[float]:
        if isinstance(value, (list, tuple)) and value:
            result = [max(float(item), 0.05) for item in value]
            return result or [float(item) for item in default]
        return [float(item) for item in default]

    def _coerce_optional_hsv_triplet(self, value: Any, *, default: HsvTriplet | None = None) -> HsvTriplet | None:
        if value is None:
            return default
        return self._coerce_hsv_triplet(value, default=default or (0, 0, 0))

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
                    {"if_available": "ultimate"},
                    {"if_available": "skill"},
                    {
                        "normal": {
                            "mode": "tap_spam",
                            "duration_ms": 560,
                            "interval_ms": 70,
                        }
                    },
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

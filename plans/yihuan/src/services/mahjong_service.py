"""Mahjong recognition and v1 strategy helpers for the Yihuan plan."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from packages.aura_core.api import service_info


Region = tuple[int, int, int, int]
Point = tuple[int, int]

SUITS = ("wan", "tong", "tiao")
SUIT_LABELS = {
    "wan": "万",
    "tong": "筒",
    "tiao": "条",
}


@service_info(
    alias="yihuan_mahjong",
    public=True,
    singleton=True,
    description="Recognize and plan basic actions for the Yihuan Mahjong mini-game.",
)
class YihuanMahjongService:
    """Screenshot detectors and pure strategy helpers for Blood Flow Mahjong v1."""

    _DEFAULT_PROFILE = "default_1280x720_cn"

    def __init__(self) -> None:
        self._plan_root = Path(__file__).resolve().parents[2]
        self._profile_dir = self._plan_root / "data" / "mahjong"
        self._profile_cache: dict[str, dict[str, Any]] = {}
        self._template_cache: dict[tuple[str, str, tuple[int, int]], np.ndarray | None] = {}

    def load_profile(self, profile_name: str | None = None) -> dict[str, Any]:
        resolved_name = str(profile_name or self._DEFAULT_PROFILE).strip() or self._DEFAULT_PROFILE
        cached = self._profile_cache.get(resolved_name)
        if cached is not None:
            return dict(cached)

        profile_path = self._profile_dir / f"{resolved_name}.yaml"
        if not profile_path.is_file():
            raise FileNotFoundError(f"Mahjong profile not found: {profile_path}")

        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            payload = {}

        switches_payload = dict(payload.get("auto_switches") or {})
        buttons_payload = dict(payload.get("dingque_buttons") or {})
        normalized = {
            "profile_name": str(payload.get("profile_name") or resolved_name),
            "client_size": self._coerce_size(payload.get("client_size"), default=(1280, 720)),
            "template_dir": str(
                payload.get("template_dir")
                or f"data/mahjong/{resolved_name}/templates"
            ),
            "template_threshold": min(max(self._coerce_float(payload.get("template_threshold"), 0.82), 0.0), 1.0),
            "ready_button_region": self._coerce_region(
                payload.get("ready_button_region"),
                default=(1045, 610, 165, 100),
            ),
            "ready_button_point": self._coerce_point(payload.get("ready_button_point"), default=(1100, 650)),
            "ready_min_bright_ratio": self._coerce_float(payload.get("ready_min_bright_ratio"), 0.10),
            "ready_min_green_ratio": self._coerce_float(payload.get("ready_min_green_ratio"), 0.01),
            "ready_bright_s_max": self._coerce_int(payload.get("ready_bright_s_max"), 80),
            "ready_bright_v_min": self._coerce_int(payload.get("ready_bright_v_min"), 150),
            "ready_green_hsv_lower": self._coerce_hsv_triplet(
                payload.get("ready_green_hsv_lower"),
                default=(35, 45, 70),
            ),
            "ready_green_hsv_upper": self._coerce_hsv_triplet(
                payload.get("ready_green_hsv_upper"),
                default=(90, 255, 255),
            ),
            "dingque_buttons": {
                "wan": self._coerce_button_profile(
                    buttons_payload.get("wan"),
                    default_point=(540, 493),
                    default_region=(510, 465, 90, 95),
                    default_bounds=((0, 60, 80), (12, 255, 255)),
                    extra_bounds=((165, 60, 80), (179, 255, 255)),
                ),
                "tong": self._coerce_button_profile(
                    buttons_payload.get("tong"),
                    default_point=(637, 493),
                    default_region=(602, 465, 90, 95),
                    default_bounds=((14, 60, 90), (36, 255, 255)),
                ),
                "tiao": self._coerce_button_profile(
                    buttons_payload.get("tiao"),
                    default_point=(735, 493),
                    default_region=(695, 465, 90, 95),
                    default_bounds=((38, 60, 70), (95, 255, 255)),
                ),
            },
            "dingque_button_min_dark_ratio": self._coerce_float(payload.get("dingque_button_min_dark_ratio"), 0.18),
            "dingque_button_min_color_ratio": self._coerce_float(payload.get("dingque_button_min_color_ratio"), 0.025),
            "dingque_required_button_count": max(self._coerce_int(payload.get("dingque_required_button_count"), 3), 1),
            "hand_region": self._coerce_region(payload.get("hand_region"), default=(315, 612, 690, 105)),
            "hand_tile_min_width": max(self._coerce_int(payload.get("hand_tile_min_width"), 24), 1),
            "hand_tile_max_width": max(self._coerce_int(payload.get("hand_tile_max_width"), 78), 1),
            "hand_tile_expected_width": max(self._coerce_int(payload.get("hand_tile_expected_width"), 46), 1),
            "hand_tile_min_height": max(self._coerce_int(payload.get("hand_tile_min_height"), 34), 1),
            "hand_tile_s_max": max(self._coerce_int(payload.get("hand_tile_s_max"), 95), 0),
            "hand_tile_v_min": max(self._coerce_int(payload.get("hand_tile_v_min"), 125), 0),
            "hand_tile_projection_min_ratio": min(
                max(self._coerce_float(payload.get("hand_tile_projection_min_ratio"), 0.22), 0.01),
                1.0,
            ),
            "suit_hsv_bounds": self._coerce_suit_hsv_bounds(payload.get("suit_hsv_bounds")),
            "suit_min_pixels": max(self._coerce_int(payload.get("suit_min_pixels"), 22), 1),
            "suit_min_ratio": min(max(self._coerce_float(payload.get("suit_min_ratio"), 0.012), 0.0), 1.0),
            "missing_suit_tie_break_order": self._coerce_suit_order(
                payload.get("missing_suit_tie_break_order"),
                default=("tong", "tiao", "wan"),
            ),
            "exchange_prompt_region": self._coerce_region(
                payload.get("exchange_prompt_region"),
                default=(505, 415, 285, 55),
            ),
            "exchange_confirm_region": self._coerce_region(
                payload.get("exchange_confirm_region"),
                default=(556, 485, 168, 39),
            ),
            "exchange_confirm_point": self._coerce_point(
                payload.get("exchange_confirm_point"),
                default=(640, 504),
            ),
            "exchange_hand_region": self._coerce_region(
                payload.get("exchange_hand_region"),
                default=(315, 585, 690, 135),
            ),
            "exchange_required_tile_count": max(
                self._coerce_int(payload.get("exchange_required_tile_count"), 3),
                1,
            ),
            "exchange_wait_timeout_sec": max(
                self._coerce_float(payload.get("exchange_wait_timeout_sec"), 8.0),
                0.1,
            ),
            "exchange_prompt_min_white_ratio": self._coerce_float(
                payload.get("exchange_prompt_min_white_ratio"),
                0.015,
            ),
            "exchange_prompt_min_orange_ratio": self._coerce_float(
                payload.get("exchange_prompt_min_orange_ratio"),
                0.010,
            ),
            "exchange_confirm_present_min_body_ratio": self._coerce_float(
                payload.get("exchange_confirm_present_min_body_ratio"),
                0.35,
            ),
            "exchange_confirm_enabled_white_ratio": self._coerce_float(
                payload.get("exchange_confirm_enabled_white_ratio"),
                0.38,
            ),
            "exchange_suit_tie_break_order": self._coerce_suit_order(
                payload.get("exchange_suit_tie_break_order"),
                default=("tong", "tiao", "wan"),
            ),
            "auto_switches": {
                "hu": self._coerce_switch_profile(
                    switches_payload.get("hu"),
                    default_point=(56, 310),
                    default_region=(28, 290, 68, 58),
                    default_enabled_template="switch_hu_enabled.png",
                    default_disabled_template="switch_hu_disabled.png",
                ),
                "peng": self._coerce_switch_profile(
                    switches_payload.get("peng"),
                    default_point=(56, 377),
                    default_region=(28, 357, 68, 58),
                    default_enabled_template="switch_peng_enabled.png",
                    default_disabled_template="switch_peng_disabled.png",
                ),
                "discard": self._coerce_switch_profile(
                    switches_payload.get("discard"),
                    default_point=(56, 444),
                    default_region=(28, 424, 68, 58),
                    default_enabled_template="switch_discard_enabled.png",
                    default_disabled_template="switch_discard_disabled.png",
                ),
            },
            "switch_stack_region": self._coerce_region(payload.get("switch_stack_region"), default=(20, 280, 90, 235)),
            "switch_template_threshold": min(
                max(self._coerce_float(payload.get("switch_template_threshold"), 0.86), 0.0),
                1.0,
            ),
            "switch_template_score_margin": min(
                max(self._coerce_float(payload.get("switch_template_score_margin"), 0.08), 0.0),
                1.0,
            ),
            "switch_retry_limit": max(self._coerce_int(payload.get("switch_retry_limit"), 2), 1),
            "switch_verify_delay_ms": max(self._coerce_int(payload.get("switch_verify_delay_ms"), 120), 0),
            "result_panel_region": self._coerce_region(payload.get("result_panel_region"), default=(565, 80, 690, 575)),
            "result_first_row_region": self._coerce_region(
                payload.get("result_first_row_region"),
                default=(590, 145, 640, 95),
            ),
            "result_buttons_region": self._coerce_region(
                payload.get("result_buttons_region"),
                default=(675, 575, 500, 70),
            ),
            "result_auto_exit_region": self._coerce_region(
                payload.get("result_auto_exit_region"),
                default=(820, 645, 270, 55),
            ),
            "result_panel_min_dark_ratio": self._coerce_float(payload.get("result_panel_min_dark_ratio"), 0.42),
            "result_first_row_min_cyan_ratio": self._coerce_float(
                payload.get("result_first_row_min_cyan_ratio"),
                0.08,
            ),
            "result_buttons_min_white_ratio": self._coerce_float(payload.get("result_buttons_min_white_ratio"), 0.18),
            "result_auto_exit_min_white_ratio": self._coerce_float(
                payload.get("result_auto_exit_min_white_ratio"),
                0.015,
            ),
            "result_cyan_hsv_lower": self._coerce_hsv_triplet(
                payload.get("result_cyan_hsv_lower"),
                default=(78, 35, 80),
            ),
            "result_cyan_hsv_upper": self._coerce_hsv_triplet(
                payload.get("result_cyan_hsv_upper"),
                default=(105, 255, 255),
            ),
            "poll_ms": max(self._coerce_int(payload.get("poll_ms"), 120), 0),
            "post_click_delay_ms": max(self._coerce_int(payload.get("post_click_delay_ms"), 250), 0),
            "start_wait_timeout_sec": max(self._coerce_float(payload.get("start_wait_timeout_sec"), 12.0), 0.1),
            "dingque_wait_timeout_sec": max(self._coerce_float(payload.get("dingque_wait_timeout_sec"), 8.0), 0.1),
            "phase_timeout_sec": max(self._coerce_float(payload.get("phase_timeout_sec"), 20.0), 0.1),
            "debug_snapshot_dir": str(payload.get("debug_snapshot_dir") or "tmp/mahjong_debug"),
        }
        self._profile_cache[resolved_name] = normalized
        return dict(normalized)

    def analyze_phase(self, source_image: np.ndarray, *, profile_name: str | None = None) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        result = self.analyze_result_screen(source_image, profile_name=profile["profile_name"])
        exchange = self.analyze_exchange(source_image, profile_name=profile["profile_name"])
        ready = self.analyze_ready(source_image, profile_name=profile["profile_name"])
        dingque = self.analyze_dingque(source_image, profile_name=profile["profile_name"])
        playing = self.analyze_playing(source_image, profile_name=profile["profile_name"])

        phase = "unknown"
        if result["found"]:
            phase = "result"
        elif exchange["found"]:
            phase = "exchange"
        elif dingque["found"]:
            phase = "dingque"
        elif playing["found"]:
            phase = "playing"
        elif ready["found"]:
            phase = "ready"

        return {
            "phase": phase,
            "profile_name": profile["profile_name"],
            "capture_size": [int(source_image.shape[1]), int(source_image.shape[0])],
            "ready": ready,
            "exchange": exchange,
            "dingque": dingque,
            "playing": playing,
            "result": result,
        }

    def analyze_exchange(self, source_image: np.ndarray, *, profile_name: str | None = None) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        prompt_region = self.scale_region(source_image, profile["exchange_prompt_region"], profile=profile)
        confirm_region = self.scale_region(source_image, profile["exchange_confirm_region"], profile=profile)
        prompt = self._crop_region(source_image, prompt_region)
        confirm = self._crop_region(source_image, confirm_region)
        if prompt.size == 0 or confirm.size == 0:
            return {
                "found": False,
                "reason": "empty_region",
                "prompt_region": list(prompt_region),
                "confirm_region": list(confirm_region),
                "confirm_enabled": False,
                "prompt_match": {},
                "confirm_enabled_match": {},
                "confirm_disabled_match": {},
                "prompt_white_ratio": 0.0,
                "prompt_orange_ratio": 0.0,
                "confirm_body_ratio": 0.0,
                "confirm_white_ratio": 0.0,
            }

        threshold = float(profile["template_threshold"])
        prompt_match = self._match_template(
            source_image,
            "exchange_prompt.png",
            region=profile["exchange_prompt_region"],
            profile=profile,
            threshold=threshold,
        )
        confirm_enabled_match = self._match_template(
            source_image,
            "exchange_confirm_enabled.png",
            region=profile["exchange_confirm_region"],
            profile=profile,
            threshold=threshold,
        )
        confirm_disabled_match = self._match_template(
            source_image,
            "exchange_confirm_disabled.png",
            region=profile["exchange_confirm_region"],
            profile=profile,
            threshold=threshold,
        )

        prompt_hsv = cv2.cvtColor(prompt, cv2.COLOR_RGB2HSV)
        prompt_white_ratio = self._ratio((prompt_hsv[:, :, 1] <= 80) & (prompt_hsv[:, :, 2] >= 150))
        prompt_orange_ratio = self._mask_ratio(prompt_hsv, [[(10, 45, 90), (35, 255, 255)]])
        prompt_visual_found = (
            prompt_white_ratio >= float(profile["exchange_prompt_min_white_ratio"])
            and prompt_white_ratio <= 0.35
            and prompt_orange_ratio >= float(profile["exchange_prompt_min_orange_ratio"])
        )

        confirm_hsv = cv2.cvtColor(confirm, cv2.COLOR_RGB2HSV)
        confirm_body_ratio = self._ratio((confirm_hsv[:, :, 1] <= 90) & (confirm_hsv[:, :, 2] >= 105))
        confirm_white_ratio = self._ratio((confirm_hsv[:, :, 1] <= 70) & (confirm_hsv[:, :, 2] >= 190))
        confirm_present = (
            confirm_body_ratio >= float(profile["exchange_confirm_present_min_body_ratio"])
            or bool(confirm_enabled_match.get("found"))
            or bool(confirm_disabled_match.get("found"))
        )
        confirm_enabled = (
            confirm_white_ratio >= float(profile["exchange_confirm_enabled_white_ratio"])
        )
        prompt_found = bool(prompt_match.get("found")) or prompt_visual_found
        found = prompt_found and confirm_present
        if found:
            reason = "ok"
        elif not prompt_found:
            reason = "prompt_missing"
        else:
            reason = "confirm_missing"
        return {
            "found": found,
            "reason": reason,
            "prompt_region": list(prompt_region),
            "confirm_region": list(confirm_region),
            "confirm_point": list(self.scale_point(source_image, profile["exchange_confirm_point"], profile=profile)),
            "confirm_enabled": bool(confirm_enabled),
            "prompt_match": prompt_match,
            "confirm_enabled_match": confirm_enabled_match,
            "confirm_disabled_match": confirm_disabled_match,
            "prompt_white_ratio": prompt_white_ratio,
            "prompt_orange_ratio": prompt_orange_ratio,
            "confirm_body_ratio": confirm_body_ratio,
            "confirm_white_ratio": confirm_white_ratio,
        }

    def analyze_ready(self, source_image: np.ndarray, *, profile_name: str | None = None) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        region = self.scale_region(source_image, profile["ready_button_region"], profile=profile)
        crop = self._crop_region(source_image, region)
        if crop.size == 0:
            return self._ready_result(False, "empty_region", region, 0.0, 0.0)

        template_match = self._match_template(
            source_image,
            "ready_button.png",
            region=profile["ready_button_region"],
            profile=profile,
            threshold=float(profile["template_threshold"]),
        )
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        bright_mask = (
            (hsv[:, :, 1] <= int(profile["ready_bright_s_max"]))
            & (hsv[:, :, 2] >= int(profile["ready_bright_v_min"]))
        )
        bright_ratio = self._ratio(bright_mask)
        green_ratio = self._mask_ratio(hsv, [profile["ready_green_hsv_lower"], profile["ready_green_hsv_upper"]])
        visual_found = (
            bright_ratio >= float(profile["ready_min_bright_ratio"])
            and green_ratio >= float(profile["ready_min_green_ratio"])
        )
        found = bool(template_match.get("found")) or visual_found
        reason = "ok" if found else "visual_threshold_low"
        result = self._ready_result(found, reason, region, bright_ratio, green_ratio)
        result["template_match"] = template_match
        return result

    def analyze_dingque(self, source_image: np.ndarray, *, profile_name: str | None = None) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        buttons: dict[str, dict[str, Any]] = {}
        found_count = 0
        for suit, button_profile in dict(profile["dingque_buttons"]).items():
            region = self.scale_region(source_image, button_profile["region"], profile=profile)
            crop = self._crop_region(source_image, region)
            if crop.size == 0:
                buttons[suit] = {
                    "found": False,
                    "reason": "empty_region",
                    "region": list(region),
                    "dark_ratio": 0.0,
                    "color_ratio": 0.0,
                }
                continue
            template_match = self._match_template(
                source_image,
                f"dingque_{suit}.png",
                region=button_profile["region"],
                profile=profile,
                threshold=float(profile["template_threshold"]),
            )
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            dark_ratio = self._ratio(hsv[:, :, 2] <= 80)
            color_ratio = self._mask_ratio(hsv, button_profile["hsv_bounds"])
            button_found = (
                bool(template_match.get("found"))
                or (
                    dark_ratio >= float(profile["dingque_button_min_dark_ratio"])
                    and color_ratio >= float(profile["dingque_button_min_color_ratio"])
                )
            )
            if button_found:
                found_count += 1
            buttons[suit] = {
                "found": button_found,
                "reason": "ok" if button_found else "visual_threshold_low",
                "region": list(region),
                "dark_ratio": dark_ratio,
                "color_ratio": color_ratio,
                "template_match": template_match,
            }

        required = min(int(profile["dingque_required_button_count"]), len(SUITS))
        found = found_count >= required
        return {
            "found": found,
            "reason": "ok" if found else "buttons_missing",
            "found_button_count": found_count,
            "required_button_count": required,
            "buttons": buttons,
        }

    def analyze_playing(self, source_image: np.ndarray, *, profile_name: str | None = None) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        switches: dict[str, dict[str, Any]] = {}
        present_count = 0
        enabled_count = 0
        threshold = float(profile["switch_template_threshold"])
        score_margin = float(profile["switch_template_score_margin"])
        for name, switch_profile in dict(profile["auto_switches"]).items():
            region = self.scale_region(source_image, switch_profile["region"], profile=profile)
            crop = self._crop_region(source_image, region)
            if crop.size == 0:
                switches[name] = {
                    "present": False,
                    "enabled": False,
                    "reason": "empty_region",
                    "region": list(region),
                    "enabled_template_score": 0.0,
                    "disabled_template_score": 0.0,
                }
                continue
            enabled_match = self._match_template(
                source_image,
                str(switch_profile["enabled_template"]),
                region=switch_profile["region"],
                profile=profile,
                threshold=threshold,
                color=True,
                match_method="sqdiff",
            )
            disabled_match = self._match_template(
                source_image,
                str(switch_profile["disabled_template"]),
                region=switch_profile["region"],
                profile=profile,
                threshold=threshold,
                color=True,
                match_method="sqdiff",
            )
            enabled_score = float(enabled_match.get("score") or 0.0)
            disabled_score = float(disabled_match.get("score") or 0.0)
            enabled_template_found = bool(enabled_match.get("found"))
            disabled_template_found = bool(disabled_match.get("found"))
            enabled = enabled_template_found and enabled_score >= disabled_score + score_margin
            present = enabled_template_found or disabled_template_found
            if present:
                present_count += 1
            if enabled:
                enabled_count += 1
            if enabled:
                reason = "enabled_template"
            elif disabled_template_found:
                reason = "disabled_template"
            elif enabled_template_found:
                reason = "ambiguous_template"
            else:
                reason = "template_score_below_threshold"
            switches[name] = {
                "present": present,
                "enabled": enabled,
                "reason": reason,
                "region": list(region),
                "enabled_template_score": enabled_score,
                "disabled_template_score": disabled_score,
                "template_score_margin": score_margin,
                "template_match": enabled_match,
                "enabled_template_match": enabled_match,
                "disabled_template_match": disabled_match,
            }

        stack_region = self.scale_region(source_image, profile["switch_stack_region"], profile=profile)

        found = present_count >= 2
        return {
            "found": found,
            "reason": "ok" if found else "switches_missing",
            "present_switch_count": present_count,
            "enabled_switch_count": enabled_count,
            "switch_stack_region": list(stack_region),
            "switch_template_threshold": threshold,
            "switch_template_score_margin": score_margin,
            "switches": switches,
        }

    def analyze_result_screen(self, source_image: np.ndarray, *, profile_name: str | None = None) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        panel_region = self.scale_region(source_image, profile["result_panel_region"], profile=profile)
        first_row_region = self.scale_region(source_image, profile["result_first_row_region"], profile=profile)
        buttons_region = self.scale_region(source_image, profile["result_buttons_region"], profile=profile)
        auto_exit_region = self.scale_region(source_image, profile["result_auto_exit_region"], profile=profile)

        panel = self._crop_region(source_image, panel_region)
        first_row = self._crop_region(source_image, first_row_region)
        buttons = self._crop_region(source_image, buttons_region)
        auto_exit = self._crop_region(source_image, auto_exit_region)
        if panel.size == 0 or buttons.size == 0:
            return {
                "found": False,
                "reason": "empty_region",
                "panel_region": list(panel_region),
                "first_row_region": list(first_row_region),
                "buttons_region": list(buttons_region),
                "auto_exit_region": list(auto_exit_region),
                "panel_dark_ratio": 0.0,
                "first_row_cyan_ratio": 0.0,
                "buttons_white_ratio": 0.0,
                "auto_exit_white_ratio": 0.0,
            }

        panel_hsv = cv2.cvtColor(panel, cv2.COLOR_RGB2HSV)
        panel_dark_ratio = self._ratio(panel_hsv[:, :, 2] <= 75)

        first_row_cyan_ratio = 0.0
        if first_row.size:
            first_row_hsv = cv2.cvtColor(first_row, cv2.COLOR_RGB2HSV)
            first_row_cyan_ratio = self._mask_ratio(
                first_row_hsv,
                [[profile["result_cyan_hsv_lower"], profile["result_cyan_hsv_upper"]]],
            )

        buttons_hsv = cv2.cvtColor(buttons, cv2.COLOR_RGB2HSV)
        buttons_white_ratio = self._ratio((buttons_hsv[:, :, 1] <= 65) & (buttons_hsv[:, :, 2] >= 160))

        auto_exit_white_ratio = 0.0
        if auto_exit.size:
            auto_exit_hsv = cv2.cvtColor(auto_exit, cv2.COLOR_RGB2HSV)
            auto_exit_white_ratio = self._ratio(
                (auto_exit_hsv[:, :, 1] <= 80) & (auto_exit_hsv[:, :, 2] >= 150)
            )

        threshold = float(profile["template_threshold"])
        title_match = self._match_template(
            source_image,
            "result_ranking_title.png",
            region=profile["result_panel_region"],
            profile=profile,
            threshold=threshold,
        )
        exit_match = self._match_template(
            source_image,
            "result_exit_button.png",
            region=profile["result_buttons_region"],
            profile=profile,
            threshold=threshold,
        )
        retry_match = self._match_template(
            source_image,
            "result_retry_button.png",
            region=profile["result_buttons_region"],
            profile=profile,
            threshold=threshold,
        )
        auto_exit_match = self._match_template(
            source_image,
            "result_auto_exit_text.png",
            region=profile["result_auto_exit_region"],
            profile=profile,
            threshold=threshold,
        )

        panel_pass = panel_dark_ratio >= float(profile["result_panel_min_dark_ratio"])
        buttons_pass = buttons_white_ratio >= float(profile["result_buttons_min_white_ratio"])
        cyan_pass = first_row_cyan_ratio >= float(profile["result_first_row_min_cyan_ratio"])
        auto_exit_pass = auto_exit_white_ratio >= float(profile["result_auto_exit_min_white_ratio"])
        template_pass = bool(title_match.get("found")) and (
            bool(exit_match.get("found"))
            or bool(retry_match.get("found"))
            or bool(auto_exit_match.get("found"))
        )
        found = bool(template_pass or (panel_pass and buttons_pass and (cyan_pass or auto_exit_pass)))
        if found:
            reason = "ok"
        elif not panel_pass:
            reason = "panel_dark_ratio_low"
        elif not buttons_pass:
            reason = "buttons_white_ratio_low"
        else:
            reason = "title_features_missing"
        return {
            "found": found,
            "reason": reason,
            "panel_region": list(panel_region),
            "first_row_region": list(first_row_region),
            "buttons_region": list(buttons_region),
            "auto_exit_region": list(auto_exit_region),
            "panel_dark_ratio": panel_dark_ratio,
            "panel_min_dark_ratio": float(profile["result_panel_min_dark_ratio"]),
            "first_row_cyan_ratio": first_row_cyan_ratio,
            "first_row_min_cyan_ratio": float(profile["result_first_row_min_cyan_ratio"]),
            "buttons_white_ratio": buttons_white_ratio,
            "buttons_min_white_ratio": float(profile["result_buttons_min_white_ratio"]),
            "auto_exit_white_ratio": auto_exit_white_ratio,
            "auto_exit_min_white_ratio": float(profile["result_auto_exit_min_white_ratio"]),
            "template_pass": template_pass,
            "title_match": title_match,
            "exit_match": exit_match,
            "retry_match": retry_match,
            "auto_exit_match": auto_exit_match,
        }

    def analyze_hand_suits(self, source_image: np.ndarray, *, profile_name: str | None = None) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        return self._analyze_hand_suits_in_region(
            source_image,
            profile=profile,
            logical_region=profile["hand_region"],
        )

    def analyze_exchange_hand_suits(self, source_image: np.ndarray, *, profile_name: str | None = None) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        return self._analyze_hand_suits_in_region(
            source_image,
            profile=profile,
            logical_region=profile["exchange_hand_region"],
        )

    def _analyze_hand_suits_in_region(
        self,
        source_image: np.ndarray,
        *,
        profile: Mapping[str, Any],
        logical_region: Region,
    ) -> dict[str, Any]:
        region = self.scale_region(source_image, logical_region, profile=profile)
        crop = self._crop_region(source_image, region)
        counts = {suit: 0 for suit in SUITS}
        candidates: list[dict[str, Any]] = []
        if crop.size == 0:
            return {
                "counts": counts,
                "candidates": candidates,
                "region": list(region),
                "reason": "empty_region",
            }

        tile_rects = self._extract_hand_tile_rects(crop, profile)
        for index, rect in enumerate(tile_rects):
            x, y, width, height = rect
            tile_crop = crop[y : y + height, x : x + width]
            classification = self.classify_tile_suit(tile_crop, profile_name=profile["profile_name"])
            suit = classification.get("suit")
            if suit in counts:
                counts[str(suit)] += 1
            candidates.append(
                {
                    "index": index,
                    "rect": [int(region[0] + x), int(region[1] + y), int(width), int(height)],
                    **classification,
                }
            )

        return {
            "counts": counts,
            "candidates": candidates,
            "region": list(region),
            "reason": "ok" if candidates else "no_tile_candidates",
        }

    def classify_tile_suit(self, tile_image: np.ndarray, *, profile_name: str | None = None) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        if tile_image.size == 0:
            return {"suit": None, "reason": "empty_tile", "scores": {}}

        height, width = tile_image.shape[:2]
        x0 = int(round(width * 0.08))
        x1 = int(round(width * 0.92))
        y0 = int(round(height * 0.08))
        y1 = int(round(height * 0.95))
        inner = tile_image[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
        hsv = cv2.cvtColor(inner, cv2.COLOR_RGB2HSV)
        total_pixels = max(int(hsv.shape[0] * hsv.shape[1]), 1)
        scores: dict[str, dict[str, Any]] = {}
        best_suit: str | None = None
        best_pixels = 0
        for suit in SUITS:
            mask = self._combined_mask(hsv, profile["suit_hsv_bounds"][suit])
            pixels = int(np.count_nonzero(mask))
            ratio = float(pixels) / float(total_pixels)
            scores[suit] = {"pixels": pixels, "ratio": ratio}
            if pixels > best_pixels:
                best_pixels = pixels
                best_suit = suit

        if best_suit is None:
            return {"suit": None, "reason": "no_suit_pixels", "scores": scores}

        best_ratio = float(scores[best_suit]["ratio"])
        if best_pixels < int(profile["suit_min_pixels"]) or best_ratio < float(profile["suit_min_ratio"]):
            return {
                "suit": None,
                "reason": "suit_threshold_low",
                "scores": scores,
            }
        return {
            "suit": best_suit,
            "reason": "ok",
            "label": SUIT_LABELS[best_suit],
            "scores": scores,
        }

    def choose_missing_suit(
        self,
        counts: Mapping[str, Any],
        candidates: list[Mapping[str, Any]] | None = None,
        *,
        profile_name: str | None = None,
    ) -> str:
        return str(
            self.choose_missing_suit_detail(
                counts,
                candidates,
                profile_name=profile_name,
            )["selected_suit"]
        )

    def choose_missing_suit_detail(
        self,
        counts: Mapping[str, Any],
        candidates: list[Mapping[str, Any]] | None = None,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        normalized_counts = {suit: max(self._coerce_int(counts.get(suit), 0), 0) for suit in SUITS}
        min_count = min(normalized_counts.values())
        tied = [suit for suit in SUITS if normalized_counts[suit] == min_count]
        weakness = self._suit_weakness_scores(candidates or [])
        tie_order = list(profile["missing_suit_tie_break_order"])
        tie_rank = {suit: index for index, suit in enumerate(tie_order)}
        selected = sorted(
            tied,
            key=lambda suit: (
                -float(weakness.get(suit, {}).get("score") or 0.0),
                tie_rank.get(suit, len(tie_rank)),
            ),
        )[0]
        return {
            "selected_suit": selected,
            "selected_label": SUIT_LABELS[selected],
            "counts": normalized_counts,
            "tied_suits": tied,
            "tie_break_order": tie_order,
            "weakness": weakness,
            "reason": "fewest_tiles" if len(tied) == 1 else "tie_break",
        }

    def choose_exchange_three_detail(
        self,
        candidates: list[Mapping[str, Any]] | None,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        required = int(profile["exchange_required_tile_count"])
        grouped: dict[str, list[dict[str, Any]]] = {suit: [] for suit in SUITS}
        for tile in candidates or []:
            if not isinstance(tile, Mapping):
                continue
            suit = str(tile.get("suit") or "").strip()
            if suit not in grouped:
                continue
            grouped[suit].append(dict(tile))

        counts = {suit: len(tiles) for suit, tiles in grouped.items()}
        eligible = [suit for suit in SUITS if counts[suit] >= required]
        if not eligible:
            return {
                "found": False,
                "reason": "insufficient_same_suit",
                "required_tile_count": required,
                "counts": counts,
                "selected_suit": None,
                "selected_label": None,
                "selected_tiles": [],
                "tile_indices": [],
                "tile_points": [],
            }

        min_count = min(counts[suit] for suit in eligible)
        tied = [suit for suit in eligible if counts[suit] == min_count]
        tie_order = list(profile["exchange_suit_tie_break_order"])
        tie_rank = {suit: index for index, suit in enumerate(tie_order)}
        selected_suit = sorted(tied, key=lambda suit: tie_rank.get(suit, len(tie_rank)))[0]
        selected_tiles = sorted(grouped[selected_suit], key=self._tile_left_to_right_key)[:required]
        tile_indices = [int(tile.get("index", index)) for index, tile in enumerate(selected_tiles)]
        tile_points = [self._tile_center_point(tile) for tile in selected_tiles]
        return {
            "found": True,
            "reason": "fewest_eligible_suit" if len(tied) == 1 else "tie_break",
            "required_tile_count": required,
            "counts": counts,
            "eligible_suits": eligible,
            "tied_suits": tied,
            "tie_break_order": tie_order,
            "selected_suit": selected_suit,
            "selected_label": SUIT_LABELS[selected_suit],
            "selected_tiles": selected_tiles,
            "tile_indices": tile_indices,
            "tile_points": [[int(x), int(y)] for x, y in tile_points],
        }

    def plan_exchange_three(
        self,
        tiles: list[Mapping[str, Any]],
        *,
        missing_suit: str | None = None,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        """Pure future strategy: pick three low-value tiles for the exchange-three phase."""

        selected_missing = str(missing_suit or "").strip()
        candidates = [dict(tile) for tile in tiles if isinstance(tile, Mapping)]
        if selected_missing:
            preferred = [tile for tile in candidates if str(tile.get("suit")) == selected_missing]
        else:
            counts = Counter(str(tile.get("suit")) for tile in candidates if str(tile.get("suit")) in SUITS)
            selected_missing = self.choose_missing_suit(counts, candidates, profile_name=profile_name)
            preferred = [tile for tile in candidates if str(tile.get("suit")) == selected_missing]

        ranked = sorted(preferred or candidates, key=self._exchange_tile_rank, reverse=True)
        return {
            "selected_tiles": ranked[:3],
            "missing_suit": selected_missing,
            "reason": "prefer_missing_or_weak_suit",
        }

    def recommend_discard(
        self,
        tiles: list[Mapping[str, Any]],
        *,
        missing_suit: str,
    ) -> dict[str, Any]:
        """Pure future strategy: clear the missing suit first, then discard weak isolated tiles."""

        candidates = [dict(tile) for tile in tiles if isinstance(tile, Mapping)]
        missing_tiles = [tile for tile in candidates if str(tile.get("suit")) == str(missing_suit)]
        ranked = sorted(missing_tiles or candidates, key=self._discard_tile_rank, reverse=True)
        if not ranked:
            return {"found": False, "reason": "empty_hand", "tile": None}
        return {
            "found": True,
            "reason": "clear_missing_suit" if missing_tiles else "weakest_tile",
            "tile": ranked[0],
        }

    def scale_point(self, source_image: np.ndarray | None, point: Point, *, profile: Mapping[str, Any] | None = None) -> Point:
        profile_map = dict(profile or self.load_profile())
        if source_image is None:
            return int(point[0]), int(point[1])
        scale_x, scale_y = self._scale_factors(source_image, profile_map)
        return int(round(float(point[0]) * scale_x)), int(round(float(point[1]) * scale_y))

    def scale_region(self, source_image: np.ndarray, region: Region, *, profile: Mapping[str, Any] | None = None) -> Region:
        profile_map = dict(profile or self.load_profile())
        scale_x, scale_y = self._scale_factors(source_image, profile_map)
        x, y, width, height = region
        return (
            int(round(float(x) * scale_x)),
            int(round(float(y) * scale_y)),
            max(int(round(float(width) * scale_x)), 1),
            max(int(round(float(height) * scale_y)), 1),
        )

    def _extract_hand_tile_rects(self, crop: np.ndarray, profile: Mapping[str, Any]) -> list[Region]:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        tile_mask = (
            (hsv[:, :, 1] <= int(profile["hand_tile_s_max"]))
            & (hsv[:, :, 2] >= int(profile["hand_tile_v_min"]))
        )
        tile_mask = tile_mask.astype(np.uint8) * 255
        if tile_mask.size == 0:
            return []

        min_col_pixels = max(int(round(tile_mask.shape[0] * float(profile["hand_tile_projection_min_ratio"]))), 2)
        active_cols = np.where(np.sum(tile_mask > 0, axis=0) >= min_col_pixels)[0].tolist()
        rects: list[Region] = []
        for left, right in self._contiguous_runs(active_cols):
            width = int(right - left + 1)
            if width < int(profile["hand_tile_min_width"]):
                continue
            local = tile_mask[:, left : right + 1]
            active_rows = np.where(np.sum(local > 0, axis=1) >= max(int(width * 0.20), 2))[0]
            if active_rows.size == 0:
                continue
            top = int(active_rows[0])
            bottom = int(active_rows[-1])
            height = int(bottom - top + 1)
            if height < int(profile["hand_tile_min_height"]):
                continue
            if width <= int(profile["hand_tile_max_width"]):
                rects.append((int(left), top, width, height))
                continue
            rects.extend(self._split_wide_tile_run(int(left), top, width, height, profile))

        if rects:
            return sorted(self._dedupe_rects(rects), key=lambda item: (item[1], item[0]))

        return sorted(self._component_tile_rects(tile_mask, profile), key=lambda item: (item[1], item[0]))

    def _split_wide_tile_run(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        profile: Mapping[str, Any],
    ) -> list[Region]:
        expected_width = max(int(profile["hand_tile_expected_width"]), 1)
        split_count = max(int(round(float(width) / float(expected_width))), 1)
        if split_count <= 1:
            return [(left, top, width, height)]
        rects: list[Region] = []
        for index in range(split_count):
            x0 = left + int(round(index * width / split_count))
            x1 = left + int(round((index + 1) * width / split_count))
            part_width = max(int(x1 - x0), 1)
            if part_width >= int(profile["hand_tile_min_width"]):
                rects.append((x0, top, part_width, height))
        return rects

    def _component_tile_rects(self, mask: np.ndarray, profile: Mapping[str, Any]) -> list[Region]:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        rects: list[Region] = []
        for label in range(1, num_labels):
            x, y, width, height, area = [int(value) for value in stats[label]]
            if width < int(profile["hand_tile_min_width"]) or height < int(profile["hand_tile_min_height"]):
                continue
            if area < int(profile["hand_tile_min_width"] * profile["hand_tile_min_height"] * 0.35):
                continue
            if width <= int(profile["hand_tile_max_width"]):
                rects.append((x, y, width, height))
            else:
                rects.extend(self._split_wide_tile_run(x, y, width, height, profile))
        return self._dedupe_rects(rects)

    @staticmethod
    def _dedupe_rects(rects: list[Region]) -> list[Region]:
        unique: list[Region] = []
        seen: set[tuple[int, int]] = set()
        for rect in rects:
            x, y, width, height = rect
            key = (int(round(x / 4.0)), int(round(y / 4.0)))
            if key in seen:
                continue
            seen.add(key)
            unique.append((int(x), int(y), int(width), int(height)))
        return unique

    def _suit_weakness_scores(self, candidates: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        by_suit: dict[str, list[int]] = {suit: [] for suit in SUITS}
        for tile in candidates:
            suit = str(tile.get("suit") or "").strip()
            if suit not in by_suit:
                continue
            try:
                rank = int(tile.get("rank"))
            except (TypeError, ValueError):
                continue
            if 1 <= rank <= 9:
                by_suit[suit].append(rank)

        result: dict[str, dict[str, Any]] = {}
        for suit, ranks in by_suit.items():
            if not ranks:
                result[suit] = {"score": 0.0, "isolated": 0, "tatzu": 0, "pairs": 0}
                continue
            counts = Counter(ranks)
            pairs = sum(1 for value in counts.values() if value >= 2)
            tatzu = 0
            for rank in range(1, 9):
                if counts.get(rank, 0) and counts.get(rank + 1, 0):
                    tatzu += 1
            isolated = 0
            for rank in ranks:
                if counts[rank] >= 2:
                    continue
                if counts.get(rank - 2, 0) or counts.get(rank - 1, 0) or counts.get(rank + 1, 0) or counts.get(rank + 2, 0):
                    continue
                isolated += 1
            score = isolated * 10.0 - tatzu * 3.0 - pairs * 5.0
            result[suit] = {"score": score, "isolated": isolated, "tatzu": tatzu, "pairs": pairs}
        return result

    @staticmethod
    def _exchange_tile_rank(tile: Mapping[str, Any]) -> tuple[int, int, int]:
        rank = _safe_rank(tile.get("rank"))
        edge_penalty = 1 if rank in {1, 2, 8, 9} else 0
        isolated = 1 if bool(tile.get("isolated", True)) else 0
        paired = 1 if bool(tile.get("paired", False)) else 0
        return isolated, edge_penalty, -paired

    @staticmethod
    def _discard_tile_rank(tile: Mapping[str, Any]) -> tuple[int, int, int]:
        rank = _safe_rank(tile.get("rank"))
        edge = 1 if rank in {1, 9} else 0
        near_edge = 1 if rank in {2, 8} else 0
        isolated = 1 if bool(tile.get("isolated", True)) else 0
        return isolated, edge, near_edge

    @staticmethod
    def _tile_left_to_right_key(tile: Mapping[str, Any]) -> tuple[int, int, int]:
        rect = tile.get("rect")
        if isinstance(rect, (list, tuple)) and len(rect) >= 4:
            return int(rect[1]), int(rect[0]), int(tile.get("index", 0) or 0)
        return 0, 0, int(tile.get("index", 0) or 0)

    @staticmethod
    def _tile_center_point(tile: Mapping[str, Any]) -> Point:
        rect = tile.get("rect")
        if isinstance(rect, (list, tuple)) and len(rect) >= 4:
            x, y, width, height = [int(value) for value in rect[:4]]
            return int(round(x + width / 2.0)), int(round(y + height / 2.0))
        point = tile.get("center_point")
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            return int(point[0]), int(point[1])
        return 0, 0

    def _match_template(
        self,
        source_image: np.ndarray,
        template_name: str,
        *,
        region: Region,
        profile: Mapping[str, Any],
        threshold: float,
        color: bool = False,
        match_method: str = "ccoeff",
    ) -> dict[str, Any]:
        scaled_region = self.scale_region(source_image, region, profile=profile)
        crop = self._crop_region(source_image, scaled_region)
        if crop.size == 0:
            return {"found": False, "reason": "empty_region", "score": 0.0, "region": list(scaled_region)}
        template = self._load_template(template_name, profile=profile, source_image=source_image)
        if template is None or template.size == 0:
            return {"found": False, "reason": "template_missing", "score": 0.0, "region": list(scaled_region)}
        if template.shape[0] > crop.shape[0] or template.shape[1] > crop.shape[1]:
            return {
                "found": False,
                "reason": "template_larger_than_region",
                "score": 0.0,
                "region": list(scaled_region),
                "template_size": [int(template.shape[1]), int(template.shape[0])],
            }
        if color:
            match_crop = crop
            match_template = template
        else:
            match_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            match_template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        if str(match_method).strip().lower() == "sqdiff":
            result = cv2.matchTemplate(match_crop, match_template, cv2.TM_SQDIFF_NORMED)
            min_value, _, min_location, _ = cv2.minMaxLoc(result)
            score = 1.0 - float(min_value)
            match_location = min_location
            raw_distance = float(min_value)
        else:
            result = cv2.matchTemplate(match_crop, match_template, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, max_location = cv2.minMaxLoc(result)
            score = float(max_value)
            match_location = max_location
            raw_distance = None
        left = int(scaled_region[0] + match_location[0])
        top = int(scaled_region[1] + match_location[1])
        width = int(template.shape[1])
        height = int(template.shape[0])
        payload = {
            "found": score >= float(threshold),
            "reason": "ok" if score >= float(threshold) else "score_below_threshold",
            "score": score,
            "threshold": float(threshold),
            "region": list(scaled_region),
            "match_region": [left, top, width, height],
            "center_point": [int(left + width / 2), int(top + height / 2)],
            "color": bool(color),
            "match_method": str(match_method),
        }
        if raw_distance is not None:
            payload["distance"] = raw_distance
        return payload

    def _load_template(
        self,
        template_name: str,
        *,
        profile: Mapping[str, Any],
        source_image: np.ndarray,
    ) -> np.ndarray | None:
        scale_x, scale_y = self._scale_factors(source_image, profile)
        cache_key = (
            str(profile["profile_name"]),
            str(template_name),
            (int(round(scale_x * 1000)), int(round(scale_y * 1000))),
        )
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]
        template_dir = Path(str(profile["template_dir"]))
        if not template_dir.is_absolute():
            template_dir = self._plan_root / template_dir
        path = template_dir / template_name
        if not path.is_file():
            self._template_cache[cache_key] = None
            return None
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            self._template_cache[cache_key] = None
            return None
        template = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if abs(scale_x - 1.0) > 0.001 or abs(scale_y - 1.0) > 0.001:
            width = max(int(round(template.shape[1] * scale_x)), 1)
            height = max(int(round(template.shape[0] * scale_y)), 1)
            template = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        self._template_cache[cache_key] = template
        return template

    @staticmethod
    def _scale_factors(source_image: np.ndarray, profile: Mapping[str, Any]) -> tuple[float, float]:
        client_w, client_h = profile["client_size"]
        image_h, image_w = source_image.shape[:2]
        return float(image_w) / float(client_w), float(image_h) / float(client_h)

    @staticmethod
    def _crop_region(source_image: np.ndarray, region: Region) -> np.ndarray:
        x, y, width, height = region
        image_h, image_w = source_image.shape[:2]
        left = max(int(x), 0)
        top = max(int(y), 0)
        right = min(max(int(x + width), left), image_w)
        bottom = min(max(int(y + height), top), image_h)
        return source_image[top:bottom, left:right].copy()

    @staticmethod
    def _combined_mask(hsv: np.ndarray, bounds_list: list[list[tuple[int, int, int]]]) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in bounds_list:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)))
        return mask

    def _mask_ratio(self, hsv: np.ndarray, bounds_list: list[Any]) -> float:
        if hsv.size == 0:
            return 0.0
        normalized_bounds = []
        if self._looks_hsv_pair(bounds_list):
            normalized_bounds = [[bounds_list[0], bounds_list[1]]]
        else:
            normalized_bounds = bounds_list
        return self._ratio(self._combined_mask(hsv, normalized_bounds) > 0)

    @staticmethod
    def _ratio(mask: np.ndarray) -> float:
        if mask.size == 0:
            return 0.0
        return float(np.count_nonzero(mask)) / float(mask.size)

    @staticmethod
    def _contiguous_runs(values: list[int]) -> list[tuple[int, int]]:
        if not values:
            return []
        runs: list[tuple[int, int]] = []
        start = int(values[0])
        previous = int(values[0])
        for raw_value in values[1:]:
            value = int(raw_value)
            if value == previous + 1:
                previous = value
                continue
            runs.append((start, previous))
            start = value
            previous = value
        runs.append((start, previous))
        return runs

    @staticmethod
    def _ready_result(found: bool, reason: str, region: Region, bright_ratio: float, green_ratio: float) -> dict[str, Any]:
        return {
            "found": bool(found),
            "reason": reason,
            "region": list(region),
            "bright_ratio": float(bright_ratio),
            "green_ratio": float(green_ratio),
        }

    def _coerce_button_profile(
        self,
        payload: Any,
        *,
        default_point: Point,
        default_region: Region,
        default_bounds: tuple[tuple[int, int, int], tuple[int, int, int]],
        extra_bounds: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    ) -> dict[str, Any]:
        data = dict(payload or {}) if isinstance(payload, Mapping) else {}
        bounds = [self._coerce_hsv_pair(data.get("hsv_bounds"), default=default_bounds)]
        if extra_bounds is not None:
            bounds.append(self._coerce_hsv_pair(data.get("hsv_bounds_2"), default=extra_bounds))
        return {
            "point": self._coerce_point(data.get("point"), default=default_point),
            "region": self._coerce_region(data.get("region"), default=default_region),
            "hsv_bounds": bounds,
        }

    def _coerce_switch_profile(
        self,
        payload: Any,
        *,
        default_point: Point,
        default_region: Region,
        default_enabled_template: str,
        default_disabled_template: str,
    ) -> dict[str, Any]:
        data = dict(payload or {}) if isinstance(payload, Mapping) else {}
        return {
            "point": self._coerce_point(data.get("point"), default=default_point),
            "region": self._coerce_region(data.get("region"), default=default_region),
            "enabled_template": str(data.get("enabled_template") or default_enabled_template),
            "disabled_template": str(data.get("disabled_template") or default_disabled_template),
        }

    def _coerce_suit_hsv_bounds(self, value: Any) -> dict[str, list[list[tuple[int, int, int]]]]:
        payload = dict(value or {}) if isinstance(value, Mapping) else {}
        defaults = {
            "wan": [
                ((0, 65, 70), (12, 255, 255)),
                ((165, 65, 70), (179, 255, 255)),
            ],
            "tong": [
                ((100, 35, 55), (145, 255, 255)),
            ],
            "tiao": [
                ((35, 35, 45), (95, 255, 230)),
            ],
        }
        result: dict[str, list[list[tuple[int, int, int]]]] = {}
        for suit in SUITS:
            raw_bounds = payload.get(suit)
            pairs: list[list[tuple[int, int, int]]] = []
            if isinstance(raw_bounds, list) and raw_bounds:
                if self._looks_hsv_pair(raw_bounds):
                    pairs.append(list(self._coerce_hsv_pair(raw_bounds, default=defaults[suit][0])))
                else:
                    for item in raw_bounds:
                        if self._looks_hsv_pair(item):
                            pairs.append(list(self._coerce_hsv_pair(item, default=defaults[suit][0])))
            if not pairs:
                pairs = [[lower, upper] for lower, upper in defaults[suit]]
            result[suit] = pairs
        return result

    @staticmethod
    def _looks_hsv_pair(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], (list, tuple))
            and isinstance(value[1], (list, tuple))
            and len(value[0]) >= 3
            and len(value[1]) >= 3
        )

    def _coerce_suit_order(self, value: Any, *, default: tuple[str, ...]) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return list(default)
        seen: set[str] = set()
        order: list[str] = []
        for item in value:
            suit = str(item or "").strip()
            if suit in SUITS and suit not in seen:
                seen.add(suit)
                order.append(suit)
        for suit in default:
            if suit not in seen:
                order.append(suit)
        return order

    @staticmethod
    def _coerce_size(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return max(int(value[0]), 1), max(int(value[1]), 1)
        return default

    @staticmethod
    def _coerce_point(value: Any, *, default: Point) -> Point:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
        return default

    @staticmethod
    def _coerce_region(value: Any, *, default: Region) -> Region:
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return int(value[0]), int(value[1]), max(int(value[2]), 1), max(int(value[3]), 1)
        return default

    def _coerce_hsv_pair(
        self,
        value: Any,
        *,
        default: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return self._coerce_hsv_triplet(value[0], default=default[0]), self._coerce_hsv_triplet(
                value[1],
                default=default[1],
            )
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
            return float(default)

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)


def _safe_rank(value: Any) -> int:
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return 5
    return min(max(rank, 1), 9)

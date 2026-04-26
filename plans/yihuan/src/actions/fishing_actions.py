"""Fishing actions for the Yihuan plan."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any

import cv2

from packages.aura_core.api import action_info, requires_services
from packages.aura_core.observability.logging.core_logger import logger

from ..services.fishing_service import YihuanFishingService


_DEBUG_WINDOW_NAME = "Yihuan Fishing Duel Debug"
_LIVE_MONITOR_WINDOW_NAME = "Yihuan Fishing Live Monitor"


def _read_state_snapshot(
    app: Any,
    ocr: Any,
    vision: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    profile_name: str | None,
    include_capture_image: bool = False,
    enable_ocr: bool = True,
) -> dict[str, Any]:
    capture = app.capture()
    if not capture.success or capture.image is None:
        raise RuntimeError("Failed to capture the Yihuan fishing screen.")

    profile = yihuan_fishing.load_profile(profile_name)
    bite_texts: list[str] = []
    result_texts = _recognize_region_texts(capture.image, profile["result_text_region"], ocr) if enable_ocr else []
    bite_marker = yihuan_fishing.analyze_bite_marker(capture.image, profile_name=profile["profile_name"])
    duel = yihuan_fishing.analyze_duel_meter(capture.image, profile_name=profile["profile_name"])
    ready_anchor = yihuan_fishing.analyze_ready_anchor_with_vision(
        capture.image,
        vision,
        profile_name=profile["profile_name"],
    )
    result_match = yihuan_fishing._match_required_texts(result_texts, profile["ocr_texts"]["result_required"])

    phase = "unknown"
    ready_confidence = float(ready_anchor.get("confidence") or 0.0)
    if result_match["matched"]:
        phase = "result"
    elif ready_anchor["found"] and ready_confidence >= YihuanFishingService._READY_PRIORITY_THRESHOLD:
        phase = "ready"
    elif bite_marker["found"]:
        phase = "bite"
    elif duel["found"]:
        phase = "duel"
    elif ready_anchor["found"]:
        phase = "ready"

    state = {
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
    state["capture_size"] = [int(capture.image.shape[1]), int(capture.image.shape[0])]
    if include_capture_image:
        state["_capture_image"] = capture.image
    return state


def _read_duel_snapshot_fast(
    app: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    profile_name: str | None,
    include_capture_image: bool = False,
) -> dict[str, Any]:
    capture = app.capture()
    if not capture.success or capture.image is None:
        raise RuntimeError("Failed to capture the Yihuan fishing screen.")

    profile = yihuan_fishing.load_profile(profile_name)
    duel = yihuan_fishing.analyze_duel_meter(capture.image, profile_name=profile["profile_name"])
    state = {
        "phase": "duel" if duel["found"] else "unknown",
        "profile_name": profile["profile_name"],
        "bite": {"matched": False, "reason": "skipped_fast_duel"},
        "result": {"matched": False, "texts": [], "required": [], "missing": []},
        "ready_anchor": {"found": False, "confidence": 0.0, "backend": "skipped_fast_duel"},
        "duel": duel,
        "control_advice": duel["control_advice"] if duel["found"] else "none",
    }
    state["capture_size"] = [int(capture.image.shape[1]), int(capture.image.shape[0])]
    if include_capture_image:
        state["_capture_image"] = capture.image
    return state


def _recognize_region_texts(
    source_image,
    region,
    ocr: Any,
) -> list[str]:
    x, y, width, height = region
    crop = source_image[int(y): int(y + height), int(x): int(x + width)]
    if crop.size == 0:
        return []
    multi_result = ocr.recognize_all(crop)
    return [str(result.text).strip() for result in multi_result.results if str(result.text).strip()]


def _press_input_action(
    input_mapping: Any,
    app: Any,
    *,
    action_name: str,
    profile_name: str | None,
) -> None:
    input_mapping.execute_action(action_name, phase="press", app=app, profile=profile_name)


def _tap_input_action(
    input_mapping: Any,
    app: Any,
    *,
    action_name: str,
    profile_name: str | None,
    press_ms: int = 0,
) -> None:
    hold_ms = max(int(press_ms), 0)
    if hold_ms <= 0:
        input_mapping.execute_action(action_name, phase="tap", app=app, profile=profile_name)
        return

    input_mapping.execute_action(action_name, phase="hold", app=app, profile=profile_name)
    time.sleep(float(hold_ms) / 1000.0)
    input_mapping.execute_action(action_name, phase="release", app=app, profile=profile_name)


def _set_held_direction(
    input_mapping: Any,
    app: Any,
    *,
    current_action_name: str | None,
    desired_action_name: str | None,
    profile_name: str | None,
 ) -> tuple[str | None, bool]:
    if current_action_name == desired_action_name:
        return current_action_name, False
    if current_action_name is not None:
        input_mapping.execute_action(current_action_name, phase="release", app=app, profile=profile_name)
    if desired_action_name is not None:
        input_mapping.execute_action(desired_action_name, phase="hold", app=app, profile=profile_name)
    return desired_action_name, True


def _clamp_int(value: float, lower: int, upper: int) -> int:
    low = min(int(lower), int(upper))
    high = max(int(lower), int(upper))
    return max(min(int(round(value)), high), low)


def _compute_duel_control_command(
    state: dict[str, Any],
    profile: dict[str, Any],
    control_memory: dict[str, Any],
    now: float,
) -> dict[str, Any]:
    duel = state.get("duel") or {}
    current_hold_action = control_memory.get("held_action_name")
    command: dict[str, Any] = {
        "action_name": None,
        "hold_action_name": None,
        "tap_action_name": None,
        "execute": False,
        "control_phase": "none",
        "reason": "not_duel",
        "detection_trusted": False,
        "suspicious": False,
        "suspicious_reason": "none",
        "keep_hold": False,
        "keep_hold_age_ms": None,
        "zone_left": duel.get("zone_left"),
        "zone_right": duel.get("zone_right"),
        "zone_center": duel.get("zone_center"),
        "indicator_x": duel.get("indicator_x"),
        "center_error_px": duel.get("center_error_px") or duel.get("error_px"),
        "boundary_error_px": duel.get("boundary_error_px"),
        "predicted_error_px": None,
        "velocity_px_per_sec": 0.0,
        "zone_width": None,
        "zone_jump_px": 0,
        "indicator_jump_px": 0,
        "width_jump_px": 0,
    }
    if not bool(duel.get("found")):
        grace_ms = max(int(profile.get("control_hold_grace_ms", 150) or 0), 0)
        last_trusted_at = control_memory.get("last_trusted_at")
        if current_hold_action is not None and last_trusted_at is not None:
            age_ms = (float(now) - float(last_trusted_at)) * 1000.0
            if age_ms <= grace_ms:
                command.update(
                    {
                        "action_name": current_hold_action,
                        "hold_action_name": current_hold_action,
                        "control_phase": "keep_hold",
                        "reason": "missing_keep_hold",
                        "keep_hold": True,
                        "keep_hold_age_ms": round(age_ms, 1),
                    }
                )
                return command
        command["reason"] = "missing_release"
        return command

    try:
        zone_left = int(duel["zone_left"])
        zone_right = int(duel["zone_right"])
        indicator_x = int(duel["indicator_x"])
    except (TypeError, ValueError, KeyError):
        command["reason"] = "missing_geometry"
        return command

    zone_center = int(duel.get("zone_center") or round((zone_left + zone_right) / 2.0))
    zone_width = int(zone_right - zone_left)
    center_error_px = int(duel.get("center_error_px") if duel.get("center_error_px") is not None else indicator_x - zone_center)
    if duel.get("boundary_error_px") is not None:
        boundary_error_px = int(duel["boundary_error_px"])
    elif indicator_x < zone_left:
        boundary_error_px = int(indicator_x - zone_left)
    elif indicator_x > zone_right:
        boundary_error_px = int(indicator_x - zone_right)
    else:
        boundary_error_px = 0

    previous_geometry = control_memory.get("trusted_geometry")
    previous_sample_at = control_memory.get("last_trusted_at")
    velocity_px_per_sec = 0.0
    zone_jump_px = 0
    indicator_jump_px = 0
    width_jump_px = 0
    if isinstance(previous_geometry, dict) and previous_sample_at is not None:
        dt = float(now - float(previous_sample_at))
        if dt >= 0.005:
            previous_center_error = float(previous_geometry.get("center_error_px", center_error_px))
            velocity_px_per_sec = (float(center_error_px) - previous_center_error) / dt
        max_compare_dt = max(float(profile.get("control_detection_jump_window_ms", 250) or 0) / 1000.0, 0.0)
        if max_compare_dt <= 0 or dt <= max_compare_dt:
            zone_jump_px = max(
                abs(int(previous_geometry.get("zone_left", zone_left)) - zone_left),
                abs(int(previous_geometry.get("zone_right", zone_right)) - zone_right),
            )
            indicator_jump_px = abs(int(previous_geometry.get("indicator_x", indicator_x)) - indicator_x)
            width_jump_px = abs(int(previous_geometry.get("zone_width", zone_width)) - zone_width)

    lookahead_sec = float(profile.get("control_lookahead_sec") or 0.0)
    predicted_error_px = float(center_error_px) + velocity_px_per_sec * lookahead_sec

    command.update(
        {
            "zone_left": zone_left,
            "zone_right": zone_right,
            "zone_center": zone_center,
            "indicator_x": indicator_x,
            "center_error_px": center_error_px,
            "boundary_error_px": boundary_error_px,
            "predicted_error_px": predicted_error_px,
            "velocity_px_per_sec": velocity_px_per_sec,
            "zone_width": zone_width,
            "zone_jump_px": zone_jump_px,
            "indicator_jump_px": indicator_jump_px,
            "width_jump_px": width_jump_px,
        }
    )

    width_min = max(int(profile.get("control_zone_width_min_px", 70) or 0), 0)
    width_max = max(int(profile.get("control_zone_width_max_px", 120) or 0), width_min)
    zone_jump_limit = max(int(profile.get("control_zone_jump_px", 55) or 0), 0)
    indicator_jump_limit = max(int(profile.get("control_indicator_jump_px", 55) or 0), 0)
    width_jump_limit = max(int(profile.get("control_zone_width_jump_px", 30) or 0), 0)
    suspicious_reason = "none"
    if zone_width < width_min or zone_width > width_max:
        suspicious_reason = "zone_width_out_of_range"
    elif zone_jump_limit > 0 and zone_jump_px > zone_jump_limit:
        suspicious_reason = "zone_jump"
    elif indicator_jump_limit > 0 and indicator_jump_px > indicator_jump_limit:
        suspicious_reason = "indicator_jump"
    elif width_jump_limit > 0 and width_jump_px > width_jump_limit:
        suspicious_reason = "zone_width_jump"

    if suspicious_reason != "none":
        command["suspicious"] = True
        command["suspicious_reason"] = suspicious_reason
        grace_ms = max(int(profile.get("control_hold_grace_ms", 150) or 0), 0)
        last_trusted_at = control_memory.get("last_trusted_at")
        if current_hold_action is not None and last_trusted_at is not None:
            age_ms = (float(now) - float(last_trusted_at)) * 1000.0
            if age_ms <= grace_ms:
                command.update(
                    {
                        "action_name": current_hold_action,
                        "hold_action_name": current_hold_action,
                        "control_phase": "keep_hold",
                        "reason": "suspicious_keep_hold",
                        "keep_hold": True,
                        "keep_hold_age_ms": round(age_ms, 1),
                    }
                )
                return command
        command["reason"] = "suspicious_release"
        return command

    control_memory["last_trusted_at"] = float(now)
    control_memory["trusted_geometry"] = {
        "zone_left": zone_left,
        "zone_right": zone_right,
        "zone_center": zone_center,
        "zone_width": zone_width,
        "indicator_x": indicator_x,
        "center_error_px": center_error_px,
    }
    command["detection_trusted"] = True

    deadband_px = max(int(profile.get("control_deadband_px", profile.get("deadband_px", 6)) or 0), 0)
    hold_margin_px = max(int(profile.get("control_hold_margin_px", 0) or 0), 0)
    left_hold_threshold = zone_left + hold_margin_px
    right_hold_threshold = zone_right - hold_margin_px

    action_name: str | None = None
    hold_action_name: str | None = None
    tap_action_name: str | None = None
    reason = "stable"
    if indicator_x <= left_hold_threshold:
        action_name = "fish_right"
        hold_action_name = action_name
        reason = "outside_hold" if indicator_x < zone_left else "edge_hold"
    elif indicator_x >= right_hold_threshold:
        action_name = "fish_left"
        hold_action_name = action_name
        reason = "outside_hold" if indicator_x > zone_right else "edge_hold"
    elif predicted_error_px < -deadband_px:
        action_name = "fish_right"
        tap_action_name = action_name
        reason = "inside_tap"
    elif predicted_error_px > deadband_px:
        action_name = "fish_left"
        tap_action_name = action_name
        reason = "inside_tap"

    command["reason"] = reason
    command["action_name"] = action_name
    command["hold_action_name"] = hold_action_name
    command["tap_action_name"] = tap_action_name
    if hold_action_name is not None:
        command["control_phase"] = "hold"
        command["execute"] = hold_action_name != current_hold_action
    elif tap_action_name is not None:
        command["control_phase"] = "tap"
        command["execute"] = True
    elif current_hold_action is not None:
        command["control_phase"] = "release"
        command["execute"] = True
    return command


def _update_debug_window(
    yihuan_fishing: YihuanFishingService,
    *,
    source_image,
    state: dict[str, Any],
    profile_name: str | None,
    held_action_name: str | None,
) -> bool:
    try:
        frame = yihuan_fishing.build_duel_debug_view(
            source_image,
            state=state,
            profile_name=profile_name,
            held_action_name=held_action_name,
        )
        cv2.namedWindow(_DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(_DEBUG_WINDOW_NAME, frame)
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def _close_debug_window() -> None:
    try:
        cv2.destroyWindow(_DEBUG_WINDOW_NAME)
        cv2.waitKey(1)
    except cv2.error:
        pass


def _update_live_monitor_window(frame) -> tuple[bool, bool]:
    try:
        cv2.namedWindow(_LIVE_MONITOR_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(_LIVE_MONITOR_WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            return True, True
        if cv2.getWindowProperty(_LIVE_MONITOR_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False, True
        return True, False
    except cv2.error:
        return False, True


def _close_live_monitor_window() -> None:
    try:
        cv2.destroyWindow(_LIVE_MONITOR_WINDOW_NAME)
        cv2.waitKey(1)
    except cv2.error:
        pass


def _log_state_details(stage: str, state: dict[str, Any]) -> None:
    bite = state.get("bite") or {}
    duel = state.get("duel") or {}
    ready_anchor = state.get("ready_anchor") or {}
    logger.info(
        "Fishing[%s] phase=%s bite_matched=%s bite_reason=%s bite_pixels=%s bite_component_area=%s ready_conf=%.4f duel_reason=%s zone_detected=%s indicator_detected=%s indicator_raw_detected=%s indicator_source=%s zone_left=%s zone_center=%s zone_right=%s indicator_x=%s error_px=%s advice=%s",
        stage,
        state.get("phase"),
        bite.get("matched"),
        bite.get("reason"),
        bite.get("pixel_count"),
        bite.get("largest_component_area"),
        float(ready_anchor.get("confidence") or 0.0),
        duel.get("reason"),
        duel.get("zone_detected"),
        duel.get("indicator_detected"),
        duel.get("indicator_raw_detected"),
        duel.get("indicator_source"),
        duel.get("zone_left"),
        duel.get("zone_center"),
        duel.get("zone_right"),
        duel.get("indicator_x"),
        duel.get("error_px"),
        state.get("control_advice"),
    )


def _create_monitor_output_dir() -> Path:
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "logs" / "yihuan_fishing_monitor" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_monitor_debug_frame(
    yihuan_fishing: YihuanFishingService,
    *,
    source_image,
    state: dict[str, Any],
    profile_name: str | None,
    held_action_name: str | None,
    out_dir: Path,
    frame_index: int,
    elapsed_sec: float,
) -> str | None:
    try:
        frame = yihuan_fishing.build_duel_debug_view(
            source_image,
            state=state,
            profile_name=profile_name,
            held_action_name=held_action_name,
        )
        phase = str(state.get("phase") or "unknown")
        path = out_dir / f"frame_{frame_index:03d}_{elapsed_sec:06.2f}s_{phase}.png"
        cv2.imwrite(str(path), frame)
        return str(path)
    except cv2.error:
        return None


def _new_detection_stats() -> dict[str, Any]:
    return {
        "observation_sec": 0.0,
        "samples": 0,
        "zone_detected_sec": 0.0,
        "zone_missing_sec": 0.0,
        "indicator_detected_sec": 0.0,
        "indicator_missing_sec": 0.0,
        "indicator_raw_detected_sec": 0.0,
        "indicator_raw_missing_sec": 0.0,
        "zone_missing_with_raw_indicator_sec": 0.0,
        "zone_missing_without_raw_indicator_sec": 0.0,
        "indicator_missing_with_raw_indicator_sec": 0.0,
        "indicator_missing_without_raw_indicator_sec": 0.0,
        "zone_lost_events": 0,
        "zone_recovered_events": 0,
        "indicator_lost_events": 0,
        "indicator_recovered_events": 0,
        "indicator_raw_lost_events": 0,
        "indicator_raw_recovered_events": 0,
        "reason_sec": {},
    }


def _extract_detection_status(state: dict[str, Any]) -> dict[str, Any]:
    duel = state.get("duel") or {}
    phase = str(state.get("phase") or "unknown")
    reason = str(duel.get("reason") or "none")
    duel_found = bool(duel.get("found"))
    active = duel_found or (phase == "unknown" and reason in {"zone_missing", "indicator_missing"})
    return {
        "active": active,
        "phase": phase,
        "reason": reason,
        "duel_found": duel_found,
        "zone_detected": bool(duel.get("zone_detected")),
        "indicator_detected": bool(duel.get("indicator_detected")),
        "indicator_raw_detected": bool(duel.get("indicator_raw_detected")),
    }


def _accumulate_detection_stats(
    detection_stats: dict[str, Any],
    *,
    previous_status: dict[str, Any] | None,
    previous_sample_at: float | None,
    current_sample_at: float,
    current_status: dict[str, Any] | None = None,
) -> None:
    if previous_status is None or previous_sample_at is None:
        return
    if not bool(previous_status.get("active")):
        return
    delta_sec = max(float(current_sample_at - previous_sample_at), 0.0)
    detection_stats["observation_sec"] += delta_sec
    if previous_status["zone_detected"]:
        detection_stats["zone_detected_sec"] += delta_sec
    else:
        detection_stats["zone_missing_sec"] += delta_sec
    if previous_status["indicator_detected"]:
        detection_stats["indicator_detected_sec"] += delta_sec
    else:
        detection_stats["indicator_missing_sec"] += delta_sec
    if previous_status["indicator_raw_detected"]:
        detection_stats["indicator_raw_detected_sec"] += delta_sec
    else:
        detection_stats["indicator_raw_missing_sec"] += delta_sec

    reason = str(previous_status.get("reason") or "none")
    reason_sec = detection_stats.setdefault("reason_sec", {})
    reason_sec[reason] = float(reason_sec.get(reason, 0.0)) + delta_sec
    if not previous_status["zone_detected"]:
        if previous_status["indicator_raw_detected"]:
            detection_stats["zone_missing_with_raw_indicator_sec"] += delta_sec
        else:
            detection_stats["zone_missing_without_raw_indicator_sec"] += delta_sec
    if not previous_status["indicator_detected"]:
        if previous_status["indicator_raw_detected"]:
            detection_stats["indicator_missing_with_raw_indicator_sec"] += delta_sec
        else:
            detection_stats["indicator_missing_without_raw_indicator_sec"] += delta_sec

    if current_status is None:
        return
    if not bool(current_status.get("active")):
        return
    if bool(previous_status["zone_detected"]) and not bool(current_status["zone_detected"]):
        detection_stats["zone_lost_events"] += 1
    elif not bool(previous_status["zone_detected"]) and bool(current_status["zone_detected"]):
        detection_stats["zone_recovered_events"] += 1
    if bool(previous_status["indicator_detected"]) and not bool(current_status["indicator_detected"]):
        detection_stats["indicator_lost_events"] += 1
    elif not bool(previous_status["indicator_detected"]) and bool(current_status["indicator_detected"]):
        detection_stats["indicator_recovered_events"] += 1
    if bool(previous_status["indicator_raw_detected"]) and not bool(current_status["indicator_raw_detected"]):
        detection_stats["indicator_raw_lost_events"] += 1
    elif not bool(previous_status["indicator_raw_detected"]) and bool(current_status["indicator_raw_detected"]):
        detection_stats["indicator_raw_recovered_events"] += 1


def _finalize_detection_stats(
    detection_stats: dict[str, Any] | None,
    *,
    previous_status: dict[str, Any] | None,
    previous_sample_at: float | None,
) -> dict[str, Any] | None:
    if detection_stats is None:
        return None
    _accumulate_detection_stats(
        detection_stats,
        previous_status=previous_status,
        previous_sample_at=previous_sample_at,
        current_sample_at=time.monotonic(),
    )
    observation_sec = max(float(detection_stats["observation_sec"]), 0.0)

    def _ratio(part_key: str) -> float:
        if observation_sec <= 0:
            return 0.0
        return round(float(detection_stats[part_key]) / observation_sec, 4)

    return {
        "observation_sec": round(observation_sec, 3),
        "samples": int(detection_stats["samples"]),
        "zone": {
            "detected_sec": round(float(detection_stats["zone_detected_sec"]), 3),
            "missing_sec": round(float(detection_stats["zone_missing_sec"]), 3),
            "detected_ratio": _ratio("zone_detected_sec"),
            "missing_ratio": _ratio("zone_missing_sec"),
            "lost_events": int(detection_stats["zone_lost_events"]),
            "recovered_events": int(detection_stats["zone_recovered_events"]),
            "interrupted_then_recovered": bool(
                int(detection_stats["zone_lost_events"]) > 0 and int(detection_stats["zone_recovered_events"]) > 0
            ),
        },
        "indicator": {
            "detected_sec": round(float(detection_stats["indicator_detected_sec"]), 3),
            "missing_sec": round(float(detection_stats["indicator_missing_sec"]), 3),
            "detected_ratio": _ratio("indicator_detected_sec"),
            "missing_ratio": _ratio("indicator_missing_sec"),
            "missing_with_raw_indicator_sec": round(
                float(detection_stats["indicator_missing_with_raw_indicator_sec"]),
                3,
            ),
            "missing_without_raw_indicator_sec": round(
                float(detection_stats["indicator_missing_without_raw_indicator_sec"]),
                3,
            ),
            "lost_events": int(detection_stats["indicator_lost_events"]),
            "recovered_events": int(detection_stats["indicator_recovered_events"]),
            "interrupted_then_recovered": bool(
                int(detection_stats["indicator_lost_events"]) > 0
                and int(detection_stats["indicator_recovered_events"]) > 0
            ),
        },
        "indicator_raw": {
            "detected_sec": round(float(detection_stats["indicator_raw_detected_sec"]), 3),
            "missing_sec": round(float(detection_stats["indicator_raw_missing_sec"]), 3),
            "detected_ratio": _ratio("indicator_raw_detected_sec"),
            "missing_ratio": _ratio("indicator_raw_missing_sec"),
            "lost_events": int(detection_stats["indicator_raw_lost_events"]),
            "recovered_events": int(detection_stats["indicator_raw_recovered_events"]),
            "interrupted_then_recovered": bool(
                int(detection_stats["indicator_raw_lost_events"]) > 0
                and int(detection_stats["indicator_raw_recovered_events"]) > 0
            ),
        },
        "zone_missing_breakdown": {
            "with_raw_indicator_sec": round(
                float(detection_stats["zone_missing_with_raw_indicator_sec"]),
                3,
            ),
            "without_raw_indicator_sec": round(
                float(detection_stats["zone_missing_without_raw_indicator_sec"]),
                3,
            ),
        },
        "reason_sec": {
            str(reason): round(float(duration_sec), 3)
            for reason, duration_sec in sorted((detection_stats.get("reason_sec") or {}).items())
        },
    }


def _sleep_for_target_period(next_tick_at: float | None, poll_sec: float) -> float:
    now = time.monotonic()
    if poll_sec <= 0:
        return now
    if next_tick_at is None:
        return now + poll_sec

    sleep_sec = float(next_tick_at - now)
    if sleep_sec > 0:
        time.sleep(sleep_sec)
        now = time.monotonic()

    updated_tick = float(next_tick_at)
    while updated_tick <= now:
        updated_tick += poll_sec
    return updated_tick


def _append_trace(
    trace: list[dict[str, Any]],
    *,
    start_time: float,
    state: dict[str, Any] | None = None,
    note: str | None = None,
) -> None:
    phase = str((state or {}).get("phase") or "n/a")
    entry: dict[str, Any] = {
        "t_ms": int((time.monotonic() - start_time) * 1000),
        "phase": phase,
    }
    if note:
        entry["note"] = note
    if state is not None:
        bite = state.get("bite") or {}
        if bite.get("matched") is not None:
            entry["bite_matched"] = bool(bite.get("matched"))
        if bite.get("reason"):
            entry["bite_reason"] = bite.get("reason")
        if bite.get("pixel_count") is not None:
            entry["bite_pixels"] = bite.get("pixel_count")
        if bite.get("largest_component_area") is not None:
            entry["bite_component_area"] = bite.get("largest_component_area")
        if state.get("control_advice"):
            entry["control_advice"] = state.get("control_advice")
        duel = state.get("duel") or {}
        if duel.get("reason"):
            entry["duel_reason"] = duel.get("reason")
        if duel.get("zone_detected") is not None:
            entry["zone_detected"] = bool(duel.get("zone_detected"))
        if duel.get("indicator_detected") is not None:
            entry["indicator_detected"] = bool(duel.get("indicator_detected"))
        if duel.get("indicator_raw_detected") is not None:
            entry["indicator_raw_detected"] = bool(duel.get("indicator_raw_detected"))
        if duel.get("indicator_source"):
            entry["indicator_source"] = duel.get("indicator_source")
        if duel.get("error_px") is not None:
            entry["error_px"] = duel.get("error_px")
        if duel.get("zone_left") is not None:
            entry["zone_left"] = duel.get("zone_left")
        if duel.get("zone_center") is not None:
            entry["zone_center"] = duel.get("zone_center")
        if duel.get("zone_right") is not None:
            entry["zone_right"] = duel.get("zone_right")
        if duel.get("indicator_x") is not None:
            entry["indicator_x"] = duel.get("indicator_x")
    if trace and trace[-1].get("phase") == entry["phase"] and trace[-1].get("note") == entry.get("note"):
        return
    trace.append(entry)


def _hook_success_zone_width(state: dict[str, Any]) -> int | None:
    duel = state.get("duel") or {}
    zone_left = duel.get("zone_left")
    zone_right = duel.get("zone_right")
    if zone_left is None or zone_right is None:
        return None
    return int(zone_right) - int(zone_left)


def _is_hook_success_state(state: dict[str, Any], profile: dict[str, Any]) -> bool:
    duel = state.get("duel") or {}
    if not bool(duel.get("found")):
        return False
    if duel.get("indicator_x") is None:
        return False
    width = _hook_success_zone_width(state)
    if width is None:
        return False
    min_width = int(profile.get("hook_success_min_zone_width") or 1)
    max_width = int(profile.get("hook_success_max_zone_width") or min_width)
    lower_bound = min(min_width, max_width)
    upper_bound = max(min_width, max_width)
    return lower_bound <= int(width) <= upper_bound


def _wait_for_hook_success(
    app: Any,
    vision: Any,
    input_mapping: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    profile: dict[str, Any],
    profile_name: str,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    poll_sec: float,
) -> dict[str, Any]:
    hook_started_at = time.monotonic()
    hook_timeout_sec = float(profile["hook_timeout_sec"])
    hook_press_interval_sec = max(float(profile["hook_press_interval_ms"]) / 1000.0, 0.01)
    hook_poll_sec = min(max(float(poll_sec), 0.0), hook_press_interval_sec)
    confirm_frames = max(int(profile["hook_success_confirm_frames"]), 1)

    next_hook_press_at: float | None = None
    next_hook_poll_at: float | None = None
    success_streak = 0
    last_state: dict[str, Any] | None = None

    while True:
        now = time.monotonic()
        if now - hook_started_at > hook_timeout_sec:
            return {
                "ok": False,
                "failure_reason": "hook_timeout",
                "elapsed_sec": round(now - hook_started_at, 3),
                "state": last_state,
            }

        if next_hook_press_at is None or now >= next_hook_press_at:
            _press_input_action(input_mapping, app, action_name="fish_interact", profile_name=profile_name)
            _append_trace(phase_trace, start_time=start_time, state=last_state, note="hook_spam")
            logger.info("Fishing[input] action=fish_interact phase=press note=hook_spam")
            next_hook_press_at = now + hook_press_interval_sec

        state = _read_state_snapshot(
            app,
            ocr=None,
            vision=vision,
            yihuan_fishing=yihuan_fishing,
            profile_name=profile_name,
            enable_ocr=False,
        )
        last_state = state
        _append_trace(phase_trace, start_time=start_time, state=state)
        _log_state_details("hook_wait", state)

        if _is_hook_success_state(state, profile):
            success_streak += 1
            width = _hook_success_zone_width(state)
            logger.info(
                "Fishing[hook_wait] success_candidate streak=%s width=%s zone_left=%s zone_right=%s indicator_x=%s",
                success_streak,
                width,
                (state.get("duel") or {}).get("zone_left"),
                (state.get("duel") or {}).get("zone_right"),
                (state.get("duel") or {}).get("indicator_x"),
            )
            if success_streak >= confirm_frames:
                _append_trace(phase_trace, start_time=start_time, state=state, note="hook_success")
                logger.info(
                    "Fishing[hook_wait] hook_success_confirmed width=%s streak=%s",
                    width,
                    success_streak,
                )
                return {
                    "ok": True,
                    "failure_reason": None,
                    "elapsed_sec": round(time.monotonic() - hook_started_at, 3),
                    "state": state,
                }
        else:
            success_streak = 0

        next_hook_poll_at = _sleep_for_target_period(next_hook_poll_at, hook_poll_sec)


def _has_ready_template(state: dict[str, Any] | None) -> bool:
    if not state:
        return False
    ready_anchor = (state.get("ready_anchor") or {}) if isinstance(state, dict) else {}
    if ready_anchor:
        return bool(ready_anchor.get("found"))
    return str(state.get("phase") or "unknown") == "ready"


def _run_post_duel_cleanup(
    app: Any,
    ocr: Any,
    vision: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    profile_name: str | None,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    cleanup_timeout_sec: float | None = None,
) -> dict[str, Any]:
    profile = yihuan_fishing.load_profile(profile_name)
    close_x, close_y = profile["result_close_point"]
    click_interval_sec = max(float(profile["post_duel_click_interval_ms"]) / 1000.0, 0.01)
    deadline = (
        time.monotonic() + max(float(cleanup_timeout_sec), 0.1)
        if cleanup_timeout_sec is not None and cleanup_timeout_sec > 0
        else None
    )
    next_click_at: float | None = None
    click_count = 0
    last_state: dict[str, Any] | None = None

    while True:
        now = time.monotonic()
        if deadline is not None and now > deadline:
            break
        if next_click_at is None or now >= next_click_at:
            app.click(x=close_x, y=close_y, button="left", clicks=1, interval=0.0)
            click_count += 1
            _append_trace(phase_trace, start_time=start_time, note="post_duel_click")
            logger.info(
                "Fishing[input] action=click button=left point=(%s,%s) note=post_duel_cleanup click_index=%s",
                close_x,
                close_y,
                click_count,
            )
            next_click_at = now + click_interval_sec

        state = _read_state_snapshot(
            app,
            ocr,
            vision,
            yihuan_fishing,
            profile_name=profile["profile_name"],
            enable_ocr=False,
        )
        last_state = state
        if _has_ready_template(state):
            _append_trace(phase_trace, start_time=start_time, state=state, note="post_duel_cleanup_done")
            return {
                "ok": True,
                "click_count": click_count,
                "final_state": state,
            }

        if next_click_at is None:
            continue
        sleep_sec = min(0.1, max(next_click_at - time.monotonic(), 0.0))
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if last_state is None:
        last_state = _read_state_snapshot(
            app,
            ocr,
            vision,
            yihuan_fishing,
            profile_name=profile["profile_name"],
            enable_ocr=False,
        )
    if _has_ready_template(last_state):
        _append_trace(phase_trace, start_time=start_time, state=last_state, note="post_duel_cleanup_done")
        return {
            "ok": True,
            "click_count": click_count,
            "final_state": last_state,
        }
    return {
        "ok": False,
        "click_count": click_count,
        "final_state": last_state,
    }



def _run_fishing_round_impl(
    app: Any,
    ocr: Any,
    vision: Any,
    input_mapping: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    round_index: int,
    profile_name: str | None,
    bite_timeout_sec: float | None,
    duel_timeout_sec: float | None,
    show_debug_window: bool = False,
) -> dict[str, Any]:
    profile = yihuan_fishing.load_profile(profile_name)
    resolved_profile = profile["profile_name"]
    poll_sec = float(profile["poll_ms"]) / 1000.0
    duel_timeout = float(duel_timeout_sec) if duel_timeout_sec and duel_timeout_sec > 0 else float(profile["duel_timeout_sec"])

    start_time = time.monotonic()
    phase_trace: list[dict[str, Any]] = []
    timings = {
        "ready_wait_sec": 0.0,
        "bite_wait_sec": 0.0,
        "hook_wait_sec": 0.0,
        "duel_sec": 0.0,
        "total_sec": 0.0,
    }
    failure_reason: str | None = None
    duel_started_at: float | None = None
    duel_debug_artifact: dict[str, Any] | None = None
    held_direction_action: str | None = None
    debug_window_active = False
    detection_stats: dict[str, Any] | None = None
    last_detection_status: dict[str, Any] | None = None
    last_detection_sample_at: float | None = None
    control_memory: dict[str, Any] = {}

    try:
        state = _read_state_snapshot(app, ocr, vision, yihuan_fishing, profile_name=resolved_profile)
        _append_trace(phase_trace, start_time=start_time, state=state, note="initial")
        _log_state_details("initial", state)

        if state["phase"] == "result":
            cleanup_result = _run_post_duel_cleanup(
                app,
                ocr,
                vision,
                yihuan_fishing,
                profile_name=resolved_profile,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            if not cleanup_result["ok"]:
                failure_reason = "result_close_timeout"
                return _round_result("failed", round_index, phase_trace, failure_reason, timings, start_time)
            state = dict(cleanup_result["final_state"] or {})

        ready_wait_started = time.monotonic()
        ready_deadline = ready_wait_started + 5.0
        next_ready_poll_at: float | None = None
        while state["phase"] != "ready":
            if time.monotonic() > ready_deadline:
                failure_reason = "not_ready"
                logger.info("Fishing[ready_wait] timed_out phase=%s", state.get("phase"))
                return _round_result("failed", round_index, phase_trace, failure_reason, timings, start_time)
            next_ready_poll_at = _sleep_for_target_period(next_ready_poll_at, poll_sec)
            state = _read_state_snapshot(app, ocr, vision, yihuan_fishing, profile_name=resolved_profile)
            _append_trace(phase_trace, start_time=start_time, state=state)
            _log_state_details("ready_wait", state)
        timings["ready_wait_sec"] = round(time.monotonic() - ready_wait_started, 3)

        _press_input_action(input_mapping, app, action_name="fish_interact", profile_name=resolved_profile)
        _append_trace(phase_trace, start_time=start_time, state=state, note="cast")
        logger.info("Fishing[input] action=fish_interact phase=press note=cast")
        logger.info("Fishing[hook_wait] starting_immediately_after_cast")

        hook_result = _wait_for_hook_success(
            app,
            vision,
            input_mapping,
            yihuan_fishing,
            profile=profile,
            profile_name=resolved_profile,
            phase_trace=phase_trace,
            start_time=start_time,
            poll_sec=poll_sec,
        )
        timings["hook_wait_sec"] = float(hook_result["elapsed_sec"])
        state = dict(hook_result.get("state") or state)
        if not bool(hook_result["ok"]):
            failure_reason = str(hook_result["failure_reason"] or "hook_timeout")
            logger.info("Fishing[hook_wait] failed reason=%s", failure_reason)
            return _round_result("failed", round_index, phase_trace, failure_reason, timings, start_time, duel_debug_artifact)

        duel_deadline = time.monotonic() + duel_timeout
        duel_started = bool((state.get("duel") or {}).get("found"))
        if duel_started:
            duel_started_at = time.monotonic()
        duel_missing_started_at: float | None = None
        duel_end_missing_sec = max(float(profile["duel_end_missing_sec"]), 0.0)
        duel_terminal_phase = str(state.get("phase") or "unknown")
        detection_stats = _new_detection_stats()
        next_duel_poll_at: float | None = None
        while True:
            if time.monotonic() > duel_deadline:
                held_direction_action, _ = _set_held_direction(
                    input_mapping,
                    app,
                    current_action_name=held_direction_action,
                    desired_action_name=None,
                    profile_name=resolved_profile,
                )
                failure_reason = "duel_timeout"
                if duel_started_at is not None:
                    timings["duel_sec"] = round(time.monotonic() - duel_started_at, 3)
                return _round_result(
                    "failed",
                    round_index,
                    phase_trace,
                    failure_reason,
                    timings,
                    start_time,
                    duel_debug_artifact,
                    _finalize_detection_stats(
                        detection_stats,
                        previous_status=last_detection_status,
                        previous_sample_at=last_detection_sample_at,
                    ),
                )

            state = _read_duel_snapshot_fast(
                app,
                yihuan_fishing,
                profile_name=resolved_profile,
                include_capture_image=show_debug_window or duel_debug_artifact is None,
            )
            capture_image = state.pop("_capture_image", None)
            if capture_image is not None and (
                state["phase"] == "duel"
                or (duel_debug_artifact is None and state["phase"] == "unknown")
            ):
                duel_debug_artifact = yihuan_fishing.save_duel_debug_artifact(
                    capture_image,
                    profile_name=resolved_profile,
                    tag=f"round_{round_index}",
                )
            current_sample_at = time.monotonic()
            current_detection_status = _extract_detection_status(state)
            _accumulate_detection_stats(
                detection_stats,
                previous_status=last_detection_status,
                previous_sample_at=last_detection_sample_at,
                current_sample_at=current_sample_at,
                current_status=current_detection_status,
            )
            if detection_stats is not None and current_detection_status["active"]:
                detection_stats["samples"] += 1
            last_detection_status = current_detection_status
            last_detection_sample_at = current_sample_at
            _append_trace(phase_trace, start_time=start_time, state=state)
            _log_state_details("duel_loop", state)
            phase = state["phase"]

            duel_found = bool((state.get("duel") or {}).get("found"))
            if duel_found:
                duel_missing_started_at = None
                if not duel_started:
                    duel_started = True
                    duel_started_at = time.monotonic()
                control_memory["held_action_name"] = held_direction_action
                control_command = _compute_duel_control_command(
                    state,
                    profile,
                    control_memory,
                    current_sample_at,
                )
                desired_hold_action = control_command.get("hold_action_name")
                tap_action_name = control_command.get("tap_action_name")
                previous_action_name = held_direction_action
                held_direction_action, hold_changed = _set_held_direction(
                    input_mapping,
                    app,
                    current_action_name=held_direction_action,
                    desired_action_name=str(desired_hold_action) if desired_hold_action is not None else None,
                    profile_name=resolved_profile,
                )
                if hold_changed:
                    _append_trace(
                        phase_trace,
                        start_time=start_time,
                        state=state,
                        note=f"control_{control_command.get('reason')}",
                    )
                    if held_direction_action is not None:
                        logger.info(
                            "Fishing[input] action=%s phase=hold reason=%s",
                            held_direction_action,
                            control_command.get("reason"),
                        )
                    else:
                        logger.info(
                            "Fishing[input] action=%s phase=release reason=%s",
                            previous_action_name,
                            control_command.get("reason"),
                        )

                if tap_action_name is not None and held_direction_action is None:
                    input_mapping.execute_action(
                        str(tap_action_name),
                        phase="tap",
                        app=app,
                        profile=resolved_profile,
                    )
                    _append_trace(
                        phase_trace,
                        start_time=start_time,
                        state=state,
                        note=f"control_{control_command.get('reason')}",
                    )
                    logger.info(
                        "Fishing[input] action=%s phase=tap reason=%s",
                        tap_action_name,
                        control_command.get("reason"),
                    )

                logger.info(
                    "Fishing[control] action=%s phase=%s reason=%s trusted=%s suspicious=%s suspicious_reason=%s keep_hold=%s keep_hold_age_ms=%s zone=[%s,%s] width=%s zone_jump=%s width_jump=%s center=%s indicator=%s indicator_jump=%s center_error=%s boundary_error=%s predicted_error=%.1f velocity=%.1f",
                    control_command.get("action_name") or "none",
                    control_command.get("control_phase"),
                    control_command.get("reason"),
                    control_command.get("detection_trusted"),
                    control_command.get("suspicious"),
                    control_command.get("suspicious_reason"),
                    control_command.get("keep_hold"),
                    control_command.get("keep_hold_age_ms"),
                    control_command.get("zone_left"),
                    control_command.get("zone_right"),
                    control_command.get("zone_width"),
                    control_command.get("zone_jump_px"),
                    control_command.get("width_jump_px"),
                    control_command.get("zone_center"),
                    control_command.get("indicator_x"),
                    control_command.get("indicator_jump_px"),
                    control_command.get("center_error_px"),
                    control_command.get("boundary_error_px"),
                    float(control_command.get("predicted_error_px") or 0.0),
                    float(control_command.get("velocity_px_per_sec") or 0.0),
                )
            else:
                control_memory["held_action_name"] = held_direction_action
                control_command = _compute_duel_control_command(
                    state,
                    profile,
                    control_memory,
                    current_sample_at,
                )
                desired_hold_action = control_command.get("hold_action_name")
                duel_missing_elapsed_sec = 0.0
                if duel_started:
                    if duel_missing_started_at is None:
                        duel_missing_started_at = current_sample_at
                    duel_missing_elapsed_sec = max(current_sample_at - duel_missing_started_at, 0.0)
                    duel_terminal_phase = str(phase or "unknown")
                    logger.info(
                        "Fishing[duel_end] missing_sec=%.3f/%.3f phase=%s duel_reason=%s",
                        duel_missing_elapsed_sec,
                        duel_end_missing_sec,
                        duel_terminal_phase,
                        (state.get("duel") or {}).get("reason"),
                    )
                if (
                    bool(control_command.get("keep_hold"))
                    and desired_hold_action == held_direction_action
                    and duel_missing_elapsed_sec < duel_end_missing_sec
                ):
                    logger.info(
                        "Fishing[control] action=%s phase=%s reason=%s trusted=%s suspicious=%s suspicious_reason=%s keep_hold=%s keep_hold_age_ms=%s zone=[%s,%s] width=%s zone_jump=%s width_jump=%s center=%s indicator=%s indicator_jump=%s center_error=%s boundary_error=%s predicted_error=%.1f velocity=%.1f",
                        control_command.get("action_name") or "none",
                        control_command.get("control_phase"),
                        control_command.get("reason"),
                        control_command.get("detection_trusted"),
                        control_command.get("suspicious"),
                        control_command.get("suspicious_reason"),
                        control_command.get("keep_hold"),
                        control_command.get("keep_hold_age_ms"),
                        control_command.get("zone_left"),
                        control_command.get("zone_right"),
                        control_command.get("zone_width"),
                        control_command.get("zone_jump_px"),
                        control_command.get("width_jump_px"),
                        control_command.get("zone_center"),
                        control_command.get("indicator_x"),
                        control_command.get("indicator_jump_px"),
                        control_command.get("center_error_px"),
                        control_command.get("boundary_error_px"),
                        float(control_command.get("predicted_error_px") or 0.0),
                        float(control_command.get("velocity_px_per_sec") or 0.0),
                    )
                    if show_debug_window and capture_image is not None:
                        debug_window_active = _update_debug_window(
                            yihuan_fishing,
                            source_image=capture_image,
                            state=state,
                            profile_name=resolved_profile,
                            held_action_name=held_direction_action,
                        ) or debug_window_active
                    next_duel_poll_at = _sleep_for_target_period(next_duel_poll_at, poll_sec)
                    continue

                released_action_name = held_direction_action
                held_direction_action, released = _set_held_direction(
                    input_mapping,
                    app,
                    current_action_name=held_direction_action,
                    desired_action_name=None,
                    profile_name=resolved_profile,
                )
                if released:
                    _append_trace(phase_trace, start_time=start_time, state=state, note="release_hold")
                    logger.info("Fishing[input] action=%s phase=release note=release_hold", released_action_name)
                logger.info(
                    "Fishing[control] action=%s phase=%s reason=%s trusted=%s suspicious=%s suspicious_reason=%s keep_hold=%s keep_hold_age_ms=%s zone=[%s,%s] width=%s zone_jump=%s width_jump=%s center=%s indicator=%s indicator_jump=%s center_error=%s boundary_error=%s predicted_error=%.1f velocity=%.1f",
                    control_command.get("action_name") or "none",
                    control_command.get("control_phase"),
                    control_command.get("reason"),
                    control_command.get("detection_trusted"),
                    control_command.get("suspicious"),
                    control_command.get("suspicious_reason"),
                    control_command.get("keep_hold"),
                    control_command.get("keep_hold_age_ms"),
                    control_command.get("zone_left"),
                    control_command.get("zone_right"),
                    control_command.get("zone_width"),
                    control_command.get("zone_jump_px"),
                    control_command.get("width_jump_px"),
                    control_command.get("zone_center"),
                    control_command.get("indicator_x"),
                    control_command.get("indicator_jump_px"),
                    control_command.get("center_error_px"),
                    control_command.get("boundary_error_px"),
                    float(control_command.get("predicted_error_px") or 0.0),
                    float(control_command.get("velocity_px_per_sec") or 0.0),
                )
                if duel_started:
                    if duel_missing_elapsed_sec >= duel_end_missing_sec:
                        if duel_started_at is not None:
                            timings["duel_sec"] = round(time.monotonic() - duel_started_at, 3)
                        cleanup_result = _run_post_duel_cleanup(
                            app,
                            ocr,
                            vision,
                            yihuan_fishing,
                            profile_name=resolved_profile,
                            phase_trace=phase_trace,
                            start_time=start_time,
                        )
                        final_state = dict(cleanup_result["final_state"] or {})
                        logger.info(
                            "Fishing[duel_end] cleanup_ok=%s click_count=%s terminal_phase=%s final_phase=%s",
                            cleanup_result["ok"],
                            cleanup_result["click_count"],
                            duel_terminal_phase,
                            final_state.get("phase"),
                        )
                        if not cleanup_result["ok"]:
                            failure_reason = "post_duel_cleanup_timeout"
                            return _round_result(
                                "failed",
                                round_index,
                                phase_trace,
                                failure_reason,
                                timings,
                                start_time,
                                duel_debug_artifact,
                                _finalize_detection_stats(
                                    detection_stats,
                                    previous_status=last_detection_status,
                                    previous_sample_at=last_detection_sample_at,
                                ),
                            )
                        return _round_result(
                            "success",
                            round_index,
                            phase_trace,
                            None,
                            timings,
                            start_time,
                            duel_debug_artifact,
                            _finalize_detection_stats(
                                detection_stats,
                                previous_status=last_detection_status,
                                previous_sample_at=last_detection_sample_at,
                            ),
                        )

            if show_debug_window and capture_image is not None:
                debug_window_active = _update_debug_window(
                    yihuan_fishing,
                    source_image=capture_image,
                    state=state,
                    profile_name=resolved_profile,
                    held_action_name=held_direction_action,
                ) or debug_window_active

            next_duel_poll_at = _sleep_for_target_period(next_duel_poll_at, poll_sec)
    finally:
        if debug_window_active or show_debug_window:
            _close_debug_window()
        app.release_all()


def _round_result(
    status: str,
    round_index: int,
    phase_trace: list[dict[str, Any]],
    failure_reason: str | None,
    timings: dict[str, Any],
    start_time: float,
    duel_debug_artifact: dict[str, Any] | None = None,
    detection_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timings["total_sec"] = round(time.monotonic() - start_time, 3)
    return {
        "status": status,
        "round_index": int(round_index),
        "phase_trace": phase_trace,
        "failure_reason": failure_reason,
        "timings": timings,
        "duel_debug_artifact": duel_debug_artifact,
        "detection_stats": detection_stats,
    }


@action_info(
    name="yihuan_fishing_debug_monitor",
    public=True,
    read_only=False,
    description="Auto cast and hook once, then show a live fishing debug window for a fixed duration.",
)
@requires_services(
    app="plans/aura_base/app",
    ocr="plans/aura_base/ocr",
    vision="plans/aura_base/vision",
    input_mapping="plans/aura_base/input_mapping",
    yihuan_fishing="yihuan_fishing",
)
def yihuan_fishing_debug_monitor(
    app: Any,
    ocr: Any,
    vision: Any,
    input_mapping: Any,
    yihuan_fishing: YihuanFishingService,
    profile_name: str = "default_1280x720_cn",
    duration_sec: float = 60.0,
    poll_ms: int = 80,
    bite_timeout_sec: float = 0.0,
    screenshot_interval_sec: float = 1.0,
) -> dict[str, Any]:
    profile = yihuan_fishing.load_profile(profile_name)
    resolved_profile = profile["profile_name"]
    poll_sec = max(float(poll_ms) / 1000.0, 0.0)
    duration_sec = max(float(duration_sec), 0.1)
    screenshot_interval_sec = max(float(screenshot_interval_sec), 0.1)

    start_time = time.monotonic()
    phase_trace: list[dict[str, Any]] = []
    detection_stats: dict[str, Any] | None = None
    last_detection_status: dict[str, Any] | None = None
    last_detection_sample_at: float | None = None
    last_state: dict[str, Any] | None = None
    debug_window_active = False
    screenshot_paths: list[str] = []
    monitor_output_dir: Path | None = None
    monitor_started_at: float | None = None
    next_screenshot_at: float | None = None
    screenshot_index = 0
    failure_reason: str | None = None

    try:
        state = _read_state_snapshot(
            app,
            ocr,
            vision,
            yihuan_fishing,
            profile_name=resolved_profile,
        )
        _append_trace(phase_trace, start_time=start_time, state=state, note="initial")

        if state["phase"] == "result":
            cleanup_result = _run_post_duel_cleanup(
                app,
                ocr,
                vision,
                yihuan_fishing,
                profile_name=resolved_profile,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            if not cleanup_result["ok"]:
                failure_reason = "result_close_timeout"
                return {
                    "status": "failed",
                    "profile_name": resolved_profile,
                    "failure_reason": failure_reason,
                    "phase_trace": phase_trace,
                    "last_state": state,
                    "duration_sec": round(time.monotonic() - start_time, 3),
                    "detection_stats": None,
                    "screenshot_dir": None,
                    "screenshots": [],
                }
            state = dict(cleanup_result["final_state"] or {})

        ready_wait_started = time.monotonic()
        ready_deadline = ready_wait_started + 5.0
        next_ready_poll_at: float | None = None
        while state["phase"] != "ready":
            if time.monotonic() > ready_deadline:
                failure_reason = "not_ready"
                return {
                    "status": "failed",
                    "profile_name": resolved_profile,
                    "failure_reason": failure_reason,
                    "phase_trace": phase_trace,
                    "last_state": state,
                    "duration_sec": round(time.monotonic() - start_time, 3),
                    "detection_stats": None,
                    "screenshot_dir": None,
                    "screenshots": [],
                }
            next_ready_poll_at = _sleep_for_target_period(next_ready_poll_at, poll_sec)
            state = _read_state_snapshot(
                app,
                ocr,
                vision,
                yihuan_fishing,
                profile_name=resolved_profile,
            )
            _append_trace(phase_trace, start_time=start_time, state=state)

        _press_input_action(input_mapping, app, action_name="fish_interact", profile_name=resolved_profile)
        _append_trace(phase_trace, start_time=start_time, state=state, note="cast")
        logger.info("Fishing[hook_wait] starting_immediately_after_cast")

        hook_result = _wait_for_hook_success(
            app,
            vision,
            input_mapping,
            yihuan_fishing,
            profile=profile,
            profile_name=resolved_profile,
            phase_trace=phase_trace,
            start_time=start_time,
            poll_sec=poll_sec,
        )
        state = dict(hook_result.get("state") or state)
        if not bool(hook_result["ok"]):
            failure_reason = str(hook_result["failure_reason"] or "hook_timeout")
            return {
                "status": "failed",
                "profile_name": resolved_profile,
                "failure_reason": failure_reason,
                "phase_trace": phase_trace,
                "last_state": state,
                "duration_sec": round(time.monotonic() - start_time, 3),
                "detection_stats": None,
                "screenshot_dir": None,
                "screenshots": [],
            }

        detection_stats = _new_detection_stats()
        monitor_output_dir = _create_monitor_output_dir()
        monitor_started_at = time.monotonic()
        next_screenshot_at = monitor_started_at
        monitor_deadline = monitor_started_at + duration_sec
        next_monitor_poll_at: float | None = None

        while True:
            state = _read_state_snapshot(
                app,
                ocr,
                vision,
                yihuan_fishing,
                profile_name=resolved_profile,
                include_capture_image=True,
                enable_ocr=False,
            )
            capture_image = state.pop("_capture_image", None)
            current_sample_at = time.monotonic()
            current_detection_status = _extract_detection_status(state)
            _accumulate_detection_stats(
                detection_stats,
                previous_status=last_detection_status,
                previous_sample_at=last_detection_sample_at,
                current_sample_at=current_sample_at,
                current_status=current_detection_status,
            )
            if current_detection_status["active"]:
                detection_stats["samples"] += 1
            last_detection_status = current_detection_status
            last_detection_sample_at = current_sample_at
            _append_trace(phase_trace, start_time=start_time, state=state)
            last_state = state

            held_action_name = None
            advice = str(state.get("control_advice") or "none")
            if advice == "hold_a":
                held_action_name = "fish_left"
            elif advice == "hold_d":
                held_action_name = "fish_right"
            if capture_image is not None:
                debug_window_active = _update_debug_window(
                    yihuan_fishing,
                    source_image=capture_image,
                    state=state,
                    profile_name=resolved_profile,
                    held_action_name=held_action_name,
                ) or debug_window_active
                if next_screenshot_at is not None and current_sample_at >= next_screenshot_at and monitor_output_dir is not None and monitor_started_at is not None:
                    saved_path = _save_monitor_debug_frame(
                        yihuan_fishing,
                        source_image=capture_image,
                        state=state,
                        profile_name=resolved_profile,
                        held_action_name=held_action_name,
                        out_dir=monitor_output_dir,
                        frame_index=screenshot_index,
                        elapsed_sec=current_sample_at - monitor_started_at,
                    )
                    if saved_path is not None:
                        screenshot_paths.append(saved_path)
                        screenshot_index += 1
                    next_screenshot_at += screenshot_interval_sec

            if current_sample_at >= monitor_deadline:
                break
            next_monitor_poll_at = _sleep_for_target_period(next_monitor_poll_at, poll_sec)
    finally:
        if debug_window_active:
            _close_debug_window()

    return {
        "status": "success",
        "profile_name": resolved_profile,
        "duration_sec": round(time.monotonic() - start_time, 3),
        "phase_trace": phase_trace,
        "last_state": last_state,
        "failure_reason": failure_reason,
        "detection_stats": _finalize_detection_stats(
            detection_stats,
            previous_status=last_detection_status,
            previous_sample_at=last_detection_sample_at,
        ),
        "screenshot_dir": str(monitor_output_dir) if monitor_output_dir is not None else None,
        "screenshots": screenshot_paths,
    }


@action_info(
    name="yihuan_fishing_live_monitor",
    public=True,
    read_only=False,
    description="Continuously show the fishing duel region before and after processing until the monitor window is closed.",
)
@requires_services(
    app="plans/aura_base/app",
    yihuan_fishing="yihuan_fishing",
)
def yihuan_fishing_live_monitor(
    app: Any,
    yihuan_fishing: YihuanFishingService,
    profile_name: str = "default_1280x720_cn",
    duration_sec: float = 0.0,
    poll_ms: int = 0,
) -> dict[str, Any]:
    profile = yihuan_fishing.load_profile(profile_name)
    resolved_profile = profile["profile_name"]
    poll_sec = max(float(poll_ms) / 1000.0, 0.0)
    duration_limit_sec = float(duration_sec)
    deadline = None if duration_limit_sec <= 0 else time.monotonic() + duration_limit_sec

    start_time = time.monotonic()
    phase_trace: list[dict[str, Any]] = []
    detection_stats = _new_detection_stats()
    last_detection_status: dict[str, Any] | None = None
    last_detection_sample_at: float | None = None
    last_state: dict[str, Any] | None = None
    window_active = False
    next_poll_at: float | None = None
    stop_reason = "window_closed"

    try:
        while True:
            capture = app.capture()
            if not capture.success or capture.image is None:
                raise RuntimeError("Failed to capture the Yihuan fishing screen.")

            duel = yihuan_fishing.analyze_duel_meter(capture.image, profile_name=resolved_profile)
            state = {
                "phase": "duel" if duel.get("found") else "unknown",
                "profile_name": resolved_profile,
                "duel": duel,
                "control_advice": duel.get("control_advice") if duel.get("found") else "none",
            }
            current_sample_at = time.monotonic()
            current_detection_status = _extract_detection_status(state)
            _accumulate_detection_stats(
                detection_stats,
                previous_status=last_detection_status,
                previous_sample_at=last_detection_sample_at,
                current_sample_at=current_sample_at,
                current_status=current_detection_status,
            )
            if current_detection_status["active"]:
                detection_stats["samples"] += 1
            last_detection_status = current_detection_status
            last_detection_sample_at = current_sample_at
            last_state = state
            _append_trace(phase_trace, start_time=start_time, state=state)

            frame = yihuan_fishing.build_duel_live_monitor_view(
                capture.image,
                state=state,
                profile_name=resolved_profile,
            )
            window_active, exit_requested = _update_live_monitor_window(frame)
            if exit_requested:
                stop_reason = "window_closed"
                break
            if deadline is not None and current_sample_at >= deadline:
                stop_reason = "duration_reached"
                break
            next_poll_at = _sleep_for_target_period(next_poll_at, poll_sec)
    finally:
        if window_active:
            _close_live_monitor_window()

    return {
        "status": "success",
        "profile_name": resolved_profile,
        "duration_sec": round(time.monotonic() - start_time, 3),
        "stop_reason": stop_reason,
        "phase_trace": phase_trace,
        "last_state": last_state,
        "detection_stats": _finalize_detection_stats(
            detection_stats,
            previous_status=last_detection_status,
            previous_sample_at=last_detection_sample_at,
        ),
    }


@action_info(
    name="yihuan_fishing_run_session",
    public=False,
    read_only=False,
    description="Run multiple fishing rounds until max_rounds is reached.",
)
@requires_services(
    app="plans/aura_base/app",
    ocr="plans/aura_base/ocr",
    vision="plans/aura_base/vision",
    input_mapping="plans/aura_base/input_mapping",
    yihuan_fishing="yihuan_fishing",
)
def yihuan_fishing_run_session(
    app: Any,
    ocr: Any,
    vision: Any,
    input_mapping: Any,
    yihuan_fishing: YihuanFishingService,
    max_rounds: int = 0,
    profile_name: str = "default_1280x720_cn",
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    consecutive_failures = 0
    success_count = 0
    failure_count = 0
    round_index = 1
    stop_reason = "max_rounds"
    unlimited_rounds = int(max_rounds) <= 0

    while unlimited_rounds or round_index <= int(max_rounds):
        round_result = _run_fishing_round_impl(
            app,
            ocr,
            vision,
            input_mapping,
            yihuan_fishing,
            round_index=round_index,
            profile_name=profile_name,
            bite_timeout_sec=None,
            duel_timeout_sec=None,
        )
        results.append(round_result)
        if round_result["status"] == "success":
            success_count += 1
            consecutive_failures = 0
        else:
            failure_count += 1
            consecutive_failures += 1
        round_index += 1

    status = "success"
    if success_count == 0 and failure_count > 0:
        status = "failed"
    elif success_count > 0 and failure_count > 0:
        status = "partial"

    return {
        "status": status,
        "round_count": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "consecutive_failures": consecutive_failures,
        "stopped_reason": stop_reason,
        "results": results,
    }


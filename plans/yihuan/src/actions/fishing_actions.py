"""Fishing actions for the Yihuan plan."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any

import cv2

from packages.aura_core.api import action_info, requires_services
from packages.aura_core.observability.logging.core_logger import logger
from packages.aura_core.scheduler.cancellation import is_current_task_cancel_requested

from ..services.fishing_service import YihuanFishingService


_DEBUG_WINDOW_NAME = "Yihuan Fishing Duel Debug"
_LIVE_MONITOR_WINDOW_NAME = "Yihuan Fishing Live Monitor"


class _BoundedResultBuffer(list[dict[str, Any]]):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = max(int(limit), 1)
        self.dropped_count = 0
        self.total_count = 0

    def append(self, item: dict[str, Any]) -> None:  # type: ignore[override]
        self.total_count += 1
        super().append(item)
        overflow = len(self) - self.limit
        if overflow > 0:
            del self[:overflow]
            self.dropped_count += overflow


def _new_bounded_buffer(limit: Any, *, default: int) -> _BoundedResultBuffer:
    return _BoundedResultBuffer(max(_non_negative_int(limit, default=default), 1))


def _bounded_buffer_meta(buffer: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int | None, int]:
    items = list(buffer)
    if isinstance(buffer, _BoundedResultBuffer):
        return items, int(buffer.limit), int(buffer.dropped_count)
    return items, None, 0


def _apply_phase_trace_payload(result: dict[str, Any], phase_trace: list[dict[str, Any]]) -> dict[str, Any]:
    trace_items, trace_limit, truncated_count = _bounded_buffer_meta(phase_trace)
    result["phase_trace"] = trace_items
    if trace_limit is not None:
        result["phase_trace_limit"] = trace_limit
    if truncated_count > 0:
        result["phase_trace_truncated_count"] = truncated_count
    return result


def _fishing_cancel_requested() -> bool:
    try:
        return is_current_task_cancel_requested()
    except Exception:
        return False


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


def _non_negative_int(value: Any, default: int = 0) -> int:
    if value is None:
        return max(int(default), 0)
    try:
        if isinstance(value, str) and not value.strip():
            return max(int(default), 0)
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return max(int(default), 0)


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
        _sleep_interruptibly(sleep_sec)
        now = time.monotonic()

    updated_tick = float(next_tick_at)
    while updated_tick <= now:
        updated_tick += poll_sec
    return updated_tick


def _sleep_is_mocked() -> bool:
    return hasattr(time.sleep, "mock_calls")


def _sleep_interruptibly(duration_sec: float, *, quantum_sec: float = 0.05) -> None:
    duration = max(float(duration_sec), 0.0)
    if _fishing_cancel_requested():
        return
    if duration <= 0:
        return
    if _sleep_is_mocked():
        time.sleep(duration)
        return

    deadline = time.monotonic() + duration
    while True:
        if _fishing_cancel_requested():
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, max(float(quantum_sec), 0.01)))


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
        if _fishing_cancel_requested():
            logger.info("Fishing[hook_wait] cancelled")
            return {
                "ok": False,
                "failure_reason": "cancelled",
                "cancelled": True,
                "elapsed_sec": round(time.monotonic() - hook_started_at, 3),
                "state": last_state,
            }

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
            include_capture_image=True,
            enable_ocr=False,
        )
        capture_image = state.pop("_capture_image", None)
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
            if capture_image is not None:
                bait_shortage = yihuan_fishing.analyze_bait_shortage(
                    capture_image,
                    vision,
                    profile_name=profile_name,
                )
                if bool(bait_shortage.get("found")):
                    logger.info(
                        "Fishing[bait] shortage_detected confidence=%.4f dark_bar=%s",
                        float(bait_shortage.get("confidence") or 0.0),
                        bool((bait_shortage.get("dark_bar") or {}).get("found")),
                    )
                    _append_trace(phase_trace, start_time=start_time, state=state, note="bait_shortage")
                    return {
                        "ok": False,
                        "failure_reason": "bait_shortage",
                        "event": "bait_shortage",
                        "state": state,
                        "bait_shortage": bait_shortage,
                        "elapsed_sec": round(time.monotonic() - hook_started_at, 3),
                    }

        next_hook_poll_at = _sleep_for_target_period(next_hook_poll_at, hook_poll_sec)


def _has_ready_template(state: dict[str, Any] | None) -> bool:
    if not state:
        return False
    ready_anchor = (state.get("ready_anchor") or {}) if isinstance(state, dict) else {}
    if ready_anchor:
        return bool(ready_anchor.get("found"))
    return str(state.get("phase") or "unknown") == "ready"


def _bait_step_interval_sec(profile: dict[str, Any]) -> float:
    return max(float(profile.get("bait_recovery_step_interval_ms") or 0) / 1000.0, 0.0)


def _profile_interval_sec(profile: dict[str, Any], key: str) -> float:
    fallback_ms = int(profile.get("bait_recovery_step_interval_ms") or 0)
    return max(float(profile.get(key, fallback_ms) or 0) / 1000.0, 0.0)


def _click_point(
    app: Any,
    *,
    x: int,
    y: int,
    button: str = "left",
    hold_ms: int = 0,
) -> None:
    resolved_hold_ms = max(int(hold_ms), 0)
    if resolved_hold_ms <= 0:
        app.click(x=x, y=y, button=button, clicks=1, interval=0.0)
        return

    app.move_to(int(x), int(y))
    app.mouse_down(button=button)
    try:
        _sleep_interruptibly(float(resolved_hold_ms) / 1000.0)
    finally:
        app.mouse_up(button=button)


def _click_config_point(
    app: Any,
    profile: dict[str, Any],
    point_key: str,
    *,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    log_scope: str,
    step: str,
    hold_ms: int = 0,
) -> tuple[int, int]:
    x, y = profile[point_key]
    _click_point(app, x=int(x), y=int(y), hold_ms=hold_ms)
    _append_trace(phase_trace, start_time=start_time, note=f"{log_scope}_{step}")
    logger.info(
        "Fishing[%s] step=%s ok=True point=(%s,%s) hold_ms=%s",
        log_scope,
        step,
        x,
        y,
        max(int(hold_ms), 0),
    )
    return int(x), int(y)


def _press_config_action(
    input_mapping: Any,
    app: Any,
    profile: dict[str, Any],
    action_key: str,
    *,
    profile_name: str,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    log_scope: str,
    step: str,
) -> str:
    action_name = str(profile.get(action_key) or action_key)
    _press_input_action(input_mapping, app, action_name=action_name, profile_name=profile_name)
    _append_trace(phase_trace, start_time=start_time, note=f"{log_scope}_{step}")
    logger.info("Fishing[%s] step=%s ok=True action=%s phase=press", log_scope, step, action_name)
    return action_name


def _capture_and_match_profile_template(
    app: Any,
    vision: Any,
    yihuan_fishing: YihuanFishingService,
    profile: dict[str, Any],
    *,
    template_key: str,
    region_key: str,
    threshold_key: str,
    profile_name: str,
) -> dict[str, Any]:
    template_name = str(profile.get(template_key) or "").strip()
    if not template_name:
        return {
            "found": False,
            "reason": "template_not_configured",
            "confidence": 0.0,
            "match_rect": None,
        }
    capture = app.capture()
    if not capture.success or capture.image is None:
        return {
            "found": False,
            "reason": "capture_failed",
            "confidence": 0.0,
            "match_rect": None,
        }
    return yihuan_fishing.match_template_with_vision(
        capture.image,
        vision,
        template_name=template_name,
        region=profile[region_key],
        threshold=float(profile[threshold_key]),
        profile_name=profile_name,
    )


def _wait_for_ready_template(
    app: Any,
    ocr: Any,
    vision: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    profile_name: str,
    timeout_sec: float,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    note: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(float(timeout_sec), 0.1)
    last_state: dict[str, Any] | None = None
    while True:
        if _fishing_cancel_requested():
            logger.info("Fishing[ready_wait] cancelled note=%s", note)
            return {"ok": False, "state": last_state, "cancelled": True}
        state = _read_state_snapshot(
            app,
            ocr,
            vision,
            yihuan_fishing,
            profile_name=profile_name,
            enable_ocr=False,
        )
        last_state = state
        if _has_ready_template(state):
            _append_trace(phase_trace, start_time=start_time, state=state, note=note)
            return {"ok": True, "state": state}
        if time.monotonic() >= deadline:
            return {"ok": False, "state": last_state}
        _sleep_interruptibly(0.1)


def _wait_for_profile_template(
    app: Any,
    vision: Any,
    yihuan_fishing: YihuanFishingService,
    profile: dict[str, Any],
    *,
    template_key: str,
    region_key: str,
    threshold_key: str,
    profile_name: str,
    timeout_sec: float,
    log_scope: str,
    step: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(float(timeout_sec), 0.1)
    last_match: dict[str, Any] | None = None
    while True:
        if _fishing_cancel_requested():
            logger.info("Fishing[%s] step=%s cancelled", log_scope, step)
            return {
                "found": False,
                "reason": "cancelled",
                "confidence": 0.0,
                "match_rect": None,
                "cancelled": True,
            }
        match = _capture_and_match_profile_template(
            app,
            vision,
            yihuan_fishing,
            profile,
            template_key=template_key,
            region_key=region_key,
            threshold_key=threshold_key,
            profile_name=profile_name,
        )
        last_match = match
        if bool(match.get("found")):
            logger.info(
                "Fishing[%s] step=%s ok=True confidence=%.4f",
                log_scope,
                step,
                float(match.get("confidence") or 0.0),
            )
            return match
        if time.monotonic() >= deadline:
            logger.info(
                "Fishing[%s] step=%s ok=False reason=timeout confidence=%.4f",
                log_scope,
                step,
                float(match.get("confidence") or 0.0),
            )
            return last_match
        _sleep_interruptibly(0.1)


def _click_repeated(
    app: Any,
    point: tuple[int, int],
    *,
    clicks: int,
    interval_ms: int,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    log_scope: str,
    step: str,
    hold_ms: int = 0,
) -> int:
    x, y = int(point[0]), int(point[1])
    total = max(int(clicks), 0)
    interval_sec = max(float(interval_ms) / 1000.0, 0.0)
    for index in range(total):
        if _fishing_cancel_requested():
            return index
        _click_point(app, x=x, y=y, hold_ms=hold_ms)
        _append_trace(phase_trace, start_time=start_time, note=f"{log_scope}_{step}")
        logger.info(
            "Fishing[%s] step=%s ok=True point=(%s,%s) click_index=%s/%s hold_ms=%s",
            log_scope,
            step,
            x,
            y,
            index + 1,
            total,
            max(int(hold_ms), 0),
        )
        if index + 1 < total and interval_sec > 0:
            _sleep_interruptibly(interval_sec)
    return total


def _click_config_point_repeated(
    app: Any,
    profile: dict[str, Any],
    point_key: str,
    *,
    clicks: int,
    interval_ms: int,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    log_scope: str,
    step: str,
    hold_ms: int = 0,
) -> tuple[int, int, int]:
    x, y = profile[point_key]
    total = max(int(clicks), 1)
    interval_sec = max(float(interval_ms) / 1000.0, 0.0)
    for index in range(total):
        if _fishing_cancel_requested():
            return int(x), int(y), index
        _click_point(app, x=int(x), y=int(y), hold_ms=hold_ms)
        _append_trace(phase_trace, start_time=start_time, note=f"{log_scope}_{step}")
        logger.info(
            "Fishing[%s] step=%s ok=True point=(%s,%s) click_index=%s/%s interval_ms=%s hold_ms=%s",
            log_scope,
            step,
            x,
            y,
            index + 1,
            total,
            int(interval_ms),
            max(int(hold_ms), 0),
        )
        if index + 1 < total and interval_sec > 0:
            _sleep_interruptibly(interval_sec)
    return int(x), int(y), total


def _cancelled_operation_result(
    *,
    started_at: float,
    steps: list[dict[str, Any]] | None = None,
    status: str = "cancelled",
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "ok": False,
        "status": status,
        "failure_reason": "cancelled",
        "cancelled": True,
        "steps": steps or [],
        "elapsed_sec": round(time.monotonic() - started_at, 3),
    }
    result.update(extra)
    return result


def _run_sell_fish_before_buy_bait(
    app: Any,
    ocr: Any,
    vision: Any,
    input_mapping: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    profile: dict[str, Any],
    profile_name: str,
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    started_at = time.monotonic()
    steps: list[dict[str, Any]] = []
    step_interval_sec = _profile_interval_sec(profile, "sell_step_interval_ms")
    click_hold_ms = int(profile.get("sell_click_hold_ms") or 0)
    result_status = "failed_not_ready"
    confirm_match: dict[str, Any] | None = None
    success_match: dict[str, Any] | None = None

    try:
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                confirm_match=confirm_match,
                success_match=success_match,
                state=None,
            )
        _press_config_action(
            input_mapping,
            app,
            profile,
            "sell_open_action",
            profile_name=profile_name,
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="sell",
            step="open_shop",
        )
        _sleep_interruptibly(step_interval_sec)
        steps.append({"step": "open_shop", "ok": True, "action": profile["sell_open_action"]})
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                confirm_match=confirm_match,
                success_match=success_match,
                state=None,
            )

        _click_config_point(
            app,
            profile,
            "sell_tab_point",
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="sell",
            step="open_sell_tab",
            hold_ms=click_hold_ms,
        )
        _sleep_interruptibly(step_interval_sec)
        steps.append({"step": "open_sell_tab", "ok": True, "point": list(profile["sell_tab_point"])})
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                confirm_match=confirm_match,
                success_match=success_match,
                state=None,
            )

        _click_config_point(
            app,
            profile,
            "sell_one_click_point",
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="sell",
            step="one_click_sell",
            hold_ms=click_hold_ms,
        )
        _sleep_interruptibly(step_interval_sec)
        steps.append({"step": "one_click_sell", "ok": True, "point": list(profile["sell_one_click_point"])})
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                confirm_match=confirm_match,
                success_match=success_match,
                state=None,
            )

        confirm_match = _wait_for_profile_template(
            app,
            vision,
            yihuan_fishing,
            profile,
            template_key="sell_confirm_template",
            region_key="sell_confirm_region",
            threshold_key="sell_confirm_match_threshold",
            profile_name=profile_name,
            timeout_sec=float(profile["sell_confirm_wait_timeout_sec"]),
            log_scope="sell",
            step="confirm_template",
        )
        steps.append(
            {
                "step": "confirm_template",
                "ok": bool(confirm_match.get("found")),
                "confidence": float(confirm_match.get("confidence") or 0.0),
            }
        )
        if bool(confirm_match.get("cancelled")):
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                confirm_match=confirm_match,
                success_match=success_match,
                state=None,
            )

        if bool(confirm_match.get("found")):
            _click_config_point(
                app,
                profile,
                "sell_confirm_point",
                phase_trace=phase_trace,
                start_time=start_time,
                log_scope="sell",
                step="confirm_sell",
                hold_ms=click_hold_ms,
            )
            _sleep_interruptibly(step_interval_sec)
            steps.append({"step": "confirm_sell", "ok": True, "point": list(profile["sell_confirm_point"])})
            result_status = "sold"
            if _fishing_cancel_requested():
                return _cancelled_operation_result(
                    started_at=started_at,
                    steps=steps,
                    confirm_match=confirm_match,
                    success_match=success_match,
                    state=None,
                )

            success_match = _wait_for_profile_template(
                app,
                vision,
                yihuan_fishing,
                profile,
                template_key="sell_success_template",
                region_key="sell_success_region",
                threshold_key="sell_success_match_threshold",
                profile_name=profile_name,
                timeout_sec=float(profile["sell_success_wait_timeout_sec"]),
                log_scope="sell",
                step="success_template",
            )
            steps.append(
                {
                    "step": "success_template",
                    "ok": bool(success_match.get("found")),
                    "confidence": float(success_match.get("confidence") or 0.0),
                }
            )
            if bool(success_match.get("cancelled")):
                return _cancelled_operation_result(
                    started_at=started_at,
                    steps=steps,
                    confirm_match=confirm_match,
                    success_match=success_match,
                    state=None,
                )
            close_count = _click_repeated(
                app,
                profile["sell_success_close_point"],
                clicks=int(profile["sell_success_close_clicks"]),
                interval_ms=int(profile["sell_success_close_interval_ms"]),
                phase_trace=phase_trace,
                start_time=start_time,
                log_scope="sell",
                step="close_success",
                hold_ms=click_hold_ms,
            )
            steps.append({"step": "close_success", "ok": True, "clicks": close_count})
            _sleep_interruptibly(step_interval_sec)
        else:
            result_status = "no_fish_or_no_confirm"

        _press_config_action(
            input_mapping,
            app,
            profile,
            "menu_back",
            profile_name=profile_name,
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="sell",
            step="return_ready",
        )
        _sleep_interruptibly(step_interval_sec)
        ready = _wait_for_ready_template(
            app,
            ocr,
            vision,
            yihuan_fishing,
            profile_name=profile_name,
            timeout_sec=float(profile["bait_recovery_ready_timeout_sec"]),
            phase_trace=phase_trace,
            start_time=start_time,
            note="sell_return_ready",
        )
        steps.append({"step": "return_ready", "ok": bool(ready["ok"])})
        if bool(ready.get("cancelled")):
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                confirm_match=confirm_match,
                success_match=success_match,
                state=ready.get("state"),
            )
        if not bool(ready["ok"]):
            result_status = "failed_not_ready"
        logger.info("Fishing[sell] done status=%s ready=%s", result_status, bool(ready["ok"]))
        return {
            "ok": result_status != "failed_not_ready",
            "status": result_status,
            "steps": steps,
            "confirm_match": confirm_match,
            "success_match": success_match,
            "state": ready.get("state"),
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }
    except Exception as exc:  # pragma: no cover - defensive runtime recovery path
        logger.warning("Fishing[sell] failed exception=%s", exc)
        steps.append({"step": "exception", "ok": False, "reason": str(exc)})
        try:
            _press_config_action(
                input_mapping,
                app,
                profile,
                "menu_back",
                profile_name=profile_name,
                phase_trace=phase_trace,
                start_time=start_time,
                log_scope="sell",
                step="return_ready_after_error",
            )
        except Exception:
            pass
        ready = _wait_for_ready_template(
            app,
            ocr,
            vision,
            yihuan_fishing,
            profile_name=profile_name,
            timeout_sec=float(profile["bait_recovery_ready_timeout_sec"]),
            phase_trace=phase_trace,
            start_time=start_time,
            note="sell_error_return_ready",
        )
        status = "failed_but_ready" if bool(ready["ok"]) else "failed_not_ready"
        return {
            "ok": bool(ready["ok"]),
            "status": status,
            "steps": steps,
            "confirm_match": confirm_match,
            "success_match": success_match,
            "state": ready.get("state"),
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }


def _run_buy_universal_bait(
    app: Any,
    ocr: Any,
    vision: Any,
    input_mapping: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    profile: dict[str, Any],
    profile_name: str,
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    started_at = time.monotonic()
    steps: list[dict[str, Any]] = []
    step_interval_sec = _profile_interval_sec(profile, "bait_step_interval_ms")
    click_hold_ms = int(profile.get("bait_click_hold_ms") or profile.get("sell_click_hold_ms") or 0)
    buy_confirm_match: dict[str, Any] | None = None
    try:
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                buy_confirm_match=buy_confirm_match,
                state=None,
            )
        _press_config_action(
            input_mapping,
            app,
            profile,
            "bait_shop_open_action",
            profile_name=profile_name,
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="bait",
            step="open_shop",
        )
        _sleep_interruptibly(step_interval_sec)
        steps.append({"step": "open_shop", "ok": True, "action": profile["bait_shop_open_action"]})
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                buy_confirm_match=buy_confirm_match,
                state=None,
            )

        _click_config_point(
            app,
            profile,
            "bait_item_point",
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="bait",
            step="select_universal",
            hold_ms=click_hold_ms,
        )
        item_wait_sec = max(float(profile["bait_item_after_wait_ms"]) / 1000.0, step_interval_sec)
        _sleep_interruptibly(item_wait_sec)
        steps.append(
            {
                "step": "select_universal",
                "ok": True,
                "point": list(profile["bait_item_point"]),
                "after_wait_ms": int(round(item_wait_sec * 1000)),
            }
        )
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                buy_confirm_match=buy_confirm_match,
                state=None,
            )

        _, _, max_clicks = _click_config_point_repeated(
            app,
            profile,
            "bait_max_point",
            clicks=int(profile["bait_max_clicks"]),
            interval_ms=int(profile["bait_max_click_interval_ms"]),
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="bait",
            step="max_count",
            hold_ms=click_hold_ms,
        )
        max_after_wait_sec = max(float(profile["bait_max_after_wait_ms"]) / 1000.0, step_interval_sec)
        _sleep_interruptibly(max_after_wait_sec)
        steps.append(
            {
                "step": "max_count",
                "ok": True,
                "point": list(profile["bait_max_point"]),
                "clicks": max_clicks,
                "click_interval_ms": int(profile["bait_max_click_interval_ms"]),
                "after_wait_ms": int(round(max_after_wait_sec * 1000)),
            }
        )
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                buy_confirm_match=buy_confirm_match,
                state=None,
            )

        _click_config_point(
            app,
            profile,
            "bait_buy_point",
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="bait",
            step="buy",
            hold_ms=click_hold_ms,
        )
        _sleep_interruptibly(step_interval_sec)
        steps.append({"step": "buy", "ok": True, "point": list(profile["bait_buy_point"])})
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                buy_confirm_match=buy_confirm_match,
                state=None,
            )

        buy_confirm_match = _wait_for_profile_template(
            app,
            vision,
            yihuan_fishing,
            profile,
            template_key="bait_buy_confirm_template",
            region_key="bait_buy_confirm_region",
            threshold_key="bait_buy_confirm_match_threshold",
            profile_name=profile_name,
            timeout_sec=float(profile["bait_buy_confirm_timeout_sec"]),
            log_scope="bait",
            step="buy_confirm_template",
        )
        steps.append(
            {
                "step": "buy_confirm_template",
                "ok": bool(buy_confirm_match.get("found")),
                "confidence": float(buy_confirm_match.get("confidence") or 0.0),
            }
        )
        if bool(buy_confirm_match.get("cancelled")):
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                buy_confirm_match=buy_confirm_match,
                state=None,
            )
        if not bool(buy_confirm_match.get("found")):
            return {
                "ok": False,
                "failure_reason": "buy_confirm_prompt_missing",
                "steps": steps,
                "buy_confirm_match": buy_confirm_match,
                "state": None,
                "elapsed_sec": round(time.monotonic() - started_at, 3),
            }

        _click_config_point(
            app,
            profile,
            "bait_confirm_point",
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="bait",
            step="confirm",
            hold_ms=click_hold_ms,
        )
        _sleep_interruptibly(step_interval_sec)
        steps.append({"step": "confirm", "ok": True, "point": list(profile["bait_confirm_point"])})
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                buy_confirm_match=buy_confirm_match,
                state=None,
            )

        close_count = _click_repeated(
            app,
            profile["bait_success_close_point"],
            clicks=int(profile["bait_success_close_clicks"]),
            interval_ms=int(profile["bait_success_close_interval_ms"]),
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="bait",
            step="close_success",
            hold_ms=click_hold_ms,
        )
        steps.append({"step": "close_success", "ok": True, "clicks": close_count})
        _sleep_interruptibly(step_interval_sec)
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                buy_confirm_match=buy_confirm_match,
                state=None,
            )

        _press_config_action(
            input_mapping,
            app,
            profile,
            "menu_back",
            profile_name=profile_name,
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="bait",
            step="return_ready",
        )
        _sleep_interruptibly(step_interval_sec)
        ready = _wait_for_ready_template(
            app,
            ocr,
            vision,
            yihuan_fishing,
            profile_name=profile_name,
            timeout_sec=float(profile["bait_recovery_ready_timeout_sec"]),
            phase_trace=phase_trace,
            start_time=start_time,
            note="buy_bait_return_ready",
        )
        steps.append({"step": "return_ready", "ok": bool(ready["ok"])})
        if bool(ready.get("cancelled")):
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                buy_confirm_match=buy_confirm_match,
                state=ready.get("state"),
            )
        ok = bool(ready["ok"])
        return {
            "ok": ok,
            "failure_reason": None if ok else "buy_bait_not_ready",
            "steps": steps,
            "buy_confirm_match": buy_confirm_match,
            "state": ready.get("state"),
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }
    except Exception as exc:  # pragma: no cover - defensive runtime recovery path
        logger.warning("Fishing[bait] buy_failed exception=%s", exc)
        steps.append({"step": "exception", "ok": False, "reason": str(exc)})
        return {
            "ok": False,
            "failure_reason": "buy_bait_exception",
            "steps": steps,
            "buy_confirm_match": buy_confirm_match,
            "state": None,
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }


def _run_change_universal_bait(
    app: Any,
    ocr: Any,
    vision: Any,
    input_mapping: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    profile: dict[str, Any],
    profile_name: str,
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    started_at = time.monotonic()
    steps: list[dict[str, Any]] = []
    try:
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                title_match=None,
                state=None,
            )
        logger.info("Fishing[bait] change_start")
        _press_config_action(
            input_mapping,
            app,
            profile,
            "bait_change_open_action",
            profile_name=profile_name,
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="bait",
            step="change_start",
        )
        _sleep_interruptibly(float(profile["bait_change_open_wait_sec"]))
        steps.append({"step": "change_start", "ok": True, "action": profile["bait_change_open_action"]})
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                title_match=None,
                state=None,
            )

        title_match = _capture_and_match_profile_template(
            app,
            vision,
            yihuan_fishing,
            profile,
            template_key="bait_change_template",
            region_key="bait_change_region",
            threshold_key="bait_change_match_threshold",
            profile_name=profile_name,
        )
        logger.info(
            "Fishing[bait] step=change_title ok=%s confidence=%.4f",
            bool(title_match.get("found")),
            float(title_match.get("confidence") or 0.0),
        )
        steps.append(
            {
                "step": "change_title",
                "ok": bool(title_match.get("found")),
                "confidence": float(title_match.get("confidence") or 0.0),
            }
        )

        _click_config_point(
            app,
            profile,
            "bait_change_confirm_point",
            phase_trace=phase_trace,
            start_time=start_time,
            log_scope="bait",
            step="change_confirm",
        )
        logger.info("Fishing[bait] change_confirm")
        steps.append({"step": "change_confirm", "ok": True, "point": list(profile["bait_change_confirm_point"])})
        _sleep_interruptibly(float(profile["bait_change_after_click_wait_sec"]))
        if _fishing_cancel_requested():
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                title_match=title_match,
                state=None,
            )

        ready = _wait_for_ready_template(
            app,
            ocr,
            vision,
            yihuan_fishing,
            profile_name=profile_name,
            timeout_sec=float(profile["bait_recovery_ready_timeout_sec"]),
            phase_trace=phase_trace,
            start_time=start_time,
            note="change_bait_return_ready",
        )
        steps.append({"step": "change_done", "ok": bool(ready["ok"])})
        if bool(ready.get("cancelled")):
            return _cancelled_operation_result(
                started_at=started_at,
                steps=steps,
                title_match=title_match,
                state=ready.get("state"),
            )
        logger.info("Fishing[bait] change_done ok=%s", bool(ready["ok"]))
        ok = bool(ready["ok"])
        return {
            "ok": ok,
            "failure_reason": None if ok else "change_bait_not_ready",
            "steps": steps,
            "title_match": title_match,
            "state": ready.get("state"),
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }
    except Exception as exc:  # pragma: no cover - defensive runtime recovery path
        logger.warning("Fishing[bait] change_failed exception=%s", exc)
        steps.append({"step": "exception", "ok": False, "reason": str(exc)})
        return {
            "ok": False,
            "failure_reason": "change_bait_exception",
            "steps": steps,
            "title_match": None,
            "state": None,
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }


def _run_bait_recovery(
    app: Any,
    ocr: Any,
    vision: Any,
    input_mapping: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    profile: dict[str, Any],
    profile_name: str,
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    started_at = time.monotonic()
    sell_result: dict[str, Any] | None = None
    buy_bait_result: dict[str, Any] | None = None
    change_bait_result: dict[str, Any] | None = None
    sell_before_buy = bool(profile["sell_before_buy_bait"])
    logger.info("Fishing[bait] recovery_start sell_before_buy=%s", sell_before_buy)
    app.release_all()

    if _fishing_cancel_requested():
        return _cancelled_operation_result(
            started_at=started_at,
            steps=[],
            sell_result=sell_result,
            buy_bait_result=buy_bait_result,
            change_bait_result=change_bait_result,
            state=None,
        )

    if sell_before_buy:
        sell_result = _run_sell_fish_before_buy_bait(
            app,
            ocr,
            vision,
            input_mapping,
            yihuan_fishing,
            profile=profile,
            profile_name=profile_name,
            phase_trace=phase_trace,
            start_time=start_time,
        )
        if bool(sell_result.get("cancelled")):
            return _cancelled_operation_result(
                started_at=started_at,
                steps=[],
                sell_result=sell_result,
                buy_bait_result=None,
                change_bait_result=None,
                state=sell_result.get("state"),
            )
        if str(sell_result.get("status")) == "failed_not_ready":
            elapsed = round(time.monotonic() - started_at, 3)
            logger.info("Fishing[bait] recovery_done ok=False elapsed=%.3f", elapsed)
            return {
                "ok": False,
                "failure_reason": "bait_recovery_not_ready",
                "sell_result": sell_result,
                "buy_bait_result": None,
                "change_bait_result": None,
                "state": sell_result.get("state"),
                "elapsed_sec": elapsed,
            }

    buy_bait_result = _run_buy_universal_bait(
        app,
        ocr,
        vision,
        input_mapping,
        yihuan_fishing,
        profile=profile,
        profile_name=profile_name,
        phase_trace=phase_trace,
        start_time=start_time,
    )
    if bool(buy_bait_result.get("cancelled")):
        return _cancelled_operation_result(
            started_at=started_at,
            steps=[],
            sell_result=sell_result,
            buy_bait_result=buy_bait_result,
            change_bait_result=None,
            state=buy_bait_result.get("state"),
        )
    if not bool(buy_bait_result.get("ok")):
        elapsed = round(time.monotonic() - started_at, 3)
        logger.info("Fishing[bait] recovery_done ok=False elapsed=%.3f", elapsed)
        return {
            "ok": False,
            "failure_reason": str(buy_bait_result.get("failure_reason") or "buy_bait_failed"),
            "sell_result": sell_result,
            "buy_bait_result": buy_bait_result,
            "change_bait_result": None,
            "state": buy_bait_result.get("state"),
            "elapsed_sec": elapsed,
        }

    change_bait_result = _run_change_universal_bait(
        app,
        ocr,
        vision,
        input_mapping,
        yihuan_fishing,
        profile=profile,
        profile_name=profile_name,
        phase_trace=phase_trace,
        start_time=start_time,
    )
    if bool(change_bait_result.get("cancelled")):
        return _cancelled_operation_result(
            started_at=started_at,
            steps=[],
            sell_result=sell_result,
            buy_bait_result=buy_bait_result,
            change_bait_result=change_bait_result,
            state=change_bait_result.get("state"),
        )
    elapsed = round(time.monotonic() - started_at, 3)
    ok = bool(change_bait_result.get("ok"))
    logger.info("Fishing[bait] recovery_done ok=%s elapsed=%.3f", ok, elapsed)
    return {
        "ok": ok,
        "failure_reason": None if ok else str(change_bait_result.get("failure_reason") or "change_bait_failed"),
        "sell_result": sell_result,
        "buy_bait_result": buy_bait_result,
        "change_bait_result": change_bait_result,
        "state": change_bait_result.get("state"),
        "elapsed_sec": elapsed,
    }


def _run_post_duel_cleanup(
    app: Any,
    ocr: Any,
    vision: Any,
    input_mapping: Any,
    yihuan_fishing: YihuanFishingService,
    *,
    profile_name: str | None,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    cleanup_timeout_sec: float | None = None,
) -> dict[str, Any]:
    profile = yihuan_fishing.load_profile(profile_name)
    close_action = str(profile.get("result_close_action") or "menu_back")
    close_interval_sec = max(float(profile["post_duel_close_interval_ms"]) / 1000.0, 0.01)
    ready_stable_required_sec = max(float(profile.get("post_duel_ready_stable_sec", 1.0) or 0.0), 0.0)
    deadline = (
        time.monotonic() + max(float(cleanup_timeout_sec), 0.1)
        if cleanup_timeout_sec is not None and cleanup_timeout_sec > 0
        else None
    )
    next_close_at: float | None = None
    close_count = 0
    last_state: dict[str, Any] | None = None
    ready_stable_started_at: float | None = None
    ready_stable_elapsed_sec = 0.0

    while True:
        if _fishing_cancel_requested():
            logger.info("Fishing[post_duel_cleanup] cancelled")
            return {
                "ok": False,
                "click_count": close_count,
                "close_count": close_count,
                "close_action": close_action,
                "final_state": last_state,
                "cancelled": True,
                "ready_stable_required_sec": ready_stable_required_sec,
                "ready_stable_elapsed_sec": ready_stable_elapsed_sec,
            }
        now = time.monotonic()
        if deadline is not None and now > deadline:
            break

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
            sample_at = time.monotonic()
            if ready_stable_started_at is None:
                ready_stable_started_at = sample_at
                logger.info(
                    "Fishing[post_duel_cleanup] ready_candidate stable_required_sec=%.3f close_count=%s",
                    ready_stable_required_sec,
                    close_count,
                )
            ready_stable_elapsed_sec = max(sample_at - ready_stable_started_at, 0.0)
            if ready_stable_elapsed_sec >= ready_stable_required_sec:
                _append_trace(phase_trace, start_time=start_time, state=state, note="post_duel_cleanup_done")
                logger.info(
                    "Fishing[post_duel_cleanup] ready_confirmed stable_elapsed_sec=%.3f close_count=%s",
                    ready_stable_elapsed_sec,
                    close_count,
                )
                return {
                    "ok": True,
                    "click_count": close_count,
                    "close_count": close_count,
                    "close_action": close_action,
                    "final_state": state,
                    "ready_stable_required_sec": ready_stable_required_sec,
                    "ready_stable_elapsed_sec": round(ready_stable_elapsed_sec, 3),
                }
            _sleep_interruptibly(0.05)
            continue
        else:
            if ready_stable_started_at is not None:
                logger.info(
                    "Fishing[post_duel_cleanup] ready_candidate_reset stable_elapsed_sec=%.3f phase=%s",
                    ready_stable_elapsed_sec,
                    state.get("phase"),
                )
            ready_stable_started_at = None
            ready_stable_elapsed_sec = 0.0

        now = time.monotonic()
        if next_close_at is None or now >= next_close_at:
            _press_input_action(input_mapping, app, action_name=close_action, profile_name=profile["profile_name"])
            close_count += 1
            _append_trace(phase_trace, start_time=start_time, note="post_duel_close")
            logger.info(
                "Fishing[input] action=%s phase=press note=post_duel_cleanup close_index=%s",
                close_action,
                close_count,
            )
            next_close_at = now + close_interval_sec

        sleep_sec = min(0.1, max(next_close_at - time.monotonic(), 0.0))
        if sleep_sec > 0:
            _sleep_interruptibly(sleep_sec)

    if last_state is None:
        last_state = _read_state_snapshot(
            app,
            ocr,
            vision,
            yihuan_fishing,
            profile_name=profile["profile_name"],
            enable_ocr=False,
        )
    if _has_ready_template(last_state) and (
        ready_stable_required_sec <= 0.0
        or (
            ready_stable_started_at is not None
            and time.monotonic() - ready_stable_started_at >= ready_stable_required_sec
        )
    ):
        if ready_stable_started_at is not None:
            ready_stable_elapsed_sec = max(time.monotonic() - ready_stable_started_at, 0.0)
        _append_trace(phase_trace, start_time=start_time, state=last_state, note="post_duel_cleanup_done")
        return {
            "ok": True,
            "click_count": close_count,
            "close_count": close_count,
            "close_action": close_action,
            "final_state": last_state,
            "ready_stable_required_sec": ready_stable_required_sec,
            "ready_stable_elapsed_sec": round(ready_stable_elapsed_sec, 3),
        }
    return {
        "ok": False,
        "click_count": close_count,
        "close_count": close_count,
        "close_action": close_action,
        "final_state": last_state,
        "ready_stable_required_sec": ready_stable_required_sec,
        "ready_stable_elapsed_sec": round(ready_stable_elapsed_sec, 3),
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
    phase_trace = _new_bounded_buffer(profile.get("trace_limit"), default=240)
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
    round_extra: dict[str, Any] = {
        "bait_recovery_count": 0,
        "sell_result": None,
        "buy_bait_result": None,
        "change_bait_result": None,
        "bait_recovery_result": None,
        "bait_shortage": None,
    }

    def _cancelled_round_result() -> dict[str, Any]:
        logger.info("Fishing[round] cancelled round_index=%s", round_index)
        return _round_result(
            "cancelled",
            round_index,
            phase_trace,
            "cancelled",
            timings,
            start_time,
            duel_debug_artifact,
            (
                _finalize_detection_stats(
                    detection_stats,
                    previous_status=last_detection_status,
                    previous_sample_at=last_detection_sample_at,
                )
                if detection_stats is not None
                else None
            ),
            extra=round_extra,
        )

    try:
        if _fishing_cancel_requested():
            return _cancelled_round_result()
        state = _read_state_snapshot(app, ocr, vision, yihuan_fishing, profile_name=resolved_profile)
        _append_trace(phase_trace, start_time=start_time, state=state, note="initial")
        _log_state_details("initial", state)

        if state["phase"] == "result":
            cleanup_result = _run_post_duel_cleanup(
                app,
                ocr,
                vision,
                input_mapping,
                yihuan_fishing,
                profile_name=resolved_profile,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            if bool(cleanup_result.get("cancelled")):
                return _cancelled_round_result()
            if not cleanup_result["ok"]:
                failure_reason = "result_close_timeout"
                return _round_result(
                    "failed",
                    round_index,
                    phase_trace,
                    failure_reason,
                    timings,
                    start_time,
                    extra=round_extra,
                )
            state = dict(cleanup_result["final_state"] or {})

        max_recovery_attempts = int(profile["bait_recovery_max_attempts_per_round"])
        while True:
            if _fishing_cancel_requested():
                return _cancelled_round_result()
            ready_wait_started = time.monotonic()
            ready_deadline = ready_wait_started + 5.0
            next_ready_poll_at: float | None = None
            while state["phase"] != "ready":
                if _fishing_cancel_requested():
                    return _cancelled_round_result()
                if time.monotonic() > ready_deadline:
                    failure_reason = "not_ready"
                    logger.info("Fishing[ready_wait] timed_out phase=%s", state.get("phase"))
                    return _round_result(
                        "failed",
                        round_index,
                        phase_trace,
                        failure_reason,
                        timings,
                        start_time,
                        extra=round_extra,
                    )
                next_ready_poll_at = _sleep_for_target_period(next_ready_poll_at, poll_sec)
                state = _read_state_snapshot(app, ocr, vision, yihuan_fishing, profile_name=resolved_profile)
                _append_trace(phase_trace, start_time=start_time, state=state)
                _log_state_details("ready_wait", state)
            timings["ready_wait_sec"] = round(float(timings["ready_wait_sec"]) + time.monotonic() - ready_wait_started, 3)

            if _fishing_cancel_requested():
                return _cancelled_round_result()
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
            timings["hook_wait_sec"] = round(
                float(timings["hook_wait_sec"]) + float(hook_result["elapsed_sec"]),
                3,
            )
            state = dict(hook_result.get("state") or state)
            if bool(hook_result["ok"]):
                break

            failure_reason = str(hook_result["failure_reason"] or "hook_timeout")
            if failure_reason == "cancelled" or bool(hook_result.get("cancelled")):
                return _cancelled_round_result()
            round_extra["bait_shortage"] = hook_result.get("bait_shortage")
            if (
                failure_reason == "bait_shortage"
                and bool(profile["bait_recovery_enabled"])
                and int(round_extra["bait_recovery_count"]) < max_recovery_attempts
            ):
                round_extra["bait_recovery_count"] = int(round_extra["bait_recovery_count"]) + 1
                recovery_result = _run_bait_recovery(
                    app,
                    ocr,
                    vision,
                    input_mapping,
                    yihuan_fishing,
                    profile=profile,
                    profile_name=resolved_profile,
                    phase_trace=phase_trace,
                    start_time=start_time,
                )
                round_extra["bait_recovery_result"] = recovery_result
                if bool(recovery_result.get("cancelled")):
                    round_extra["sell_result"] = recovery_result.get("sell_result")
                    round_extra["buy_bait_result"] = recovery_result.get("buy_bait_result")
                    round_extra["change_bait_result"] = recovery_result.get("change_bait_result")
                    return _cancelled_round_result()
                round_extra["sell_result"] = recovery_result.get("sell_result")
                round_extra["buy_bait_result"] = recovery_result.get("buy_bait_result")
                round_extra["change_bait_result"] = recovery_result.get("change_bait_result")
                state = dict(recovery_result.get("state") or state)
                if bool(recovery_result.get("ok")):
                    logger.info("Fishing[bait] recovery_ok retrying_round=%s", round_index)
                    continue
                failure_reason = str(recovery_result.get("failure_reason") or "bait_recovery_failed")
                logger.info("Fishing[bait] recovery_failed reason=%s", failure_reason)
                return _round_result(
                    "failed",
                    round_index,
                    phase_trace,
                    failure_reason,
                    timings,
                    start_time,
                    duel_debug_artifact,
                    extra=round_extra,
                )

            logger.info("Fishing[hook_wait] failed reason=%s", failure_reason)
            return _round_result(
                "failed",
                round_index,
                phase_trace,
                failure_reason,
                timings,
                start_time,
                duel_debug_artifact,
                extra=round_extra,
            )

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
            if _fishing_cancel_requested():
                held_direction_action, _ = _set_held_direction(
                    input_mapping,
                    app,
                    current_action_name=held_direction_action,
                    desired_action_name=None,
                    profile_name=resolved_profile,
                )
                return _cancelled_round_result()
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
                    extra=round_extra,
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
                            input_mapping,
                            yihuan_fishing,
                            profile_name=resolved_profile,
                            phase_trace=phase_trace,
                            start_time=start_time,
                        )
                        if bool(cleanup_result.get("cancelled")):
                            return _cancelled_round_result()
                        final_state = dict(cleanup_result["final_state"] or {})
                        logger.info(
                            "Fishing[duel_end] cleanup_ok=%s close_count=%s terminal_phase=%s final_phase=%s",
                            cleanup_result["ok"],
                            cleanup_result.get("close_count", cleanup_result.get("click_count")),
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
                                extra=round_extra,
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
                            extra=round_extra,
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
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timings["total_sec"] = round(time.monotonic() - start_time, 3)
    result = {
        "status": status,
        "round_index": int(round_index),
        "failure_reason": failure_reason,
        "timings": timings,
        "duel_debug_artifact": duel_debug_artifact,
        "detection_stats": detection_stats,
    }
    if extra:
        result.update(extra)
    return _apply_phase_trace_payload(result, phase_trace)


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
    phase_trace = _new_bounded_buffer(profile.get("trace_limit"), default=240)
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
                input_mapping,
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
    phase_trace = _new_bounded_buffer(profile.get("trace_limit"), default=240)
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
    sell_fish_every_rounds: int = 0,
) -> dict[str, Any]:
    consecutive_failures = 0
    success_count = 0
    failure_count = 0
    active_sell_count = 0
    active_sell_failure_count = 0
    round_index = 1
    stop_reason = "max_rounds"
    max_rounds_value = _non_negative_int(max_rounds)
    sell_interval_rounds = _non_negative_int(sell_fish_every_rounds)
    unlimited_rounds = max_rounds_value <= 0
    profile = yihuan_fishing.load_profile(profile_name)
    resolved_profile = profile["profile_name"]
    results = _new_bounded_buffer(profile.get("session_results_limit"), default=60)
    active_sell_results = _new_bounded_buffer(profile.get("active_sell_results_limit"), default=20)

    while unlimited_rounds or round_index <= max_rounds_value:
        if _fishing_cancel_requested():
            stop_reason = "cancelled"
            logger.info("Fishing[session] cancelled before round_index=%s", round_index)
            break
        round_result = _run_fishing_round_impl(
            app,
            ocr,
            vision,
            input_mapping,
            yihuan_fishing,
            round_index=round_index,
            profile_name=resolved_profile,
            bite_timeout_sec=None,
            duel_timeout_sec=None,
        )
        results.append(round_result)
        if round_result.get("status") == "cancelled":
            stop_reason = "cancelled"
            logger.info("Fishing[session] cancelled during round_index=%s", round_index)
            break
        if round_result["status"] == "success":
            success_count += 1
            consecutive_failures = 0
            if sell_interval_rounds > 0 and success_count % sell_interval_rounds == 0:
                if _fishing_cancel_requested():
                    stop_reason = "cancelled"
                    logger.info("Fishing[session] cancelled before active_sell round_index=%s", round_index)
                    break
                app.release_all()
                sell_trace = _new_bounded_buffer(profile.get("trace_limit"), default=240)
                sell_started_at = time.monotonic()
                logger.info(
                    "Fishing[active_sell] start interval_rounds=%s success_count=%s round_index=%s",
                    sell_interval_rounds,
                    success_count,
                    round_index,
                )
                sell_result = _run_sell_fish_before_buy_bait(
                    app,
                    ocr,
                    vision,
                    input_mapping,
                    yihuan_fishing,
                    profile=profile,
                    profile_name=resolved_profile,
                    phase_trace=sell_trace,
                    start_time=sell_started_at,
                )
                sell_result["trigger_round_index"] = int(round_index)
                sell_result["trigger_success_count"] = int(success_count)
                sell_result["interval_rounds"] = int(sell_interval_rounds)
                _apply_phase_trace_payload(sell_result, sell_trace)
                round_result["active_sell_result"] = sell_result
                active_sell_results.append(sell_result)
                active_sell_count += 1
                if bool(sell_result.get("cancelled")) or sell_result.get("status") == "cancelled":
                    stop_reason = "cancelled"
                    logger.info(
                        "Fishing[session] cancelled during active_sell round_index=%s success_count=%s",
                        round_index,
                        success_count,
                    )
                    break
                sell_ok = bool(sell_result.get("ok"))
                logger.info(
                    "Fishing[active_sell] done ok=%s status=%s round_index=%s success_count=%s",
                    sell_ok,
                    sell_result.get("status"),
                    round_index,
                    success_count,
                )
                if not sell_ok:
                    active_sell_failure_count += 1
                    stop_reason = "active_sell_not_ready"
                    break
        else:
            failure_count += 1
            consecutive_failures += 1
        round_index += 1

    status = "cancelled" if stop_reason == "cancelled" else "success"
    if stop_reason == "cancelled":
        pass
    elif success_count == 0 and failure_count > 0:
        status = "failed"
    elif success_count > 0 and failure_count > 0:
        status = "partial"
    if stop_reason != "cancelled" and active_sell_failure_count > 0:
        status = "failed" if success_count == 0 else "partial"

    retained_results, results_limit, results_truncated_count = _bounded_buffer_meta(results)
    retained_active_sell_results, active_sell_results_limit, active_sell_results_truncated_count = _bounded_buffer_meta(
        active_sell_results
    )
    return {
        "status": status,
        "round_count": getattr(results, "total_count", len(retained_results)),
        "success_count": success_count,
        "failure_count": failure_count,
        "consecutive_failures": consecutive_failures,
        "active_sell_interval_rounds": sell_interval_rounds,
        "active_sell_count": active_sell_count,
        "active_sell_failure_count": active_sell_failure_count,
        "active_sell_results": retained_active_sell_results,
        "active_sell_results_limit": active_sell_results_limit,
        "active_sell_results_retained_count": len(retained_active_sell_results),
        "active_sell_results_truncated_count": active_sell_results_truncated_count,
        "stopped_reason": stop_reason,
        "results": retained_results,
        "results_limit": results_limit,
        "results_retained_count": len(retained_results),
        "results_truncated_count": results_truncated_count,
    }


"""Yihuan automatic combat session action."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
from packages.aura_core.api import action_info, requires_services
from packages.aura_core.observability.logging.core_logger import current_cid, logger
from packages.aura_core.scheduler.cancellation import is_current_task_cancel_requested
from plans.aura_base.src.platform.contracts import TargetRuntimeError

from ..audio_dodge import AudioDodgeRuntime
from ..services.combat_service import YihuanCombatService


class _CombatSessionCancelled(Exception):
    """Internal marker used to stop the sync combat action cooperatively."""


def _coerce_bool(value: bool | str | int | float | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false", "no", "off", "none", "null"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return bool(value)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _combat_cancel_requested() -> bool:
    try:
        return is_current_task_cancel_requested()
    except Exception:
        return False


def _raise_if_cancelled() -> None:
    if _combat_cancel_requested():
        raise _CombatSessionCancelled()


def _sleep_is_mocked() -> bool:
    return hasattr(time.sleep, "mock_calls")


def _sleep_interruptibly(duration_sec: float, *, quantum_sec: float = 0.05) -> None:
    duration = max(float(duration_sec), 0.0)
    _raise_if_cancelled()
    if duration <= 0:
        return
    if _sleep_is_mocked():
        time.sleep(duration)
        _raise_if_cancelled()
        return
    deadline = time.monotonic() + duration
    while True:
        _raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, max(float(quantum_sec), 0.01)))


def _capture_state(app: Any, yihuan_combat: YihuanCombatService, *, profile_name: str) -> tuple[dict[str, Any], Any]:
    _raise_if_cancelled()
    capture = app.capture()
    if not capture.success or capture.image is None:
        raise RuntimeError("Failed to capture the Yihuan combat screen.")
    state = yihuan_combat.analyze_frame(capture.image, profile_name=profile_name)
    return state, capture


def _trim_trace(trace: list[dict[str, Any]], trace_limit: int) -> None:
    if len(trace) > trace_limit:
        del trace[: len(trace) - trace_limit]


def _combat_capture_root() -> Path:
    return Path(__file__).resolve().parents[4] / "logs" / "yihuan_combat_debug"


def _sanitize_capture_label(label: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(label or "").strip().lower())
    normalized = normalized.strip("_")
    return normalized or "unknown"


class _CombatCaptureLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        interval_sec: float,
        max_images: int,
        raw_enabled: bool,
        profile_name: str,
        yihuan_combat: YihuanCombatService,
    ) -> None:
        self.enabled = bool(enabled)
        self.interval_sec = max(float(interval_sec), 0.1)
        self.max_images = max(int(max_images), 1)
        self.raw_enabled = bool(raw_enabled)
        self.profile_name = str(profile_name)
        self.yihuan_combat = yihuan_combat
        self.screenshot_dir: Path | None = None
        self.index_path: Path | None = None
        self.screenshots: list[dict[str, Any]] = []
        self.saved_count = 0
        self.periodic_count = 0
        self.event_count = 0
        self.skipped_max_images_count = 0
        self.capture_failed_count = 0
        self.next_periodic_at: float | None = None
        self._last_save_key: tuple[str, str, str, float] | None = None
        self._cid = str(current_cid() or "-").strip() or "-"

    def start(self, *, start_time: float) -> None:
        if not self.enabled:
            return
        self.next_periodic_at = float(start_time) + float(self.interval_sec)

    def should_capture_periodic(self, *, now: float) -> bool:
        if not self.enabled or self.next_periodic_at is None:
            return False
        return float(now) >= float(self.next_periodic_at)

    def advance_periodic_schedule(self, *, now: float) -> None:
        if not self.enabled or self.next_periodic_at is None:
            return
        next_due = float(self.next_periodic_at)
        while next_due <= float(now):
            next_due += float(self.interval_sec)
        self.next_periodic_at = next_due

    def _ensure_output_dir(self) -> Path:
        if self.screenshot_dir is not None:
            return self.screenshot_dir
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = _combat_capture_root() / f"{timestamp}_cid{_sanitize_capture_label(self._cid)}"
        directory.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir = directory
        self.index_path = directory / "index.json"
        return directory

    def _state_summary(
        self,
        state: Mapping[str, Any] | None,
        *,
        phase: str,
        label: str,
        elapsed_sec: float,
        combat_active: bool | None = None,
    ) -> dict[str, Any]:
        payload = dict(state or {})
        return {
            "t": round(float(elapsed_sec), 3),
            "label": str(label),
            "phase": str(phase),
            "in_combat": bool(payload.get("in_combat")),
            "combat_active": bool(combat_active) if combat_active is not None else bool(payload.get("in_combat")),
            "enemy_health_found": bool(payload.get("enemy_health_found")),
            "enemy_health_count": int(payload.get("enemy_health_count") or 0),
            "enemy_direction_found": bool(payload.get("enemy_direction_found")),
            "enemy_direction_count": int(payload.get("enemy_direction_count") or 0),
            "enemy_direction_primary_side": payload.get("enemy_direction_primary_side"),
            "boss_found": bool(payload.get("boss_found")),
            "target_found": bool(payload.get("target_found")),
            "current_slot": payload.get("current_slot"),
            "skill_state": payload.get("skill_state"),
            "ultimate_state": payload.get("ultimate_state"),
            "confidence": payload.get("confidence"),
        }

    def _write_index(self) -> None:
        if self.index_path is None:
            return
        payload = {
            "screenshot_dir": str(self.screenshot_dir) if self.screenshot_dir is not None else None,
            "screenshots": list(self.screenshots),
            "capture_stats": self.result_payload()["capture_stats"],
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save(
        self,
        *,
        kind: str,
        label: str,
        phase: str,
        source_image: Any,
        state: Mapping[str, Any] | None,
        elapsed_sec: float,
        combat_active: bool | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled or source_image is None:
            return None
        if self.saved_count >= self.max_images:
            self.skipped_max_images_count += 1
            return None
        rounded_elapsed = round(float(elapsed_sec), 2)
        save_key = (str(kind), str(label), str(phase), rounded_elapsed)
        if self._last_save_key == save_key:
            return None
        out_dir = self._ensure_output_dir()
        safe_label = _sanitize_capture_label(label)
        annotated_name = f"{self.saved_count:04d}_{float(elapsed_sec):06.2f}s_{safe_label}_annotated.png"
        raw_name = f"{self.saved_count:04d}_{float(elapsed_sec):06.2f}s_{safe_label}_raw.png"
        overlay = {
            "t": round(float(elapsed_sec), 3),
            "phase": str(phase),
            "note": str(label),
            "combat_active": bool(combat_active) if combat_active is not None else bool((state or {}).get("in_combat")),
        }
        annotated_image, resolved_state = self.yihuan_combat.annotate_frame(
            source_image,
            profile_name=self.profile_name,
            state=state,
            overlay=overlay,
        )
        raw_path: Path | None = None
        if self.raw_enabled:
            raw_path = out_dir / raw_name
            cv2.imwrite(str(raw_path), cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))
        annotated_path = out_dir / annotated_name
        cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        entry = {
            "t": round(float(elapsed_sec), 3),
            "kind": str(kind),
            "label": str(label),
            "phase": str(phase),
            "image_path": str(annotated_path),
            "raw_image_path": str(raw_path) if raw_path is not None else None,
            "state_summary": self._state_summary(
                resolved_state,
                phase=phase,
                label=label,
                elapsed_sec=elapsed_sec,
                combat_active=combat_active,
            ),
        }
        self.screenshots.append(entry)
        self.saved_count += 1
        self.periodic_count += 1 if kind == "periodic" else 0
        self.event_count += 1 if kind == "event" else 0
        self._last_save_key = save_key
        self._write_index()
        return entry

    def capture_event(
        self,
        *,
        label: str,
        phase: str,
        source_image: Any,
        state: Mapping[str, Any] | None,
        elapsed_sec: float,
        combat_active: bool | None = None,
    ) -> dict[str, Any] | None:
        return self._save(
            kind="event",
            label=label,
            phase=phase,
            source_image=source_image,
            state=state,
            elapsed_sec=elapsed_sec,
            combat_active=combat_active,
        )

    def capture_periodic(
        self,
        *,
        phase: str,
        source_image: Any,
        state: Mapping[str, Any] | None,
        elapsed_sec: float,
        combat_active: bool | None = None,
    ) -> dict[str, Any] | None:
        return self._save(
            kind="periodic",
            label=f"periodic_{phase}",
            phase=phase,
            source_image=source_image,
            state=state,
            elapsed_sec=elapsed_sec,
            combat_active=combat_active,
        )

    def record_capture_failure(self) -> None:
        self.capture_failed_count += 1

    def result_payload(self) -> dict[str, Any]:
        return {
            "screenshot_dir": str(self.screenshot_dir) if self.screenshot_dir is not None else None,
            "screenshots": list(self.screenshots),
            "capture_stats": {
                "enabled": bool(self.enabled),
                "saved_count": int(self.saved_count),
                "periodic_count": int(self.periodic_count),
                "event_count": int(self.event_count),
                "skipped_max_images_count": int(self.skipped_max_images_count),
                "capture_failed_count": int(self.capture_failed_count),
                "capture_interval_sec": float(self.interval_sec),
                "capture_max_images": int(self.max_images),
                "capture_raw_enabled": bool(self.raw_enabled),
            },
        }


def _capture_periodic_if_due(
    app: Any,
    capture_debug: _CombatCaptureLogger | None,
    *,
    state: Mapping[str, Any] | None,
    phase: str,
    start_time: float,
    combat_active: bool,
) -> Any | None:
    if capture_debug is None:
        return None
    now = time.monotonic()
    if not capture_debug.should_capture_periodic(now=now):
        return None
    capture = app.capture()
    if not capture.success or capture.image is None:
        capture_debug.record_capture_failure()
        capture_debug.advance_periodic_schedule(now=now)
        return None
    capture_debug.capture_periodic(
        phase=phase,
        source_image=capture.image,
        state=state,
        elapsed_sec=now - start_time,
        combat_active=combat_active,
    )
    capture_debug.advance_periodic_schedule(now=now)
    return capture.image


def _capture_event_image(
    capture_debug: _CombatCaptureLogger | None,
    *,
    label: str,
    phase: str,
    source_image: Any,
    state: Mapping[str, Any] | None,
    start_time: float,
    combat_active: bool,
) -> None:
    if capture_debug is None:
        return
    capture_debug.capture_event(
        label=label,
        phase=phase,
        source_image=source_image,
        state=state,
        elapsed_sec=time.monotonic() - start_time,
        combat_active=combat_active,
    )


def _resolve_terminal_capture_image(app: Any, *, fallback_image: Any, capture_debug: _CombatCaptureLogger | None) -> Any:
    if fallback_image is not None or capture_debug is None or not capture_debug.enabled:
        return fallback_image
    try:
        capture = app.capture()
    except Exception:  # noqa: BLE001
        capture_debug.record_capture_failure()
        return fallback_image
    if not capture.success or capture.image is None:
        capture_debug.record_capture_failure()
        return fallback_image
    return capture.image


def _append_state_trace(
    combat_state_trace: list[dict[str, Any]],
    *,
    start_time: float,
    state: Mapping[str, Any],
    combat_active: bool,
    phase: str,
    note: str,
    trace_limit: int,
) -> None:
    combat_state_trace.append(
        {
            "t": round(time.monotonic() - start_time, 3),
            "phase": str(phase),
            "note": note,
            "combat_active": bool(combat_active),
            "in_supported_scene": bool(state.get("in_supported_scene")),
            "in_combat": bool(state.get("in_combat")),
            "target_found": bool(state.get("target_found")),
            "target_confidence": state.get("target_confidence"),
            "enemy_health_found": bool(state.get("enemy_health_found")),
            "enemy_health_count": int(state.get("enemy_health_count") or 0),
            "boss_found": bool(state.get("boss_found")),
            "challenge_success_found": bool(state.get("challenge_success_found")),
            "current_slot": state.get("current_slot"),
            "skill_available": bool(state.get("skill_available")),
            "skill_state": state.get("skill_state"),
            "ultimate_available": bool(state.get("ultimate_available")),
            "ultimate_state": state.get("ultimate_state"),
            "confidence": state.get("confidence"),
        }
    )
    _trim_trace(combat_state_trace, trace_limit)


def _final_result(
    *,
    status: str,
    stopped_reason: str,
    failure_reason: str | None,
    profile_name: str,
    strategy_name: str,
    encounters_completed: int,
    current_phase: str,
    last_state: Mapping[str, Any] | None,
    combat_state_trace: list[dict[str, Any]],
    action_trace: list[dict[str, Any]],
    start_time: float,
    dry_run: bool,
    capture_debug: _CombatCaptureLogger | None = None,
) -> dict[str, Any]:
    result = {
        "status": status,
        "stopped_reason": stopped_reason,
        "failure_reason": failure_reason,
        "profile_name": profile_name,
        "strategy_name": strategy_name,
        "encounters_completed": int(encounters_completed),
        "current_phase": str(current_phase),
        "last_state": dict(last_state or {}),
        "combat_state_trace": list(combat_state_trace),
        "action_trace": list(action_trace),
        "elapsed_sec": round(time.monotonic() - start_time, 3),
        "dry_run": bool(dry_run),
    }
    if capture_debug is not None:
        result.update(capture_debug.result_payload())
    else:
        result.update(
            {
                "screenshot_dir": None,
                "screenshots": [],
                "capture_stats": {
                    "enabled": False,
                    "saved_count": 0,
                    "periodic_count": 0,
                    "event_count": 0,
                    "skipped_max_images_count": 0,
                    "capture_failed_count": 0,
                    "capture_interval_sec": 0.0,
                    "capture_max_images": 0,
                    "capture_raw_enabled": False,
                },
            }
        )
    return result


def _record_action(
    action_trace: list[dict[str, Any]],
    *,
    start_time: float,
    action: str,
    dry_run: bool,
    details: Mapping[str, Any] | None = None,
) -> None:
    entry = {
        "t": round(time.monotonic() - start_time, 3),
        "action": action,
        "dry_run": bool(dry_run),
    }
    if details:
        entry.update(dict(details))
    action_trace.append(entry)
    logger.info(
        "Combat[action] action=%s dry_run=%s details=%s",
        action,
        bool(dry_run),
        dict(details or {}),
    )


def _perform_binding_tap(app: Any, binding: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    normalized = str(binding or "").strip().lower()
    if normalized == "mouse_left":
        app.click(button="left")
    elif normalized == "mouse_right":
        app.click(button="right")
    elif normalized == "mouse_middle":
        app.click(button="middle")
    else:
        app.press_key(str(binding), presses=1)


def _binding_to_input_mapping(binding: str) -> dict[str, Any]:
    normalized = str(binding or "").strip().lower()
    if normalized == "mouse_left":
        return {"type": "mouse_button", "button": "left", "action_name": binding}
    if normalized == "mouse_right":
        return {"type": "mouse_button", "button": "right", "action_name": binding}
    if normalized == "mouse_middle":
        return {"type": "mouse_button", "button": "middle", "action_name": binding}
    return {"type": "key", "key": str(binding), "action_name": binding}


def _execute_input_mapping_phase_with_retry(
    app: Any,
    input_mapping: Any,
    binding: Mapping[str, Any],
    *,
    phase: str,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    if dry_run:
        return
    _call_with_focus_retry(
        app,
        lambda: input_mapping.execute_binding(binding, phase=phase, app=app),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )


def _tap_input_binding_with_retry(
    app: Any,
    input_mapping: Any,
    binding: str,
    *,
    hold_ms: int,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    resolved_binding = _binding_to_input_mapping(binding)
    normalized_hold_ms = max(int(hold_ms), 0)
    if normalized_hold_ms <= 0:
        _execute_input_mapping_phase_with_retry(
            app,
            input_mapping,
            resolved_binding,
            phase="tap",
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            retry_reason=retry_reason,
        )
        return
    held = False
    _execute_input_mapping_phase_with_retry(
        app,
        input_mapping,
        resolved_binding,
        phase="hold",
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )
    held = True
    try:
        _sleep_ms_uninterruptible(normalized_hold_ms, dry_run=dry_run)
    finally:
        if held:
            _execute_input_mapping_phase_with_retry(
                app,
                input_mapping,
                resolved_binding,
                phase="release",
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason=retry_reason,
            )


def _call_with_focus_retry(
    app: Any,
    callback: Any,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    try:
        callback()
        return
    except Exception as exc:  # noqa: BLE001
        if not _is_window_focus_error(exc):
            raise
        if not _try_focus_activation(
            app,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            reason=retry_reason,
        ):
            raise
        callback()


def _is_window_focus_error(exc: BaseException) -> bool:
    if isinstance(exc, TargetRuntimeError):
        return exc.code == "window_focus_required"
    return "could not be focused" in str(exc).lower()


def _try_focus_activation(
    app: Any,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    reason: str,
) -> bool:
    if dry_run or not hasattr(app, "focus_with_input"):
        return False
    _record_action(
        action_trace,
        start_time=start_time,
        action="focus_with_input",
        dry_run=dry_run,
        details={"reason": reason},
    )
    try:
        return bool(app.focus_with_input(click_delay=0.15))
    except Exception as exc:  # noqa: BLE001
        action_trace[-1]["success"] = False
        action_trace[-1]["error"] = str(exc)
        return False


def _perform_binding_tap_with_retry(
    app: Any,
    binding: str,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    _call_with_focus_retry(
        app,
        lambda: _perform_binding_tap(app, binding, dry_run=dry_run),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )


def _perform_look_delta_with_retry(
    app: Any,
    dx: int,
    dy: int,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    if dry_run:
        return
    _call_with_focus_retry(
        app,
        lambda: app.look_delta(int(dx), int(dy)),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )


def _perform_mouse_tap_without_move_with_retry(
    app: Any,
    button: str,
    *,
    hold_ms: int,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    if dry_run:
        return
    normalized_button = str(button or "left").strip().lower()
    down_sent = False
    _call_with_focus_retry(
        app,
        lambda: app.mouse_down(normalized_button),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )
    down_sent = True
    try:
        _sleep_ms_uninterruptible(max(int(hold_ms), 0), dry_run=dry_run)
    finally:
        if down_sent:
            _call_with_focus_retry(
                app,
                lambda: app.mouse_up(normalized_button),
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason=retry_reason,
            )


def _tap_binding(
    app: Any,
    binding: str,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    action_name: str,
) -> None:
    _record_action(
        action_trace,
        start_time=start_time,
        action=action_name,
        dry_run=dry_run,
        details={"binding": binding},
    )
    _perform_binding_tap_with_retry(
        app,
        binding,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=action_name,
    )


def _normal_attack(
    app: Any,
    binding: str,
    *,
    mode: str,
    duration_ms: int,
    interval_ms: int,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    interrupt_checker: Any = None,
) -> dict[str, Any]:
    normalized_mode = str(mode or "tap_spam").strip().lower()
    tap_interval_ms = max(int(interval_ms), 1)
    planned_taps = max(1, int((max(int(duration_ms), 0) + tap_interval_ms - 1) / tap_interval_ms))
    tap_hold_ms = min(max(int(round(tap_interval_ms * 0.25)), 12), 24)
    details = {
        "binding": binding,
        "mode": normalized_mode,
        "duration_ms": int(duration_ms),
        "interval_ms": tap_interval_ms,
        "planned_taps": planned_taps,
        "tap_hold_ms": tap_hold_ms,
    }
    _record_action(
        action_trace,
        start_time=start_time,
        action="normal",
        dry_run=dry_run,
        details=details,
    )
    if normalized_mode == "hold":
        normalized = str(binding or "mouse_left").strip().lower()
        duration_sec = max(int(duration_ms), 0) / 1000.0
        if not dry_run:
            if normalized.startswith("mouse_"):
                button = normalized.replace("mouse_", "", 1)
                app.mouse_down(button)
                if duration_sec > 0:
                    time.sleep(duration_sec)
                app.mouse_up(button)
            else:
                app.key_down(str(binding))
                if duration_sec > 0:
                    time.sleep(duration_sec)
                app.key_up(str(binding))
        action_trace[-1]["tap_count"] = 1
        return {"executed": True, "action": "normal", "tap_count": 1}

    tap_count = 0
    interrupt_reason: str | None = None
    interrupt_payload: Mapping[str, Any] | None = None
    normalized_binding = str(binding or "mouse_left").strip().lower()
    _raise_if_cancelled()
    for tap_index in range(planned_taps):
        if interrupt_checker is not None:
            interrupt_payload = interrupt_checker()
            if interrupt_payload is not None:
                interrupt_reason = str(interrupt_payload.get("reason") or "interrupt")
                break
        if normalized_binding == "mouse_left":
            _perform_mouse_tap_without_move_with_retry(
                app,
                "left",
                hold_ms=tap_hold_ms,
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason="normal",
            )
        else:
            _perform_binding_tap_with_retry(
                app,
                binding,
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason="normal",
            )
        tap_count += 1
        if interrupt_checker is not None:
            interrupt_payload = interrupt_checker()
            if interrupt_payload is not None:
                interrupt_reason = str(interrupt_payload.get("reason") or "interrupt")
                break
        if tap_index + 1 < planned_taps:
            _sleep_ms_uninterruptible(tap_interval_ms, dry_run=dry_run)

    action_trace[-1]["tap_count"] = tap_count
    if interrupt_reason is not None:
        action_trace[-1]["interrupt_reason"] = interrupt_reason
    _raise_if_cancelled()
    return {
        "executed": tap_count > 0,
        "action": "normal",
        "tap_count": tap_count,
        "interrupt_reason": interrupt_reason,
        "interrupt_payload": dict(interrupt_payload or {}),
    }


def _release_inputs(app: Any, profile: Mapping[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    if hasattr(app, "release_all"):
        try:
            app.release_all()
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Combat[release] release_all fallback failed: %s", exc)
    try:
        app.mouse_up("left")
        app.mouse_up("right")
        app.mouse_up("middle")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Combat[release] failed to release mouse button: %s", exc)
    for key in profile.get("release_keys") or []:
        try:
            app.key_up(str(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Combat[release] failed to release key %s: %s", key, exc)


def _sleep_ms(milliseconds: int, *, dry_run: bool) -> None:
    _raise_if_cancelled()
    if dry_run:
        return
    seconds = max(int(milliseconds), 0) / 1000.0
    if seconds > 0:
        _sleep_interruptibly(seconds)


def _sleep_ms_uninterruptible(milliseconds: int, *, dry_run: bool) -> None:
    if dry_run:
        return
    seconds = max(int(milliseconds), 0) / 1000.0
    if seconds > 0:
        time.sleep(seconds)


def _is_available(
    state: Mapping[str, Any],
    name: str,
    *,
    ability_lockouts: Mapping[str, float] | None = None,
    now: float | None = None,
) -> bool:
    if not bool(state.get(f"{name}_available")):
        return False
    if ability_lockouts is None:
        return True
    lock_until = float(ability_lockouts.get(name, 0.0))
    current = now if now is not None else time.monotonic()
    return current >= lock_until


def _lock_ability(
    ability_lockouts: dict[str, float],
    profile: Mapping[str, Any],
    name: str,
    *,
    now: float | None = None,
) -> None:
    runtime = dict(profile.get("runtime") or {})
    lockout_ms = int(runtime.get(f"{name}_lockout_ms") or 0)
    if lockout_ms <= 0:
        return
    current = now if now is not None else time.monotonic()
    ability_lockouts[name] = current + (lockout_ms / 1000.0)


def _resolve_normal_command(profile: Mapping[str, Any], strategy_name: str) -> dict[str, Any]:
    strategies = dict(profile.get("strategies") or {})
    strategy = dict(strategies.get(strategy_name) or {})
    commands = strategy.get("loop") or []
    if isinstance(commands, list):
        for command in commands:
            if isinstance(command, Mapping) and "normal" in command:
                payload = command["normal"] or {}
                if isinstance(payload, Mapping):
                    return {
                        "mode": str(payload.get("mode") or "tap_spam"),
                        "duration_ms": _coerce_int(payload.get("duration_ms"), 560),
                        "interval_ms": _coerce_int(payload.get("interval_ms"), 70),
                    }
                return {
                    "mode": "tap_spam",
                    "duration_ms": _coerce_int(payload, 560),
                    "interval_ms": 70,
                }
    return {"mode": "tap_spam", "duration_ms": 560, "interval_ms": 70}


def _resolve_runtime_schedule(profile: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(profile.get("runtime") or {})
    ability_schedule = dict(profile.get("ability_schedule") or {})
    switch_policy = dict(profile.get("switch_policy") or {})
    input_timing = dict(profile.get("input_timing") or {})
    combat_exit = dict(profile.get("combat_exit") or {})
    enemy_direction_reacquire = dict(profile.get("enemy_direction_reacquire") or {})
    return {
        "monitor_scan_interval_sec": max(float(runtime.get("monitor_scan_interval_sec") or 2.0), 0.1),
        "combat_scan_interval_sec": max(float(runtime.get("combat_scan_interval_sec") or 3.0), 0.1),
        "action_loop_sleep_sec": max(float(runtime.get("action_loop_sleep_ms") or runtime.get("poll_ms") or 80) / 1000.0, 0.01),
        "skill_interval_sec": max(float(ability_schedule.get("skill_interval_sec") or 3.0), 0.1),
        "ultimate_interval_sec": max(float(ability_schedule.get("ultimate_interval_sec") or 5.0), 0.1),
        "immediate_on_combat_enter": bool(ability_schedule.get("immediate_on_combat_enter", True)),
        "switch_interval_sec": max(float(switch_policy.get("switch_interval_sec") or 10.0), 0.1),
        "post_switch_delay_sec": max(int(switch_policy.get("post_switch_delay_ms") or 0) / 1000.0, 0.0),
        "failed_switch_cooldown_sec": max(int(switch_policy.get("failed_switch_cooldown_ms") or 0) / 1000.0, 0.0),
        "switch_confirm_required_matches": max(int(switch_policy.get("confirm_required_matches") or 2), 1),
        "skill_press_ms": max(int(input_timing.get("skill_press_ms") or 45), 0),
        "ultimate_press_ms": max(int(input_timing.get("ultimate_press_ms") or 55), 0),
        "switch_press_ms": max(int(input_timing.get("switch_press_ms") or 35), 0),
        "exit_confirm_required_scans": max(int(combat_exit.get("confirm_required_scans") or 2), 1),
        "exit_confirm_interval_sec": max(float(combat_exit.get("confirm_interval_sec") or 1.0), 0.1),
        "exit_post_cooldown_sec": max(int(combat_exit.get("post_exit_cooldown_ms") or 0) / 1000.0, 0.0),
        "exit_challenge_success_immediate": bool(combat_exit.get("challenge_success_immediate", True)),
        "exit_reset_on_reacquire": bool(combat_exit.get("reset_on_reacquire", True)),
        "rear_enemy_turn_interval_sec": max(float(enemy_direction_reacquire.get("turn_interval_sec") or 0.9), 0.1),
        "rear_enemy_turn_pixels_side": max(int(enemy_direction_reacquire.get("turn_pixels_side") or 220), 1),
        "rear_enemy_turn_pixels_bottom": max(int(enemy_direction_reacquire.get("turn_pixels_bottom") or 340), 1),
        "rear_enemy_vertical_delta": int(enemy_direction_reacquire.get("vertical_delta") or 0),
        "rear_enemy_post_turn_delay_sec": max(
            int(enemy_direction_reacquire.get("post_turn_delay_ms") or 0) / 1000.0,
            0.0,
        ),
        "rear_enemy_auto_target_after_turn": bool(enemy_direction_reacquire.get("auto_target_after_turn", True)),
    }


def _scan_interval_sec(*, combat_active: bool, schedule: Mapping[str, Any]) -> float:
    return float(
        schedule["combat_scan_interval_sec"] if combat_active else schedule["monitor_scan_interval_sec"]
    )


def _enqueue_next_scan(*, now: float, combat_active: bool, schedule: Mapping[str, Any]) -> float:
    return now + _scan_interval_sec(combat_active=combat_active, schedule=schedule)


def _trace_scan(
    combat_state_trace: list[dict[str, Any]],
    *,
    start_time: float,
    state: Mapping[str, Any],
    combat_active: bool,
    phase: str,
    note: str,
    trace_limit: int,
) -> None:
    _append_state_trace(
        combat_state_trace,
        start_time=start_time,
        state=state,
        combat_active=combat_active,
        phase=phase,
        note=note,
        trace_limit=trace_limit,
    )
    logger.info(
        "Combat[state] phase=%s note=%s combat_active=%s in_supported_scene=%s in_combat=%s enemy_health_found=%s enemy_health_count=%s enemy_direction_found=%s enemy_direction_count=%s enemy_direction_side=%s boss_found=%s target_found=%s current_slot=%s skill_state=%s ultimate_state=%s confidence=%s",
        phase,
        note,
        bool(combat_active),
        bool(state.get("in_supported_scene")),
        bool(state.get("in_combat")),
        bool(state.get("enemy_health_found")),
        int(state.get("enemy_health_count") or 0),
        bool(state.get("enemy_direction_found")),
        int(state.get("enemy_direction_count") or 0),
        state.get("enemy_direction_primary_side"),
        bool(state.get("boss_found")),
        bool(state.get("target_found")),
        state.get("current_slot"),
        state.get("skill_state"),
        state.get("ultimate_state"),
        state.get("confidence"),
    )


def _switch_slot(
    app: Any,
    input_mapping: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    target: Any,
    *,
    yihuan_combat: YihuanCombatService,
    profile_name: str,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    press_ms: int = 0,
    fallback_current_slot: int | None = None,
) -> dict[str, Any]:
    team_size = max(_coerce_int(state.get("team_size"), len(profile.get("current_slot_regions") or [])), 1)
    current_slot = state.get("current_slot")
    slot_source = "state"
    if current_slot is None and fallback_current_slot is not None:
        current_slot = fallback_current_slot
        slot_source = "fallback"
    if current_slot is None:
        _record_action(
            action_trace,
            start_time=start_time,
            action="skip_switch",
            dry_run=dry_run,
            details={"reason": "current_slot_unknown", "slot_source": slot_source},
        )
        return {
            "attempted": False,
            "executed": False,
            "reason": "current_slot_unknown",
            "slot": None,
            "state": dict(state),
            "capture_image": None,
        }

    if str(target).lower() == "next":
        slot = (int(current_slot) % team_size) + 1
    else:
        slot = min(max(_coerce_int(target, 1), 1), team_size)

    binding = dict(profile["keys"]).get(f"switch_{slot}", str(slot))
    _record_action(
        action_trace,
        start_time=start_time,
        action="switch",
        dry_run=dry_run,
        details={"binding": binding, "slot": slot, "press_ms": max(int(press_ms), 0), "slot_source": slot_source},
    )
    _tap_input_binding_with_retry(
        app,
        input_mapping,
        binding,
        hold_ms=press_ms,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason="switch",
    )
    action_trace[-1]["slot"] = slot
    if dry_run:
        action_trace[-1]["confirmed"] = True
        return {
            "attempted": True,
            "executed": True,
            "success": True,
            "slot": slot,
            "state": dict(state),
            "capture_image": None,
        }

    confirmed, confirmed_state, confirmed_capture_image = _confirm_switch_slot(
        app,
        yihuan_combat,
        profile_name=profile_name,
        expected_slot=slot,
        confirm_timeout_ms=int(dict(profile.get("switch_policy") or {}).get("confirm_timeout_ms", 0)),
        required_matches=int(dict(profile.get("switch_policy") or {}).get("confirm_required_matches", 1)),
        poll_ms=max(int(dict(profile.get("runtime") or {}).get("poll_ms", 80)), 10),
    )
    action_trace[-1]["confirmed"] = bool(confirmed)
    if not confirmed:
        _record_action(
            action_trace,
            start_time=start_time,
            action="switch_confirm_failed",
            dry_run=dry_run,
            details={"slot": slot},
        )
    return {
        "attempted": True,
        "executed": bool(confirmed),
        "success": bool(confirmed),
        "slot": slot,
        "state": dict(confirmed_state or state),
        "capture_image": confirmed_capture_image,
    }


def _confirm_switch_slot(
    app: Any,
    yihuan_combat: YihuanCombatService,
    *,
    profile_name: str,
    expected_slot: int,
    confirm_timeout_ms: int,
    required_matches: int,
    poll_ms: int,
) -> tuple[bool, dict[str, Any] | None, Any | None]:
    if confirm_timeout_ms <= 0:
        state, capture = _capture_state(app, yihuan_combat, profile_name=profile_name)
        return state.get("current_slot") == expected_slot, state, capture.image

    attempts = max(1, int((max(confirm_timeout_ms, 1) + max(poll_ms, 1) - 1) / max(poll_ms, 1)))
    last_state: dict[str, Any] | None = None
    last_capture_image: Any = None
    consecutive_matches = 0
    required_match_count = max(int(required_matches), 1)
    for attempt in range(attempts):
        state, capture = _capture_state(app, yihuan_combat, profile_name=profile_name)
        last_state = dict(state)
        last_capture_image = capture.image
        if state.get("current_slot") == expected_slot:
            consecutive_matches += 1
            if consecutive_matches >= required_match_count:
                return True, last_state, last_capture_image
        else:
            consecutive_matches = 0
        if attempt + 1 < attempts:
            _sleep_ms(poll_ms, dry_run=False)
    return False, last_state, last_capture_image


def _check_combat_interrupt(
    app: Any,
    yihuan_combat: YihuanCombatService,
    *,
    profile_name: str,
    ability_lockouts: Mapping[str, float],
    audio_dodge_runtime: AudioDodgeRuntime | None,
) -> dict[str, Any] | None:
    if audio_dodge_runtime is not None:
        trigger_event = audio_dodge_runtime.consume_trigger()
        if trigger_event is not None:
            return {"reason": "audio_dodge", "trigger_event": dict(trigger_event)}

    refreshed_state, _capture = _capture_state(app, yihuan_combat, profile_name=profile_name)
    if not bool(refreshed_state.get("in_combat")):
        return {"reason": "combat_ended", "state": dict(refreshed_state)}
    if _is_available(refreshed_state, "ultimate", ability_lockouts=ability_lockouts) or _is_available(
        refreshed_state,
        "skill",
        ability_lockouts=ability_lockouts,
    ):
        return {"reason": "ability_ready", "state": dict(refreshed_state)}
    return None


def _execute_command(
    app: Any,
    input_mapping: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    command: Any,
    *,
    ability_lockouts: dict[str, float],
    yihuan_combat: YihuanCombatService,
    profile_name: str,
    audio_dodge_runtime: AudioDodgeRuntime | None,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    keys = dict(profile["keys"])
    cooldown_ms = int(dict(profile["runtime"])["input_cooldown_ms"])
    input_timing = dict(profile.get("input_timing") or {})

    if isinstance(command, str):
        name = command.strip()
        if name in {"skill", "ultimate"}:
            _record_action(
                action_trace,
                start_time=start_time,
                action=name,
                dry_run=dry_run,
                details={
                    "binding": keys[name],
                    "press_ms": max(int(input_timing.get(f"{name}_press_ms") or 0), 0),
                },
            )
            _tap_input_binding_with_retry(
                app,
                input_mapping,
                keys[name],
                hold_ms=max(int(input_timing.get(f"{name}_press_ms") or 0), 0),
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason=name,
            )
            _lock_ability(ability_lockouts, profile, name)
            _sleep_ms(cooldown_ms, dry_run=dry_run)
            return {"executed": True, "action": name, "used_ability": name}
        if name == "normal":
            result = _normal_attack(
                app,
                keys["normal_attack"],
                mode="tap_spam",
                duration_ms=560,
                interval_ms=70,
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                interrupt_checker=lambda: _check_combat_interrupt(
                    app,
                    yihuan_combat,
                    profile_name=profile_name,
                    ability_lockouts=ability_lockouts,
                    audio_dodge_runtime=audio_dodge_runtime,
                ),
            )
            _sleep_ms(cooldown_ms, dry_run=dry_run)
            return result
        raise ValueError(f"Unsupported combat command: {command!r}")

    if not isinstance(command, Mapping):
        raise ValueError(f"Unsupported combat command type: {type(command).__name__}")

    if "if_available" in command:
        payload = command["if_available"]
        if isinstance(payload, str):
            name = payload
            nested = [name]
        elif isinstance(payload, Mapping):
            name = str(payload.get("name") or "")
            nested_payload = payload.get("then") or [name]
            nested = nested_payload if isinstance(nested_payload, list) else [nested_payload]
        else:
            raise ValueError("if_available command must be a string or mapping.")
        if name not in {"skill", "ultimate"}:
            raise ValueError(f"Unsupported availability target: {name!r}")
        if _is_available(state, name, ability_lockouts=ability_lockouts):
            for nested_command in nested:
                result = _execute_command(
                    app,
                    input_mapping,
                    profile,
                    state,
                    nested_command,
                    ability_lockouts=ability_lockouts,
                    yihuan_combat=yihuan_combat,
                    profile_name=profile_name,
                    audio_dodge_runtime=audio_dodge_runtime,
                    dry_run=dry_run,
                    action_trace=action_trace,
                    start_time=start_time,
                )
                if result.get("executed") or result.get("interrupt_reason"):
                    return result
            return {"executed": False}
        else:
            reason = "not_available"
            if bool(state.get(f"{name}_available")):
                reason = "lockout"
            _record_action(
                action_trace,
                start_time=start_time,
                action=f"skip_{name}",
                dry_run=dry_run,
                details={"reason": reason},
            )
        return {"executed": False, "action": f"skip_{name}", "skip_reason": reason}

    if "normal" in command:
        payload = command["normal"] or {}
        duration_ms = 560
        interval_ms = 70
        mode = "tap_spam"
        if isinstance(payload, Mapping):
            duration_ms = _coerce_int(payload.get("duration_ms"), 560)
            interval_ms = _coerce_int(payload.get("interval_ms"), 70)
            mode = str(payload.get("mode") or "tap_spam")
        elif payload is not None:
            duration_ms = _coerce_int(payload, 560)
        result = _normal_attack(
            app,
            keys["normal_attack"],
            mode=mode,
            duration_ms=duration_ms,
            interval_ms=interval_ms,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            interrupt_checker=lambda: _check_combat_interrupt(
                app,
                yihuan_combat,
                profile_name=profile_name,
                ability_lockouts=ability_lockouts,
                audio_dodge_runtime=audio_dodge_runtime,
            ),
        )
        _sleep_ms(cooldown_ms, dry_run=dry_run)
        return result

    if "keypress" in command:
        payload = command["keypress"]
        binding = payload.get("key") if isinstance(payload, Mapping) else payload
        _tap_binding(
            app,
            str(binding),
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            action_name="keypress",
        )
        _sleep_ms(cooldown_ms, dry_run=dry_run)
        return {"executed": True, "action": "keypress"}

    if "wait" in command:
        payload = command["wait"]
        duration_ms = payload.get("duration_ms") if isinstance(payload, Mapping) else payload
        duration_ms = _coerce_int(duration_ms, 300)
        _record_action(
            action_trace,
            start_time=start_time,
            action="wait",
            dry_run=dry_run,
            details={"duration_ms": duration_ms},
        )
        _sleep_ms(duration_ms, dry_run=dry_run)
        return {"executed": True, "action": "wait"}

    if "switch" in command:
        result = _switch_slot(
            app,
            input_mapping,
            profile,
            state,
            command["switch"],
            yihuan_combat=yihuan_combat,
            profile_name=profile_name,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            press_ms=max(int(input_timing.get("switch_press_ms") or 0), 0),
        )
        _sleep_ms(int(dict(profile["runtime"]).get("input_cooldown_ms", 120)), dry_run=dry_run)
        return {
            "executed": bool(result.get("executed")),
            "action": "switch",
            "state": dict(result.get("state") or state),
            "switch_success": bool(result.get("success")),
            "slot": result.get("slot"),
        }

    raise ValueError(f"Unsupported combat command: {command!r}")


def _execute_strategy(
    app: Any,
    input_mapping: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    strategy_name: str,
    ability_lockouts: dict[str, float],
    yihuan_combat: YihuanCombatService,
    profile_name: str,
    audio_dodge_runtime: AudioDodgeRuntime | None,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    strategies = dict(profile.get("strategies") or {})
    strategy = dict(strategies.get(strategy_name) or {})
    if not strategy:
        raise ValueError(f"Combat strategy not found: {strategy_name}")
    commands = strategy.get("loop") or []
    if not isinstance(commands, list):
        raise ValueError(f"Combat strategy '{strategy_name}' loop must be a list.")
    for command in commands:
        result = _execute_command(
            app,
            input_mapping,
            profile,
            state,
            command,
            ability_lockouts=ability_lockouts,
            yihuan_combat=yihuan_combat,
            profile_name=profile_name,
            audio_dodge_runtime=audio_dodge_runtime,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
        )
        if result.get("interrupt_reason"):
            return result
        if result.get("executed"):
            return result
    return {"executed": False, "action": None}


def _pause_after_encounter(
    app: Any,
    profile: Mapping[str, Any],
    *,
    dry_run: bool,
    cooldown_ms: int,
) -> None:
    _release_inputs(app, profile, dry_run=dry_run)
    _sleep_ms(cooldown_ms, dry_run=dry_run)


def _consume_audio_dodge_trigger(
    audio_dodge_runtime: AudioDodgeRuntime | None,
    *,
    combat_active: bool,
) -> dict[str, Any] | None:
    if audio_dodge_runtime is None or not combat_active:
        return None
    return audio_dodge_runtime.consume_trigger()


def _schedule_next_due(now: float, interval_sec: float) -> float:
    return float(now) + max(float(interval_sec), 0.0)


def _press_scheduled_ability(
    app: Any,
    input_mapping: Any,
    profile: Mapping[str, Any],
    *,
    ability_name: str,
    action_trace: list[dict[str, Any]],
    start_time: float,
    dry_run: bool,
    press_ms: int = 0,
) -> None:
    binding = dict(profile["keys"])[ability_name]
    _record_action(
        action_trace,
        start_time=start_time,
        action=ability_name,
        dry_run=dry_run,
        details={"binding": binding, "press_ms": max(int(press_ms), 0)},
    )
    _tap_input_binding_with_retry(
        app,
        input_mapping,
        binding,
        hold_ms=press_ms,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=ability_name,
    )
    _sleep_ms(int(dict(profile["runtime"]).get("input_cooldown_ms", 120)), dry_run=dry_run)


def _execute_audio_dodge(
    app: Any,
    profile: Mapping[str, Any],
    *,
    trigger_event: Mapping[str, Any],
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> float:
    payload = dict(profile.get("audio_dodge") or {})
    _record_action(
        action_trace,
        start_time=start_time,
        action="audio_dodge",
        dry_run=dry_run,
        details={"score": round(float(trigger_event.get("score") or 0.0), 5)},
    )
    if not dry_run:
        mouse_button = str(payload.get("dodge_mouse_button") or "right")
        dodge_key = str(payload.get("dodge_key") or "shift")
        _call_with_focus_retry(
            app,
            lambda: app.mouse_down(mouse_button),
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            retry_reason="audio_dodge",
        )
        try:
            _sleep_ms_uninterruptible(int(payload.get("right_hold_ms") or 0), dry_run=dry_run)
        finally:
            _call_with_focus_retry(
                app,
                lambda: app.mouse_up(mouse_button),
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason="audio_dodge",
            )
        _sleep_ms_uninterruptible(int(payload.get("post_right_delay_ms") or 0), dry_run=dry_run)
        _call_with_focus_retry(
            app,
            lambda: app.key_down(dodge_key),
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            retry_reason="audio_dodge",
        )
        try:
            _sleep_ms_uninterruptible(int(payload.get("shift_hold_ms") or 0), dry_run=dry_run)
        finally:
            _call_with_focus_retry(
                app,
                lambda: app.key_up(dodge_key),
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason="audio_dodge",
            )
    return max(int(payload.get("dodge_pause_ms") or 0), 0) / 1000.0


def _turn_to_rear_enemy(
    app: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    auto_target_enabled: bool,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    markers = list(state.get("enemy_direction_markers") or [])
    if not markers:
        _record_action(
            action_trace,
            start_time=start_time,
            action="skip_rear_enemy_turn",
            dry_run=dry_run,
            details={"reason": "no_markers"},
        )
        return {"executed": False, "auto_targeted": False, "turn_dx": 0, "turn_dy": 0}

    capture_size = list(state.get("capture_size") or [1280, 720])
    frame_width = max(_coerce_int(capture_size[0], 1280), 1)
    primary_side = str(state.get("enemy_direction_primary_side") or "").strip().lower()
    reacquire = dict(profile.get("enemy_direction_reacquire") or {})
    side_turn = max(_coerce_int(reacquire.get("turn_pixels_side"), 220), 1)
    bottom_turn = max(_coerce_int(reacquire.get("turn_pixels_bottom"), 340), 1)
    vertical_delta = _coerce_int(reacquire.get("vertical_delta"), 0)
    auto_target_after_turn = auto_target_enabled and bool(reacquire.get("auto_target_after_turn", True))

    avg_center_x = 0.5
    if markers:
        centers = [
            (float(_coerce_int(marker.get("x"), 0)) + float(_coerce_int(marker.get("width"), 0)) / 2.0)
            / float(frame_width)
            for marker in markers
            if isinstance(marker, Mapping)
        ]
        if centers:
            avg_center_x = sum(centers) / float(len(centers))

    if primary_side == "left":
        turn_dx = -side_turn
    elif primary_side == "right":
        turn_dx = side_turn
    elif primary_side == "bottom":
        turn_dx = -bottom_turn if avg_center_x < 0.45 else bottom_turn
    else:
        turn_dx = -side_turn if avg_center_x < 0.5 else side_turn

    _record_action(
        action_trace,
        start_time=start_time,
        action="turn_to_rear_enemy",
        dry_run=dry_run,
        details={
            "primary_side": primary_side or None,
            "marker_count": len(markers),
            "turn_dx": int(turn_dx),
            "turn_dy": int(vertical_delta),
            "avg_center_x": round(float(avg_center_x), 3),
        },
    )
    _perform_look_delta_with_retry(
        app,
        int(turn_dx),
        int(vertical_delta),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason="turn_to_rear_enemy",
    )

    post_turn_delay_ms = max(_coerce_int(reacquire.get("post_turn_delay_ms"), 0), 0)
    if post_turn_delay_ms > 0:
        _sleep_ms(post_turn_delay_ms, dry_run=dry_run)

    auto_targeted = False
    if auto_target_after_turn:
        _tap_binding(
            app,
            str(dict(profile["keys"]).get("target") or "mouse_middle"),
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            action_name="auto_target_after_turn",
        )
        auto_targeted = True

    return {
        "executed": True,
        "auto_targeted": auto_targeted,
        "turn_dx": int(turn_dx),
        "turn_dy": int(vertical_delta),
    }


@action_info(
    name="yihuan_combat_run_session",
    public=False,
    read_only=False,
    description="Run the Yihuan-specific automatic combat loop.",
)
@requires_services(
    app="plans/aura_base/app",
    input_mapping="plans/aura_base/input_mapping",
    yihuan_combat="yihuan_combat",
)
def yihuan_combat_run_session(
    app: Any,
    input_mapping: Any,
    yihuan_combat: YihuanCombatService,
    profile_name: str = "default_1280x720_cn",
    strategy_name: str = "default",
    max_seconds: float | int | str = 0,
    max_encounters: int | str = 0,
    auto_target: bool | str = True,
    auto_dodge: bool | str = True,
    dry_run: bool | str = False,
    debug_enabled: bool | str = False,
    capture_debug_enabled: bool | str = False,
    capture_interval_sec: float | int | str = 2.0,
    capture_max_images: int | str = 120,
    capture_raw_enabled: bool | str = False,
) -> dict[str, Any]:
    start_time = time.monotonic()
    combat_state_trace: list[dict[str, Any]] = []
    action_trace: list[dict[str, Any]] = []
    last_state: dict[str, Any] | None = None
    encounters_completed = 0
    profile: dict[str, Any] | None = None
    current_phase = "monitor"
    ability_lockouts: dict[str, float] = {"skill": 0.0, "ultimate": 0.0}
    audio_dodge_runtime: AudioDodgeRuntime | None = None
    capture_debug: _CombatCaptureLogger | None = None
    dodge_pause_until = 0.0
    last_capture_image: Any = None
    initial_monitor_captured = False
    last_known_slot: int | None = None
    last_rear_turn_at = 0.0

    try:
        profile = yihuan_combat.load_profile(profile_name)
        resolved_profile = str(profile["profile_name"])
        runtime = dict(profile["runtime"])
        dry_run_enabled = _coerce_bool(dry_run)
        auto_target_enabled = _coerce_bool(auto_target)
        auto_dodge_enabled = _coerce_bool(auto_dodge)
        _ = _coerce_bool(debug_enabled)
        capture_debug_enabled_resolved = _coerce_bool(capture_debug_enabled)
        capture_interval_sec_resolved = max(_coerce_float(capture_interval_sec, 2.0), 0.1)
        capture_max_images_resolved = max(_coerce_int(capture_max_images, 120), 1)
        capture_raw_enabled_resolved = _coerce_bool(capture_raw_enabled)
        normal_command = _resolve_normal_command(profile, str(strategy_name))
        schedule = _resolve_runtime_schedule(profile)
        duration_limit = max(_coerce_float(max_seconds, 0.0), 0.0)
        if dry_run_enabled and duration_limit <= 0:
            duration_limit = float(runtime["dry_run_max_seconds"])
        deadline = time.monotonic() + duration_limit if duration_limit > 0 else None
        encounter_limit = max(_coerce_int(max_encounters, 0), 0)
        poll_ms = max(int(runtime["poll_ms"]), 10)
        unsupported_required = int(runtime["unsupported_scene_stable_frames"])
        trace_limit = int(runtime["trace_limit"])
        post_exit_cooldown_ms = int(round(float(schedule["exit_post_cooldown_sec"]) * 1000.0))
        target_retry_interval_sec = max(float(runtime.get("target_retry_interval_ms") or 500) / 1000.0, 0.0)

        logger.info(
            "Combat[session] start profile=%s strategy=%s max_seconds=%.3f max_encounters=%s auto_target=%s auto_dodge=%s dry_run=%s",
            resolved_profile,
            strategy_name,
            duration_limit,
            encounter_limit,
            auto_target_enabled,
            auto_dodge_enabled,
            dry_run_enabled,
        )
        logger.info(
            "Combat[config] enemy_health_search_region=%s enemy_direction_regions=%s skill_press_ms=%s ultimate_press_ms=%s switch_press_ms=%s monitor_scan_interval_sec=%.2f combat_scan_interval_sec=%.2f switch_interval_sec=%.2f switch_confirm_required_matches=%s failed_switch_cooldown_sec=%.2f rear_enemy_turn_interval_sec=%.2f exit_confirm_required_scans=%s exit_confirm_interval_sec=%.2f exit_post_cooldown_ms=%s",
            dict(profile.get("regions") or {}).get("enemy_health_search_region"),
            {
                "left": dict(profile.get("regions") or {}).get("enemy_direction_left"),
                "right": dict(profile.get("regions") or {}).get("enemy_direction_right"),
                "bottom": dict(profile.get("regions") or {}).get("enemy_direction_bottom"),
            },
            schedule["skill_press_ms"],
            schedule["ultimate_press_ms"],
            schedule["switch_press_ms"],
            float(schedule["monitor_scan_interval_sec"]),
            float(schedule["combat_scan_interval_sec"]),
            float(schedule["switch_interval_sec"]),
            int(schedule["switch_confirm_required_matches"]),
            float(schedule["failed_switch_cooldown_sec"]),
            float(schedule["rear_enemy_turn_interval_sec"]),
            int(schedule["exit_confirm_required_scans"]),
            float(schedule["exit_confirm_interval_sec"]),
            post_exit_cooldown_ms,
        )
        logger.info(
            "Combat[capture] enabled=%s interval_sec=%.2f max_images=%s raw_enabled=%s",
            capture_debug_enabled_resolved,
            capture_interval_sec_resolved,
            capture_max_images_resolved,
            capture_raw_enabled_resolved,
        )

        if not dry_run_enabled:
            _try_focus_activation(
                app,
                dry_run=dry_run_enabled,
                action_trace=action_trace,
                start_time=start_time,
                reason="session_start",
            )

        audio_dodge_runtime = AudioDodgeRuntime.from_profile(profile, enabled=auto_dodge_enabled)
        audio_dodge_runtime.start()
        capture_debug = _CombatCaptureLogger(
            enabled=capture_debug_enabled_resolved,
            interval_sec=capture_interval_sec_resolved,
            max_images=capture_max_images_resolved,
            raw_enabled=capture_raw_enabled_resolved,
            profile_name=resolved_profile,
            yihuan_combat=yihuan_combat,
        )
        capture_debug.start(start_time=start_time)

        combat_active = False
        exit_pending = False
        exit_pending_scans = 0
        unsupported_stable = 0
        last_target_attempt = 0.0
        next_scan_at = time.monotonic()
        next_skill_at = float("inf")
        next_ultimate_at = float("inf")
        next_switch_at = float("inf")
        next_exit_confirm_at = float("inf")

        while True:
            _raise_if_cancelled()
            if deadline is not None and time.monotonic() >= deadline:
                stopped_reason = "dry_run_completed" if dry_run_enabled else "max_seconds"
                return _final_result(
                    status="success" if dry_run_enabled else "partial",
                    stopped_reason=stopped_reason,
                    failure_reason=None,
                    profile_name=resolved_profile,
                    strategy_name=strategy_name,
                    encounters_completed=encounters_completed,
                    current_phase=current_phase,
                    last_state=last_state,
                    combat_state_trace=combat_state_trace,
                    action_trace=action_trace,
                    start_time=start_time,
                    dry_run=dry_run_enabled,
                    capture_debug=capture_debug,
                )

            now = time.monotonic()

            trigger_event = _consume_audio_dodge_trigger(audio_dodge_runtime, combat_active=combat_active)
            if trigger_event is not None:
                current_phase = "audio_dodge"
                dodge_pause_sec = _execute_audio_dodge(
                    app,
                    profile,
                    trigger_event=trigger_event,
                    dry_run=dry_run_enabled,
                    action_trace=action_trace,
                    start_time=start_time,
                )
                dodge_pause_until = max(dodge_pause_until, time.monotonic() + dodge_pause_sec)
                if last_state is not None:
                    _trace_scan(
                        combat_state_trace,
                        start_time=start_time,
                        state=last_state,
                        combat_active=combat_active,
                        phase=current_phase,
                        note="dodge_executed",
                        trace_limit=trace_limit,
                    )
                continue

            if last_state is None or now >= next_scan_at:
                state, _capture = _capture_state(app, yihuan_combat, profile_name=resolved_profile)
                last_state = dict(state)
                if state.get("current_slot") is not None:
                    last_known_slot = _coerce_int(state.get("current_slot"), 0) or last_known_slot
                last_capture_image = _capture.image
                current_phase = "combat" if combat_active else "monitor"
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="combat_scan" if combat_active else "monitor_scan",
                    trace_limit=trace_limit,
                )
                if not initial_monitor_captured:
                    _capture_event_image(
                        capture_debug,
                        label="initial_monitor",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=combat_active,
                    )
                    initial_monitor_captured = True
                _capture_periodic_if_due(
                    app,
                    capture_debug,
                    state=state,
                    phase=current_phase,
                    start_time=start_time,
                    combat_active=combat_active,
                )

                if bool(state.get("in_supported_scene")):
                    unsupported_stable = 0
                else:
                    unsupported_stable += 1
                    if unsupported_stable >= unsupported_required:
                        current_phase = "monitor"
                        return _final_result(
                            status="partial",
                            stopped_reason="not_in_supported_scene",
                            failure_reason=None,
                            profile_name=resolved_profile,
                            strategy_name=strategy_name,
                            encounters_completed=encounters_completed,
                            current_phase=current_phase,
                            last_state=last_state,
                            combat_state_trace=combat_state_trace,
                            action_trace=action_trace,
                            start_time=start_time,
                            dry_run=dry_run_enabled,
                            capture_debug=capture_debug,
                        )

                if combat_active:
                    if bool(state.get("challenge_success_found")) and bool(schedule["exit_challenge_success_immediate"]):
                        current_phase = "post_combat"
                        _capture_event_image(
                            capture_debug,
                            label="challenge_success",
                            phase=current_phase,
                            source_image=last_capture_image,
                            state=state,
                            start_time=start_time,
                            combat_active=False,
                        )
                        _record_action(
                            action_trace,
                            start_time=start_time,
                            action="challenge_success",
                            dry_run=dry_run_enabled,
                        )
                        combat_active = False
                        exit_pending = False
                        exit_pending_scans = 0
                        next_exit_confirm_at = float("inf")
                        encounters_completed += 1
                        next_skill_at = float("inf")
                        next_ultimate_at = float("inf")
                        next_switch_at = float("inf")
                        _trace_scan(
                            combat_state_trace,
                            start_time=start_time,
                            state=state,
                            combat_active=False,
                            phase=current_phase,
                            note="challenge_success",
                            trace_limit=trace_limit,
                        )
                        _pause_after_encounter(
                            app,
                            profile,
                            dry_run=dry_run_enabled,
                            cooldown_ms=post_exit_cooldown_ms,
                        )
                        if encounter_limit > 0 and encounters_completed >= encounter_limit:
                            return _final_result(
                                status="success",
                                stopped_reason="max_encounters",
                                failure_reason=None,
                                profile_name=resolved_profile,
                                strategy_name=strategy_name,
                                encounters_completed=encounters_completed,
                                current_phase=current_phase,
                                last_state=last_state,
                                combat_state_trace=combat_state_trace,
                                action_trace=action_trace,
                                start_time=start_time,
                                dry_run=dry_run_enabled,
                                capture_debug=capture_debug,
                            )
                        current_phase = "monitor"
                        next_scan_at = _enqueue_next_scan(
                            now=time.monotonic(),
                            combat_active=False,
                            schedule=schedule,
                        )
                        continue

                    if bool(state.get("in_combat")):
                        if exit_pending and bool(schedule["exit_reset_on_reacquire"]):
                            exit_pending = False
                            exit_pending_scans = 0
                            next_exit_confirm_at = float("inf")
                            _record_action(
                                action_trace,
                                start_time=start_time,
                                action="exit_pending_reset_on_reacquire",
                                dry_run=dry_run_enabled,
                                details={
                                    "required_scans": int(schedule["exit_confirm_required_scans"]),
                                    "target_found": bool(state.get("target_found")),
                                },
                            )
                            _trace_scan(
                                combat_state_trace,
                                start_time=start_time,
                                state=state,
                                combat_active=True,
                                phase="combat",
                                note="exit_pending_reset_on_reacquire",
                                trace_limit=trace_limit,
                            )
                    elif bool(state.get("enemy_direction_found")):
                        exit_pending = False
                        exit_pending_scans = 0
                        next_exit_confirm_at = float("inf")
                        current_phase = "reacquire_target"
                        _record_action(
                            action_trace,
                            start_time=start_time,
                            action="rear_enemy_hold_combat",
                            dry_run=dry_run_enabled,
                            details={
                                "marker_count": int(state.get("enemy_direction_count") or 0),
                                "primary_side": state.get("enemy_direction_primary_side"),
                            },
                        )
                        _trace_scan(
                            combat_state_trace,
                            start_time=start_time,
                            state=state,
                            combat_active=True,
                            phase=current_phase,
                            note="rear_enemy_hold_combat",
                            trace_limit=trace_limit,
                        )
                    else:
                        current_phase = "exit_pending"
                        if not exit_pending:
                            exit_pending = True
                            exit_pending_scans = 1
                            next_exit_confirm_at = time.monotonic() + float(schedule["exit_confirm_interval_sec"])
                            _capture_event_image(
                                capture_debug,
                                label="exit_pending_start",
                                phase=current_phase,
                                source_image=last_capture_image,
                                state=state,
                                start_time=start_time,
                                combat_active=True,
                            )
                            _record_action(
                                action_trace,
                                start_time=start_time,
                                action="exit_pending_start",
                                dry_run=dry_run_enabled,
                                details={
                                    "scan_count": exit_pending_scans,
                                    "required_scans": int(schedule["exit_confirm_required_scans"]),
                                    "confirm_interval_sec": float(schedule["exit_confirm_interval_sec"]),
                                },
                            )
                            _trace_scan(
                                combat_state_trace,
                                start_time=start_time,
                                state=state,
                                combat_active=True,
                                phase=current_phase,
                                note="exit_pending_start",
                                trace_limit=trace_limit,
                            )
                            _release_inputs(app, profile, dry_run=dry_run_enabled)
                            next_scan_at = next_exit_confirm_at
                            continue

                        exit_pending_scans += 1
                        _capture_event_image(
                            capture_debug,
                            label="exit_pending_scan",
                            phase=current_phase,
                            source_image=last_capture_image,
                            state=state,
                            start_time=start_time,
                            combat_active=True,
                        )
                        _record_action(
                            action_trace,
                            start_time=start_time,
                            action="exit_pending_scan",
                            dry_run=dry_run_enabled,
                            details={
                                "scan_count": exit_pending_scans,
                                "required_scans": int(schedule["exit_confirm_required_scans"]),
                            },
                        )
                        _trace_scan(
                            combat_state_trace,
                            start_time=start_time,
                            state=state,
                            combat_active=True,
                            phase=current_phase,
                            note="exit_pending_scan",
                            trace_limit=trace_limit,
                        )
                        if exit_pending_scans < int(schedule["exit_confirm_required_scans"]):
                            next_exit_confirm_at = time.monotonic() + float(schedule["exit_confirm_interval_sec"])
                            next_scan_at = next_exit_confirm_at
                            continue

                        current_phase = "post_combat"
                        _capture_event_image(
                            capture_debug,
                            label="exit_combat_confirmed",
                            phase=current_phase,
                            source_image=last_capture_image,
                            state=state,
                            start_time=start_time,
                            combat_active=False,
                        )
                        _record_action(
                            action_trace,
                            start_time=start_time,
                            action="exit_combat_confirmed",
                            dry_run=dry_run_enabled,
                            details={"scan_count": exit_pending_scans},
                        )
                        combat_active = False
                        exit_pending = False
                        exit_pending_scans = 0
                        next_exit_confirm_at = float("inf")
                        encounters_completed += 1
                        next_skill_at = float("inf")
                        next_ultimate_at = float("inf")
                        next_switch_at = float("inf")
                        _trace_scan(
                            combat_state_trace,
                            start_time=start_time,
                            state=state,
                            combat_active=False,
                            phase=current_phase,
                            note="exit_combat_confirmed",
                            trace_limit=trace_limit,
                        )
                        _pause_after_encounter(
                            app,
                            profile,
                            dry_run=dry_run_enabled,
                            cooldown_ms=post_exit_cooldown_ms,
                        )
                        if encounter_limit > 0 and encounters_completed >= encounter_limit:
                            return _final_result(
                                status="success",
                                stopped_reason="max_encounters",
                                failure_reason=None,
                                profile_name=resolved_profile,
                                strategy_name=strategy_name,
                                encounters_completed=encounters_completed,
                                current_phase=current_phase,
                                last_state=last_state,
                                combat_state_trace=combat_state_trace,
                                action_trace=action_trace,
                                start_time=start_time,
                                dry_run=dry_run_enabled,
                                capture_debug=capture_debug,
                            )
                        current_phase = "monitor"
                        next_scan_at = _enqueue_next_scan(
                            now=time.monotonic(),
                            combat_active=False,
                            schedule=schedule,
                        )
                        continue

                elif bool(state.get("in_combat")):
                    combat_active = True
                    exit_pending = False
                    exit_pending_scans = 0
                    next_exit_confirm_at = float("inf")
                    current_phase = "enemy_detected"
                    _trace_scan(
                        combat_state_trace,
                        start_time=start_time,
                        state=state,
                        combat_active=True,
                        phase=current_phase,
                        note="enter_combat",
                        trace_limit=trace_limit,
                    )
                    _capture_event_image(
                        capture_debug,
                        label="enter_combat",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=True,
                    )
                    if auto_target_enabled and not bool(state.get("target_found")):
                        current_phase = "acquire_target"
                        _capture_event_image(
                            capture_debug,
                            label="auto_target_attempt",
                            phase=current_phase,
                            source_image=last_capture_image,
                            state=state,
                            start_time=start_time,
                            combat_active=True,
                        )
                        _tap_binding(
                            app,
                            dict(profile["keys"])["target"],
                            dry_run=dry_run_enabled,
                            action_trace=action_trace,
                            start_time=start_time,
                            action_name="auto_target",
                        )
                        last_target_attempt = time.monotonic()
                    if bool(schedule["immediate_on_combat_enter"]):
                        next_skill_at = time.monotonic()
                        next_ultimate_at = time.monotonic()
                    else:
                        next_skill_at = _schedule_next_due(time.monotonic(), float(schedule["skill_interval_sec"]))
                        next_ultimate_at = _schedule_next_due(time.monotonic(), float(schedule["ultimate_interval_sec"]))
                    next_switch_at = _schedule_next_due(time.monotonic(), float(schedule["switch_interval_sec"]))
                    next_scan_at = _enqueue_next_scan(
                        now=time.monotonic(),
                        combat_active=True,
                        schedule=schedule,
                    )
                    continue

                next_scan_at = _enqueue_next_scan(
                    now=time.monotonic(),
                    combat_active=combat_active,
                    schedule=schedule,
                )

            if not combat_active or last_state is None:
                current_phase = "monitor"
                _capture_periodic_if_due(
                    app,
                    capture_debug,
                    state=last_state,
                    phase=current_phase,
                    start_time=start_time,
                    combat_active=False,
                )
                remaining = max(next_scan_at - time.monotonic(), 0.0)
                _sleep_interruptibly(min(float(schedule["action_loop_sleep_sec"]), remaining if remaining > 0 else float(schedule["action_loop_sleep_sec"])))
                continue

            if time.monotonic() < dodge_pause_until:
                current_phase = "combat"
                _capture_periodic_if_due(
                    app,
                    capture_debug,
                    state=last_state,
                    phase=current_phase,
                    start_time=start_time,
                    combat_active=True,
                )
                _sleep_interruptibly(min(float(schedule["action_loop_sleep_sec"]), max(dodge_pause_until - time.monotonic(), 0.0)))
                continue

            if exit_pending:
                current_phase = "exit_pending"
                _capture_periodic_if_due(
                    app,
                    capture_debug,
                    state=last_state,
                    phase=current_phase,
                    start_time=start_time,
                    combat_active=True,
                )
                remaining = max(next_scan_at - time.monotonic(), 0.0)
                _sleep_interruptibly(
                    min(
                        float(schedule["action_loop_sleep_sec"]),
                        remaining if remaining > 0 else float(schedule["action_loop_sleep_sec"]),
                    )
                )
                continue

            state = dict(last_state)
            current_phase = "combat"

            if bool(state.get("enemy_direction_found")) and not bool(state.get("in_combat")):
                current_phase = "reacquire_target"
                if time.monotonic() - last_rear_turn_at >= float(schedule["rear_enemy_turn_interval_sec"]):
                    _capture_event_image(
                        capture_debug,
                        label="auto_target_attempt_after_turn",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=True,
                    )
                    rear_turn_result = _turn_to_rear_enemy(
                        app,
                        profile,
                        state,
                        auto_target_enabled=auto_target_enabled,
                        dry_run=dry_run_enabled,
                        action_trace=action_trace,
                        start_time=start_time,
                    )
                    last_rear_turn_at = time.monotonic()
                    if bool(rear_turn_result.get("auto_targeted")):
                        last_target_attempt = time.monotonic()
                    _trace_scan(
                        combat_state_trace,
                        start_time=start_time,
                        state=state,
                        combat_active=True,
                        phase=current_phase,
                        note="rear_enemy_turn",
                        trace_limit=trace_limit,
                    )
                _capture_periodic_if_due(
                    app,
                    capture_debug,
                    state=state,
                    phase=current_phase,
                    start_time=start_time,
                    combat_active=True,
                )
                remaining = max(next_scan_at - time.monotonic(), 0.0)
                _sleep_interruptibly(
                    min(
                        float(schedule["action_loop_sleep_sec"]),
                        remaining if remaining > 0 else float(schedule["action_loop_sleep_sec"]),
                    )
                )
                continue

            if (
                auto_target_enabled
                and not bool(state.get("target_found"))
                and time.monotonic() - last_target_attempt >= target_retry_interval_sec
            ):
                current_phase = "acquire_target"
                _capture_event_image(
                    capture_debug,
                    label="auto_target_attempt",
                    phase=current_phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=True,
                )
                _tap_binding(
                    app,
                    dict(profile["keys"])["target"],
                    dry_run=dry_run_enabled,
                    action_trace=action_trace,
                    start_time=start_time,
                    action_name="auto_target",
                )
                last_target_attempt = time.monotonic()
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="auto_target_attempt",
                    trace_limit=trace_limit,
                )

            if time.monotonic() >= next_ultimate_at:
                _capture_event_image(
                    capture_debug,
                    label="ultimate_due",
                    phase=current_phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=True,
                )
                _press_scheduled_ability(
                    app,
                    input_mapping,
                    profile,
                    ability_name="ultimate",
                    action_trace=action_trace,
                    start_time=start_time,
                    dry_run=dry_run_enabled,
                    press_ms=int(schedule["ultimate_press_ms"]),
                )
                next_ultimate_at = _schedule_next_due(time.monotonic(), float(schedule["ultimate_interval_sec"]))
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="ultimate_due",
                    trace_limit=trace_limit,
                )
                continue

            if time.monotonic() >= next_skill_at:
                _capture_event_image(
                    capture_debug,
                    label="skill_due",
                    phase=current_phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=True,
                )
                _press_scheduled_ability(
                    app,
                    input_mapping,
                    profile,
                    ability_name="skill",
                    action_trace=action_trace,
                    start_time=start_time,
                    dry_run=dry_run_enabled,
                    press_ms=int(schedule["skill_press_ms"]),
                )
                next_skill_at = _schedule_next_due(time.monotonic(), float(schedule["skill_interval_sec"]))
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="skill_due",
                    trace_limit=trace_limit,
                )
                continue

            if time.monotonic() >= next_switch_at:
                _capture_event_image(
                    capture_debug,
                    label="switch_attempt",
                    phase=current_phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=True,
                )
                switch_result = _switch_slot(
                    app,
                    input_mapping,
                    profile,
                    state,
                    "next",
                    yihuan_combat=yihuan_combat,
                    profile_name=resolved_profile,
                    dry_run=dry_run_enabled,
                    action_trace=action_trace,
                    start_time=start_time,
                    press_ms=int(schedule["switch_press_ms"]),
                    fallback_current_slot=last_known_slot,
                )
                if switch_result.get("state") is not None:
                    last_state = dict(switch_result.get("state") or {})
                    state = dict(last_state)
                    if state.get("current_slot") is not None:
                        last_known_slot = _coerce_int(state.get("current_slot"), 0) or last_known_slot
                if switch_result.get("capture_image") is not None:
                    last_capture_image = switch_result.get("capture_image")
                if bool(switch_result.get("success")):
                    if switch_result.get("slot") is not None:
                        last_known_slot = _coerce_int(switch_result.get("slot"), 0) or last_known_slot
                    _capture_event_image(
                        capture_debug,
                        label="switch_success",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=True,
                    )
                    next_switch_at = _schedule_next_due(time.monotonic(), float(schedule["switch_interval_sec"]))
                else:
                    _capture_event_image(
                        capture_debug,
                        label="switch_confirm_failed",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=True,
                    )
                    retry_delay_sec = float(schedule["failed_switch_cooldown_sec"]) or float(schedule["switch_interval_sec"])
                    next_switch_at = _schedule_next_due(time.monotonic(), retry_delay_sec)
                    _record_action(
                        action_trace,
                        start_time=start_time,
                        action="switch_retry_scheduled",
                        dry_run=dry_run_enabled,
                        details={"retry_delay_sec": round(float(retry_delay_sec), 3)},
                    )
                    _capture_event_image(
                        capture_debug,
                        label="switch_retry_scheduled",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=True,
                    )
                if bool(switch_result.get("success")) and float(schedule["post_switch_delay_sec"]) > 0:
                    _sleep_interruptibly(float(schedule["post_switch_delay_sec"]))
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="switch_success" if bool(switch_result.get("success")) else "switch_failed",
                    trace_limit=trace_limit,
                )
                continue

            normal_result = _normal_attack(
                app,
                dict(profile["keys"])["normal_attack"],
                mode=str(normal_command["mode"]),
                duration_ms=_coerce_int(normal_command["duration_ms"], 560),
                interval_ms=_coerce_int(normal_command["interval_ms"], 70),
                dry_run=dry_run_enabled,
                action_trace=action_trace,
                start_time=start_time,
                interrupt_checker=lambda: (
                    {
                        "reason": "audio_dodge",
                        "trigger_event": trigger,
                    }
                    if (trigger := _consume_audio_dodge_trigger(audio_dodge_runtime, combat_active=True)) is not None
                    else None
                ),
            )
            interrupt_reason = normal_result.get("interrupt_reason")
            interrupt_payload = dict(normal_result.get("interrupt_payload") or {})
            if interrupt_reason == "audio_dodge":
                current_phase = "audio_dodge"
                _capture_event_image(
                    capture_debug,
                    label="audio_dodge",
                    phase=current_phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=True,
                )
                dodge_pause_sec = _execute_audio_dodge(
                    app,
                    profile,
                    trigger_event=dict(interrupt_payload.get("trigger_event") or {}),
                    dry_run=dry_run_enabled,
                    action_trace=action_trace,
                    start_time=start_time,
                )
                dodge_pause_until = max(dodge_pause_until, time.monotonic() + dodge_pause_sec)
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="dodge_preempted_action",
                    trace_limit=trace_limit,
                )
                continue

            _trace_scan(
                combat_state_trace,
                start_time=start_time,
                state=state,
                combat_active=combat_active,
                phase=current_phase,
                note=f"combat_loop:{normal_result.get('action') or 'idle'}",
                trace_limit=trace_limit,
            )
            _capture_periodic_if_due(
                app,
                capture_debug,
                state=state,
                phase=current_phase,
                start_time=start_time,
                combat_active=True,
            )
            remaining_to_scan = max(next_scan_at - time.monotonic(), 0.0)
            if remaining_to_scan > 0:
                _sleep_interruptibly(min(float(schedule["action_loop_sleep_sec"]), remaining_to_scan))

    except _CombatSessionCancelled:
        logger.info("Combat[session] cancelled phase=%s", current_phase)
        last_capture_image = _resolve_terminal_capture_image(app, fallback_image=last_capture_image, capture_debug=capture_debug)
        _capture_event_image(
            capture_debug,
            label="cancelled",
            phase=current_phase,
            source_image=last_capture_image,
            state=last_state,
            start_time=start_time,
            combat_active=bool((last_state or {}).get("in_combat")),
        )
        _release_inputs(app, profile or {}, dry_run=_coerce_bool(dry_run))
        return _final_result(
            status="cancelled",
            stopped_reason="cancelled",
            failure_reason="cancelled",
            profile_name=str((profile or {}).get("profile_name") or profile_name),
            strategy_name=str(strategy_name),
            encounters_completed=encounters_completed,
            current_phase=current_phase,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
            capture_debug=capture_debug,
        )

    except FileNotFoundError as exc:
        logger.warning("Combat[session] profile failed: %s", exc)
        last_capture_image = _resolve_terminal_capture_image(app, fallback_image=last_capture_image, capture_debug=capture_debug)
        _capture_event_image(
            capture_debug,
            label="failure",
            phase=current_phase,
            source_image=last_capture_image,
            state=last_state,
            start_time=start_time,
            combat_active=bool((last_state or {}).get("in_combat")),
        )
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason="profile_not_found",
            profile_name=str(profile_name),
            strategy_name=str(strategy_name),
            encounters_completed=encounters_completed,
            current_phase=current_phase,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
            capture_debug=capture_debug,
        )
    except TargetRuntimeError as exc:
        failure_reason = "window_focus_failed" if exc.code == "window_focus_required" else str(exc.code or "input_failed")
        logger.warning("Combat[session] runtime failure [%s]: %s", exc.code, exc)
        last_capture_image = _resolve_terminal_capture_image(app, fallback_image=last_capture_image, capture_debug=capture_debug)
        _capture_event_image(
            capture_debug,
            label="failure",
            phase=current_phase,
            source_image=last_capture_image,
            state=last_state,
            start_time=start_time,
            combat_active=bool((last_state or {}).get("in_combat")),
        )
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason=failure_reason,
            profile_name=str(profile_name),
            strategy_name=str(strategy_name),
            encounters_completed=encounters_completed,
            current_phase=current_phase,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
            capture_debug=capture_debug,
        )
    except RuntimeError as exc:
        logger.warning("Combat[session] capture failed: %s", exc)
        last_capture_image = _resolve_terminal_capture_image(app, fallback_image=last_capture_image, capture_debug=capture_debug)
        _capture_event_image(
            capture_debug,
            label="failure",
            phase=current_phase,
            source_image=last_capture_image,
            state=last_state,
            start_time=start_time,
            combat_active=bool((last_state or {}).get("in_combat")),
        )
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason="capture_failed",
            profile_name=str(profile_name),
            strategy_name=str(strategy_name),
            encounters_completed=encounters_completed,
            current_phase=current_phase,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
            capture_debug=capture_debug,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Combat[session] unexpected failure: %s", exc)
        last_capture_image = _resolve_terminal_capture_image(app, fallback_image=last_capture_image, capture_debug=capture_debug)
        _capture_event_image(
            capture_debug,
            label="failure",
            phase=current_phase,
            source_image=last_capture_image,
            state=last_state,
            start_time=start_time,
            combat_active=bool((last_state or {}).get("in_combat")),
        )
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason="exception",
            profile_name=str(profile_name),
            strategy_name=str(strategy_name),
            encounters_completed=encounters_completed,
            current_phase=current_phase,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
            capture_debug=capture_debug,
        )
    finally:
        if audio_dodge_runtime is not None:
            audio_dodge_runtime.stop()
        if profile is not None:
            _release_inputs(app, profile, dry_run=_coerce_bool(dry_run))

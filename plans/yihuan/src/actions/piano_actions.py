"""Piano playback actions for the Yihuan plan."""

from __future__ import annotations

import time
from typing import Any

from packages.aura_core.api import action_info, requires_services
from packages.aura_core.observability.logging.core_logger import logger
from packages.aura_core.scheduler.cancellation import is_current_task_cancel_requested

from ..piano.keymap import PianoConflictPolicy
from ..services.piano_service import YihuanPianoService


class _PianoSessionCancelled(Exception):
    """Internal control-flow marker for cooperative cancellation."""


def _piano_cancel_requested() -> bool:
    try:
        return is_current_task_cancel_requested()
    except Exception:
        return False


def _raise_if_cancelled() -> None:
    if _piano_cancel_requested():
        raise _PianoSessionCancelled()


def _sleep_is_mocked() -> bool:
    return hasattr(time.sleep, "mock_calls")


def _sleep_interruptibly(duration_sec: float, *, quantum_sec: float = 0.02) -> None:
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
        time.sleep(min(remaining, max(float(quantum_sec), 0.005)))


def _coerce_bool(value: Any) -> bool:
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


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _release_all(app: Any) -> None:
    try:
        if hasattr(app, "release_all"):
            app.release_all()
        elif hasattr(app, "controller") and hasattr(app.controller, "release_all"):
            app.controller.release_all()
    except Exception:
        logger.debug("Piano[input] release_all failed during cleanup.", exc_info=True)


def _perform_action(app: Any, action: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    kind = str(action["kind"])
    key = str(action["key"])
    if kind == "modifier_down":
        app.key_down(key)
    elif kind == "modifier_up":
        app.key_up(key)
    elif kind == "key_down":
        app.key_down(key)
    elif kind == "key_up":
        app.key_up(key)
    else:
        raise ValueError(f"Unsupported piano action kind: {kind}")


@action_info(
    name="yihuan_piano_play_midi",
    public=False,
    read_only=False,
    description="Parse a MIDI file and perform it on Yihuan's in-game piano keyboard.",
)
@requires_services(app="plans/aura_base/app", yihuan_piano="yihuan_piano")
def yihuan_piano_play_midi(
    app: Any,
    yihuan_piano: YihuanPianoService,
    file_path: str,
    conflict_policy: str = "strict",
    transpose_semitones: int | str = 0,
    tempo_scale: float | int | str = 1.0,
    start_delay_ms: int | str = 250,
    roll_note_ms: int | str = 35,
    velocity_threshold: int | str = 1,
    focus_window: bool | str = True,
    dry_run: bool | str = False,
) -> dict[str, Any]:
    started_at = time.monotonic()
    dry_run_value = _coerce_bool(dry_run)
    focus_window_value = _coerce_bool(focus_window)
    roll_note_ms_value = max(_coerce_int(roll_note_ms, 35), 1)
    start_delay_ms_value = max(_coerce_int(start_delay_ms, 250), 0)
    velocity_threshold_value = max(_coerce_int(velocity_threshold, 1), 0)
    transpose_value = _coerce_int(transpose_semitones, 0)
    tempo_scale_value = max(_coerce_float(tempo_scale, 1.0), 0.0001)

    try:
        policy = PianoConflictPolicy(str(conflict_policy).strip().lower())
    except ValueError:
        return {
            "status": "failed",
            "stopped_reason": "invalid_argument",
            "failure_reason": "invalid_conflict_policy",
            "file_path": str(file_path),
            "conflict_policy": str(conflict_policy),
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }

    try:
        _raise_if_cancelled()
        parsed = yihuan_piano.parse_midi_file(
            file_path,
            transpose_semitones=transpose_value,
            tempo_scale=tempo_scale_value,
            velocity_threshold=velocity_threshold_value,
        )
    except FileNotFoundError as exc:
        return {
            "status": "failed",
            "stopped_reason": "file_not_found",
            "failure_reason": "file_not_found",
            "message": str(exc),
            "file_path": str(file_path),
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Piano[MIDI] failed to parse file: %s", file_path)
        return {
            "status": "failed",
            "stopped_reason": "parse_failed",
            "failure_reason": type(exc).__name__,
            "message": str(exc),
            "file_path": str(file_path),
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }

    if parsed["unsupported_notes"]:
        return {
            "status": "failed",
            "stopped_reason": "unsupported_note",
            "failure_reason": "unsupported_note",
            "file_path": parsed["file_path"],
            "conflict_policy": policy.value,
            "parsed_summary": {
                "format_type": parsed["format_type"],
                "track_count": parsed["track_count"],
                "division": parsed["division"],
                "note_count": parsed["note_count"],
                "unsupported_note_count": parsed["unsupported_note_count"],
            },
            "unsupported_notes": parsed["unsupported_notes"],
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }

    try:
        plan = yihuan_piano.build_playback_plan(
            parsed["notes"],
            conflict_policy=policy,
            roll_note_ms=roll_note_ms_value,
            start_delay_ms=start_delay_ms_value,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Piano[plan] failed to build playback plan.")
        return {
            "status": "failed",
            "stopped_reason": "plan_failed",
            "failure_reason": type(exc).__name__,
            "message": str(exc),
            "file_path": parsed["file_path"],
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }

    if not plan["ok"]:
        return {
            "status": "failed",
            "stopped_reason": "unplayable_score",
            "failure_reason": plan["failure_reason"],
            "file_path": parsed["file_path"],
            "conflict_policy": policy.value,
            "parsed_summary": {
                "format_type": parsed["format_type"],
                "track_count": parsed["track_count"],
                "division": parsed["division"],
                "note_count": parsed["note_count"],
                "unsupported_note_count": parsed["unsupported_note_count"],
            },
            "conflicts": plan["conflicts"],
            "scheduled_notes": plan["scheduled_notes"],
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }

    action_plan = list(plan["action_plan"])
    performed_actions: list[dict[str, Any]] = []
    released = False

    try:
        if focus_window_value and not dry_run_value:
            focused = bool(app.focus_with_input())
            if not focused:
                focused = bool(app.focus())
            if not focused:
                return {
                    "status": "failed",
                    "stopped_reason": "focus_failed",
                    "failure_reason": "focus_failed",
                    "file_path": parsed["file_path"],
                    "elapsed_sec": round(time.monotonic() - started_at, 3),
                }

        playback_started_at = time.monotonic()
        for action in action_plan:
            _raise_if_cancelled()
            target_time_sec = float(action["t_ms"]) / 1000.0
            elapsed_sec = time.monotonic() - playback_started_at
            if target_time_sec > elapsed_sec:
                _sleep_interruptibly(target_time_sec - elapsed_sec)
            _raise_if_cancelled()
            _perform_action(app, action, dry_run=dry_run_value)
            performed_actions.append(dict(action))

        _release_all(app)
        released = True
    except _PianoSessionCancelled:
        _release_all(app)
        released = True
        return {
            "status": "cancelled",
            "stopped_reason": "cancelled",
            "failure_reason": "cancelled",
            "file_path": parsed["file_path"],
            "conflict_policy": policy.value,
            "dry_run": dry_run_value,
            "parsed_summary": {
                "format_type": parsed["format_type"],
                "track_count": parsed["track_count"],
                "division": parsed["division"],
                "note_count": parsed["note_count"],
                "unsupported_note_count": parsed["unsupported_note_count"],
            },
            "scheduled_note_count": plan["scheduled_note_count"],
            "scheduled_notes": plan["scheduled_notes"],
            "action_plan": action_plan,
            "performed_actions": performed_actions,
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Piano[playback] failed during execution.")
        return {
            "status": "failed",
            "stopped_reason": "playback_failed",
            "failure_reason": type(exc).__name__,
            "message": str(exc),
            "file_path": parsed["file_path"],
            "conflict_policy": policy.value,
            "dry_run": dry_run_value,
            "parsed_summary": {
                "format_type": parsed["format_type"],
                "track_count": parsed["track_count"],
                "division": parsed["division"],
                "note_count": parsed["note_count"],
                "unsupported_note_count": parsed["unsupported_note_count"],
            },
            "scheduled_note_count": plan["scheduled_note_count"],
            "scheduled_notes": plan["scheduled_notes"],
            "action_plan": action_plan,
            "performed_actions": performed_actions,
            "elapsed_sec": round(time.monotonic() - started_at, 3),
        }
    finally:
        if not released:
            _release_all(app)

    return {
        "status": "success",
        "stopped_reason": "completed",
        "failure_reason": None,
        "file_path": parsed["file_path"],
        "conflict_policy": policy.value,
        "dry_run": dry_run_value,
        "parsed_summary": {
            "format_type": parsed["format_type"],
            "track_count": parsed["track_count"],
            "division": parsed["division"],
            "note_count": parsed["note_count"],
            "unsupported_note_count": parsed["unsupported_note_count"],
        },
        "scheduled_note_count": plan["scheduled_note_count"],
        "scheduled_notes": plan["scheduled_notes"],
        "action_plan": action_plan,
        "performed_actions": performed_actions,
        "elapsed_sec": round(time.monotonic() - started_at, 3),
    }

"""Rhythm mini-game actions for the Yihuan plan."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import cv2

from packages.aura_core.api import action_info, requires_services
from packages.aura_core.context.plan import current_plan_name
from packages.aura_core.observability.logging.core_logger import logger
from packages.aura_core.scheduler.cancellation import is_current_task_cancel_requested

from ..services.rhythm_service import YihuanRhythmService


class _RhythmSessionCancelled(Exception):
    """Internal marker used to stop rhythm actions cooperatively."""


def _rhythm_cancel_requested() -> bool:
    try:
        return is_current_task_cancel_requested()
    except Exception:
        return False


def _raise_if_cancelled() -> None:
    if _rhythm_cancel_requested():
        raise _RhythmSessionCancelled()


def _sleep_is_mocked() -> bool:
    return hasattr(time.sleep, "mock_calls")


def _sleep_interruptibly(duration_sec: float, *, quantum_sec: float = 0.01) -> None:
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
        time.sleep(min(remaining, max(float(quantum_sec), 0.001)))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    lowered = str(value).strip().lower()
    if lowered in {"", "0", "false", "no", "off", "none", "null"}:
        return False
    if lowered in {"1", "true", "yes", "on"}:
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


def _capture_image(app: Any) -> Any:
    _raise_if_cancelled()
    capture = app.capture()
    if not capture.success or capture.image is None:
        raise RuntimeError("Failed to capture the Yihuan rhythm screen.")
    return capture


def _release_all(app: Any) -> None:
    try:
        if hasattr(app, "release_all"):
            app.release_all()
        elif hasattr(app, "controller") and hasattr(app.controller, "release_all"):
            app.controller.release_all()
    except Exception:
        logger.debug("Rhythm[input] release_all failed during cleanup.", exc_info=True)


def _focus_window(app: Any) -> bool:
    if hasattr(app, "focus_with_input"):
        try:
            if bool(app.focus_with_input(click_delay=0.05)):
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rhythm[input] focus_with_input failed: %s", exc)
    if hasattr(app, "focus"):
        try:
            return bool(app.focus())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rhythm[input] focus failed: %s", exc)
    return True


def _parse_lane_keys(raw: Any, lane_order: list[str], profile: Mapping[str, Any]) -> dict[str, str]:
    default_keys = {
        lane_id: str(dict(dict(profile.get("lanes") or {}).get(lane_id) or {}).get("key") or lane_id)
        for lane_id in lane_order
    }
    if raw is None:
        return default_keys
    if isinstance(raw, Mapping):
        keys = dict(default_keys)
        for lane_id in lane_order:
            value = raw.get(lane_id)
            if value:
                keys[lane_id] = str(value).strip().lower()
        return keys
    pieces = [item.strip().lower() for item in str(raw).replace(";", ",").split(",") if item.strip()]
    if len(pieces) != len(lane_order):
        return default_keys
    return {lane_id: pieces[index] for index, lane_id in enumerate(lane_order)}


def _apply_lane_y_offset(profile: Mapping[str, Any], offset_px: int) -> dict[str, Any]:
    shifted = dict(profile)
    lanes: dict[str, dict[str, Any]] = {}
    client_size = shifted.get("client_size") or (1280, 720)
    client_h = 720
    if isinstance(client_size, (list, tuple)) and len(client_size) > 1:
        client_h = _coerce_int(client_size[1], 720)
    client_h = max(client_h, 1)
    for lane_id, raw_lane in dict(profile.get("lanes") or {}).items():
        lane = dict(raw_lane or {})
        raw_point = lane.get("point") or (0, 0)
        if not isinstance(raw_point, (list, tuple)):
            raw_point = (0, 0)
        x = _coerce_int(raw_point[0] if len(raw_point) > 0 else 0, 0)
        y = _coerce_int(raw_point[1] if len(raw_point) > 1 else 0, 0)
        shifted_y = min(max(y + int(offset_px), 0), client_h - 1)
        lane["point"] = (x, shifted_y)
        lanes[str(lane_id)] = lane
    shifted["lanes"] = lanes
    shifted["lane_y_offset_px"] = int(offset_px)
    return shifted


class _RhythmKeyWorker:
    def __init__(self, app: Any, *, key_down_sec: float, dry_run: bool, plan_name: str | None = None) -> None:
        self._app = app
        self._key_down_sec = max(float(key_down_sec), 0.001)
        self._dry_run = bool(dry_run)
        self._plan_name = str(plan_name or current_plan_name.get() or "").strip() or None
        self._condition = threading.Condition()
        self._queue: deque[dict[str, Any]] = deque()
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self.executed: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="yihuan-rhythm-key-worker", daemon=True)
            self._thread.start()

    def press(self, key: str, *, lane: str, elapsed_sec: float, dark_ratio: float) -> None:
        event = {
            "key": str(key),
            "lane": str(lane),
            "elapsed_sec": round(float(elapsed_sec), 4),
            "dark_ratio": round(float(dark_ratio), 4),
        }
        with self._condition:
            if self._stop_requested:
                return
            self._queue.append(event)
            self._condition.notify()

    def stop(self, *, discard_pending: bool = True) -> None:
        with self._condition:
            self._stop_requested = True
            if discard_pending:
                self._queue.clear()
            self._condition.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)

    def _run(self) -> None:
        token = current_plan_name.set(self._plan_name) if self._plan_name else None
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._stop_requested:
                        self._condition.wait()
                    if self._stop_requested and not self._queue:
                        return
                    event = self._queue.popleft()
                key = str(event["key"])
                try:
                    if not self._dry_run:
                        self._app.key_down(key)
                        time.sleep(self._key_down_sec)
                        self._app.key_up(key)
                    self.executed.append(event)
                except Exception as exc:  # noqa: BLE001
                    message = f"{type(exc).__name__}: {exc}"
                    self.errors.append(message)
                    logger.warning("Rhythm[input] failed to press key=%s lane=%s: %s", key, event.get("lane"), message)
                    try:
                        if not self._dry_run:
                            self._app.key_up(key)
                    except Exception:
                        logger.debug("Rhythm[input] key_up cleanup failed for key=%s", key, exc_info=True)
        finally:
            if token is not None:
                current_plan_name.reset(token)


def _click_scaled_point(
    app: Any,
    yihuan_rhythm: YihuanRhythmService,
    capture_image: Any,
    point: tuple[int, int],
    *,
    profile: Mapping[str, Any],
    dry_run: bool,
    note: str,
    planned_actions: list[dict[str, Any]],
    executed_actions: list[dict[str, Any]],
) -> None:
    click_point = yihuan_rhythm.scale_point(capture_image, point, profile=profile)
    record = {
        "action": note,
        "logical_point": [int(point[0]), int(point[1])],
        "click_point": [int(click_point[0]), int(click_point[1])],
    }
    planned_actions.append(record)
    if dry_run:
        logger.info("Rhythm[action] dry_run skip click note=%s point=%s", note, click_point)
        return
    logger.info("Rhythm[action] click note=%s point=%s", note, click_point)
    app.click(int(click_point[0]), int(click_point[1]), button="left", clicks=1)
    executed_actions.append(record)


def _save_debug_snapshot(
    *,
    profile: Mapping[str, Any],
    yihuan_rhythm: YihuanRhythmService,
    capture_image: Any,
    note_state: Mapping[str, Any],
    label: str,
) -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        out_dir = root / str(profile.get("debug_snapshot_dir") or "tmp/rhythm_debug")
        out_dir.mkdir(parents=True, exist_ok=True)
        normalized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = out_dir / f"{timestamp}_{normalized}.png"
        overlay = yihuan_rhythm.draw_debug_overlay(
            capture_image,
            note_state,
            profile=profile,
        )
        cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rhythm[debug] failed to save snapshot: %s", exc)
        return None


def _wait_until_not_song_select(
    app: Any,
    yihuan_rhythm: YihuanRhythmService,
    *,
    profile: Mapping[str, Any],
) -> tuple[str, Any, dict[str, Any]]:
    deadline = time.monotonic() + float(profile["start_timeout_sec"])
    last_phase: dict[str, Any] = {}
    while True:
        capture = _capture_image(app)
        phase = yihuan_rhythm.analyze_phase(capture.image, profile=profile)
        last_phase = phase
        if phase["phase"] != "song_select":
            return str(phase["phase"]), capture, phase
        if time.monotonic() >= deadline:
            return "song_select", capture, last_phase
        _sleep_interruptibly(0.05)


def _wait_for_song_select(
    app: Any,
    yihuan_rhythm: YihuanRhythmService,
    *,
    profile: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    deadline = time.monotonic() + float(profile["result_return_timeout_sec"])
    last_phase: dict[str, Any] = {}
    while True:
        capture = _capture_image(app)
        phase = yihuan_rhythm.analyze_phase(capture.image, profile=profile)
        last_phase = phase
        if phase["phase"] == "song_select":
            return True, phase
        if time.monotonic() >= deadline:
            return False, last_phase
        _sleep_interruptibly(0.10)


def _run_single_song(
    app: Any,
    yihuan_rhythm: YihuanRhythmService,
    *,
    profile: Mapping[str, Any],
    lane_keys: Mapping[str, str],
    session_start: float,
    total_deadline: float | None,
    dry_run: bool,
    debug_enabled: bool,
    loop_index: int,
) -> dict[str, Any]:
    lane_order = list(profile["lane_order"])
    key_worker = _RhythmKeyWorker(
        app,
        key_down_sec=float(profile["key_down_ms"]) / 1000.0,
        dry_run=dry_run,
    )
    key_worker.start()
    song_started = time.monotonic()
    song_deadline = song_started + float(profile["song_timeout_sec"])
    prev_state = {lane_id: False for lane_id in lane_order}
    last_press_at = {lane_id: -9999.0 for lane_id in lane_order}
    hits_by_lane = {lane_id: 0 for lane_id in lane_order}
    frames = 0
    last_note_state: dict[str, Any] = {}
    phase_trace: list[dict[str, Any]] = []
    debug_snapshots: list[str] = []
    debug_next_at = song_started
    next_finish_check = song_started
    stopped_reason = "song_timeout"
    final_phase: dict[str, Any] = {}

    retrigger_sec = float(profile["retrigger_interval_ms"]) / 1000.0
    frame_interval_sec = float(profile["frame_interval_ms"]) / 1000.0
    finish_interval_sec = float(profile["finish_check_interval_ms"]) / 1000.0

    try:
        while True:
            _raise_if_cancelled()
            now = time.monotonic()
            if total_deadline is not None and now >= total_deadline:
                stopped_reason = "max_seconds"
                break
            if now >= song_deadline:
                stopped_reason = "song_timeout"
                break

            capture = _capture_image(app)
            frames += 1
            if now >= next_finish_check:
                phase = yihuan_rhythm.analyze_phase(capture.image, profile=profile)
                final_phase = phase
                phase_trace.append({"t": round(now - session_start, 3), "phase": phase["phase"]})
                if phase["phase"] == "result":
                    stopped_reason = "level_end"
                    break
                if phase["phase"] == "song_select":
                    played_long_enough = now - song_started >= float(profile["song_select_completion_min_sec"])
                    pressed_any_note = any(int(value or 0) > 0 for value in hits_by_lane.values())
                    stopped_reason = "level_end" if played_long_enough and pressed_any_note else "not_playing"
                    break
                if finish_interval_sec <= 0:
                    next_finish_check = now
                else:
                    next_finish_check = now + finish_interval_sec

            note_state = yihuan_rhythm.analyze_notes(capture.image, profile=profile)
            last_note_state = note_state
            lanes = dict(note_state.get("lanes") or {})
            elapsed_sec = now - session_start
            for lane_id in lane_order:
                lane_state = dict(lanes.get(lane_id) or {})
                has_note = bool(lane_state.get("has_note"))
                dark_ratio = float(lane_state.get("dark_ratio") or 0.0)
                should_press = has_note and (
                    not prev_state[lane_id]
                    or now - float(last_press_at[lane_id]) >= retrigger_sec
                )
                if should_press:
                    key_worker.press(
                        str(lane_keys[lane_id]),
                        lane=lane_id,
                        elapsed_sec=elapsed_sec,
                        dark_ratio=dark_ratio,
                    )
                    last_press_at[lane_id] = now
                    hits_by_lane[lane_id] += 1
                prev_state[lane_id] = has_note

            if debug_enabled and int(profile["debug_snapshot_max_count"]) > len(debug_snapshots):
                interval = float(profile["debug_snapshot_interval_sec"])
                if interval <= 0 or now >= debug_next_at:
                    path = _save_debug_snapshot(
                        profile=profile,
                        yihuan_rhythm=yihuan_rhythm,
                        capture_image=capture.image,
                        note_state=note_state,
                        label=f"loop_{loop_index:03d}_frame_{frames:06d}",
                    )
                    if path:
                        debug_snapshots.append(path)
                    debug_next_at = now + max(interval, 0.001)

            if frame_interval_sec > 0:
                _sleep_interruptibly(frame_interval_sec)
    finally:
        key_worker.stop(discard_pending=True)
        _release_all(app)

    elapsed_song_sec = time.monotonic() - song_started
    return {
        "stopped_reason": stopped_reason,
        "frames": frames,
        "song_elapsed_sec": round(elapsed_song_sec, 3),
        "hits_by_lane": hits_by_lane,
        "press_count": sum(hits_by_lane.values()),
        "executed_presses": list(key_worker.executed),
        "worker_errors": list(key_worker.errors),
        "last_note_state": last_note_state,
        "last_phase": final_phase,
        "phase_trace": phase_trace[-20:],
        "debug_snapshots": debug_snapshots,
    }


@action_info(
    name="yihuan_rhythm_run_session",
    public=False,
    read_only=False,
    description="自动运行异环鼓组四键下落音游，识别判定线附近音符并按键。",
)
@requires_services(
    app="plans/aura_base/app",
    yihuan_rhythm="yihuan_rhythm",
)
def yihuan_rhythm_run_session(
    app: Any,
    yihuan_rhythm: YihuanRhythmService,
    profile_name: str = "default_1280x720_cn",
    loop_count: int | str = 1,
    max_seconds: float | int | str = 240,
    start_game: bool | str = True,
    close_result: bool | str = True,
    lane_keys: str = "d,f,j,k",
    lane_y_offset_px: int | str = 0,
    dry_run: bool | str = False,
    debug_enabled: bool | str = False,
) -> dict[str, Any]:
    profile = yihuan_rhythm.load_profile(profile_name)
    resolved_lane_y_offset_px = _coerce_int(lane_y_offset_px, 0)
    profile = _apply_lane_y_offset(profile, resolved_lane_y_offset_px)
    resolved_profile = str(profile["profile_name"])
    lane_order = list(profile["lane_order"])
    resolved_lane_keys = _parse_lane_keys(lane_keys, lane_order, profile)
    resolved_lane_points = {
        lane_id: [int(dict(profile["lanes"][lane_id])["point"][0]), int(dict(profile["lanes"][lane_id])["point"][1])]
        for lane_id in lane_order
    }
    requested_loop_count = max(_coerce_int(loop_count, 1), 0)
    duration_limit = max(_coerce_float(max_seconds, float(profile["song_timeout_sec"])), 0.0)
    total_deadline = time.monotonic() + duration_limit if duration_limit > 0 else None
    start_game_enabled = _coerce_bool(start_game)
    close_result_enabled = _coerce_bool(close_result)
    dry_run_enabled = _coerce_bool(dry_run)
    debug_enabled_value = _coerce_bool(debug_enabled)
    session_start = time.monotonic()
    planned_actions: list[dict[str, Any]] = []
    executed_actions: list[dict[str, Any]] = []
    loops: list[dict[str, Any]] = []
    status = "success"
    stopped_reason = "completed"
    failure_reason: str | None = None
    failure_message = ""
    final_phase: dict[str, Any] = {}
    result_closed_count = 0

    logger.info(
        "Rhythm[session] start profile=%s loops=%s max_seconds=%.3f start_game=%s close_result=%s dry_run=%s keys=%s lane_y_offset_px=%s",
        resolved_profile,
        requested_loop_count,
        duration_limit,
        start_game_enabled,
        close_result_enabled,
        dry_run_enabled,
        resolved_lane_keys,
        resolved_lane_y_offset_px,
    )

    try:
        if not dry_run_enabled and not _focus_window(app):
            return {
                "status": "failed",
                "stopped_reason": "focus_failed",
                "failure_reason": "focus_failed",
                "profile_name": resolved_profile,
                "lane_y_offset_px": resolved_lane_y_offset_px,
                "elapsed_sec": round(time.monotonic() - session_start, 3),
            }

        loop_index = 0
        while requested_loop_count <= 0 or loop_index < requested_loop_count:
            _raise_if_cancelled()
            if total_deadline is not None and time.monotonic() >= total_deadline:
                stopped_reason = "max_seconds"
                break

            initial_capture = _capture_image(app)
            initial_phase = yihuan_rhythm.analyze_phase(initial_capture.image, profile=profile)
            final_phase = initial_phase
            if initial_phase["phase"] == "result" and close_result_enabled:
                _click_scaled_point(
                    app,
                    yihuan_rhythm,
                    initial_capture.image,
                    profile["result_close_point"],
                    profile=profile,
                    dry_run=dry_run_enabled,
                    note="close_existing_result",
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                )
                result_closed_count += 1
                _sleep_interruptibly(float(profile["result_exit_delay_ms"]) / 1000.0)
                if start_game_enabled:
                    _wait_for_song_select(app, yihuan_rhythm, profile=profile)

            if start_game_enabled:
                start_capture = _capture_image(app)
                _click_scaled_point(
                    app,
                    yihuan_rhythm,
                    start_capture.image,
                    profile["start_song_point"],
                    profile=profile,
                    dry_run=dry_run_enabled,
                    note="start_song",
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                )
                delay = float(profile["post_start_delay_ms"]) / 1000.0
                if delay > 0:
                    _sleep_interruptibly(delay)
                phase_name, waited_capture, waited_phase = _wait_until_not_song_select(
                    app,
                    yihuan_rhythm,
                    profile=profile,
                )
                final_phase = waited_phase
                if phase_name == "song_select":
                    status = "failed" if not loops else "partial"
                    stopped_reason = "game_start_timeout"
                    failure_reason = "game_start_timeout"
                    failure_message = "Timed out waiting for rhythm gameplay after clicking start."
                    break
                if phase_name == "result":
                    song_result = {
                        "loop_index": loop_index + 1,
                        "stopped_reason": "level_end",
                        "frames": 0,
                        "song_elapsed_sec": 0.0,
                        "hits_by_lane": {lane_id: 0 for lane_id in lane_order},
                        "press_count": 0,
                        "executed_presses": [],
                        "worker_errors": [],
                        "last_note_state": {},
                        "last_phase": waited_phase,
                        "phase_trace": [{"t": round(time.monotonic() - session_start, 3), "phase": "result"}],
                        "debug_snapshots": [],
                    }
                    loops.append(song_result)
                else:
                    song_result = _run_single_song(
                        app,
                        yihuan_rhythm,
                        profile=profile,
                        lane_keys=resolved_lane_keys,
                        session_start=session_start,
                        total_deadline=total_deadline,
                        dry_run=dry_run_enabled,
                        debug_enabled=debug_enabled_value,
                        loop_index=loop_index + 1,
                    )
                    loops.append({"loop_index": loop_index + 1, **song_result})
            else:
                song_result = _run_single_song(
                    app,
                    yihuan_rhythm,
                    profile=profile,
                    lane_keys=resolved_lane_keys,
                    session_start=session_start,
                    total_deadline=total_deadline,
                    dry_run=dry_run_enabled,
                    debug_enabled=debug_enabled_value,
                    loop_index=loop_index + 1,
                )
                loops.append({"loop_index": loop_index + 1, **song_result})

            current_result = dict(loops[-1])
            stopped_reason = str(current_result.get("stopped_reason") or stopped_reason)
            if stopped_reason != "level_end":
                if stopped_reason == "max_seconds":
                    status = "partial" if current_result.get("press_count", 0) else "failed"
                    failure_reason = None if status == "partial" else "max_seconds"
                else:
                    status = "partial" if current_result.get("press_count", 0) else "failed"
                    failure_reason = stopped_reason
                break

            loop_index += 1
            if close_result_enabled:
                result_capture = _capture_image(app)
                result_phase = yihuan_rhythm.analyze_phase(result_capture.image, profile=profile)
                final_phase = result_phase
                if result_phase["phase"] == "result":
                    _click_scaled_point(
                        app,
                        yihuan_rhythm,
                        result_capture.image,
                        profile["result_close_point"],
                        profile=profile,
                        dry_run=dry_run_enabled,
                        note="close_result",
                        planned_actions=planned_actions,
                        executed_actions=executed_actions,
                    )
                    result_closed_count += 1
                    _sleep_interruptibly(float(profile["result_exit_delay_ms"]) / 1000.0)

            if requested_loop_count > 0 and loop_index >= requested_loop_count:
                stopped_reason = "loop_count"
                break

            if start_game_enabled and close_result_enabled:
                returned, return_phase = _wait_for_song_select(app, yihuan_rhythm, profile=profile)
                final_phase = return_phase
                if not returned:
                    status = "partial" if loops else "failed"
                    stopped_reason = "song_select_return_timeout"
                    failure_reason = "song_select_return_timeout"
                    break
            elif not start_game_enabled:
                stopped_reason = "level_end"
                break

    except _RhythmSessionCancelled:
        status = "cancelled"
        stopped_reason = "cancelled"
        failure_reason = "cancelled"
        _release_all(app)
    except RuntimeError as exc:
        status = "partial" if loops else "failed"
        stopped_reason = "capture_failed"
        failure_reason = "capture_failed"
        failure_message = str(exc)
        logger.warning("Rhythm[session] capture failed: %s", exc)
        _release_all(app)
    except Exception as exc:  # noqa: BLE001
        status = "partial" if loops else "failed"
        stopped_reason = "exception"
        failure_reason = "exception"
        failure_message = str(exc)
        logger.exception("Rhythm auto session failed.")
        _release_all(app)

    elapsed_sec = round(time.monotonic() - session_start, 3)
    total_hits = {lane_id: 0 for lane_id in lane_order}
    total_press_count = 0
    debug_snapshots: list[str] = []
    for item in loops:
        total_press_count += int(item.get("press_count") or 0)
        for lane_id, value in dict(item.get("hits_by_lane") or {}).items():
            total_hits[str(lane_id)] = int(total_hits.get(str(lane_id), 0)) + int(value or 0)
        debug_snapshots.extend(str(path) for path in item.get("debug_snapshots") or [])

    logger.info(
        "Rhythm[session] stop status=%s reason=%s loops=%s presses=%s elapsed=%.3f",
        status,
        stopped_reason,
        len(loops),
        total_press_count,
        elapsed_sec,
    )
    return {
        "status": status,
        "stopped_reason": stopped_reason,
        "failure_reason": failure_reason,
        "failure_message": failure_message,
        "profile_name": resolved_profile,
        "lane_y_offset_px": resolved_lane_y_offset_px,
        "lane_points": resolved_lane_points,
        "loop_count_requested": requested_loop_count,
        "loops_completed": sum(1 for item in loops if item.get("stopped_reason") == "level_end"),
        "loops": loops,
        "lane_keys": dict(resolved_lane_keys),
        "hits_by_lane": total_hits,
        "press_count": total_press_count,
        "planned_actions": planned_actions,
        "executed_actions": executed_actions,
        "result_closed_count": result_closed_count,
        "start_game": start_game_enabled,
        "close_result": close_result_enabled,
        "dry_run": dry_run_enabled,
        "debug_enabled": debug_enabled_value,
        "debug_snapshots": debug_snapshots[-int(profile["debug_snapshot_max_count"]) :] if debug_snapshots else [],
        "last_phase": final_phase,
        "elapsed_sec": elapsed_sec,
    }

"""Mahjong actions for the Yihuan plan."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Mapping

import cv2

from packages.aura_core.api import action_info, requires_services
from packages.aura_core.observability.logging.core_logger import logger
from packages.aura_core.scheduler.cancellation import is_current_task_cancel_requested

from ..services.mahjong_service import YihuanMahjongService


class _MahjongSessionStop(Exception):
    """Internal control-flow marker for an already classified session stop."""


class _MahjongSessionCancelled(Exception):
    """Internal marker used to stop the sync mahjong action cooperatively."""


def _mahjong_cancel_requested() -> bool:
    try:
        return is_current_task_cancel_requested()
    except Exception:
        return False


def _raise_if_cancelled() -> None:
    if _mahjong_cancel_requested():
        raise _MahjongSessionCancelled()


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


def _release_all(app: Any) -> None:
    try:
        if hasattr(app, "release_all"):
            app.release_all()
        elif hasattr(app, "controller") and hasattr(app.controller, "release_all"):
            app.controller.release_all()
    except Exception:
        logger.debug("Mahjong[input] release_all failed during cancellation cleanup.", exc_info=True)


def _capture_phase(
    app: Any,
    yihuan_mahjong: YihuanMahjongService,
    *,
    profile_name: str,
) -> tuple[dict[str, Any], Any]:
    _raise_if_cancelled()
    capture = app.capture()
    if not capture.success or capture.image is None:
        raise RuntimeError("Failed to capture the Yihuan Mahjong screen.")
    phase = yihuan_mahjong.analyze_phase(capture.image, profile_name=profile_name)
    return phase, capture


def _append_trace(
    phase_trace: list[dict[str, Any]],
    *,
    start_time: float,
    phase: Mapping[str, Any],
    note: str | None = None,
) -> None:
    entry = {
        "t": round(time.monotonic() - start_time, 3),
        "phase": phase.get("phase"),
    }
    if note:
        entry["note"] = note
    ready = dict(phase.get("ready") or {})
    exchange = dict(phase.get("exchange") or {})
    dingque = dict(phase.get("dingque") or {})
    playing = dict(phase.get("playing") or {})
    result = dict(phase.get("result") or {})
    entry["ready_found"] = bool(ready.get("found"))
    entry["exchange_found"] = bool(exchange.get("found"))
    entry["dingque_found"] = bool(dingque.get("found"))
    entry["playing_found"] = bool(playing.get("found"))
    entry["result_found"] = bool(result.get("found"))
    if exchange:
        entry["exchange_confirm_enabled"] = bool(exchange.get("confirm_enabled"))
    if dingque:
        entry["dingque_button_count"] = dingque.get("found_button_count")
    if playing:
        entry["enabled_switch_count"] = playing.get("enabled_switch_count")
    phase_trace.append(entry)


def _final_result(
    *,
    status: str,
    stopped_reason: str,
    failure_reason: str | None,
    profile_name: str,
    selected_missing_suit: str | None,
    hand_suit_counts: Mapping[str, Any] | None,
    auto_toggles_enabled: Mapping[str, Any] | None,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    planned_actions: list[dict[str, Any]] | None = None,
    executed_actions: list[dict[str, Any]] | None = None,
    last_phase: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    missing_suit_decision: Mapping[str, Any] | None = None,
    exchange_selected_suit: str | None = None,
    exchange_tile_indices: list[int] | None = None,
    exchange_tile_points: list[list[int]] | None = None,
    exchange_result: str = "skipped",
    click_verifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "stopped_reason": stopped_reason,
        "failure_reason": failure_reason,
        "profile_name": profile_name,
        "selected_missing_suit": selected_missing_suit,
        "hand_suit_counts": dict(hand_suit_counts or {"wan": 0, "tong": 0, "tiao": 0}),
        "auto_toggles_enabled": dict(auto_toggles_enabled or {}),
        "phase_trace": phase_trace,
        "elapsed_sec": round(time.monotonic() - start_time, 3),
        "planned_actions": list(planned_actions or []),
        "executed_actions": list(executed_actions or []),
        "last_phase": dict(last_phase or {}),
        "dry_run": bool(dry_run),
        "missing_suit_decision": dict(missing_suit_decision or {}),
        "exchange_selected_suit": exchange_selected_suit,
        "exchange_tile_indices": list(exchange_tile_indices or []),
        "exchange_tile_points": list(exchange_tile_points or []),
        "exchange_result": exchange_result,
        "click_verifications": list(click_verifications or []),
    }


def _record_mouse_action(
    app: Any,
    yihuan_mahjong: YihuanMahjongService,
    *,
    profile: Mapping[str, Any],
    capture_image: Any,
    action_name: str,
    logical_point: tuple[int, int],
    dry_run: bool,
    planned_actions: list[dict[str, Any]],
    executed_actions: list[dict[str, Any]],
    point_is_scaled: bool = False,
) -> None:
    _raise_if_cancelled()
    click_point = (
        (int(logical_point[0]), int(logical_point[1]))
        if point_is_scaled
        else yihuan_mahjong.scale_point(capture_image, logical_point, profile=profile)
    )
    record = {
        "action": action_name,
        "logical_point": [int(logical_point[0]), int(logical_point[1])],
        "click_point": [int(click_point[0]), int(click_point[1])],
        "button": "left",
    }
    planned_actions.append(record)
    if dry_run:
        logger.info("Mahjong[action] dry_run skip action=%s point=%s", action_name, click_point)
        return
    logger.info("Mahjong[action] click action=%s point=%s", action_name, click_point)
    app.click(int(click_point[0]), int(click_point[1]), button="left", clicks=1)
    _raise_if_cancelled()
    executed_actions.append(record)
    delay_sec = float(profile["post_click_delay_ms"]) / 1000.0
    if delay_sec > 0:
        _sleep_interruptibly(delay_sec)


def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _wait_for_phase(
    app: Any,
    yihuan_mahjong: YihuanMahjongService,
    *,
    profile: Mapping[str, Any],
    target_phases: set[str],
    timeout_sec: float,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    note: str,
) -> tuple[dict[str, Any], Any] | None:
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    poll_sec = float(profile["poll_ms"]) / 1000.0
    while True:
        _raise_if_cancelled()
        phase, capture = _capture_phase(app, yihuan_mahjong, profile_name=str(profile["profile_name"]))
        _append_trace(phase_trace, start_time=start_time, phase=phase, note=note)
        if str(phase.get("phase")) in target_phases:
            return phase, capture
        if time.monotonic() >= deadline:
            return None
        if poll_sec > 0:
            _sleep_interruptibly(poll_sec)


def _wait_for_exchange_confirm_or_next_phase(
    app: Any,
    yihuan_mahjong: YihuanMahjongService,
    *,
    profile: Mapping[str, Any],
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> tuple[dict[str, Any], Any] | None:
    deadline = time.monotonic() + float(profile["exchange_wait_timeout_sec"])
    poll_sec = float(profile["poll_ms"]) / 1000.0
    while True:
        _raise_if_cancelled()
        phase, capture = _capture_phase(app, yihuan_mahjong, profile_name=str(profile["profile_name"]))
        _append_trace(phase_trace, start_time=start_time, phase=phase, note="exchange_confirm_wait")
        if phase.get("phase") in {"dingque", "playing", "result"}:
            return phase, capture
        exchange = dict(phase.get("exchange") or {})
        if exchange.get("found") and exchange.get("confirm_enabled"):
            return phase, capture
        if time.monotonic() >= deadline:
            return None
        if poll_sec > 0:
            _sleep_interruptibly(poll_sec)


def _ensure_auto_switches(
    app: Any,
    yihuan_mahjong: YihuanMahjongService,
    *,
    profile: Mapping[str, Any],
    phase: Mapping[str, Any],
    capture: Any,
    requested: Mapping[str, bool],
    dry_run: bool,
    planned_actions: list[dict[str, Any]],
    executed_actions: list[dict[str, Any]],
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> tuple[dict[str, bool | None], dict[str, Any], Any]:
    current_phase = dict(phase)
    current_capture = capture
    attempts: dict[str, int] = {name: 0 for name in requested}
    retry_limit = max(int(profile["switch_retry_limit"]), 1)
    verify_delay_sec = float(profile["switch_verify_delay_ms"]) / 1000.0

    while True:
        _raise_if_cancelled()
        playing = dict(current_phase.get("playing") or {})
        switches = dict(playing.get("switches") or {})
        remaining = []
        for name, enabled in requested.items():
            if not enabled:
                continue
            switch = dict(switches.get(name) or {})
            if switch.get("enabled") is True:
                continue
            if attempts[name] >= retry_limit:
                continue
            remaining.append(name)

        if not remaining:
            break

        for name in remaining:
            _raise_if_cancelled()
            switch_profile = dict(dict(profile["auto_switches"]).get(name) or {})
            point = switch_profile.get("point")
            if not point:
                continue
            attempts[name] += 1
            _record_mouse_action(
                app,
                yihuan_mahjong,
                profile=profile,
                capture_image=current_capture.image,
                action_name=f"enable_{name}",
                logical_point=point,
                dry_run=dry_run,
                planned_actions=planned_actions,
                executed_actions=executed_actions,
            )

        if dry_run:
            break
        if verify_delay_sec > 0:
            _sleep_interruptibly(verify_delay_sec)
        current_phase, current_capture = _capture_phase(
            app,
            yihuan_mahjong,
            profile_name=str(profile["profile_name"]),
        )
        _append_trace(phase_trace, start_time=start_time, phase=current_phase, note="switch_verify")
        if current_phase.get("phase") == "result":
            break

    final_switches = dict(dict(current_phase.get("playing") or {}).get("switches") or {})
    enabled_map: dict[str, bool | None] = {}
    for name, enabled in requested.items():
        if not enabled:
            enabled_map[name] = None
        else:
            enabled_map[name] = bool(dict(final_switches.get(name) or {}).get("enabled"))
    return enabled_map, current_phase, current_capture


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    return bool(value)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _save_debug_snapshot(
    *,
    profile: Mapping[str, Any],
    capture_image: Any,
    phase: Mapping[str, Any],
    label: str,
) -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        out_dir = root / str(profile.get("debug_snapshot_dir") or "tmp/mahjong_debug")
        out_dir.mkdir(parents=True, exist_ok=True)
        normalized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = out_dir / f"{timestamp}_{normalized}_{phase.get('phase') or 'unknown'}.png"
        cv2.imwrite(str(path), cv2.cvtColor(capture_image, cv2.COLOR_RGB2BGR))
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Mahjong[debug] failed to save snapshot: %s", exc)
        return None


@action_info(
    name="yihuan_mahjong_run_session",
    public=False,
    read_only=False,
    description="Run the Yihuan Mahjong mini-game loop using the built-in auto switches.",
)
@requires_services(
    app="plans/aura_base/app",
    yihuan_mahjong="yihuan_mahjong",
)
def yihuan_mahjong_run_session(
    app: Any,
    yihuan_mahjong: YihuanMahjongService,
    profile_name: str = "default_1280x720_cn",
    max_seconds: float | int | str = 0,
    start_game: bool | str = True,
    auto_hu: bool | str = True,
    auto_peng: bool | str = True,
    auto_discard: bool | str = True,
    dry_run: bool | str = False,
    debug_enabled: bool | str = False,
) -> dict[str, Any]:
    profile = yihuan_mahjong.load_profile(profile_name)
    resolved_profile = str(profile["profile_name"])
    duration_limit = max(_coerce_float(max_seconds, 0.0), 0.0)
    deadline = time.monotonic() + duration_limit if duration_limit > 0 else None
    start_game_enabled = _coerce_bool(start_game)
    dry_run_enabled = _coerce_bool(dry_run)
    debug_enabled = _coerce_bool(debug_enabled)
    requested_switches = {
        "hu": _coerce_bool(auto_hu),
        "peng": _coerce_bool(auto_peng),
        "discard": _coerce_bool(auto_discard),
    }

    start_time = time.monotonic()
    planned_actions: list[dict[str, Any]] = []
    executed_actions: list[dict[str, Any]] = []
    phase_trace: list[dict[str, Any]] = []
    selected_missing_suit: str | None = None
    hand_suit_counts: dict[str, Any] = {"wan": 0, "tong": 0, "tiao": 0}
    auto_toggles_enabled: dict[str, bool | None] = {}
    missing_suit_decision: dict[str, Any] | None = None
    debug_snapshots: list[str] = []
    last_phase: dict[str, Any] | None = None
    exchange_selected_suit: str | None = None
    exchange_tile_indices: list[int] = []
    exchange_tile_points: list[list[int]] = []
    exchange_result = "skipped"
    click_verifications: list[dict[str, Any]] = []

    def _finish(**kwargs: Any) -> dict[str, Any]:
        return _final_result(
            **kwargs,
            exchange_selected_suit=exchange_selected_suit,
            exchange_tile_indices=exchange_tile_indices,
            exchange_tile_points=exchange_tile_points,
            exchange_result=exchange_result,
            click_verifications=click_verifications,
        )

    logger.info(
        "Mahjong[session] start profile=%s max_seconds=%.3f start_game=%s auto=%s dry_run=%s debug=%s",
        resolved_profile,
        duration_limit,
        start_game_enabled,
        requested_switches,
        dry_run_enabled,
        debug_enabled,
    )

    try:
        _raise_if_cancelled()
        phase, capture = _capture_phase(app, yihuan_mahjong, profile_name=resolved_profile)
        last_phase = phase
        _append_trace(phase_trace, start_time=start_time, phase=phase, note="initial")
        if debug_enabled:
            path = _save_debug_snapshot(profile=profile, capture_image=capture.image, phase=phase, label="initial")
            if path:
                debug_snapshots.append(path)

        if phase["phase"] == "result":
            return _finish(
                status="success",
                stopped_reason="level_end",
                failure_reason=None,
                profile_name=resolved_profile,
                selected_missing_suit=selected_missing_suit,
                hand_suit_counts=hand_suit_counts,
                auto_toggles_enabled=auto_toggles_enabled,
                phase_trace=phase_trace,
                start_time=start_time,
                planned_actions=planned_actions,
                executed_actions=executed_actions,
                last_phase=phase,
                dry_run=dry_run_enabled,
                missing_suit_decision=missing_suit_decision,
            )

        if phase["phase"] == "ready":
            if not start_game_enabled:
                return _finish(
                    status="failed",
                    stopped_reason="phase_timeout",
                    failure_reason="phase_timeout",
                    profile_name=resolved_profile,
                    selected_missing_suit=selected_missing_suit,
                    hand_suit_counts=hand_suit_counts,
                    auto_toggles_enabled=auto_toggles_enabled,
                    phase_trace=phase_trace,
                    start_time=start_time,
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                    last_phase=phase,
                    dry_run=dry_run_enabled,
                    missing_suit_decision=missing_suit_decision,
                )

            _record_mouse_action(
                app,
                yihuan_mahjong,
                profile=profile,
                capture_image=capture.image,
                action_name="click_ready",
                logical_point=profile["ready_button_point"],
                dry_run=dry_run_enabled,
                planned_actions=planned_actions,
                executed_actions=executed_actions,
            )
            waited = _wait_for_phase(
                app,
                yihuan_mahjong,
                profile=profile,
                target_phases={"exchange", "dingque", "playing", "result"},
                timeout_sec=float(profile["start_wait_timeout_sec"]),
                phase_trace=phase_trace,
                start_time=start_time,
                note="after_ready",
            )
            click_verifications.append(
                {
                    "action": "click_ready",
                    "before_phase": "ready",
                    "expected_phases": ["exchange", "dingque", "playing", "result"],
                    "after_phase": None if waited is None else waited[0].get("phase"),
                    "success": waited is not None,
                }
            )
            if waited is None:
                return _finish(
                    status="failed",
                    stopped_reason="phase_timeout",
                    failure_reason="phase_timeout",
                    profile_name=resolved_profile,
                    selected_missing_suit=selected_missing_suit,
                    hand_suit_counts=hand_suit_counts,
                    auto_toggles_enabled=auto_toggles_enabled,
                    phase_trace=phase_trace,
                    start_time=start_time,
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                    last_phase=last_phase,
                    dry_run=dry_run_enabled,
                    missing_suit_decision=missing_suit_decision,
                )
            phase, capture = waited
            last_phase = phase

        if phase["phase"] == "result":
            return _finish(
                status="success",
                stopped_reason="level_end",
                failure_reason=None,
                profile_name=resolved_profile,
                selected_missing_suit=selected_missing_suit,
                hand_suit_counts=hand_suit_counts,
                auto_toggles_enabled=auto_toggles_enabled,
                phase_trace=phase_trace,
                start_time=start_time,
                planned_actions=planned_actions,
                executed_actions=executed_actions,
                last_phase=phase,
                dry_run=dry_run_enabled,
                missing_suit_decision=missing_suit_decision,
            )

        if phase["phase"] == "exchange":
            hand = yihuan_mahjong.analyze_exchange_hand_suits(capture.image, profile_name=resolved_profile)
            exchange_decision = yihuan_mahjong.choose_exchange_three_detail(
                hand.get("candidates") or [],
                profile_name=resolved_profile,
            )
            if not exchange_decision.get("found"):
                exchange_result = "selection_failed"
                return _finish(
                    status="failed",
                    stopped_reason="exchange_selection_failed",
                    failure_reason="exchange_selection_failed",
                    profile_name=resolved_profile,
                    selected_missing_suit=selected_missing_suit,
                    hand_suit_counts=hand_suit_counts,
                    auto_toggles_enabled=auto_toggles_enabled,
                    phase_trace=phase_trace,
                    start_time=start_time,
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                    last_phase=phase,
                    dry_run=dry_run_enabled,
                    missing_suit_decision=missing_suit_decision,
                )

            exchange_selected_suit = str(exchange_decision["selected_suit"])
            exchange_tile_indices = list(exchange_decision.get("tile_indices") or [])
            exchange_tile_points = list(exchange_decision.get("tile_points") or [])
            for index, point in enumerate(exchange_tile_points):
                _record_mouse_action(
                    app,
                    yihuan_mahjong,
                    profile=profile,
                    capture_image=capture.image,
                    action_name=f"exchange_select_{exchange_selected_suit}_{index + 1}",
                    logical_point=(int(point[0]), int(point[1])),
                    dry_run=dry_run_enabled,
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                    point_is_scaled=True,
                )

            confirm_waited = _wait_for_exchange_confirm_or_next_phase(
                app,
                yihuan_mahjong,
                profile=profile,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            if confirm_waited is None:
                exchange_result = "confirm_unavailable"
                return _finish(
                    status="failed",
                    stopped_reason="exchange_confirm_unavailable",
                    failure_reason="exchange_confirm_unavailable",
                    profile_name=resolved_profile,
                    selected_missing_suit=selected_missing_suit,
                    hand_suit_counts=hand_suit_counts,
                    auto_toggles_enabled=auto_toggles_enabled,
                    phase_trace=phase_trace,
                    start_time=start_time,
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                    last_phase=phase,
                    dry_run=dry_run_enabled,
                    missing_suit_decision=missing_suit_decision,
                )

            phase, capture = confirm_waited
            last_phase = phase
            if phase["phase"] == "exchange":
                _record_mouse_action(
                    app,
                    yihuan_mahjong,
                    profile=profile,
                    capture_image=capture.image,
                    action_name="exchange_confirm",
                    logical_point=profile["exchange_confirm_point"],
                    dry_run=dry_run_enabled,
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                )
                exchange_result = "planned" if dry_run_enabled else "confirmed"
                waited = _wait_for_phase(
                    app,
                    yihuan_mahjong,
                    profile=profile,
                    target_phases={"dingque", "playing", "result"},
                    timeout_sec=float(profile["exchange_wait_timeout_sec"]),
                    phase_trace=phase_trace,
                    start_time=start_time,
                    note="after_exchange_confirm",
                )
                click_verifications.append(
                    {
                        "action": "exchange_confirm",
                        "before_phase": "exchange",
                        "expected_phases": ["dingque", "playing", "result"],
                        "after_phase": None if waited is None else waited[0].get("phase"),
                        "success": waited is not None,
                    }
                )
                if waited is None:
                    return _finish(
                        status="failed",
                        stopped_reason="phase_timeout",
                        failure_reason="phase_timeout",
                        profile_name=resolved_profile,
                        selected_missing_suit=selected_missing_suit,
                        hand_suit_counts=hand_suit_counts,
                        auto_toggles_enabled=auto_toggles_enabled,
                        phase_trace=phase_trace,
                        start_time=start_time,
                        planned_actions=planned_actions,
                        executed_actions=executed_actions,
                        last_phase=last_phase,
                        dry_run=dry_run_enabled,
                        missing_suit_decision=missing_suit_decision,
                    )
                phase, capture = waited
                last_phase = phase
            else:
                exchange_result = "completed"

        if phase["phase"] == "result":
            return _finish(
                status="success",
                stopped_reason="level_end",
                failure_reason=None,
                profile_name=resolved_profile,
                selected_missing_suit=selected_missing_suit,
                hand_suit_counts=hand_suit_counts,
                auto_toggles_enabled=auto_toggles_enabled,
                phase_trace=phase_trace,
                start_time=start_time,
                planned_actions=planned_actions,
                executed_actions=executed_actions,
                last_phase=phase,
                dry_run=dry_run_enabled,
                missing_suit_decision=missing_suit_decision,
            )

        if phase["phase"] == "dingque":
            hand = yihuan_mahjong.analyze_hand_suits(capture.image, profile_name=resolved_profile)
            hand_suit_counts = dict(hand.get("counts") or hand_suit_counts)
            missing_suit_decision = yihuan_mahjong.choose_missing_suit_detail(
                hand_suit_counts,
                hand.get("candidates") or [],
                profile_name=resolved_profile,
            )
            selected_missing_suit = str(missing_suit_decision["selected_suit"])
            button_profile = dict(dict(profile["dingque_buttons"])[selected_missing_suit])
            _record_mouse_action(
                app,
                yihuan_mahjong,
                profile=profile,
                capture_image=capture.image,
                action_name=f"select_missing_{selected_missing_suit}",
                logical_point=button_profile["point"],
                dry_run=dry_run_enabled,
                planned_actions=planned_actions,
                executed_actions=executed_actions,
            )
            waited = _wait_for_phase(
                app,
                yihuan_mahjong,
                profile=profile,
                target_phases={"playing", "result"},
                timeout_sec=float(profile["dingque_wait_timeout_sec"]),
                phase_trace=phase_trace,
                start_time=start_time,
                note="after_dingque",
            )
            click_verifications.append(
                {
                    "action": f"select_missing_{selected_missing_suit}",
                    "before_phase": "dingque",
                    "expected_phases": ["playing", "result"],
                    "after_phase": None if waited is None else waited[0].get("phase"),
                    "success": waited is not None,
                }
            )
            if waited is None:
                return _finish(
                    status="failed",
                    stopped_reason="phase_timeout",
                    failure_reason="phase_timeout",
                    profile_name=resolved_profile,
                    selected_missing_suit=selected_missing_suit,
                    hand_suit_counts=hand_suit_counts,
                    auto_toggles_enabled=auto_toggles_enabled,
                    phase_trace=phase_trace,
                    start_time=start_time,
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                    last_phase=last_phase,
                    dry_run=dry_run_enabled,
                    missing_suit_decision=missing_suit_decision,
                )
            phase, capture = waited
            last_phase = phase

        if phase["phase"] == "result":
            return _finish(
                status="success",
                stopped_reason="level_end",
                failure_reason=None,
                profile_name=resolved_profile,
                selected_missing_suit=selected_missing_suit,
                hand_suit_counts=hand_suit_counts,
                auto_toggles_enabled=auto_toggles_enabled,
                phase_trace=phase_trace,
                start_time=start_time,
                planned_actions=planned_actions,
                executed_actions=executed_actions,
                last_phase=phase,
                dry_run=dry_run_enabled,
                missing_suit_decision=missing_suit_decision,
            )

        if phase["phase"] != "playing":
            phase_timeout = time.monotonic() + float(profile["phase_timeout_sec"])
            poll_sec = float(profile["poll_ms"]) / 1000.0
            while phase["phase"] not in {"playing", "result"}:
                _raise_if_cancelled()
                if _deadline_reached(deadline) or time.monotonic() >= phase_timeout:
                    return _finish(
                        status="partial" if _deadline_reached(deadline) else "failed",
                        stopped_reason="max_seconds" if _deadline_reached(deadline) else "phase_timeout",
                        failure_reason=None if _deadline_reached(deadline) else "phase_timeout",
                        profile_name=resolved_profile,
                        selected_missing_suit=selected_missing_suit,
                        hand_suit_counts=hand_suit_counts,
                        auto_toggles_enabled=auto_toggles_enabled,
                        phase_trace=phase_trace,
                        start_time=start_time,
                        planned_actions=planned_actions,
                        executed_actions=executed_actions,
                        last_phase=phase,
                        dry_run=dry_run_enabled,
                        missing_suit_decision=missing_suit_decision,
                    )
                if poll_sec > 0:
                    _sleep_interruptibly(poll_sec)
                phase, capture = _capture_phase(app, yihuan_mahjong, profile_name=resolved_profile)
                last_phase = phase
                _append_trace(phase_trace, start_time=start_time, phase=phase, note="phase_wait")

        if phase["phase"] == "playing":
            auto_toggles_enabled, phase, capture = _ensure_auto_switches(
                app,
                yihuan_mahjong,
                profile=profile,
                phase=phase,
                capture=capture,
                requested=requested_switches,
                dry_run=dry_run_enabled,
                planned_actions=planned_actions,
                executed_actions=executed_actions,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            last_phase = phase

        poll_sec = float(profile["poll_ms"]) / 1000.0
        unknown_started_at: float | None = None
        while True:
            _raise_if_cancelled()
            if phase.get("phase") == "result":
                return _finish(
                    status="success",
                    stopped_reason="level_end",
                    failure_reason=None,
                    profile_name=resolved_profile,
                    selected_missing_suit=selected_missing_suit,
                    hand_suit_counts=hand_suit_counts,
                    auto_toggles_enabled=auto_toggles_enabled,
                    phase_trace=phase_trace,
                    start_time=start_time,
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                    last_phase=phase,
                    dry_run=dry_run_enabled,
                    missing_suit_decision=missing_suit_decision,
                )
            if _deadline_reached(deadline):
                return _finish(
                    status="partial",
                    stopped_reason="max_seconds",
                    failure_reason=None,
                    profile_name=resolved_profile,
                    selected_missing_suit=selected_missing_suit,
                    hand_suit_counts=hand_suit_counts,
                    auto_toggles_enabled=auto_toggles_enabled,
                    phase_trace=phase_trace,
                    start_time=start_time,
                    planned_actions=planned_actions,
                    executed_actions=executed_actions,
                    last_phase=phase,
                    dry_run=dry_run_enabled,
                    missing_suit_decision=missing_suit_decision,
                )
            if phase.get("phase") == "unknown":
                unknown_started_at = unknown_started_at or time.monotonic()
                if time.monotonic() - unknown_started_at >= float(profile["phase_timeout_sec"]):
                    return _finish(
                        status="failed",
                        stopped_reason="phase_timeout",
                        failure_reason="phase_timeout",
                        profile_name=resolved_profile,
                        selected_missing_suit=selected_missing_suit,
                        hand_suit_counts=hand_suit_counts,
                        auto_toggles_enabled=auto_toggles_enabled,
                        phase_trace=phase_trace,
                        start_time=start_time,
                        planned_actions=planned_actions,
                        executed_actions=executed_actions,
                        last_phase=phase,
                        dry_run=dry_run_enabled,
                        missing_suit_decision=missing_suit_decision,
                    )
            else:
                unknown_started_at = None
            if poll_sec > 0:
                _sleep_interruptibly(poll_sec)
            phase, capture = _capture_phase(app, yihuan_mahjong, profile_name=resolved_profile)
            last_phase = phase
            _append_trace(phase_trace, start_time=start_time, phase=phase, note="monitor")
    except _MahjongSessionCancelled:
        logger.info("Mahjong[session] cancelled phase=%s", (last_phase or {}).get("phase"))
        _release_all(app)
        return _finish(
            status="cancelled",
            stopped_reason="cancelled",
            failure_reason="cancelled",
            profile_name=resolved_profile,
            selected_missing_suit=selected_missing_suit,
            hand_suit_counts=hand_suit_counts,
            auto_toggles_enabled=auto_toggles_enabled,
            phase_trace=phase_trace,
            start_time=start_time,
            planned_actions=planned_actions,
            executed_actions=executed_actions,
            last_phase=last_phase,
            dry_run=dry_run_enabled,
            missing_suit_decision=missing_suit_decision,
        )
    except RuntimeError as exc:
        logger.warning("Mahjong[session] capture failed: %s", exc)
        return _finish(
            status="failed",
            stopped_reason="failure",
            failure_reason="capture_failed",
            profile_name=resolved_profile,
            selected_missing_suit=selected_missing_suit,
            hand_suit_counts=hand_suit_counts,
            auto_toggles_enabled=auto_toggles_enabled,
            phase_trace=phase_trace,
            start_time=start_time,
            planned_actions=planned_actions,
            executed_actions=executed_actions,
            last_phase=last_phase,
            dry_run=dry_run_enabled,
            missing_suit_decision=missing_suit_decision,
        )
    except _MahjongSessionStop:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Mahjong auto session failed.")
        return _finish(
            status="failed",
            stopped_reason="exception",
            failure_reason="exception",
            profile_name=resolved_profile,
            selected_missing_suit=selected_missing_suit,
            hand_suit_counts=hand_suit_counts,
            auto_toggles_enabled=auto_toggles_enabled,
            phase_trace=phase_trace,
            start_time=start_time,
            planned_actions=planned_actions,
            executed_actions=executed_actions,
            last_phase=last_phase,
            dry_run=dry_run_enabled,
            missing_suit_decision=missing_suit_decision,
        ) | {"failure_message": str(exc), "debug_snapshots": debug_snapshots}

"""Yihuan automatic combat session action."""

from __future__ import annotations

import time
from typing import Any, Mapping

from packages.aura_core.api import action_info, requires_services
from packages.aura_core.observability.logging.core_logger import logger

from ..services.combat_service import YihuanCombatService


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


def _capture_state(app: Any, yihuan_combat: YihuanCombatService, *, profile_name: str) -> tuple[dict[str, Any], Any]:
    capture = app.capture()
    if not capture.success or capture.image is None:
        raise RuntimeError("Failed to capture the Yihuan combat screen.")
    state = yihuan_combat.analyze_frame(capture.image, profile_name=profile_name)
    return state, capture


def _trim_trace(trace: list[dict[str, Any]], trace_limit: int) -> None:
    if len(trace) > trace_limit:
        del trace[: len(trace) - trace_limit]


def _append_state_trace(
    combat_state_trace: list[dict[str, Any]],
    *,
    start_time: float,
    state: Mapping[str, Any],
    combat_active: bool,
    note: str,
    trace_limit: int,
) -> None:
    combat_state_trace.append(
        {
            "t": round(time.monotonic() - start_time, 3),
            "note": note,
            "combat_active": bool(combat_active),
            "in_supported_scene": bool(state.get("in_supported_scene")),
            "in_combat": bool(state.get("in_combat")),
            "target_found": bool(state.get("target_found")),
            "enemy_health_found": bool(state.get("enemy_health_found")),
            "boss_found": bool(state.get("boss_found")),
            "current_slot": state.get("current_slot"),
            "skill_available": bool(state.get("skill_available")),
            "ultimate_available": bool(state.get("ultimate_available")),
            "arc_available": bool(state.get("arc_available")),
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
    last_state: Mapping[str, Any] | None,
    combat_state_trace: list[dict[str, Any]],
    action_trace: list[dict[str, Any]],
    start_time: float,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "status": status,
        "stopped_reason": stopped_reason,
        "failure_reason": failure_reason,
        "profile_name": profile_name,
        "strategy_name": strategy_name,
        "encounters_completed": int(encounters_completed),
        "last_state": dict(last_state or {}),
        "combat_state_trace": list(combat_state_trace),
        "action_trace": list(action_trace),
        "elapsed_sec": round(time.monotonic() - start_time, 3),
        "dry_run": bool(dry_run),
    }


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


def _tap_binding(
    app: Any,
    binding: str,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    action_name: str,
) -> None:
    normalized = str(binding or "").strip().lower()
    _record_action(
        action_trace,
        start_time=start_time,
        action=action_name,
        dry_run=dry_run,
        details={"binding": binding},
    )
    if dry_run:
        return
    if normalized == "mouse_left":
        app.click(button="left")
    elif normalized == "mouse_right":
        app.click(button="right")
    elif normalized == "mouse_middle":
        app.click(button="middle")
    else:
        app.press_key(str(binding), presses=1)


def _normal_attack(
    app: Any,
    binding: str,
    *,
    duration_ms: int,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> None:
    normalized = str(binding or "mouse_left").strip().lower()
    duration_sec = max(int(duration_ms), 0) / 1000.0
    _record_action(
        action_trace,
        start_time=start_time,
        action="normal",
        dry_run=dry_run,
        details={"binding": binding, "duration_ms": int(duration_ms)},
    )
    if dry_run:
        return
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


def _release_inputs(app: Any, profile: Mapping[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
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
    if dry_run:
        return
    seconds = max(int(milliseconds), 0) / 1000.0
    if seconds > 0:
        time.sleep(seconds)


def _is_available(state: Mapping[str, Any], name: str) -> bool:
    return bool(state.get(f"{name}_available"))


def _switch_slot(
    app: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    target: Any,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> None:
    team_size = max(_coerce_int(state.get("team_size"), len(profile.get("current_slot_regions") or [])), 1)
    current_slot = state.get("current_slot")
    if str(target).lower() == "next":
        if current_slot is None:
            slot = 1
        else:
            slot = (int(current_slot) % team_size) + 1
    else:
        slot = min(max(_coerce_int(target, 1), 1), team_size)

    binding = dict(profile["keys"]).get(f"switch_{slot}", str(slot))
    _tap_binding(
        app,
        binding,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        action_name="switch",
    )
    action_trace[-1]["slot"] = slot


def _execute_command(
    app: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    command: Any,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> None:
    keys = dict(profile["keys"])
    cooldown_ms = int(dict(profile["runtime"])["input_cooldown_ms"])

    if isinstance(command, str):
        name = command.strip()
        if name in {"skill", "ultimate", "arc"}:
            _tap_binding(
                app,
                keys[name],
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                action_name=name,
            )
            _sleep_ms(cooldown_ms, dry_run=dry_run)
            return
        if name == "normal":
            _normal_attack(
                app,
                keys["normal_attack"],
                duration_ms=900,
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
            )
            _sleep_ms(cooldown_ms, dry_run=dry_run)
            return
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
        if name not in {"skill", "ultimate", "arc"}:
            raise ValueError(f"Unsupported availability target: {name!r}")
        if _is_available(state, name):
            for nested_command in nested:
                _execute_command(
                    app,
                    profile,
                    state,
                    nested_command,
                    dry_run=dry_run,
                    action_trace=action_trace,
                    start_time=start_time,
                )
        else:
            _record_action(
                action_trace,
                start_time=start_time,
                action=f"skip_{name}",
                dry_run=dry_run,
                details={"reason": "not_available"},
            )
        return

    if "normal" in command:
        payload = command["normal"] or {}
        duration_ms = 900
        if isinstance(payload, Mapping):
            duration_ms = _coerce_int(payload.get("duration_ms"), 900)
        elif payload is not None:
            duration_ms = _coerce_int(payload, 900)
        _normal_attack(
            app,
            keys["normal_attack"],
            duration_ms=duration_ms,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
        )
        _sleep_ms(cooldown_ms, dry_run=dry_run)
        return

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
        return

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
        return

    if "switch" in command:
        _switch_slot(
            app,
            profile,
            state,
            command["switch"],
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
        )
        _sleep_ms(int(dict(profile["runtime"])["post_switch_delay_ms"]), dry_run=dry_run)
        return

    raise ValueError(f"Unsupported combat command: {command!r}")


def _execute_strategy(
    app: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    strategy_name: str,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> None:
    strategies = dict(profile.get("strategies") or {})
    strategy = dict(strategies.get(strategy_name) or {})
    if not strategy:
        raise ValueError(f"Combat strategy not found: {strategy_name}")
    commands = strategy.get("loop") or []
    if not isinstance(commands, list):
        raise ValueError(f"Combat strategy '{strategy_name}' loop must be a list.")
    for command in commands:
        _execute_command(
            app,
            profile,
            state,
            command,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
        )


@action_info(
    name="yihuan_combat_run_session",
    public=False,
    read_only=False,
    description="Run the Yihuan-specific automatic combat loop.",
)
@requires_services(
    app="plans/aura_base/app",
    yihuan_combat="yihuan_combat",
)
def yihuan_combat_run_session(
    app: Any,
    yihuan_combat: YihuanCombatService,
    profile_name: str = "default_1280x720_cn",
    strategy_name: str = "default",
    max_seconds: float | int | str = 0,
    max_encounters: int | str = 0,
    auto_target: bool | str = True,
    dry_run: bool | str = False,
    debug_enabled: bool | str = False,
) -> dict[str, Any]:
    start_time = time.monotonic()
    combat_state_trace: list[dict[str, Any]] = []
    action_trace: list[dict[str, Any]] = []
    last_state: dict[str, Any] | None = None
    encounters_completed = 0
    profile: dict[str, Any] | None = None

    try:
        profile = yihuan_combat.load_profile(profile_name)
        resolved_profile = str(profile["profile_name"])
        runtime = dict(profile["runtime"])
        dry_run_enabled = _coerce_bool(dry_run)
        auto_target_enabled = _coerce_bool(auto_target)
        _ = _coerce_bool(debug_enabled)
        duration_limit = max(_coerce_float(max_seconds, 0.0), 0.0)
        if dry_run_enabled and duration_limit <= 0:
            duration_limit = float(runtime["dry_run_max_seconds"])
        deadline = time.monotonic() + duration_limit if duration_limit > 0 else None
        encounter_limit = max(_coerce_int(max_encounters, 0), 0)
        poll_sec = max(float(runtime["poll_ms"]) / 1000.0, 0.01)
        enter_required = int(runtime["combat_enter_stable_frames"])
        exit_required = int(runtime["combat_exit_stable_frames"])
        unsupported_required = int(runtime["unsupported_scene_stable_frames"])
        idle_timeout = float(runtime["idle_timeout_sec"])
        trace_limit = int(runtime["trace_limit"])

        logger.info(
            "Combat[session] start profile=%s strategy=%s max_seconds=%.3f max_encounters=%s auto_target=%s dry_run=%s",
            resolved_profile,
            strategy_name,
            duration_limit,
            encounter_limit,
            auto_target_enabled,
            dry_run_enabled,
        )

        combat_active = False
        saw_combat = False
        enter_stable = 0
        exit_stable = 0
        unsupported_stable = 0
        last_target_attempt = 0.0
        idle_started_at = time.monotonic()

        while True:
            if deadline is not None and time.monotonic() >= deadline:
                stopped_reason = "dry_run_completed" if dry_run_enabled else "max_seconds"
                return _final_result(
                    status="success" if dry_run_enabled else "partial",
                    stopped_reason=stopped_reason,
                    failure_reason=None,
                    profile_name=resolved_profile,
                    strategy_name=strategy_name,
                    encounters_completed=encounters_completed,
                    last_state=last_state,
                    combat_state_trace=combat_state_trace,
                    action_trace=action_trace,
                    start_time=start_time,
                    dry_run=dry_run_enabled,
                )

            state, _capture = _capture_state(app, yihuan_combat, profile_name=resolved_profile)
            last_state = dict(state)
            raw_in_combat = bool(state.get("in_combat"))
            in_supported_scene = bool(state.get("in_supported_scene"))

            if in_supported_scene:
                unsupported_stable = 0
            else:
                unsupported_stable += 1

            if not combat_active and not saw_combat and unsupported_stable >= unsupported_required:
                _append_state_trace(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    note="not_in_supported_scene",
                    trace_limit=trace_limit,
                )
                return _final_result(
                    status="partial",
                    stopped_reason="not_in_supported_scene",
                    failure_reason=None,
                    profile_name=resolved_profile,
                    strategy_name=strategy_name,
                    encounters_completed=encounters_completed,
                    last_state=last_state,
                    combat_state_trace=combat_state_trace,
                    action_trace=action_trace,
                    start_time=start_time,
                    dry_run=dry_run_enabled,
                )

            if raw_in_combat:
                enter_stable += 1
                exit_stable = 0
            else:
                exit_stable += 1
                enter_stable = 0

            if not combat_active and enter_stable >= enter_required:
                combat_active = True
                saw_combat = True
                _append_state_trace(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    note="enter_combat",
                    trace_limit=trace_limit,
                )

            combat_hint = raw_in_combat or bool(state.get("enemy_health_found")) or bool(state.get("boss_found"))
            if (
                auto_target_enabled
                and combat_hint
                and not bool(state.get("target_found"))
                and time.monotonic() - last_target_attempt >= float(runtime["target_retry_interval_ms"]) / 1000.0
            ):
                _tap_binding(
                    app,
                    dict(profile["keys"])["target"],
                    dry_run=dry_run_enabled,
                    action_trace=action_trace,
                    start_time=start_time,
                    action_name="auto_target",
                )
                last_target_attempt = time.monotonic()

            if combat_active:
                if exit_stable >= exit_required:
                    combat_active = False
                    encounters_completed += 1
                    _append_state_trace(
                        combat_state_trace,
                        start_time=start_time,
                        state=state,
                        combat_active=combat_active,
                        note="exit_combat",
                        trace_limit=trace_limit,
                    )
                    if encounter_limit > 0 and encounters_completed >= encounter_limit:
                        return _final_result(
                            status="success",
                            stopped_reason="max_encounters",
                            failure_reason=None,
                            profile_name=resolved_profile,
                            strategy_name=strategy_name,
                            encounters_completed=encounters_completed,
                            last_state=last_state,
                            combat_state_trace=combat_state_trace,
                            action_trace=action_trace,
                            start_time=start_time,
                            dry_run=dry_run_enabled,
                        )
                    if encounter_limit == 0:
                        return _final_result(
                            status="success",
                            stopped_reason="out_of_combat",
                            failure_reason=None,
                            profile_name=resolved_profile,
                            strategy_name=strategy_name,
                            encounters_completed=encounters_completed,
                            last_state=last_state,
                            combat_state_trace=combat_state_trace,
                            action_trace=action_trace,
                            start_time=start_time,
                            dry_run=dry_run_enabled,
                        )
                    _append_state_trace(
                        combat_state_trace,
                        start_time=start_time,
                        state=state,
                        combat_active=combat_active,
                        note="monitor",
                        trace_limit=trace_limit,
                    )
                    time.sleep(poll_sec)
                    continue

                if not raw_in_combat:
                    _append_state_trace(
                        combat_state_trace,
                        start_time=start_time,
                        state=state,
                        combat_active=combat_active,
                        note="combat_exit_pending",
                        trace_limit=trace_limit,
                    )
                    time.sleep(poll_sec)
                    continue

                try:
                    _execute_strategy(
                        app,
                        profile,
                        state,
                        strategy_name=strategy_name,
                        dry_run=dry_run_enabled,
                        action_trace=action_trace,
                        start_time=start_time,
                    )
                except ValueError as exc:
                    logger.warning("Combat[strategy] failed: %s", exc)
                    return _final_result(
                        status="failed",
                        stopped_reason="failed",
                        failure_reason="strategy_failed",
                        profile_name=resolved_profile,
                        strategy_name=strategy_name,
                        encounters_completed=encounters_completed,
                        last_state=last_state,
                        combat_state_trace=combat_state_trace,
                        action_trace=action_trace,
                        start_time=start_time,
                        dry_run=dry_run_enabled,
                    )

            elif not saw_combat and idle_timeout > 0 and time.monotonic() - idle_started_at >= idle_timeout:
                _append_state_trace(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    note="idle_out_of_combat",
                    trace_limit=trace_limit,
                )
                return _final_result(
                    status="partial",
                    stopped_reason="out_of_combat",
                    failure_reason=None,
                    profile_name=resolved_profile,
                    strategy_name=strategy_name,
                    encounters_completed=encounters_completed,
                    last_state=last_state,
                    combat_state_trace=combat_state_trace,
                    action_trace=action_trace,
                    start_time=start_time,
                    dry_run=dry_run_enabled,
                )

            _append_state_trace(
                combat_state_trace,
                start_time=start_time,
                state=state,
                combat_active=combat_active,
                note="monitor",
                trace_limit=trace_limit,
            )
            time.sleep(poll_sec)

    except FileNotFoundError as exc:
        logger.warning("Combat[session] profile failed: %s", exc)
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason="profile_not_found",
            profile_name=str(profile_name),
            strategy_name=strategy_name,
            encounters_completed=encounters_completed,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
        )
    except RuntimeError as exc:
        logger.warning("Combat[session] capture failed: %s", exc)
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason="capture_failed",
            profile_name=str(profile_name),
            strategy_name=strategy_name,
            encounters_completed=encounters_completed,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Combat[session] unexpected failure: %s", exc)
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason="exception",
            profile_name=str(profile_name),
            strategy_name=strategy_name,
            encounters_completed=encounters_completed,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
        )
    finally:
        if profile is not None:
            _release_inputs(app, profile, dry_run=_coerce_bool(dry_run))

"""Cafe automation actions for the Yihuan plan."""

from __future__ import annotations

import time
from typing import Any

from packages.aura_core.api import action_info, requires_services
from packages.aura_core.observability.logging.core_logger import logger

from ..services.cafe_service import YihuanCafeService


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _click_point(app: Any, point: tuple[int, int], *, note: str, profile: dict[str, Any]) -> None:
    x, y = int(point[0]), int(point[1])
    repeat_count = max(int(profile.get("click_repeat_count", 2) or 2), 1)
    hold_ms = max(int(profile.get("click_hold_ms", 100) or 100), 0)
    repeat_interval_ms = max(int(profile.get("click_repeat_interval_ms", 100) or 100), 0)
    logger.info(
        "Cafe[input] click point=(%s,%s) note=%s repeat=%s hold_ms=%s interval_ms=%s",
        x,
        y,
        note,
        repeat_count,
        hold_ms,
        repeat_interval_ms,
    )
    app.move_to(x, y, duration=0.0)
    for index in range(repeat_count):
        app.mouse_down(button="left")
        _sleep_ms(hold_ms)
        app.mouse_up(button="left")
        if index < repeat_count - 1:
            _sleep_ms(repeat_interval_ms)


def _sleep_ms(ms: int) -> None:
    if ms > 0:
        time.sleep(float(ms) / 1000.0)


def _append_trace(
    phase_trace: list[dict[str, Any]],
    *,
    start_time: float,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    phase_trace.append(
        {
            "t": round(time.monotonic() - start_time, 3),
            "event": event,
            **dict(payload or {}),
        }
    )


def _coffee_stock_point(profile: dict[str, Any], coffee_stock: int) -> tuple[int, int]:
    stock_point = profile.get("coffee_stock_point")
    if stock_point:
        return int(stock_point[0]), int(stock_point[1])

    stock_points = list(profile.get("coffee_stock_points") or [])
    if not stock_points:
        raise ValueError("Cafe profile requires coffee_stock_point or coffee_stock_points")
    batch_size = int(profile["coffee_batch_size"])
    stock_index = max(batch_size - int(coffee_stock), 0)
    stock_index = min(stock_index, len(stock_points) - 1)
    return stock_points[stock_index]


def _ensure_coffee_stock(
    app: Any,
    profile: dict[str, Any],
    *,
    coffee_stock: int,
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> tuple[int, bool]:
    if coffee_stock > 0:
        return coffee_stock, False

    _click_point(app, profile["coffee_station_point"], note="make_coffee_batch", profile=profile)
    _append_trace(
        phase_trace,
        start_time=start_time,
        event="coffee_batch_started",
        payload={"point": list(profile["coffee_station_point"])},
    )
    time.sleep(float(profile["coffee_make_sec"]))
    return int(profile["coffee_batch_size"]), True


def _should_keep_stocking(
    *,
    orders_completed: int,
    max_orders: int,
    deadline: float,
) -> bool:
    if max_orders > 0 and orders_completed >= max_orders:
        return False
    return time.monotonic() < deadline


def _check_level_end(
    yihuan_cafe: YihuanCafeService,
    profile: dict[str, Any],
    image: Any,
    *,
    profile_name: str,
    stable_count: int,
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> tuple[int, bool, bool, dict[str, Any]]:
    detection = yihuan_cafe.detect_level_end(image, profile_name=profile_name)
    if bool(detection.get("ended")):
        stable_count += 1
        logger.info(
            "Cafe[level_end] detected outcome=%s reason=%s stable=%s/%s buttons=%s success_area=%s failure_area=%s",
            detection.get("outcome"),
            detection.get("reason"),
            stable_count,
            int(profile["level_end_stable_frames"]),
            detection.get("buttons_count"),
            dict(detection.get("success_title") or {}).get("area"),
            dict(detection.get("failure_title") or {}).get("area"),
        )
        if stable_count >= int(profile["level_end_stable_frames"]):
            _append_trace(
                phase_trace,
                start_time=start_time,
                event="level_end_detected",
                payload={
                    "outcome": detection.get("outcome"),
                    "reason": detection.get("reason"),
                    "buttons_count": detection.get("buttons_count"),
                    "button_bboxes": detection.get("button_bboxes"),
                    "success_area": dict(detection.get("success_title") or {}).get("area"),
                    "failure_area": dict(detection.get("failure_title") or {}).get("area"),
                },
            )
            return stable_count, True, True, detection
        return stable_count, False, True, detection

    if stable_count > 0:
        logger.info(
            "Cafe[level_end] stable reset reason=%s buttons=%s",
            detection.get("reason"),
            detection.get("buttons_count"),
        )
    return 0, False, False, detection


def _execute_recipe(
    app: Any,
    profile: dict[str, Any],
    recipe_id: str,
    *,
    coffee_stock: int,
) -> tuple[int, bool]:
    step_delay_ms = int(profile["step_delay_ms"])

    if recipe_id == "latte_coffee":
        _click_point(app, profile["glass_point"], note="latte_glass", profile=profile)
        _sleep_ms(step_delay_ms)
        _click_point(app, _coffee_stock_point(profile, coffee_stock), note="latte_coffee_stock", profile=profile)
        _sleep_ms(step_delay_ms)
        _click_point(app, profile["latte_art_point"], note="latte_art", profile=profile)
    elif recipe_id == "cream_coffee":
        _click_point(app, profile["coffee_cup_point"], note="cream_cup", profile=profile)
        _sleep_ms(step_delay_ms)
        _click_point(app, _coffee_stock_point(profile, coffee_stock), note="cream_coffee_stock", profile=profile)
        _sleep_ms(step_delay_ms)
        _click_point(app, profile["cream_point"], note="cream", profile=profile)
    else:
        raise ValueError(f"Unsupported cafe recipe: {recipe_id}")

    coffee_stock = max(int(coffee_stock) - 1, 0)
    _sleep_ms(int(profile["craft_delay_ms"]))
    return coffee_stock, False


def _wait_level_started(
    app: Any,
    yihuan_cafe: YihuanCafeService,
    profile: dict[str, Any],
    *,
    profile_name: str,
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> tuple[bool, str | None, dict[str, Any]]:
    timeout_sec = float(profile["level_started_timeout_sec"])
    poll_ms = int(profile["level_started_poll_ms"])
    stable_needed = int(profile["level_started_stable_frames"])
    deadline = time.monotonic() + timeout_sec
    stable_count = 0
    last_detection: dict[str, Any] = {
        "started": False,
        "reason": "not_checked",
        "green_area": 0,
        "bbox": None,
        "aspect_ratio": 0.0,
    }

    logger.info(
        "Cafe[level_start] waiting timeout=%.3f stable_frames=%s region=%s",
        timeout_sec,
        stable_needed,
        list(profile["level_started_region"]),
    )
    _append_trace(
        phase_trace,
        start_time=start_time,
        event="level_start_waiting",
        payload={
            "timeout_sec": timeout_sec,
            "stable_frames": stable_needed,
            "region": list(profile["level_started_region"]),
        },
    )

    while time.monotonic() < deadline:
        capture = app.capture()
        if not capture.success or capture.image is None:
            logger.error("Cafe[level_start] capture failed")
            return False, "level_start_capture_failed", last_detection

        last_detection = yihuan_cafe.detect_level_started(capture.image, profile_name=profile_name)
        if bool(last_detection.get("started")):
            stable_count += 1
            logger.info(
                "Cafe[level_start] detected area=%s bbox=%s aspect=%s stable=%s/%s",
                last_detection.get("green_area"),
                last_detection.get("bbox"),
                last_detection.get("aspect_ratio"),
                stable_count,
                stable_needed,
            )
            if stable_count >= stable_needed:
                _append_trace(
                    phase_trace,
                    start_time=start_time,
                    event="level_started_confirmed",
                    payload={
                        "green_area": last_detection.get("green_area"),
                        "bbox": last_detection.get("bbox"),
                        "aspect_ratio": last_detection.get("aspect_ratio"),
                    },
                )
                return True, None, last_detection
        else:
            if stable_count > 0:
                logger.info(
                    "Cafe[level_start] stable reset reason=%s area=%s",
                    last_detection.get("reason"),
                    last_detection.get("green_area"),
                )
            stable_count = 0

        _sleep_ms(poll_ms)

    logger.error(
        "Cafe[level_start] timeout last_reason=%s last_area=%s last_bbox=%s",
        last_detection.get("reason"),
        last_detection.get("green_area"),
        last_detection.get("bbox"),
    )
    _append_trace(
        phase_trace,
        start_time=start_time,
        event="level_start_timeout",
        payload={
            "reason": last_detection.get("reason"),
            "green_area": last_detection.get("green_area"),
            "bbox": last_detection.get("bbox"),
        },
    )
    return False, "level_start_timeout", last_detection


@action_info(
    name="yihuan_cafe_run_session",
    public=False,
    read_only=False,
    description="Run the Yihuan cafe mini-game loop for the coffee-only v1 profile.",
)
@requires_services(app="plans/aura_base/app", yihuan_cafe="yihuan_cafe")
def yihuan_cafe_run_session(
    app: Any,
    yihuan_cafe: YihuanCafeService,
    profile_name: str = "default_1280x720_cn",
    max_seconds: float | int | str = 0.0,
    max_orders: int | str = 0,
    start_game: bool | str = True,
    wait_level_started: bool | str = True,
) -> dict[str, Any]:
    profile = yihuan_cafe.load_profile(profile_name)
    resolved_profile = str(profile["profile_name"])
    max_seconds_value = _coerce_float(max_seconds, 0.0)
    if max_seconds_value <= 0:
        max_seconds_value = float(profile["max_seconds"])
    max_orders_value = max(_coerce_int(max_orders, 0), 0)
    start_game_value = _coerce_bool(start_game)
    wait_level_started_value = _coerce_bool(wait_level_started)

    start_time = time.monotonic()
    deadline = start_time + max_seconds_value
    phase_trace: list[dict[str, Any]] = []
    coffee_stock = 0
    coffee_batches_made = 0
    orders_completed = 0
    unknown_scan_count = 0
    recognized_counts = {"latte_coffee": 0, "cream_coffee": 0}
    failure_reason: str | None = None
    level_outcome: str | None = None
    level_end_detection: dict[str, Any] | None = None
    level_end_stable_count = 0

    logger.info(
        "Cafe[session] start profile=%s max_seconds=%.3f max_orders=%s start_game=%s wait_level_started=%s",
        resolved_profile,
        max_seconds_value,
        max_orders_value,
        start_game_value,
        wait_level_started_value,
    )

    try:
        if start_game_value:
            _click_point(app, profile["start_game_point"], note="start_game", profile=profile)
            _append_trace(
                phase_trace,
                start_time=start_time,
                event="start_game_clicked",
                payload={"point": list(profile["start_game_point"])},
            )
            time.sleep(float(profile["start_game_delay_sec"]))

        if wait_level_started_value:
            level_started, failure_reason, detection = _wait_level_started(
                app,
                yihuan_cafe,
                profile,
                profile_name=resolved_profile,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            if not level_started:
                return {
                    "status": "failed",
                    "stopped_reason": "failure",
                    "failure_reason": failure_reason,
                    "level_start_detection": detection,
                    "orders_completed": orders_completed,
                    "coffee_batches_made": coffee_batches_made,
                    "coffee_stock_remaining": coffee_stock,
                    "recognized_counts": recognized_counts,
                    "unknown_scan_count": unknown_scan_count,
                    "level_outcome": level_outcome,
                    "level_end_detection": level_end_detection,
                    "phase_trace": phase_trace,
                    "elapsed_sec": round(time.monotonic() - start_time, 3),
                    "profile_name": resolved_profile,
                }

        while True:
            now = time.monotonic()
            if max_orders_value > 0 and orders_completed >= max_orders_value:
                stopped_reason = "max_orders"
                break
            if now >= deadline:
                stopped_reason = "max_seconds"
                break

            capture = app.capture()
            if not capture.success or capture.image is None:
                failure_reason = "capture_failed"
                logger.error("Cafe[session] capture failed")
                return {
                    "status": "failed",
                    "stopped_reason": "failure",
                    "failure_reason": failure_reason,
                    "orders_completed": orders_completed,
                    "coffee_batches_made": coffee_batches_made,
                    "coffee_stock_remaining": coffee_stock,
                    "recognized_counts": recognized_counts,
                    "unknown_scan_count": unknown_scan_count,
                    "level_outcome": level_outcome,
                    "level_end_detection": level_end_detection,
                    "phase_trace": phase_trace,
                    "elapsed_sec": round(time.monotonic() - start_time, 3),
                    "profile_name": resolved_profile,
                }

            level_end_stable_count, should_stop, pending_level_end, detection = _check_level_end(
                yihuan_cafe,
                profile,
                capture.image,
                profile_name=resolved_profile,
                stable_count=level_end_stable_count,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            level_end_detection = detection
            if should_stop:
                level_outcome = str(detection.get("outcome") or "unknown")
                stopped_reason = "level_end"
                break
            if pending_level_end:
                _sleep_ms(int(profile["level_end_poll_ms"]))
                continue

            coffee_stock, made_batch = _ensure_coffee_stock(
                app,
                profile,
                coffee_stock=coffee_stock,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            if made_batch:
                coffee_batches_made += 1

            capture = app.capture()
            if not capture.success or capture.image is None:
                failure_reason = "capture_failed"
                logger.error("Cafe[session] capture failed")
                return {
                    "status": "failed",
                    "stopped_reason": "failure",
                    "failure_reason": failure_reason,
                    "orders_completed": orders_completed,
                    "coffee_batches_made": coffee_batches_made,
                    "coffee_stock_remaining": coffee_stock,
                    "recognized_counts": recognized_counts,
                    "unknown_scan_count": unknown_scan_count,
                    "level_outcome": level_outcome,
                    "level_end_detection": level_end_detection,
                    "phase_trace": phase_trace,
                    "elapsed_sec": round(time.monotonic() - start_time, 3),
                    "profile_name": resolved_profile,
                }

            level_end_stable_count, should_stop, pending_level_end, detection = _check_level_end(
                yihuan_cafe,
                profile,
                capture.image,
                profile_name=resolved_profile,
                stable_count=level_end_stable_count,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            level_end_detection = detection
            if should_stop:
                level_outcome = str(detection.get("outcome") or "unknown")
                stopped_reason = "level_end"
                break
            if pending_level_end:
                _sleep_ms(int(profile["level_end_poll_ms"]))
                continue

            orders = yihuan_cafe.analyze_orders(capture.image, profile_name=resolved_profile)
            if not orders:
                unknown_scan_count += 1
                if unknown_scan_count <= 5 or unknown_scan_count % 20 == 0:
                    logger.info("Cafe[scan] no supported order unknown_scan_count=%s", unknown_scan_count)
                _sleep_ms(int(profile["poll_ms"]))
                continue

            selected = orders[0]
            recipe_id = str(selected["recipe_id"])
            logger.info(
                "Cafe[order] selected=%s score=%.4f center=(%s,%s) all=%s stock=%s",
                recipe_id,
                float(selected["score"]),
                selected["center_x"],
                selected["center_y"],
                [
                    {
                        "recipe_id": order["recipe_id"],
                        "score": order["score"],
                        "center_x": order["center_x"],
                        "center_y": order["center_y"],
                    }
                    for order in orders
                ],
                coffee_stock,
            )
            _append_trace(
                phase_trace,
                start_time=start_time,
                event="order_selected",
                payload={
                    "recipe_id": recipe_id,
                    "score": selected["score"],
                    "center_x": selected["center_x"],
                    "center_y": selected["center_y"],
                    "coffee_stock_before": coffee_stock,
                },
            )

            coffee_stock, made_batch = _execute_recipe(
                app,
                profile,
                recipe_id,
                coffee_stock=coffee_stock,
            )
            orders_completed += 1
            recognized_counts[recipe_id] = int(recognized_counts.get(recipe_id, 0)) + 1
            logger.info(
                "Cafe[order] completed=%s orders_completed=%s stock_remaining=%s batches=%s",
                recipe_id,
                orders_completed,
                coffee_stock,
                coffee_batches_made,
            )
            _append_trace(
                phase_trace,
                start_time=start_time,
                event="order_completed",
                payload={
                    "recipe_id": recipe_id,
                    "coffee_stock_after": coffee_stock,
                    "coffee_batches_made": coffee_batches_made,
                    "orders_completed": orders_completed,
                },
            )

            capture = app.capture()
            if not capture.success or capture.image is None:
                failure_reason = "capture_failed"
                logger.error("Cafe[session] capture failed after order")
                return {
                    "status": "failed",
                    "stopped_reason": "failure",
                    "failure_reason": failure_reason,
                    "orders_completed": orders_completed,
                    "coffee_batches_made": coffee_batches_made,
                    "coffee_stock_remaining": coffee_stock,
                    "recognized_counts": recognized_counts,
                    "unknown_scan_count": unknown_scan_count,
                    "level_outcome": level_outcome,
                    "level_end_detection": level_end_detection,
                    "phase_trace": phase_trace,
                    "elapsed_sec": round(time.monotonic() - start_time, 3),
                    "profile_name": resolved_profile,
                }

            level_end_stable_count, should_stop, pending_level_end, detection = _check_level_end(
                yihuan_cafe,
                profile,
                capture.image,
                profile_name=resolved_profile,
                stable_count=level_end_stable_count,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            level_end_detection = detection
            if should_stop:
                level_outcome = str(detection.get("outcome") or "unknown")
                stopped_reason = "level_end"
                break
            if pending_level_end:
                _sleep_ms(int(profile["level_end_poll_ms"]))
                continue

            if coffee_stock <= 0 and _should_keep_stocking(
                orders_completed=orders_completed,
                max_orders=max_orders_value,
                deadline=deadline,
            ):
                logger.info(
                    "Cafe[stock] depleted after order=%s orders_completed=%s; replenishing before next scan",
                    recipe_id,
                    orders_completed,
                )
                _append_trace(
                    phase_trace,
                    start_time=start_time,
                    event="coffee_stock_depleted_after_order",
                    payload={"orders_completed": orders_completed},
                )
                coffee_stock, made_batch = _ensure_coffee_stock(
                    app,
                    profile,
                    coffee_stock=coffee_stock,
                    phase_trace=phase_trace,
                    start_time=start_time,
                )
                if made_batch:
                    coffee_batches_made += 1

        logger.info("Cafe[session] stopped reason=%s orders_completed=%s", stopped_reason, orders_completed)
        return {
            "status": "success",
            "stopped_reason": stopped_reason,
            "failure_reason": None,
            "orders_completed": orders_completed,
            "coffee_batches_made": coffee_batches_made,
            "coffee_stock_remaining": coffee_stock,
            "recognized_counts": recognized_counts,
            "unknown_scan_count": unknown_scan_count,
            "level_outcome": level_outcome,
            "level_end_detection": level_end_detection,
            "phase_trace": phase_trace,
            "elapsed_sec": round(time.monotonic() - start_time, 3),
            "profile_name": resolved_profile,
        }
    except Exception as exc:  # noqa: BLE001
        failure_reason = type(exc).__name__
        logger.exception("Cafe[session] failed: %s", exc)
        return {
            "status": "failed",
            "stopped_reason": "exception",
            "failure_reason": failure_reason,
            "failure_message": str(exc),
            "orders_completed": orders_completed,
            "coffee_batches_made": coffee_batches_made,
            "coffee_stock_remaining": coffee_stock,
            "recognized_counts": recognized_counts,
            "unknown_scan_count": unknown_scan_count,
            "level_outcome": level_outcome,
            "level_end_detection": level_end_detection,
            "phase_trace": phase_trace,
            "elapsed_sec": round(time.monotonic() - start_time, 3),
            "profile_name": resolved_profile,
        }

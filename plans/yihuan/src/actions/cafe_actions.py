"""Cafe automation actions for the Yihuan plan."""

from __future__ import annotations

import time
from typing import Any

from packages.aura_core.api import action_info, requires_services
from packages.aura_core.observability.logging.core_logger import logger

from ..services.cafe_service import YihuanCafeService


RECIPE_STOCK: dict[str, str] = {
    "latte_coffee": "coffee",
    "cream_coffee": "coffee",
    "bacon_bread": "bread",
    "egg_croissant": "croissant",
    "jam_cake": "cake",
}


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


def _perf_time(perf_stats: dict[str, Any], key: str, elapsed_sec: float) -> None:
    time_sec = perf_stats.setdefault("time_sec", {})
    time_sec[key] = float(time_sec.get(key, 0.0)) + max(float(elapsed_sec), 0.0)


def _perf_count(perf_stats: dict[str, Any], key: str, amount: int = 1) -> None:
    counts = perf_stats.setdefault("counts", {})
    counts[key] = int(counts.get(key, 0)) + int(amount)


def _perf_observe(perf_stats: dict[str, Any], group: str, key: str, value_sec: float) -> None:
    value = max(float(value_sec), 0.0)
    bucket = perf_stats.setdefault(group, {})
    item = bucket.setdefault(key, {"count": 0, "total_sec": 0.0, "min_sec": None, "max_sec": None})
    item["count"] = int(item.get("count", 0)) + 1
    item["total_sec"] = float(item.get("total_sec", 0.0)) + value
    item["min_sec"] = value if item.get("min_sec") is None else min(float(item["min_sec"]), value)
    item["max_sec"] = value if item.get("max_sec") is None else max(float(item["max_sec"]), value)


def _round_perf_stats(perf_stats: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {"time_sec": {}, "counts": dict(perf_stats.get("counts") or {})}
    for key, value in dict(perf_stats.get("time_sec") or {}).items():
        rounded["time_sec"][key] = round(float(value), 3)

    for group in ("recipe_elapsed", "order_gap"):
        rounded[group] = {}
        for key, raw_item in dict(perf_stats.get(group) or {}).items():
            item = dict(raw_item)
            count = int(item.get("count", 0))
            total = float(item.get("total_sec", 0.0))
            rounded[group][key] = {
                "count": count,
                "total_sec": round(total, 3),
                "avg_sec": round(total / count, 3) if count else 0.0,
                "min_sec": round(float(item["min_sec"]), 3) if item.get("min_sec") is not None else None,
                "max_sec": round(float(item["max_sec"]), 3) if item.get("max_sec") is not None else None,
            }
    return rounded


def _enabled_stock_profiles(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = dict(profile.get("stock_profiles") or {})
    return {
        str(stock_id): dict(stock_profile)
        for stock_id, stock_profile in profiles.items()
        if bool(dict(stock_profile).get("enabled", True))
    }


def _start_stock_batch(
    app: Any,
    profile: dict[str, Any],
    *,
    stock_id: str,
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    stock_profile = dict(profile["stock_profiles"][stock_id])
    station_point = tuple(stock_profile["station_point"])
    batch_size = int(stock_profile["batch_size"])
    make_sec = float(stock_profile["make_sec"])
    _click_point(app, station_point, note=f"make_{stock_id}_batch", profile=profile)
    _append_trace(
        phase_trace,
        start_time=start_time,
        event="stock_batch_started",
        payload={
            "stock_id": stock_id,
            "point": list(station_point),
            "batch_size": batch_size,
            "make_sec": make_sec,
        },
    )
    logger.info(
        "Cafe[stock] batch_started stock_id=%s point=%s batch_size=%s make_sec=%.3f",
        stock_id,
        list(station_point),
        batch_size,
        make_sec,
    )
    return {"stock_id": stock_id, "batch_size": batch_size, "make_sec": make_sec}


def _ensure_all_depleted_stocks(
    app: Any,
    profile: dict[str, Any],
    *,
    stocks: dict[str, int],
    batches_made: dict[str, int],
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> list[str]:
    started: list[dict[str, Any]] = []
    for stock_id in _enabled_stock_profiles(profile):
        if int(stocks.get(stock_id, 0)) > 0:
            continue
        started.append(
            _start_stock_batch(
                app,
                profile,
                stock_id=stock_id,
                phase_trace=phase_trace,
                start_time=start_time,
            )
        )

    if not started:
        return []

    wait_sec = max(float(item["make_sec"]) for item in started)
    logger.info(
        "Cafe[stock] parallel_wait stock_ids=%s wait_sec=%.3f",
        [item["stock_id"] for item in started],
        wait_sec,
    )
    if wait_sec > 0:
        time.sleep(wait_sec)

    made: list[str] = []
    for item in started:
        stock_id = str(item["stock_id"])
        stocks[stock_id] = int(item["batch_size"])
        batches_made[stock_id] = int(batches_made.get(stock_id, 0)) + 1
        made.append(stock_id)
    return made


def _sync_stocks_from_visual(
    yihuan_cafe: YihuanCafeService,
    image: Any,
    *,
    profile_name: str,
    stocks: dict[str, int],
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    detection = yihuan_cafe.detect_stock_status(image, profile_name=profile_name)
    corrections: list[dict[str, Any]] = []
    for stock_id, stock_detection in dict(detection.get("stocks") or {}).items():
        if int(stocks.get(str(stock_id), 0)) <= 0:
            continue
        if stock_detection.get("present") is not False:
            continue
        previous_stock = int(stocks.get(str(stock_id), 0))
        stocks[str(stock_id)] = 0
        correction = {
            "stock_id": str(stock_id),
            "previous_stock": previous_stock,
            "product_pixels": stock_detection.get("product_pixels"),
            "min_pixels": stock_detection.get("min_pixels"),
            "region": stock_detection.get("region"),
        }
        corrections.append(correction)
        logger.warning(
            "Cafe[stock_monitor] visual_empty stock_id=%s previous_stock=%s product_pixels=%s min_pixels=%s region=%s",
            correction["stock_id"],
            previous_stock,
            correction["product_pixels"],
            correction["min_pixels"],
            correction["region"],
        )

    if corrections:
        _append_trace(
            phase_trace,
            start_time=start_time,
            event="stock_visual_empty_correction",
            payload={"corrections": corrections},
        )
    return {"detection": detection, "corrections": corrections}


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


def _click_stock_item(
    app: Any,
    profile: dict[str, Any],
    *,
    stock_id: str,
    note: str,
) -> None:
    stock_profile = dict(profile["stock_profiles"][stock_id])
    _click_point(app, tuple(stock_profile["stock_point"]), note=note, profile=profile)


def _execute_recipe(
    app: Any,
    profile: dict[str, Any],
    recipe_id: str,
    *,
    stocks: dict[str, int],
) -> str:
    if recipe_id not in RECIPE_STOCK:
        raise ValueError(f"Unsupported cafe recipe: {recipe_id}")

    stock_id = RECIPE_STOCK[recipe_id]
    if stock_id not in _enabled_stock_profiles(profile):
        raise ValueError(f"Cafe stock disabled for recipe {recipe_id}: {stock_id}")
    if int(stocks.get(stock_id, 0)) <= 0:
        raise ValueError(f"Cafe stock depleted before recipe {recipe_id}: {stock_id}")

    step_delay_ms = int(profile["step_delay_ms"])

    if recipe_id == "latte_coffee":
        _click_point(app, profile["glass_point"], note="latte_glass", profile=profile)
        _sleep_ms(step_delay_ms)
        _click_stock_item(app, profile, stock_id=stock_id, note="latte_coffee_stock")
        _sleep_ms(step_delay_ms)
        _click_point(app, profile["latte_art_point"], note="latte_art", profile=profile)
    elif recipe_id == "cream_coffee":
        _click_point(app, profile["coffee_cup_point"], note="cream_cup", profile=profile)
        _sleep_ms(step_delay_ms)
        _click_stock_item(app, profile, stock_id=stock_id, note="cream_coffee_stock")
        _sleep_ms(step_delay_ms)
        _click_point(app, profile["cream_point"], note="cream", profile=profile)
    elif recipe_id == "bacon_bread":
        _click_stock_item(app, profile, stock_id=stock_id, note="bacon_bread_stock")
        _sleep_ms(step_delay_ms)
        _click_point(app, profile["bacon_point"], note="bacon", profile=profile)
    elif recipe_id == "egg_croissant":
        _click_stock_item(app, profile, stock_id=stock_id, note="egg_croissant_stock")
        _sleep_ms(step_delay_ms)
        _click_point(app, profile["egg_point"], note="egg", profile=profile)
    elif recipe_id == "jam_cake":
        _click_stock_item(app, profile, stock_id=stock_id, note="jam_cake_stock")
        _sleep_ms(step_delay_ms)
        _click_point(app, profile["jam_point"], note="jam", profile=profile)

    stocks[stock_id] = max(int(stocks.get(stock_id, 0)) - 1, 0)
    _sleep_ms(int(profile["craft_delay_ms"]))
    return stock_id


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
    description="Run the Yihuan cafe mini-game loop with supported stock and recipe automation.",
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
    stock_profiles = _enabled_stock_profiles(profile)
    stocks = {stock_id: 0 for stock_id in stock_profiles}
    batches_made = {stock_id: 0 for stock_id in stock_profiles}
    orders_completed = 0
    unknown_scan_count = 0
    recognized_counts = {recipe_id: 0 for recipe_id in RECIPE_STOCK}
    failure_reason: str | None = None
    level_outcome: str | None = None
    level_end_detection: dict[str, Any] | None = None
    level_end_stable_count = 0
    perf_stats: dict[str, Any] = {
        "time_sec": {},
        "counts": {},
        "recipe_elapsed": {},
        "order_gap": {},
    }
    last_order_completed_at: float | None = None

    def result_payload(
        *,
        status: str,
        stopped_reason: str,
        failure_reason_value: str | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "stopped_reason": stopped_reason,
            "failure_reason": failure_reason_value,
            "orders_completed": orders_completed,
            "batches_made": dict(batches_made),
            "stocks_remaining": dict(stocks),
            "coffee_batches_made": int(batches_made.get("coffee", 0)),
            "coffee_stock_remaining": int(stocks.get("coffee", 0)),
            "recognized_counts": dict(recognized_counts),
            "unknown_scan_count": unknown_scan_count,
            "level_outcome": level_outcome,
            "level_end_detection": level_end_detection,
            "perf_stats": _round_perf_stats(perf_stats),
            "phase_trace": phase_trace,
            "elapsed_sec": round(time.monotonic() - start_time, 3),
            "profile_name": resolved_profile,
        }
        payload.update(extra or {})
        return payload

    logger.info(
        "Cafe[session] start profile=%s max_seconds=%.3f max_orders=%s start_game=%s wait_level_started=%s stocks=%s",
        resolved_profile,
        max_seconds_value,
        max_orders_value,
        start_game_value,
        wait_level_started_value,
        list(stock_profiles),
    )

    try:
        if start_game_value:
            click_start = time.monotonic()
            _click_point(app, profile["start_game_point"], note="start_game", profile=profile)
            _perf_time(perf_stats, "start_game_click", time.monotonic() - click_start)
            _perf_count(perf_stats, "start_game_click_count")
            _append_trace(
                phase_trace,
                start_time=start_time,
                event="start_game_clicked",
                payload={"point": list(profile["start_game_point"])},
            )
            delay_start = time.monotonic()
            time.sleep(float(profile["start_game_delay_sec"]))
            _perf_time(perf_stats, "start_game_delay", time.monotonic() - delay_start)

        if wait_level_started_value:
            wait_start = time.monotonic()
            level_started, failure_reason, detection = _wait_level_started(
                app,
                yihuan_cafe,
                profile,
                profile_name=resolved_profile,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            _perf_time(perf_stats, "level_start_wait", time.monotonic() - wait_start)
            if not level_started:
                return result_payload(
                    status="failed",
                    stopped_reason="failure",
                    failure_reason_value=failure_reason,
                    extra={"level_start_detection": detection},
                )

        while True:
            now = time.monotonic()
            if max_orders_value > 0 and orders_completed >= max_orders_value:
                stopped_reason = "max_orders"
                break
            if now >= deadline:
                stopped_reason = "max_seconds"
                break

            capture_start = time.monotonic()
            capture = app.capture()
            _perf_time(perf_stats, "capture_before_stock", time.monotonic() - capture_start)
            _perf_count(perf_stats, "capture_count")
            if not capture.success or capture.image is None:
                failure_reason = "capture_failed"
                logger.error("Cafe[session] capture failed")
                return result_payload(
                    status="failed",
                    stopped_reason="failure",
                    failure_reason_value=failure_reason,
                )

            level_end_start = time.monotonic()
            level_end_stable_count, should_stop, pending_level_end, detection = _check_level_end(
                yihuan_cafe,
                profile,
                capture.image,
                profile_name=resolved_profile,
                stable_count=level_end_stable_count,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            _perf_time(perf_stats, "level_end_check_before_stock", time.monotonic() - level_end_start)
            _perf_count(perf_stats, "level_end_check_count")
            level_end_detection = detection
            if should_stop:
                level_outcome = str(detection.get("outcome") or "unknown")
                stopped_reason = "level_end"
                break
            if pending_level_end:
                sleep_start = time.monotonic()
                _sleep_ms(int(profile["level_end_poll_ms"]))
                _perf_time(perf_stats, "level_end_pending_sleep", time.monotonic() - sleep_start)
                continue

            stock_sync_start = time.monotonic()
            stock_sync_result = _sync_stocks_from_visual(
                yihuan_cafe,
                capture.image,
                profile_name=resolved_profile,
                stocks=stocks,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            _perf_time(perf_stats, "stock_visual_sync_before_scan", time.monotonic() - stock_sync_start)
            _perf_count(perf_stats, "stock_visual_sync_count")
            _perf_count(
                perf_stats,
                "stock_visual_correction_count",
                len(list(stock_sync_result.get("corrections") or [])),
            )

            restock_start = time.monotonic()
            made_stocks = _ensure_all_depleted_stocks(
                app,
                profile,
                stocks=stocks,
                batches_made=batches_made,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            restock_elapsed = time.monotonic() - restock_start
            _perf_time(perf_stats, "restock_before_scan", restock_elapsed)
            _perf_count(perf_stats, "restock_before_scan_call_count")
            _perf_count(perf_stats, "restock_stock_started_count", len(made_stocks))
            if made_stocks:
                logger.info(
                    "Cafe[stock] replenished_before_scan stocks=%s remaining=%s elapsed_sec=%.3f",
                    made_stocks,
                    stocks,
                    restock_elapsed,
                )

            capture_start = time.monotonic()
            capture = app.capture()
            _perf_time(perf_stats, "capture_before_scan", time.monotonic() - capture_start)
            _perf_count(perf_stats, "capture_count")
            if not capture.success or capture.image is None:
                failure_reason = "capture_failed"
                logger.error("Cafe[session] capture failed")
                return result_payload(
                    status="failed",
                    stopped_reason="failure",
                    failure_reason_value=failure_reason,
                )

            level_end_start = time.monotonic()
            level_end_stable_count, should_stop, pending_level_end, detection = _check_level_end(
                yihuan_cafe,
                profile,
                capture.image,
                profile_name=resolved_profile,
                stable_count=level_end_stable_count,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            _perf_time(perf_stats, "level_end_check_before_scan", time.monotonic() - level_end_start)
            _perf_count(perf_stats, "level_end_check_count")
            level_end_detection = detection
            if should_stop:
                level_outcome = str(detection.get("outcome") or "unknown")
                stopped_reason = "level_end"
                break
            if pending_level_end:
                sleep_start = time.monotonic()
                _sleep_ms(int(profile["level_end_poll_ms"]))
                _perf_time(perf_stats, "level_end_pending_sleep", time.monotonic() - sleep_start)
                continue

            scan_start = time.monotonic()
            orders = yihuan_cafe.analyze_orders(capture.image, profile_name=resolved_profile)
            scan_elapsed = time.monotonic() - scan_start
            scan_debug = {}
            if hasattr(yihuan_cafe, "get_last_order_scan_debug"):
                scan_debug = dict(yihuan_cafe.get_last_order_scan_debug() or {})
            _perf_time(perf_stats, "order_scan", scan_elapsed)
            _perf_count(perf_stats, "order_scan_count")
            _perf_time(perf_stats, "order_scan_locator", float(scan_debug.get("locator_sec", 0.0) or 0.0))
            _perf_time(perf_stats, "order_scan_classify", float(scan_debug.get("classification_sec", 0.0) or 0.0))
            _perf_time(perf_stats, "order_scan_fallback", float(scan_debug.get("fallback_sec", 0.0) or 0.0))
            if bool(scan_debug.get("fallback_used")):
                _perf_count(perf_stats, "order_scan_fallback_count")
            if scan_debug.get("fallback_skipped_reason"):
                _perf_count(perf_stats, "order_scan_fallback_skipped_count")
            logger.info(
                "Cafe[perf] scan elapsed_sec=%.3f orders=%s unknown_scan_count=%s stocks=%s",
                scan_elapsed,
                len(orders),
                unknown_scan_count,
                dict(stocks),
            )
            if scan_debug:
                logger.info(
                    "Cafe[scan_debug] strategy=%s locators=%s classified=%s fallback_used=%s fallback_allowed=%s "
                    "fallback_reason=%s fallback_skipped_reason=%s miss_count=%s locator_ms=%.1f classify_ms=%.1f "
                    "fallback_ms=%.1f total_ms=%.1f",
                    scan_debug.get("strategy"),
                    scan_debug.get("locator_count"),
                    scan_debug.get("classified_count"),
                    bool(scan_debug.get("fallback_used")),
                    bool(scan_debug.get("fallback_allowed")),
                    scan_debug.get("fallback_reason"),
                    scan_debug.get("fallback_skipped_reason"),
                    scan_debug.get("two_stage_miss_count"),
                    float(scan_debug.get("locator_sec", 0.0) or 0.0) * 1000.0,
                    float(scan_debug.get("classification_sec", 0.0) or 0.0) * 1000.0,
                    float(scan_debug.get("fallback_sec", 0.0) or 0.0) * 1000.0,
                    float(scan_debug.get("total_sec", 0.0) or 0.0) * 1000.0,
                )
            if not orders:
                unknown_scan_count += 1
                _perf_count(perf_stats, "no_order_scan_count")
                if unknown_scan_count <= 5 or unknown_scan_count % 20 == 0:
                    logger.info("Cafe[scan] no supported order unknown_scan_count=%s", unknown_scan_count)
                sleep_start = time.monotonic()
                _sleep_ms(int(profile["poll_ms"]))
                _perf_time(perf_stats, "no_order_sleep", time.monotonic() - sleep_start)
                continue

            selected = orders[0]
            recipe_id = str(selected["recipe_id"])
            order_selected_at = time.monotonic()
            if last_order_completed_at is not None:
                gap_sec = order_selected_at - last_order_completed_at
                _perf_observe(perf_stats, "order_gap", recipe_id, gap_sec)
                logger.info(
                    "Cafe[perf] order_gap recipe=%s previous_completed_to_selected_sec=%.3f",
                    recipe_id,
                    gap_sec,
                )
            logger.info(
                "Cafe[order] selected=%s score=%.4f center=(%s,%s) all=%s stocks=%s batches=%s",
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
                dict(stocks),
                dict(batches_made),
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
                    "stocks_before": dict(stocks),
                },
            )

            recipe_start = time.monotonic()
            consumed_stock_id = _execute_recipe(app, profile, recipe_id, stocks=stocks)
            recipe_elapsed = time.monotonic() - recipe_start
            selected_to_completed_sec = time.monotonic() - order_selected_at
            _perf_time(perf_stats, "recipe_execute", recipe_elapsed)
            _perf_observe(perf_stats, "recipe_elapsed", recipe_id, selected_to_completed_sec)
            logger.info(
                "Cafe[perf] order_timing recipe=%s selected_to_completed_sec=%.3f recipe_execute_sec=%.3f",
                recipe_id,
                selected_to_completed_sec,
                recipe_elapsed,
            )
            orders_completed += 1
            recognized_counts[recipe_id] = int(recognized_counts.get(recipe_id, 0)) + 1
            last_order_completed_at = time.monotonic()
            logger.info(
                "Cafe[order] completed=%s consumed_stock=%s orders_completed=%s stocks=%s batches=%s",
                recipe_id,
                consumed_stock_id,
                orders_completed,
                dict(stocks),
                dict(batches_made),
            )
            _append_trace(
                phase_trace,
                start_time=start_time,
                event="order_completed",
                payload={
                    "recipe_id": recipe_id,
                    "consumed_stock_id": consumed_stock_id,
                    "stocks_after": dict(stocks),
                    "batches_made": dict(batches_made),
                    "orders_completed": orders_completed,
                },
            )

            capture_start = time.monotonic()
            capture = app.capture()
            _perf_time(perf_stats, "capture_after_order", time.monotonic() - capture_start)
            _perf_count(perf_stats, "capture_count")
            if not capture.success or capture.image is None:
                failure_reason = "capture_failed"
                logger.error("Cafe[session] capture failed after order")
                return result_payload(
                    status="failed",
                    stopped_reason="failure",
                    failure_reason_value=failure_reason,
                )

            level_end_start = time.monotonic()
            level_end_stable_count, should_stop, pending_level_end, detection = _check_level_end(
                yihuan_cafe,
                profile,
                capture.image,
                profile_name=resolved_profile,
                stable_count=level_end_stable_count,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            _perf_time(perf_stats, "level_end_check_after_order", time.monotonic() - level_end_start)
            _perf_count(perf_stats, "level_end_check_count")
            level_end_detection = detection
            if should_stop:
                level_outcome = str(detection.get("outcome") or "unknown")
                stopped_reason = "level_end"
                break
            if pending_level_end:
                sleep_start = time.monotonic()
                _sleep_ms(int(profile["level_end_poll_ms"]))
                _perf_time(perf_stats, "level_end_pending_sleep", time.monotonic() - sleep_start)
                continue

            stock_sync_start = time.monotonic()
            stock_sync_result = _sync_stocks_from_visual(
                yihuan_cafe,
                capture.image,
                profile_name=resolved_profile,
                stocks=stocks,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            _perf_time(perf_stats, "stock_visual_sync_after_order", time.monotonic() - stock_sync_start)
            _perf_count(perf_stats, "stock_visual_sync_count")
            _perf_count(
                perf_stats,
                "stock_visual_correction_count",
                len(list(stock_sync_result.get("corrections") or [])),
            )

            depleted_stock_ids = [
                stock_id
                for stock_id in stock_profiles
                if int(stocks.get(stock_id, 0)) <= 0
            ]
            if depleted_stock_ids and _should_keep_stocking(
                orders_completed=orders_completed,
                max_orders=max_orders_value,
                deadline=deadline,
            ):
                logger.info(
                    "Cafe[stock] depleted_after_order=%s orders_completed=%s; replenishing before next scan",
                    depleted_stock_ids,
                    orders_completed,
                )
                _append_trace(
                    phase_trace,
                    start_time=start_time,
                    event="stock_depleted_after_order",
                    payload={
                        "stock_ids": depleted_stock_ids,
                        "orders_completed": orders_completed,
                    },
                )
                restock_start = time.monotonic()
                _ensure_all_depleted_stocks(
                    app,
                    profile,
                    stocks=stocks,
                    batches_made=batches_made,
                    phase_trace=phase_trace,
                    start_time=start_time,
                )
                restock_elapsed = time.monotonic() - restock_start
                _perf_time(perf_stats, "restock_after_order", restock_elapsed)
                _perf_count(perf_stats, "restock_after_order_call_count")
                _perf_count(perf_stats, "restock_stock_started_count", len(depleted_stock_ids))
                logger.info(
                    "Cafe[perf] restock_after_order elapsed_sec=%.3f depleted=%s remaining=%s",
                    restock_elapsed,
                    depleted_stock_ids,
                    dict(stocks),
                )

        logger.info(
            "Cafe[session] stopped reason=%s orders_completed=%s stocks=%s batches=%s",
            stopped_reason,
            orders_completed,
            dict(stocks),
            dict(batches_made),
        )
        logger.info("Cafe[perf] summary=%s", _round_perf_stats(perf_stats))
        return result_payload(
            status="success",
            stopped_reason=stopped_reason,
            failure_reason_value=None,
        )
    except Exception as exc:  # noqa: BLE001
        failure_reason = type(exc).__name__
        logger.exception("Cafe[session] failed: %s", exc)
        logger.info("Cafe[perf] summary=%s", _round_perf_stats(perf_stats))
        return result_payload(
            status="failed",
            stopped_reason="exception",
            failure_reason_value=failure_reason,
            extra={"failure_message": str(exc)},
        )

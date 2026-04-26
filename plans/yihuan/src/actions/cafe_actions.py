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
    single_click_notes = {str(item) for item in list(profile.get("single_click_notes") or [])}
    if str(note) in single_click_notes:
        repeat_count = 1
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


def _complete_ready_stock_batches(
    pending_batches: dict[str, dict[str, Any]],
    stocks: dict[str, int],
    *,
    now: float | None = None,
    phase_trace: list[dict[str, Any]],
    start_time: float,
) -> list[str]:
    ready_time = time.monotonic() if now is None else float(now)
    completed: list[str] = []
    for stock_id, item in list(pending_batches.items()):
        if float(item["ready_at"]) > ready_time:
            continue
        stocks[stock_id] = int(item["batch_size"])
        pending_batches.pop(stock_id, None)
        completed.append(stock_id)
        _append_trace(
            phase_trace,
            start_time=start_time,
            event="stock_batch_ready",
            payload={
                "stock_id": stock_id,
                "batch_size": int(item["batch_size"]),
                "ready_at": round(float(item["ready_at"]) - start_time, 3),
            },
        )
        logger.info(
            "Cafe[stock] batch_ready stock_id=%s batch_size=%s remaining=%s",
            stock_id,
            int(item["batch_size"]),
            dict(stocks),
        )
    return completed


def _start_depleted_stocks_async(
    app: Any,
    profile: dict[str, Any],
    *,
    stocks: dict[str, int],
    batches_made: dict[str, int],
    pending_batches: dict[str, dict[str, Any]],
    phase_trace: list[dict[str, Any]],
    start_time: float,
    stock_ids: list[str] | None = None,
) -> list[str]:
    started: list[str] = []
    enabled_profiles = _enabled_stock_profiles(profile)
    requested_stock_ids = list(stock_ids) if stock_ids is not None else list(enabled_profiles)
    for stock_id in requested_stock_ids:
        if stock_id not in enabled_profiles:
            continue
        if int(stocks.get(stock_id, 0)) > 0:
            continue
        if stock_id in pending_batches:
            continue
        item = _start_stock_batch(
            app,
            profile,
            stock_id=stock_id,
            phase_trace=phase_trace,
            start_time=start_time,
        )
        now = time.monotonic()
        item["started_at"] = now
        item["ready_at"] = now + float(item["make_sec"])
        pending_batches[stock_id] = item
        batches_made[stock_id] = int(batches_made.get(stock_id, 0)) + 1
        started.append(stock_id)
        logger.info(
            "Cafe[stock] batch_pending stock_id=%s ready_in_sec=%.3f pending=%s batches=%s",
            stock_id,
            float(item["make_sec"]),
            sorted(pending_batches),
            dict(batches_made),
        )

    return started


def _wait_for_stock_ready(
    app: Any,
    profile: dict[str, Any],
    *,
    stock_id: str,
    stocks: dict[str, int],
    batches_made: dict[str, int],
    pending_batches: dict[str, dict[str, Any]],
    phase_trace: list[dict[str, Any]],
    start_time: float,
    perf_stats: dict[str, Any],
) -> bool:
    _complete_ready_stock_batches(
        pending_batches,
        stocks,
        phase_trace=phase_trace,
        start_time=start_time,
    )
    if int(stocks.get(stock_id, 0)) > 0:
        return True

    if stock_id not in pending_batches:
        _start_depleted_stocks_async(
            app,
            profile,
            stocks=stocks,
            batches_made=batches_made,
            pending_batches=pending_batches,
            phase_trace=phase_trace,
            start_time=start_time,
            stock_ids=[stock_id],
        )

    pending = pending_batches.get(stock_id)
    if pending is None:
        return int(stocks.get(stock_id, 0)) > 0

    remaining_sec = max(float(pending["ready_at"]) - time.monotonic(), 0.0)
    logger.info(
        "Cafe[stock] wait_for_batch stock_id=%s wait_sec=%.3f pending=%s",
        stock_id,
        remaining_sec,
        sorted(pending_batches),
    )
    wait_start = time.monotonic()
    if remaining_sec > 0:
        time.sleep(remaining_sec)
    _perf_time(perf_stats, "stock_async_wait", time.monotonic() - wait_start)
    _perf_count(perf_stats, "stock_async_wait_count")
    _complete_ready_stock_batches(
        pending_batches,
        stocks,
        now=float(pending["ready_at"]),
        phase_trace=phase_trace,
        start_time=start_time,
    )
    return int(stocks.get(stock_id, 0)) > 0


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


def _candidate_brief(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "center_x": candidate.get("center_x"),
        "center_y": candidate.get("center_y"),
        "area": candidate.get("area"),
        "bbox": candidate.get("bbox"),
        "red_pixels": candidate.get("red_pixels"),
        "blue_pixels": candidate.get("blue_pixels"),
    }


def _candidate_center(candidate: dict[str, Any]) -> tuple[int, int]:
    return int(candidate.get("center_x", 0) or 0), int(candidate.get("center_y", 0) or 0)


def _prune_recent_fake_hammers(
    fake_customer_state: dict[str, Any],
    *,
    now: float,
    cooldown_sec: float,
) -> list[dict[str, Any]]:
    if cooldown_sec <= 0:
        fake_customer_state["recent_hammers"] = []
        return []

    recent: list[dict[str, Any]] = []
    for item in list(fake_customer_state.get("recent_hammers") or []):
        try:
            hammered_at = float(item.get("at", 0.0) or 0.0)
        except AttributeError:
            continue
        if now - hammered_at < cooldown_sec:
            recent.append(
                {
                    "center_x": int(item.get("center_x", 0) or 0),
                    "center_y": int(item.get("center_y", 0) or 0),
                    "at": hammered_at,
                }
            )
    fake_customer_state["recent_hammers"] = recent
    return recent


def _find_same_spot_hammer(
    candidate: dict[str, Any],
    recent_hammers: list[dict[str, Any]],
    *,
    distance_px: int,
) -> dict[str, Any] | None:
    center_x, center_y = _candidate_center(candidate)
    distance_sq_limit = float(distance_px * distance_px)
    best: dict[str, Any] | None = None
    best_distance_sq: float | None = None
    for item in recent_hammers:
        dx = center_x - int(item.get("center_x", 0) or 0)
        dy = center_y - int(item.get("center_y", 0) or 0)
        distance_sq = float(dx * dx + dy * dy)
        if distance_sq > distance_sq_limit:
            continue
        if best is None or distance_sq < float(best_distance_sq or 0.0):
            best = item
            best_distance_sq = distance_sq
    return best


def _drive_fake_customer_if_needed(
    app: Any,
    yihuan_cafe: YihuanCafeService,
    profile: dict[str, Any],
    image: Any,
    *,
    profile_name: str,
    phase_trace: list[dict[str, Any]],
    start_time: float,
    fake_customer_state: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    if not bool(profile.get("fake_customer_enabled", True)):
        return {
            "detected": False,
            "reason": "disabled",
            "region": list(profile.get("fake_customer_region") or []),
            "red_pixels": 0,
            "component_count": 0,
            "kept_count": 0,
            "candidates": [],
        }, False

    detection = yihuan_cafe.detect_fake_customers(image, profile_name=profile_name)
    raw_candidates = list(detection.get("candidates") or [])
    if not bool(detection.get("detected")):
        clear_frames = int(fake_customer_state.get("clear_frames", 0)) + 1
        fake_customer_state["clear_frames"] = clear_frames
        fake_customer_state["visible_frames"] = 0
        if clear_frames >= int(profile.get("fake_customer_clear_frames", 2) or 2):
            fake_customer_state["last_signature"] = []
        return detection, False

    now = time.monotonic()
    fake_customer_state["clear_frames"] = 0
    fake_customer_state["visible_frames"] = int(fake_customer_state.get("visible_frames", 0)) + 1
    fake_customer_state["last_signature"] = [
        [int(candidate.get("center_x", 0) or 0), int(candidate.get("center_y", 0) or 0)]
        for candidate in raw_candidates
    ]
    same_spot_cooldown_sec = max(float(profile.get("fake_customer_same_spot_cooldown_sec", 3.0) or 3.0), 0.0)
    same_spot_distance_px = max(int(profile.get("fake_customer_same_spot_distance_px", 85) or 85), 1)
    recent_hammers = _prune_recent_fake_hammers(
        fake_customer_state,
        now=now,
        cooldown_sec=same_spot_cooldown_sec,
    )

    confirm_frames = max(int(profile.get("fake_customer_confirm_frames", 2) or 2), 1)
    if int(fake_customer_state["visible_frames"]) < confirm_frames:
        detection["reason"] = "confirming"
        logger.info(
            "Cafe[fake_customer] skipped reason=confirming visible_frames=%s/%s candidates=%s",
            int(fake_customer_state["visible_frames"]),
            confirm_frames,
            [_candidate_brief(candidate) for candidate in raw_candidates],
        )
        return detection, False

    actionable_candidates: list[dict[str, Any]] = []
    cooldown_matches: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        matched = _find_same_spot_hammer(
            candidate,
            recent_hammers,
            distance_px=same_spot_distance_px,
        )
        if matched is None:
            actionable_candidates.append(candidate)
            continue
        remaining_sec = max(same_spot_cooldown_sec - (now - float(matched.get("at", now) or now)), 0.0)
        cooldown_matches.append(
            {
                "candidate": _candidate_brief(candidate),
                "matched_center": [int(matched.get("center_x", 0) or 0), int(matched.get("center_y", 0) or 0)],
                "remaining_sec": round(remaining_sec, 3),
            }
        )

    if not actionable_candidates:
        detection["reason"] = "same_spot_cooldown"
        detection["cooldown_matches"] = cooldown_matches
        logger.info(
            "Cafe[fake_customer] skipped reason=same_spot_cooldown distance_px=%s matches=%s",
            same_spot_distance_px,
            cooldown_matches,
        )
        return detection, False

    selected_candidate = actionable_candidates[0]
    candidates = [selected_candidate]
    if len(actionable_candidates) < len(raw_candidates):
        detection["same_spot_cooldown_matches"] = cooldown_matches
    detection["all_candidates"] = raw_candidates
    detection["candidates"] = candidates
    hammer_point = tuple(profile.get("fake_customer_hammer_point") or (70, 325))
    logger.info(
        "Cafe[fake_customer] detected actionable=%s raw=%s candidates=%s; clicking hammer point=%s",
        len(actionable_candidates),
        len(raw_candidates),
        [
            {
                "center_x": candidate.get("center_x"),
                "center_y": candidate.get("center_y"),
                "area": candidate.get("area"),
                "bbox": candidate.get("bbox"),
                "body_bbox": candidate.get("body_bbox"),
                "red_pixels": candidate.get("red_pixels"),
                "blue_pixels": candidate.get("blue_pixels"),
                "red_blue_vertical_gap_px": candidate.get("red_blue_vertical_gap_px"),
            }
            for candidate in candidates
        ],
        list(hammer_point),
    )
    debug_image_path = None
    if bool(profile.get("fake_customer_hammer_debug_enabled", True)) and hasattr(
        yihuan_cafe,
        "save_fake_customer_hammer_debug_image",
    ):
        try:
            debug_sequence = int(fake_customer_state.get("hammer_debug_index", 0) or 0) + 1
            fake_customer_state["hammer_debug_index"] = debug_sequence
            debug_image_path = yihuan_cafe.save_fake_customer_hammer_debug_image(
                image,
                detection,
                profile_name=profile_name,
                sequence=debug_sequence,
            )
            if debug_image_path:
                detection["debug_image_path"] = debug_image_path
                logger.info("Cafe[fake_customer] hammer_debug_image saved path=%s", debug_image_path)
        except Exception as exc:  # pragma: no cover - diagnostics must never block gameplay.
            logger.warning("Cafe[fake_customer] failed to save hammer debug image: %s", exc)

    _click_point(app, hammer_point, note="fake_customer_hammer", profile=profile)
    fake_customer_state["last_hammer_at"] = now
    fake_customer_state["last_hammer_center"] = [
        int(selected_candidate.get("center_x", 0) or 0),
        int(selected_candidate.get("center_y", 0) or 0),
    ]
    recent_hammers.append(
        {
            "center_x": int(selected_candidate.get("center_x", 0) or 0),
            "center_y": int(selected_candidate.get("center_y", 0) or 0),
            "at": now,
        }
    )
    fake_customer_state["recent_hammers"] = recent_hammers
    _append_trace(
        phase_trace,
        start_time=start_time,
        event="fake_customer_hammer_clicked",
        payload={
            "hammer_point": list(hammer_point),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "all_candidate_count": len(raw_candidates),
            "red_pixels": detection.get("red_pixels"),
            "component_count": detection.get("component_count"),
            "debug_image_path": debug_image_path,
        },
    )
    _sleep_ms(int(profile.get("fake_customer_after_drive_ms", 0) or 0))
    return detection, True


def _click_stock_item(
    app: Any,
    profile: dict[str, Any],
    *,
    stock_id: str,
    note: str,
) -> None:
    stock_profile = dict(profile["stock_profiles"][stock_id])
    _click_point(app, tuple(stock_profile["stock_point"]), note=note, profile=profile)


def _stock_id_for_recipe(recipe_id: str) -> str:
    if recipe_id not in RECIPE_STOCK:
        raise ValueError(f"Unsupported cafe recipe: {recipe_id}")
    return RECIPE_STOCK[recipe_id]


def _select_order_with_ready_stock(
    orders: list[dict[str, Any]],
    *,
    stocks: dict[str, int],
) -> dict[str, Any]:
    for order in orders:
        stock_id = _stock_id_for_recipe(str(order["recipe_id"]))
        if int(stocks.get(stock_id, 0)) > 0:
            return order
    return orders[0]


def _execute_recipe(
    app: Any,
    profile: dict[str, Any],
    recipe_id: str,
    *,
    stocks: dict[str, int],
) -> str:
    stock_id = _stock_id_for_recipe(recipe_id)
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
    min_order_interval_sec: float | int | str | None = None,
    min_order_duration_sec: float | int | str | None = None,
) -> dict[str, Any]:
    profile = yihuan_cafe.load_profile(profile_name)
    resolved_profile = str(profile["profile_name"])
    if hasattr(yihuan_cafe, "reset_order_scan_state"):
        yihuan_cafe.reset_order_scan_state(resolved_profile)
    max_seconds_value = _coerce_float(max_seconds, 0.0)
    if max_seconds_value <= 0:
        max_seconds_value = float(profile["max_seconds"])
    max_orders_value = max(_coerce_int(max_orders, 0), 0)
    start_game_value = _coerce_bool(start_game)
    wait_level_started_value = _coerce_bool(wait_level_started)
    profile_order_interval_sec = float(profile.get("min_order_interval_sec", 0.0) or 0.0)
    min_order_interval_sec_value = max(
        _coerce_float(min_order_interval_sec, profile_order_interval_sec),
        0.0,
    )
    profile_order_duration_sec = float(profile.get("min_order_duration_sec", 0.0) or 0.0)
    min_order_duration_sec_value = max(
        _coerce_float(min_order_duration_sec, profile_order_duration_sec),
        0.0,
    )

    start_time = time.monotonic()
    deadline = start_time + max_seconds_value
    phase_trace: list[dict[str, Any]] = []
    stock_profiles = _enabled_stock_profiles(profile)
    stocks = {stock_id: 0 for stock_id in stock_profiles}
    batches_made = {stock_id: 0 for stock_id in stock_profiles}
    pending_batches: dict[str, dict[str, Any]] = {}
    orders_completed = 0
    unknown_scan_count = 0
    recognized_counts = {recipe_id: 0 for recipe_id in RECIPE_STOCK}
    failure_reason: str | None = None
    level_outcome: str | None = None
    level_end_detection: dict[str, Any] | None = None
    level_end_stable_count = 0
    fake_customers_detected = 0
    fake_customers_driven = 0
    last_fake_customer_detection: dict[str, Any] | None = None
    fake_customer_state: dict[str, Any] = {
        "visible_frames": 0,
        "clear_frames": 0,
        "last_hammer_at": 0.0,
        "recent_hammers": [],
    }
    perf_stats: dict[str, Any] = {
        "time_sec": {},
        "counts": {},
        "recipe_elapsed": {},
        "order_gap": {},
    }
    last_order_completed_at: float | None = None
    last_order_started_at: float | None = None
    order_pacing_wait_applied = True

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
            "pending_batches": {
                stock_id: {
                    "batch_size": int(item["batch_size"]),
                    "ready_in_sec": round(max(float(item["ready_at"]) - time.monotonic(), 0.0), 3),
                }
                for stock_id, item in pending_batches.items()
            },
            "coffee_batches_made": int(batches_made.get("coffee", 0)),
            "coffee_stock_remaining": int(stocks.get("coffee", 0)),
            "recognized_counts": dict(recognized_counts),
            "unknown_scan_count": unknown_scan_count,
            "fake_customers_detected": fake_customers_detected,
            "fake_customers_driven": fake_customers_driven,
            "last_fake_customer_detection": last_fake_customer_detection,
            "level_outcome": level_outcome,
            "level_end_detection": level_end_detection,
            "perf_stats": _round_perf_stats(perf_stats),
            "phase_trace": phase_trace,
            "elapsed_sec": round(time.monotonic() - start_time, 3),
            "profile_name": resolved_profile,
            "min_order_interval_sec": min_order_interval_sec_value,
            "min_order_duration_sec": min_order_duration_sec_value,
        }
        payload.update(extra or {})
        return payload

    logger.info(
        "Cafe[session] start profile=%s max_seconds=%.3f max_orders=%s start_game=%s "
        "wait_level_started=%s min_order_interval_sec=%.3f min_order_duration_sec=%.3f stocks=%s",
        resolved_profile,
        max_seconds_value,
        max_orders_value,
        start_game_value,
        wait_level_started_value,
        min_order_interval_sec_value,
        min_order_duration_sec_value,
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
            ready_stocks = _complete_ready_stock_batches(
                pending_batches,
                stocks,
                now=now,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            _perf_count(perf_stats, "stock_batch_ready_count", len(ready_stocks))
            if max_orders_value > 0 and orders_completed >= max_orders_value:
                stopped_reason = "max_orders"
                break
            if now >= deadline:
                stopped_reason = "max_seconds"
                break
            if (
                (min_order_interval_sec_value > 0 or min_order_duration_sec_value > 0)
                and last_order_completed_at is not None
                and not order_pacing_wait_applied
            ):
                wait_now = time.monotonic()
                elapsed_since_completed = wait_now - last_order_completed_at
                elapsed_since_started = (
                    wait_now - last_order_started_at
                    if last_order_started_at is not None
                    else elapsed_since_completed
                )
                interval_remaining_sec = max(
                    min_order_interval_sec_value - elapsed_since_completed,
                    0.0,
                )
                duration_remaining_sec = max(
                    min_order_duration_sec_value - elapsed_since_started,
                    0.0,
                )
                wait_sec = max(interval_remaining_sec, duration_remaining_sec)
                order_pacing_wait_applied = True
                if wait_sec > 0:
                    wait_sec = min(wait_sec, max(deadline - time.monotonic(), 0.0))
                    if wait_sec > 0:
                        limiting_rule = (
                            "duration"
                            if duration_remaining_sec >= interval_remaining_sec
                            else "interval"
                        )
                        logger.info(
                            "Cafe[order] pacing_wait wait_sec=%.3f rule=%s "
                            "interval_remaining_sec=%.3f duration_remaining_sec=%.3f "
                            "min_order_interval_sec=%.3f min_order_duration_sec=%.3f "
                            "elapsed_since_completed_sec=%.3f elapsed_since_started_sec=%.3f",
                            wait_sec,
                            limiting_rule,
                            interval_remaining_sec,
                            duration_remaining_sec,
                            min_order_interval_sec_value,
                            min_order_duration_sec_value,
                            elapsed_since_completed,
                            elapsed_since_started,
                        )
                        _append_trace(
                            phase_trace,
                            start_time=start_time,
                            event="order_pacing_wait",
                            payload={
                                "wait_sec": round(wait_sec, 3),
                                "rule": limiting_rule,
                                "min_order_interval_sec": min_order_interval_sec_value,
                                "min_order_duration_sec": min_order_duration_sec_value,
                                "interval_remaining_sec": round(interval_remaining_sec, 3),
                                "duration_remaining_sec": round(duration_remaining_sec, 3),
                                "elapsed_since_completed_sec": round(elapsed_since_completed, 3),
                                "elapsed_since_started_sec": round(elapsed_since_started, 3),
                            },
                        )
                        wait_start = time.monotonic()
                        time.sleep(wait_sec)
                        actual_wait_sec = time.monotonic() - wait_start
                        _perf_time(perf_stats, "order_pacing_wait", actual_wait_sec)
                        _perf_count(perf_stats, "order_pacing_wait_count")
                        if interval_remaining_sec > 0:
                            _perf_count(perf_stats, "min_order_interval_wait_count")
                        if duration_remaining_sec > 0:
                            _perf_count(perf_stats, "min_order_duration_wait_count")
                now = time.monotonic()
                ready_stocks = _complete_ready_stock_batches(
                    pending_batches,
                    stocks,
                    now=now,
                    phase_trace=phase_trace,
                    start_time=start_time,
                )
                _perf_count(perf_stats, "stock_batch_ready_count", len(ready_stocks))
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

            fake_scan_start = time.monotonic()
            fake_detection, fake_driven = _drive_fake_customer_if_needed(
                app,
                yihuan_cafe,
                profile,
                capture.image,
                profile_name=resolved_profile,
                phase_trace=phase_trace,
                start_time=start_time,
                fake_customer_state=fake_customer_state,
            )
            _perf_time(perf_stats, "fake_customer_scan_before_stock", time.monotonic() - fake_scan_start)
            _perf_count(perf_stats, "fake_customer_scan_count")
            last_fake_customer_detection = fake_detection
            if fake_driven:
                candidates = list(fake_detection.get("candidates") or [])
                fake_customers_detected += len(candidates)
                fake_customers_driven += 1
                _perf_count(perf_stats, "fake_customer_detected_count", len(candidates))
                _perf_count(perf_stats, "fake_customer_hammer_click_count")
                continue
            if str(fake_detection.get("reason") or "") in {
                "confirming",
                "same_spot_cooldown",
            }:
                _perf_count(perf_stats, "fake_customer_cooldown_skip_count")

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
            started_stocks = _start_depleted_stocks_async(
                app,
                profile,
                stocks=stocks,
                batches_made=batches_made,
                pending_batches=pending_batches,
                phase_trace=phase_trace,
                start_time=start_time,
            )
            restock_elapsed = time.monotonic() - restock_start
            _perf_time(perf_stats, "restock_before_scan", restock_elapsed)
            _perf_count(perf_stats, "restock_before_scan_call_count")
            _perf_count(perf_stats, "restock_stock_started_count", len(started_stocks))
            if started_stocks:
                logger.info(
                    "Cafe[stock] restock_started_before_scan stocks=%s remaining=%s pending=%s elapsed_sec=%.3f",
                    started_stocks,
                    stocks,
                    sorted(pending_batches),
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

            fake_scan_start = time.monotonic()
            fake_detection, fake_driven = _drive_fake_customer_if_needed(
                app,
                yihuan_cafe,
                profile,
                capture.image,
                profile_name=resolved_profile,
                phase_trace=phase_trace,
                start_time=start_time,
                fake_customer_state=fake_customer_state,
            )
            _perf_time(perf_stats, "fake_customer_scan_before_order", time.monotonic() - fake_scan_start)
            _perf_count(perf_stats, "fake_customer_scan_count")
            last_fake_customer_detection = fake_detection
            if fake_driven:
                candidates = list(fake_detection.get("candidates") or [])
                fake_customers_detected += len(candidates)
                fake_customers_driven += 1
                _perf_count(perf_stats, "fake_customer_detected_count", len(candidates))
                _perf_count(perf_stats, "fake_customer_hammer_click_count")
                continue
            if str(fake_detection.get("reason") or "") in {
                "confirming",
                "same_spot_cooldown",
            }:
                _perf_count(perf_stats, "fake_customer_cooldown_skip_count")

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
                    "fallback_ms=%.1f total_ms=%.1f order_sort=%s track_count=%s returned_tracks=%s arrivals=%s",
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
                    scan_debug.get("order_sort_strategy"),
                    scan_debug.get("order_track_count"),
                    scan_debug.get("returned_track_ids"),
                    scan_debug.get("returned_arrival_indices"),
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

            selected = _select_order_with_ready_stock(orders, stocks=stocks)
            recipe_id = str(selected["recipe_id"])
            selected_index = orders.index(selected)
            selected_stock_id = _stock_id_for_recipe(recipe_id)
            if selected_index > 0:
                logger.info(
                    "Cafe[order] selected_ready_stock recipe=%s stock_id=%s selected_index=%s "
                    "fifo_head_recipe=%s fifo_head_stock=%s stocks=%s pending=%s",
                    recipe_id,
                    selected_stock_id,
                    selected_index,
                    orders[0].get("recipe_id"),
                    _stock_id_for_recipe(str(orders[0]["recipe_id"])),
                    dict(stocks),
                    sorted(pending_batches),
                )
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
                        "track_id": order.get("track_id"),
                        "arrival_index": order.get("arrival_index"),
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
                    "track_id": selected.get("track_id"),
                    "arrival_index": selected.get("arrival_index"),
                    "stocks_before": dict(stocks),
                },
            )

            recipe_start = time.monotonic()
            stock_wait_start = time.monotonic()
            stock_ready = _wait_for_stock_ready(
                app,
                profile,
                stock_id=selected_stock_id,
                stocks=stocks,
                batches_made=batches_made,
                pending_batches=pending_batches,
                phase_trace=phase_trace,
                start_time=start_time,
                perf_stats=perf_stats,
            )
            stock_wait_elapsed = time.monotonic() - stock_wait_start
            if stock_wait_elapsed > 0:
                logger.info(
                    "Cafe[perf] stock_wait_before_recipe recipe=%s stock_id=%s elapsed_sec=%.3f stocks=%s pending=%s",
                    recipe_id,
                    selected_stock_id,
                    stock_wait_elapsed,
                    dict(stocks),
                    sorted(pending_batches),
                )
            if not stock_ready:
                raise RuntimeError(f"Cafe stock not ready for recipe {recipe_id}: {selected_stock_id}")
            order_make_started_at = time.monotonic()
            consumed_stock_id = _execute_recipe(app, profile, recipe_id, stocks=stocks)
            order_track_removed = False
            if hasattr(yihuan_cafe, "mark_order_completed"):
                order_track_removed = bool(
                    yihuan_cafe.mark_order_completed(selected, profile_name=resolved_profile)
                )
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
            last_order_started_at = order_make_started_at
            order_pacing_wait_applied = False
            logger.info(
                "Cafe[order] completed=%s consumed_stock=%s orders_completed=%s track_id=%s "
                "arrival_index=%s track_removed=%s stocks=%s batches=%s",
                recipe_id,
                consumed_stock_id,
                orders_completed,
                selected.get("track_id"),
                selected.get("arrival_index"),
                order_track_removed,
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
                    "track_id": selected.get("track_id"),
                    "arrival_index": selected.get("arrival_index"),
                    "track_removed": order_track_removed,
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
                    "Cafe[stock] depleted_after_order=%s orders_completed=%s; starting async restock before next scan",
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
                started_stocks = _start_depleted_stocks_async(
                    app,
                    profile,
                    stocks=stocks,
                    batches_made=batches_made,
                    pending_batches=pending_batches,
                    phase_trace=phase_trace,
                    start_time=start_time,
                    stock_ids=depleted_stock_ids,
                )
                restock_elapsed = time.monotonic() - restock_start
                _perf_time(perf_stats, "restock_after_order", restock_elapsed)
                _perf_count(perf_stats, "restock_after_order_call_count")
                _perf_count(perf_stats, "restock_stock_started_count", len(started_stocks))
                logger.info(
                    "Cafe[perf] restock_after_order elapsed_sec=%.3f depleted=%s started=%s remaining=%s pending=%s",
                    restock_elapsed,
                    depleted_stock_ids,
                    started_stocks,
                    dict(stocks),
                    sorted(pending_batches),
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

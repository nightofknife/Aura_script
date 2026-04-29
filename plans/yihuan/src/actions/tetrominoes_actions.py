"""Tetrominoes actions for the Yihuan plan."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from packages.aura_core.api import action_info, requires_services
from packages.aura_core.observability.logging.core_logger import logger

from ..services.tetrominoes_service import YihuanTetrominoesService


class _TetrominoesSessionStop(Exception):
    """Internal control-flow marker for an already classified session stop."""


def _capture_state(
    app: Any,
    yihuan_tetrominoes: YihuanTetrominoesService,
    *,
    profile_name: str | None,
    update_tracker: bool,
) -> tuple[dict[str, Any], Any]:
    capture = app.capture()
    if not capture.success or capture.image is None:
        raise RuntimeError("Failed to capture the Yihuan Tetrominoes screen.")
    state = yihuan_tetrominoes.analyze_state(
        capture.image,
        profile_name=profile_name,
        update_tracker=update_tracker,
    )
    return state, capture


def _capture_image(app: Any) -> Any:
    capture = app.capture()
    if not capture.success or capture.image is None:
        raise RuntimeError("Failed to capture the Yihuan Tetrominoes screen.")
    return capture


def _click_start_game(app: Any, profile: dict[str, Any]) -> None:
    x, y = profile["start_game_point"]
    logger.info("Tetrominoes[start] click_start point=(%s,%s)", x, y)
    app.click(int(x), int(y), button="left", clicks=1)
    delay_sec = float(profile["start_game_delay_ms"]) / 1000.0
    if delay_sec > 0:
        logger.info("Tetrominoes[start] wait_after_click sec=%.3f", delay_sec)
        time.sleep(delay_sec)


def _click_result_exit(app: Any, profile: dict[str, Any]) -> None:
    x, y = profile["result_exit_point"]
    logger.info("Tetrominoes[start] click_result_exit point=(%s,%s)", x, y)
    app.click(int(x), int(y), button="left", clicks=1)
    delay_sec = float(profile["result_exit_delay_ms"]) / 1000.0
    if delay_sec > 0:
        logger.info("Tetrominoes[start] wait_after_exit sec=%.3f", delay_sec)
        time.sleep(delay_sec)


def _clear_result_screen_before_start(
    app: Any,
    yihuan_tetrominoes: YihuanTetrominoesService,
    *,
    profile: dict[str, Any],
) -> bool:
    capture = _capture_image(app)
    result_screen = yihuan_tetrominoes.analyze_result_screen(
        capture.image,
        profile_name=profile["profile_name"],
    )
    if not result_screen["found"]:
        return False
    logger.info(
        "Tetrominoes[start] existing result_screen detected before start panel_ratio=%.3f exit_ratio=%.3f",
        float(result_screen.get("panel_purple_ratio") or 0.0),
        float(result_screen.get("exit_white_ratio") or 0.0),
    )
    _click_result_exit(app, profile)
    return True


def _state_is_playing(state: dict[str, Any], profile: dict[str, Any]) -> bool:
    board = dict(state.get("board") or {})
    current_piece = dict(state.get("current_piece") or {})
    try:
        origin_row = int(current_piece.get("origin_row"))
    except (TypeError, ValueError):
        origin_row = int(profile["board_rows"])
    return (
        float(board.get("confidence") or 0.0) >= float(profile["board_confidence_min"])
        and int(board.get("occupied_count") or 0) >= 4
        and bool(current_piece.get("found"))
        and origin_row <= int(profile["start_piece_max_origin_row"])
    )


def _wait_for_game_start(
    app: Any,
    yihuan_tetrominoes: YihuanTetrominoesService,
    *,
    profile: dict[str, Any],
) -> dict[str, Any]:
    deadline = time.monotonic() + float(profile["start_timeout_sec"])
    poll_sec = float(profile["start_poll_ms"]) / 1000.0
    last_result: dict[str, Any] | None = None
    last_state: dict[str, Any] | None = None
    while True:
        capture = _capture_image(app)
        result_screen = yihuan_tetrominoes.analyze_result_screen(
            capture.image,
            profile_name=profile["profile_name"],
        )
        last_result = result_screen
        if result_screen["found"]:
            logger.info(
                "Tetrominoes[start] result_screen detected reason=%s panel_ratio=%.3f exit_ratio=%.3f",
                result_screen.get("reason"),
                float(result_screen.get("panel_purple_ratio") or 0.0),
                float(result_screen.get("exit_white_ratio") or 0.0),
            )
            return {
                "phase": "result",
                "capture": capture,
                "state": None,
                "result_screen": result_screen,
            }

        state = yihuan_tetrominoes.analyze_state(
            capture.image,
            profile_name=profile["profile_name"],
            update_tracker=True,
        )
        last_state = state
        if _state_is_playing(state, profile):
            piece = dict(state.get("current_piece") or {})
            board = dict(state.get("board") or {})
            logger.info(
                "Tetrominoes[start] playing detected piece=%s rotation=%s origin=(%s,%s) board_confidence=%.3f occupied=%s",
                piece.get("shape"),
                piece.get("rotation"),
                piece.get("origin_row"),
                piece.get("origin_col"),
                float(board.get("confidence") or 0.0),
                board.get("occupied_count"),
            )
            return {
                "phase": "playing",
                "capture": capture,
                "state": state,
                "result_screen": result_screen,
            }

        if time.monotonic() >= deadline:
            logger.warning(
                "Tetrominoes[start] timeout waiting for playing/result last_result=%s",
                (last_result or {}).get("reason"),
            )
            return {
                "phase": "unknown",
                "capture": capture,
                "state": last_state,
                "result_screen": last_result,
            }
        if poll_sec > 0:
            time.sleep(poll_sec)


def _wait_for_result_screen(
    app: Any,
    yihuan_tetrominoes: YihuanTetrominoesService,
    *,
    profile: dict[str, Any],
    timeout_sec: float,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    poll_sec = float(profile["recognition_timeout_result_poll_ms"]) / 1000.0
    while True:
        capture = _capture_image(app)
        result_screen = yihuan_tetrominoes.analyze_result_screen(
            capture.image,
            profile_name=profile["profile_name"],
        )
        if result_screen["found"]:
            logger.info(
                "Tetrominoes[result_wait] result_screen detected reason=%s panel_ratio=%.3f exit_ratio=%.3f",
                result_screen.get("reason"),
                float(result_screen.get("panel_purple_ratio") or 0.0),
                float(result_screen.get("exit_white_ratio") or 0.0),
            )
            return result_screen
        if time.monotonic() >= deadline:
            logger.warning(
                "Tetrominoes[result_wait] timeout waiting for result screen last_reason=%s",
                result_screen.get("reason"),
            )
            return None
        if poll_sec > 0:
            time.sleep(poll_sec)


def _execute_input_sequence(
    input_mapping: Any,
    app: Any,
    *,
    sequence: list[str],
    profile_name: str | None,
    inter_key_delay_ms: int,
    key_press_ms: int,
) -> list[str]:
    executed: list[str] = []
    hold_sec = max(float(key_press_ms), 0.0) / 1000.0
    gap_sec = max(float(inter_key_delay_ms), 0.0) / 1000.0
    for index, action_name in enumerate(sequence):
        if hold_sec > 0:
            logger.info(
                "Tetrominoes[input] hold action=%s profile=%s hold_ms=%s gap_ms=%s",
                action_name,
                profile_name,
                key_press_ms,
                inter_key_delay_ms,
            )
            input_mapping.execute_action(action_name, phase="hold", app=app, profile=profile_name)
            time.sleep(hold_sec)
            logger.info("Tetrominoes[input] release action=%s profile=%s", action_name, profile_name)
            input_mapping.execute_action(action_name, phase="release", app=app, profile=profile_name)
        else:
            logger.info(
                "Tetrominoes[input] tap action=%s profile=%s gap_ms=%s",
                action_name,
                profile_name,
                inter_key_delay_ms,
            )
            input_mapping.execute_action(action_name, phase="tap", app=app, profile=profile_name)
        executed.append(action_name)
        if gap_sec > 0 and index < len(sequence) - 1:
            time.sleep(gap_sec)
    return executed


def _release_all(app: Any) -> None:
    try:
        if hasattr(app, "release_all"):
            app.release_all()
        elif hasattr(app, "controller") and hasattr(app.controller, "release_all"):
            app.controller.release_all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to release all Tetrominoes inputs: %s", exc)


def _sanitize_snapshot_label(label: str | None) -> str:
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label or "").strip().lower())
    return normalized.strip("_") or "snapshot"


def _color_bgr(name: str | None) -> tuple[int, int, int]:
    palette = {
        "red": (60, 60, 230),
        "orange": (40, 140, 255),
        "yellow": (70, 210, 255),
        "green": (80, 220, 100),
        "cyan": (255, 220, 80),
        "blue": (255, 130, 70),
        "purple": (220, 90, 220),
        "pink": (210, 120, 255),
        "empty": (45, 45, 45),
        "unknown": (110, 110, 110),
    }
    return palette.get(str(name or "unknown").strip().lower(), palette["unknown"])


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.size == 0:
        return np.zeros((max(target_height, 1), 1, 3), dtype=np.uint8)
    if image.shape[0] == target_height:
        return image
    scale = float(target_height) / float(max(image.shape[0], 1))
    target_width = max(int(round(image.shape[1] * scale)), 1)
    return cv2.resize(image, (target_width, max(target_height, 1)), interpolation=cv2.INTER_NEAREST)


def _pad_to_height(image: np.ndarray, target_height: int, *, fill: tuple[int, int, int] = (18, 18, 18)) -> np.ndarray:
    if image.shape[0] >= target_height:
        return image
    pad_bottom = target_height - image.shape[0]
    return cv2.copyMakeBorder(image, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=fill)


def _draw_text_block(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int] = (12, 24),
    line_height: int = 22,
    color: tuple[int, int, int] = (236, 236, 236),
) -> None:
    x, y = origin
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            str(line),
            (int(x), int(y + index * line_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def _projected_piece_cells(decision: dict[str, Any] | None) -> list[tuple[int, int]]:
    if not isinstance(decision, dict) or not bool(decision.get("found")):
        return []
    shape = str(decision.get("shape") or "")
    rotations = YihuanTetrominoesService.SHAPES.get(shape)
    if not rotations:
        return []
    try:
        rotation_index = int(decision.get("target_rotation") or 0) % len(rotations)
        top_row = int(decision.get("target_row"))
        left_col = int(decision.get("target_col"))
    except (TypeError, ValueError):
        return []
    return [(top_row + int(cell_row), left_col + int(cell_col)) for cell_row, cell_col in rotations[rotation_index]]


def _cell_rect(grid: dict[str, Any], row: int, col: int) -> tuple[int, int, int, int]:
    origin_x = float(grid.get("origin_x") or 0.0)
    origin_y = float(grid.get("origin_y") or 0.0)
    cell_w = float(grid.get("cell_w") or 1.0)
    cell_h = float(grid.get("cell_h") or 1.0)
    x0 = int(round(origin_x + col * cell_w))
    y0 = int(round(origin_y + row * cell_h))
    x1 = int(round(origin_x + (col + 1) * cell_w))
    y1 = int(round(origin_y + (row + 1) * cell_h))
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


def _board_region(grid: dict[str, Any]) -> tuple[int, int, int, int]:
    x, y, width, height = grid.get("board_region") or [0, 0, 1, 1]
    return int(x), int(y), max(int(width), 1), max(int(height), 1)


def _build_debug_text_lines(
    state: dict[str, Any],
    decision: dict[str, Any] | None,
    *,
    snapshot_label: str | None,
) -> list[str]:
    board = dict(state.get("board") or {})
    piece = dict(state.get("current_piece") or {})
    metrics = dict(state.get("metrics") or {})
    debug = dict(state.get("debug") or {})
    piece_detection = dict(debug.get("piece_detection") or {})
    tracker_candidate = dict(piece_detection.get("tracker_candidate") or {})
    top_candidates = list((decision or {}).get("top_candidates") or [])
    next_queue = list(state.get("next_queue") or [])
    lines = [
        f"snapshot: {snapshot_label or '-'}",
        f"phase: {state.get('phase')}",
        (
            "piece: "
            f"{piece.get('shape')} rot={piece.get('rotation')} origin=({piece.get('origin_row')},{piece.get('origin_col')}) "
            f"src={piece.get('source')} conf={float(piece.get('confidence') or 0.0):.3f}"
        ),
        (
            "board: "
            f"conf={float(board.get('confidence') or 0.0):.3f} occupied={int(board.get('occupied_count') or 0)} "
            f"agg={metrics.get('aggregate_height')} max={metrics.get('max_height')} holes={metrics.get('holes')}"
        ),
        (
            "tracker: "
            f"available={piece_detection.get('tracker_available')} "
            f"diff_cells={len(piece_detection.get('tracker_diff_cells') or [])} "
            f"selected={piece_detection.get('selected_strategy')}"
        ),
    ]
    if tracker_candidate:
        lines.append(
            "tracker_candidate: "
            f"found={tracker_candidate.get('found')} shape={tracker_candidate.get('shape')} "
            f"rot={tracker_candidate.get('rotation')} reason={tracker_candidate.get('reason')}"
        )
    component_candidates = list(piece_detection.get("component_candidates") or [])
    lines.append(f"components: total={piece_detection.get('component_count')} candidates={len(component_candidates)}")
    for component in component_candidates[:5]:
        candidate = dict(component.get("candidate") or {})
        score = component.get("score")
        lines.append(
            "  comp#{idx} size={size} shape={shape} rot={rot} score={score} reason={reason}".format(
                idx=component.get("component_index"),
                size=component.get("size"),
                shape=candidate.get("shape"),
                rot=candidate.get("rotation"),
                score="-" if score is None else f"{float(score):.3f}",
                reason=candidate.get("reason"),
            )
        )
    if decision:
        lines.append(
            "decision: "
            f"shape={decision.get('shape')} cur_rot={decision.get('current_rotation')} target_rot={decision.get('target_rotation')} "
            f"cur_col={decision.get('current_col')} target_col={decision.get('target_col')} row={decision.get('target_row')}"
        )
        lines.append(
            "decision2: "
            f"lines={decision.get('lines_cleared')} score={float(decision.get('score') or 0.0):.3f} "
            f"cand={decision.get('candidate_count')} seq={list(decision.get('input_sequence') or [])}"
        )
        for candidate in top_candidates[:3]:
            lines.append(
                "  top#{rank} rot={rot} col={col} row={row} lines={lines_cleared} score={score:.3f}".format(
                    rank=int(candidate.get("rank") or 0),
                    rot=int(candidate.get("target_rotation") or 0),
                    col=int(candidate.get("target_col") or 0),
                    row=int(candidate.get("target_row") or 0),
                    lines_cleared=int(candidate.get("lines_cleared") or 0),
                    score=float(candidate.get("score") or 0.0),
                )
            )
    if next_queue:
        visible = [
            f"{int(item.get('index') or 0)}:{item.get('dominant_color')}"
            for item in next_queue
            if bool(item.get("visible"))
        ]
        lines.append(f"next_queue: {' | '.join(visible) if visible else '-'}")
    return lines


def _build_annotated_capture(
    capture_image: Any,
    *,
    state: dict[str, Any],
    decision: dict[str, Any] | None,
    profile: dict[str, Any],
    snapshot_label: str | None,
) -> np.ndarray:
    image = cv2.cvtColor(capture_image, cv2.COLOR_RGB2BGR)
    board = dict(state.get("board") or {})
    debug = dict(state.get("debug") or {})
    grid = dict(debug.get("grid") or {})
    piece = dict(state.get("current_piece") or {})
    projected_cells = set(_projected_piece_cells(decision))
    current_cells = {tuple(cell) for cell in piece.get("cells") or []}
    occupied_matrix = board.get("occupied_matrix") or []
    color_matrix = board.get("color_matrix") or []

    if grid:
        board_x, board_y, board_w, board_h = _board_region(grid)
        cv2.rectangle(image, (board_x, board_y), (board_x + board_w, board_y + board_h), (240, 240, 240), 2)
        rows = int(grid.get("rows") or 0)
        cols = int(grid.get("cols") or 0)
        for row in range(rows + 1):
            x0, y0, _, _ = _cell_rect(grid, row if row < rows else rows - 1, 0)
            y = board_y if row == 0 else int(round(float(grid.get("origin_y") or 0.0) + row * float(grid.get("cell_h") or 1.0)))
            cv2.line(image, (board_x, y), (board_x + board_w, y), (95, 95, 95), 1)
        for col in range(cols + 1):
            x = board_x if col == 0 else int(round(float(grid.get("origin_x") or 0.0) + col * float(grid.get("cell_w") or 1.0)))
            cv2.line(image, (x, board_y), (x, board_y + board_h), (95, 95, 95), 1)
        for row, row_values in enumerate(occupied_matrix):
            for col, occupied in enumerate(row_values):
                if not bool(occupied):
                    continue
                x, y, width, height = _cell_rect(grid, row, col)
                color_name = "unknown"
                if row < len(color_matrix) and col < len(color_matrix[row]):
                    color_name = str(color_matrix[row][col])
                cv2.rectangle(image, (x, y), (x + width, y + height), _color_bgr(color_name), 1)
                if (row, col) in current_cells:
                    cv2.rectangle(image, (x + 1, y + 1), (x + width - 1, y + height - 1), (60, 255, 60), 3)
                elif (row, col) in projected_cells:
                    cv2.rectangle(image, (x + 1, y + 1), (x + width - 1, y + height - 1), (40, 220, 255), 2)
        for row, col in projected_cells:
            x, y, width, height = _cell_rect(grid, row, col)
            cv2.rectangle(image, (x + 3, y + 3), (x + width - 3, y + height - 3), (40, 220, 255), 2)

    scale_x = float(grid.get("scale_x") or 1.0)
    scale_y = float(grid.get("scale_y") or 1.0)
    for item in profile.get("next_regions") or []:
        x, y, width, height = item
        cv2.rectangle(
            image,
            (int(round(x * scale_x)), int(round(y * scale_y))),
            (int(round((x + width) * scale_x)), int(round((y + height) * scale_y))),
            (180, 180, 40),
            1,
        )

    text_lines = _build_debug_text_lines(state, decision, snapshot_label=snapshot_label)
    text_panel = np.full((max(280, min(540, image.shape[0])), 620, 3), 18, dtype=np.uint8)
    _draw_text_block(text_panel, text_lines)
    if text_panel.shape[0] < image.shape[0]:
        text_panel = _pad_to_height(text_panel, image.shape[0])
    elif text_panel.shape[0] > image.shape[0]:
        image = _pad_to_height(image, text_panel.shape[0])
    return np.hstack((image, text_panel))


def _build_matrix_panel(
    state: dict[str, Any],
    decision: dict[str, Any] | None,
    *,
    target_height: int,
) -> np.ndarray:
    board = dict(state.get("board") or {})
    debug = dict(state.get("debug") or {})
    piece_debug = dict(debug.get("piece_detection") or {})
    rows = int(board.get("rows") or 0)
    cols = int(board.get("cols") or 0)
    if rows <= 0 or cols <= 0:
        return np.zeros((max(target_height, 1), 1, 3), dtype=np.uint8)

    cell_px = max(int(target_height / max(rows, 1)), 18)
    panel_height = rows * cell_px + 1
    panel_width = cols * cell_px + 1
    panel = np.full((panel_height, panel_width, 3), 16, dtype=np.uint8)

    occupied_matrix = board.get("occupied_matrix") or []
    settled_matrix = board.get("settled_matrix") or []
    color_matrix = board.get("color_matrix") or []
    diff_matrix = piece_debug.get("tracker_diff_matrix") or []
    projected_cells = set(_projected_piece_cells(decision))
    current_cells = {tuple(cell) for cell in dict(state.get("current_piece") or {}).get("cells") or []}

    for row in range(rows):
        for col in range(cols):
            x0 = col * cell_px
            y0 = row * cell_px
            x1 = x0 + cell_px
            y1 = y0 + cell_px
            occupied = bool(occupied_matrix[row][col]) if row < len(occupied_matrix) and col < len(occupied_matrix[row]) else False
            settled = bool(settled_matrix[row][col]) if row < len(settled_matrix) and col < len(settled_matrix[row]) else False
            diff = bool(diff_matrix[row][col]) if row < len(diff_matrix) and col < len(diff_matrix[row]) else False
            fill = (28, 28, 28)
            if settled:
                fill = (90, 90, 90)
            elif occupied:
                color_name = "unknown"
                if row < len(color_matrix) and col < len(color_matrix[row]):
                    color_name = str(color_matrix[row][col])
                fill = _color_bgr(color_name)
            cv2.rectangle(panel, (x0, y0), (x1, y1), fill, thickness=-1)
            cv2.rectangle(panel, (x0, y0), (x1, y1), (55, 55, 55), thickness=1)
            if diff:
                cv2.rectangle(panel, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (255, 80, 255), thickness=2)
            if (row, col) in current_cells:
                cv2.rectangle(panel, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (60, 255, 60), thickness=3)
            if (row, col) in projected_cells:
                cv2.rectangle(panel, (x0 + 6, y0 + 6), (x1 - 6, y1 - 6), (40, 220, 255), thickness=2)
    return panel


def _build_board_debug_composite(
    capture_image: Any,
    annotated_image: np.ndarray,
    *,
    state: dict[str, Any],
    decision: dict[str, Any] | None,
    snapshot_label: str | None,
) -> np.ndarray:
    debug = dict(state.get("debug") or {})
    grid = dict(debug.get("grid") or {})
    image_bgr = cv2.cvtColor(capture_image, cv2.COLOR_RGB2BGR)
    if grid:
        board_x, board_y, board_w, board_h = _board_region(grid)
        margin = 8
        left = max(board_x - margin, 0)
        top = max(board_y - margin, 0)
        right = min(board_x + board_w + margin, image_bgr.shape[1])
        bottom = min(board_y + board_h + margin, image_bgr.shape[0])
        raw_crop = image_bgr[top:bottom, left:right]
        annotated_crop = annotated_image[top:bottom, left:right]
    else:
        raw_crop = image_bgr
        annotated_crop = annotated_image
    target_height = max(440, min(720, raw_crop.shape[0]))
    raw_scaled = _resize_to_height(raw_crop, target_height)
    annotated_scaled = _resize_to_height(annotated_crop, target_height)
    matrix_panel = _resize_to_height(_build_matrix_panel(state, decision, target_height=target_height), target_height)
    text_panel = np.full((target_height, 620, 3), 16, dtype=np.uint8)
    _draw_text_block(text_panel, _build_debug_text_lines(state, decision, snapshot_label=snapshot_label))
    return np.hstack(
        (
            _pad_to_height(raw_scaled, target_height),
            _pad_to_height(annotated_scaled, target_height),
            _pad_to_height(matrix_panel, target_height),
            text_panel,
        )
    )


def _save_debug_snapshot(
    capture_image: Any,
    *,
    state: dict[str, Any],
    decision: dict[str, Any] | None,
    profile: dict[str, Any],
    pieces_played: int,
    snapshot_label: str | None = None,
) -> dict[str, Any]:
    debug_dir = Path(str(profile["debug_snapshot_dir"]))
    if not debug_dir.is_absolute():
        debug_dir = Path.cwd() / debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    label = _sanitize_snapshot_label(snapshot_label)
    stem = f"tetrominoes_{stamp}_{pieces_played:04d}_{label}"
    raw_image_path = debug_dir / f"{stem}_raw.png"
    annotated_image_path = debug_dir / f"{stem}_annotated.png"
    board_debug_path = debug_dir / f"{stem}_board_debug.png"
    json_path = debug_dir / f"{stem}.json"
    raw_bgr = cv2.cvtColor(capture_image, cv2.COLOR_RGB2BGR)
    annotated_image = _build_annotated_capture(
        capture_image,
        state=state,
        decision=decision,
        profile=profile,
        snapshot_label=label,
    )
    board_debug_image = _build_board_debug_composite(
        capture_image,
        annotated_image,
        state=state,
        decision=decision,
        snapshot_label=label,
    )
    cv2.imwrite(str(raw_image_path), raw_bgr)
    cv2.imwrite(str(annotated_image_path), annotated_image)
    cv2.imwrite(str(board_debug_path), board_debug_image)
    payload = {
        "snapshot_label": label,
        "raw_image_path": str(raw_image_path),
        "annotated_image_path": str(annotated_image_path),
        "board_debug_path": str(board_debug_path),
        "state": _json_safe_state(state),
        "decision": decision,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "label": label,
        "raw_image_path": str(raw_image_path),
        "annotated_image_path": str(annotated_image_path),
        "board_debug_path": str(board_debug_path),
        "json_path": str(json_path),
    }


def _append_debug_snapshot(
    debug_snapshots: list[dict[str, Any]],
    capture_image: Any,
    *,
    state: dict[str, Any],
    decision: dict[str, Any] | None,
    profile: dict[str, Any],
    pieces_played: int,
    snapshot_label: str | None,
) -> None:
    try:
        snapshot = _save_debug_snapshot(
            capture_image,
            state=state,
            decision=decision,
            profile=profile,
            pieces_played=pieces_played,
            snapshot_label=snapshot_label,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tetrominoes[debug] failed to save snapshot label=%s: %s", snapshot_label, exc)
        return
    debug_snapshots.append(snapshot)
    logger.info(
        "Tetrominoes[debug] snapshot saved label=%s raw=%s annotated=%s board=%s json=%s",
        snapshot.get("label"),
        snapshot.get("raw_image_path"),
        snapshot.get("annotated_image_path"),
        snapshot.get("board_debug_path"),
        snapshot.get("json_path"),
    )


def _json_safe_state(state: dict[str, Any]) -> dict[str, Any]:
    payload = dict(state)
    return payload


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _decision_summary(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "found": bool(decision.get("found")),
        "reason": decision.get("reason"),
        "shape": decision.get("shape"),
        "current_rotation": decision.get("current_rotation"),
        "target_rotation": decision.get("target_rotation"),
        "current_col": decision.get("current_col"),
        "target_col": decision.get("target_col"),
        "target_row": decision.get("target_row"),
        "lines_cleared": decision.get("lines_cleared"),
        "score": decision.get("score"),
        "score_parts": decision.get("score_parts"),
        "input_sequence": list(decision.get("input_sequence") or []),
        "projected_metrics": dict(decision.get("projected_metrics") or {}),
    }


def _piece_summary(piece: dict[str, Any]) -> dict[str, Any]:
    return {
        "found": bool(piece.get("found")),
        "reason": piece.get("reason"),
        "source": piece.get("source"),
        "shape": piece.get("shape"),
        "rotation": piece.get("rotation"),
        "origin_row": piece.get("origin_row"),
        "origin_col": piece.get("origin_col"),
        "color": piece.get("color"),
        "confidence": piece.get("confidence"),
        "cells": list(piece.get("cells") or []),
    }


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "aggregate_height": metrics.get("aggregate_height"),
        "max_height": metrics.get("max_height"),
        "holes": metrics.get("holes"),
        "bumpiness": metrics.get("bumpiness"),
        "well_depth": metrics.get("well_depth"),
        "top_occupied_count": metrics.get("top_occupied_count"),
    }


def _operation_record(
    *,
    piece_index: int,
    elapsed_sec: float,
    state: dict[str, Any],
    decision: dict[str, Any],
    executed: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    board = dict(state.get("board") or {})
    piece = dict(state.get("current_piece") or {})
    return {
        "piece_index": int(piece_index),
        "elapsed_sec": round(float(elapsed_sec), 3),
        "detected_piece": _piece_summary(piece),
        "board": {
            "confidence": board.get("confidence"),
            "occupied_count": board.get("occupied_count"),
            "metrics": _metrics_summary(dict(state.get("metrics") or {})),
        },
        "decision": _decision_summary(decision),
        "executed_sequence": list(executed),
        "dry_run": bool(dry_run),
    }


@action_info(
    name="yihuan_tetrominoes_analyze_screen",
    public=True,
    read_only=True,
    description="Capture and analyze one Yihuan Tetrominoes screen.",
)
@requires_services(app="plans/aura_base/app", yihuan_tetrominoes="yihuan_tetrominoes")
def yihuan_tetrominoes_analyze_screen(
    app: Any,
    yihuan_tetrominoes: YihuanTetrominoesService,
    profile_name: str = "default_1280x720_cn",
) -> dict[str, Any]:
    capture = _capture_image(app)
    result_screen = yihuan_tetrominoes.analyze_result_screen(
        capture.image,
        profile_name=profile_name,
    )
    if result_screen["found"]:
        return {
            "profile_name": yihuan_tetrominoes.load_profile(profile_name)["profile_name"],
            "capture_size": [int(capture.image.shape[1]), int(capture.image.shape[0])],
            "phase": "result",
            "result_screen": result_screen,
            "decision": {"found": False, "reason": "level_end"},
        }

    state = yihuan_tetrominoes.analyze_state(
        capture.image,
        profile_name=profile_name,
        update_tracker=False,
    )
    state["phase"] = "playing" if _state_is_playing(state, yihuan_tetrominoes.load_profile(profile_name)) else "unknown"
    decision = yihuan_tetrominoes.choose_best_move(state)
    state["decision"] = _decision_summary(decision) if decision.get("found") else decision
    return state


@action_info(
    name="yihuan_tetrominoes_run_session",
    public=False,
    read_only=False,
    description="Run the Yihuan Tetrominoes mini-game loop with survival-first planning.",
)
@requires_services(
    app="plans/aura_base/app",
    input_mapping="plans/aura_base/input_mapping",
    yihuan_tetrominoes="yihuan_tetrominoes",
)
def yihuan_tetrominoes_run_session(
    app: Any,
    input_mapping: Any,
    yihuan_tetrominoes: YihuanTetrominoesService,
    profile_name: str = "default_1280x720_cn",
    max_seconds: float | int | str = 0.0,
    max_pieces: int | str = 0,
    start_game: bool | str = True,
    dry_run: bool | str = False,
    debug_enabled: bool | str = False,
) -> dict[str, Any]:
    profile = yihuan_tetrominoes.load_profile(profile_name)
    yihuan_tetrominoes.reset_tracker(profile["profile_name"])
    start_game_enabled = _coerce_bool(start_game)
    dry_run_enabled = _coerce_bool(dry_run)
    debug_enabled = _coerce_bool(debug_enabled)
    duration_limit = _coerce_float(max_seconds, 0.0)
    if duration_limit <= 0:
        duration_limit = float(profile["max_seconds"])
    piece_limit = max(_coerce_int(max_pieces, 0), 0)
    if dry_run_enabled and piece_limit <= 0:
        piece_limit = 1

    start = time.monotonic()
    pieces_played = 0
    recognition_failures = 0
    decisions_tail: list[dict[str, Any]] = []
    operation_log: list[dict[str, Any]] = []
    debug_snapshots: list[dict[str, Any]] = []
    final_state: dict[str, Any] | None = None
    final_result_screen: dict[str, Any] | None = None
    pending_state: dict[str, Any] | None = None
    pending_capture: Any | None = None
    start_clicked = False
    result_screen_cleared_before_start = False
    stopped_reason = "max_seconds"
    status = "success"
    failure_message = ""
    logger.info(
        "Tetrominoes[session] start profile=%s max_seconds=%.3f max_pieces=%s start_game=%s dry_run=%s debug=%s",
        profile["profile_name"],
        duration_limit,
        piece_limit,
        start_game_enabled,
        dry_run_enabled,
        debug_enabled,
    )

    try:
        if start_game_enabled:
            if not dry_run_enabled:
                try:
                    result_screen_cleared_before_start = _clear_result_screen_before_start(
                        app,
                        yihuan_tetrominoes,
                        profile=profile,
                    )
                except RuntimeError as exc:
                    status = "failed"
                    stopped_reason = "capture_failed"
                    failure_message = str(exc)
                    raise _TetrominoesSessionStop() from exc
                _click_start_game(app, profile)
                start_clicked = True
            try:
                start_phase = _wait_for_game_start(app, yihuan_tetrominoes, profile=profile)
            except RuntimeError as exc:
                status = "failed"
                stopped_reason = "capture_failed"
                failure_message = str(exc)
                raise _TetrominoesSessionStop() from exc
            final_result_screen = start_phase.get("result_screen")
            if start_phase["phase"] == "result":
                if debug_enabled and start_phase.get("capture") is not None:
                    _append_debug_snapshot(
                        debug_snapshots,
                        start_phase["capture"].image,
                        state={
                            "profile_name": profile["profile_name"],
                            "capture_size": [int(start_phase["capture"].image.shape[1]), int(start_phase["capture"].image.shape[0])],
                            "phase": "result",
                            "result_screen": final_result_screen,
                            "board": {},
                            "current_piece": {},
                            "metrics": {},
                            "debug": {},
                        },
                        decision=None,
                        profile=profile,
                        pieces_played=pieces_played,
                        snapshot_label="start_result_screen",
                    )
                stopped_reason = "level_end"
                final_state = start_phase.get("state")
                logger.info(
                    "Tetrominoes[session] stop status=%s reason=%s pieces=%s elapsed=%.3f",
                    status,
                    stopped_reason,
                    pieces_played,
                    time.monotonic() - start,
                )
                return {
                    "status": status,
                    "stopped_reason": stopped_reason,
                    "failure_reason": None,
                    "failure_message": "",
                    "pieces_played": pieces_played,
                    "elapsed_sec": time.monotonic() - start,
                    "profile_name": profile["profile_name"],
                    "start_game": start_game_enabled,
                    "start_clicked": start_clicked,
                    "result_screen_cleared_before_start": result_screen_cleared_before_start,
                    "dry_run": dry_run_enabled,
                    "final_metrics": {"last_confidence": 0.0},
                    "result_screen": final_result_screen,
                    "decisions_tail": decisions_tail,
                    "operation_log": operation_log,
                    "debug_snapshots": debug_snapshots,
                }
            if start_phase["phase"] != "playing":
                status = "failed"
                stopped_reason = "game_start_timeout"
                failure_message = "Timed out waiting for Tetrominoes pieces after clicking start."
                final_state = start_phase.get("state")
                raise _TetrominoesSessionStop()
            pending_state = start_phase.get("state")
            pending_capture = start_phase.get("capture")
            if debug_enabled and pending_state is not None and pending_capture is not None:
                _append_debug_snapshot(
                    debug_snapshots,
                    pending_capture.image,
                    state=pending_state,
                    decision=None,
                    profile=profile,
                    pieces_played=pieces_played,
                    snapshot_label="start_playing",
                )

        while True:
            elapsed = time.monotonic() - start
            if piece_limit > 0 and pieces_played >= piece_limit:
                stopped_reason = "max_pieces"
                break
            if duration_limit > 0 and elapsed >= duration_limit:
                stopped_reason = "max_seconds"
                break

            if pending_state is not None and pending_capture is not None:
                state = pending_state
                capture = pending_capture
                pending_state = None
                pending_capture = None
            else:
                try:
                    capture = _capture_image(app)
                except RuntimeError as exc:
                    status = "partial" if pieces_played else "failed"
                    stopped_reason = "capture_failed"
                    failure_message = str(exc)
                    break
                result_screen = yihuan_tetrominoes.analyze_result_screen(
                    capture.image,
                    profile_name=profile["profile_name"],
                )
                final_result_screen = result_screen
                if result_screen["found"]:
                    if debug_enabled:
                        _append_debug_snapshot(
                            debug_snapshots,
                            capture.image,
                            state={
                                "profile_name": profile["profile_name"],
                                "capture_size": [int(capture.image.shape[1]), int(capture.image.shape[0])],
                                "phase": "result",
                                "result_screen": result_screen,
                                "board": {},
                                "current_piece": {},
                                "metrics": {},
                                "debug": {},
                            },
                            decision=None,
                            profile=profile,
                            pieces_played=pieces_played,
                            snapshot_label=f"result_screen_piece_{pieces_played:04d}",
                        )
                    logger.info(
                        "Tetrominoes[session] level_end detected pieces=%s panel_ratio=%.3f exit_ratio=%.3f",
                        pieces_played,
                        float(result_screen.get("panel_purple_ratio") or 0.0),
                        float(result_screen.get("exit_white_ratio") or 0.0),
                    )
                    stopped_reason = "level_end"
                    break
                state = yihuan_tetrominoes.analyze_state(
                    capture.image,
                    profile_name=profile["profile_name"],
                    update_tracker=True,
                )

            final_state = state
            board_confidence = float(dict(state.get("board") or {}).get("confidence") or 0.0)
            current_piece = dict(state.get("current_piece") or {})
            if board_confidence < float(profile["board_confidence_min"]) or not current_piece.get("found"):
                recognition_failures += 1
                logger.warning(
                    "Tetrominoes[recognition] failure count=%s/%s board_confidence=%.3f piece_found=%s reason=%s",
                    recognition_failures,
                    int(profile["recognition_retry_count"]),
                    board_confidence,
                    bool(current_piece.get("found")),
                    current_piece.get("reason"),
                )
                if recognition_failures >= int(profile["recognition_retry_count"]):
                    result_wait_sec = float(profile.get("recognition_timeout_result_wait_sec") or 0.0)
                    if result_wait_sec > 0:
                        try:
                            waited_result = _wait_for_result_screen(
                                app,
                                yihuan_tetrominoes,
                                profile=profile,
                                timeout_sec=result_wait_sec,
                            )
                        except RuntimeError as exc:
                            waited_result = None
                            logger.warning("Tetrominoes[result_wait] capture failed while waiting: %s", exc)
                        if waited_result is not None:
                            if debug_enabled:
                                _append_debug_snapshot(
                                    debug_snapshots,
                                    capture.image,
                                    state=state,
                                    decision=None,
                                    profile=profile,
                                    pieces_played=pieces_played,
                                    snapshot_label=f"result_wait_piece_{pieces_played + 1:04d}",
                                )
                            status = "success"
                            stopped_reason = "level_end"
                            final_result_screen = waited_result
                            failure_message = ""
                            break
                    status = "partial" if pieces_played else "failed"
                    stopped_reason = "recognition_timeout"
                    failure_message = str(current_piece.get("reason") or "low_confidence")
                    break
                if debug_enabled:
                    _append_debug_snapshot(
                        debug_snapshots,
                        capture.image,
                        state=state,
                        decision=None,
                        profile=profile,
                        pieces_played=pieces_played,
                        snapshot_label=f"recognition_failure_piece_{pieces_played + 1:04d}_attempt_{recognition_failures:02d}",
                    )
                retry_sec = float(profile["recognition_retry_interval_ms"]) / 1000.0
                if retry_sec > 0:
                    time.sleep(retry_sec)
                continue

            recognition_failures = 0
            metrics = dict(state.get("metrics") or {})
            piece_index = pieces_played + 1
            logger.info(
                "Tetrominoes[piece %s] detected shape=%s rotation=%s origin=(%s,%s) color=%s confidence=%.3f board_confidence=%.3f metrics=%s",
                piece_index,
                current_piece.get("shape"),
                current_piece.get("rotation"),
                current_piece.get("origin_row"),
                current_piece.get("origin_col"),
                current_piece.get("color"),
                float(current_piece.get("confidence") or 0.0),
                board_confidence,
                _metrics_summary(metrics),
            )
            if int(metrics.get("max_height") or 0) >= int(profile["board_rows"]):
                status = "partial" if pieces_played else "failed"
                stopped_reason = "top_out_risk"
                failure_message = "settled board reached the top row"
                break

            decision = yihuan_tetrominoes.choose_best_move(state)
            if not decision.get("found"):
                logger.warning(
                    "Tetrominoes[piece %s] no_decision reason=%s shape=%s",
                    piece_index,
                    decision.get("reason"),
                    decision.get("shape"),
                )
                status = "partial" if pieces_played else "failed"
                stopped_reason = "top_out_risk" if decision.get("reason") == "no_valid_placement" else "recognition_timeout"
                failure_message = str(decision.get("reason") or "no_decision")
                break
            logger.info(
                "Tetrominoes[piece %s] decision shape=%s target_rotation=%s target_col=%s target_row=%s lines=%s score=%.3f sequence=%s",
                piece_index,
                decision.get("shape"),
                decision.get("target_rotation"),
                decision.get("target_col"),
                decision.get("target_row"),
                decision.get("lines_cleared"),
                float(decision.get("score") or 0.0),
                list(decision.get("input_sequence") or []),
            )

            if debug_enabled:
                _append_debug_snapshot(
                    debug_snapshots,
                    capture.image,
                    state=state,
                    decision=decision,
                    profile=profile,
                    pieces_played=pieces_played,
                    snapshot_label=f"piece_{piece_index:04d}_decision",
                )

            sequence = [str(item) for item in decision.get("input_sequence") or []]
            executed: list[str] = []
            if not dry_run_enabled:
                executed = _execute_input_sequence(
                    input_mapping,
                    app,
                    sequence=sequence,
                    profile_name=profile["profile_name"],
                    inter_key_delay_ms=int(profile["inter_key_delay_ms"]),
                    key_press_ms=int(profile["key_press_ms"]),
                )
                yihuan_tetrominoes.commit_settled_matrix(
                    decision.get("projected_board") or [],
                    profile_name=profile["profile_name"],
                )

            summary = _decision_summary(decision)
            summary["executed_sequence"] = executed
            summary["dry_run"] = dry_run_enabled
            decisions_tail.append(summary)
            decisions_tail = decisions_tail[-12:]
            operation_log.append(
                _operation_record(
                    piece_index=piece_index,
                    elapsed_sec=time.monotonic() - start,
                    state=state,
                    decision=decision,
                    executed=executed,
                    dry_run=dry_run_enabled,
                )
            )
            logger.info(
                "Tetrominoes[piece %s] executed sequence=%s dry_run=%s",
                piece_index,
                executed,
                dry_run_enabled,
            )
            pieces_played += 1

            post_drop_sec = float(profile["post_drop_delay_ms"]) / 1000.0
            if post_drop_sec > 0 and not dry_run_enabled:
                time.sleep(post_drop_sec)
            if debug_enabled and not dry_run_enabled:
                try:
                    post_capture = _capture_image(app)
                    post_state = yihuan_tetrominoes.analyze_state(
                        post_capture.image,
                        profile_name=profile["profile_name"],
                        update_tracker=False,
                    )
                    _append_debug_snapshot(
                        debug_snapshots,
                        post_capture.image,
                        state=post_state,
                        decision=None,
                        profile=profile,
                        pieces_played=pieces_played,
                        snapshot_label=f"piece_{piece_index:04d}_post_drop",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Tetrominoes[debug] failed to capture post-drop snapshot piece=%s: %s", piece_index, exc)
    except _TetrominoesSessionStop:
        pass
    except Exception as exc:  # noqa: BLE001
        status = "partial" if pieces_played else "failed"
        stopped_reason = "exception"
        failure_message = str(exc)
        logger.exception("Tetrominoes auto session failed.")
    finally:
        _release_all(app)

    elapsed_sec = time.monotonic() - start
    logger.info(
        "Tetrominoes[session] stop status=%s reason=%s pieces=%s elapsed=%.3f failure=%s",
        status,
        stopped_reason,
        pieces_played,
        elapsed_sec,
        failure_message,
    )
    final_metrics = dict((final_state or {}).get("metrics") or {})
    final_confidence = float(dict(dict(final_state or {}).get("board") or {}).get("confidence") or 0.0)
    return {
        "status": status,
        "stopped_reason": stopped_reason,
        "failure_reason": stopped_reason if status != "success" else None,
        "failure_message": failure_message,
        "pieces_played": pieces_played,
        "elapsed_sec": elapsed_sec,
        "profile_name": profile["profile_name"],
        "start_game": start_game_enabled,
        "start_clicked": start_clicked,
        "result_screen_cleared_before_start": result_screen_cleared_before_start,
        "dry_run": dry_run_enabled,
        "final_metrics": {
            **final_metrics,
            "last_confidence": final_confidence,
        },
        "result_screen": final_result_screen,
        "decisions_tail": decisions_tail,
        "operation_log": operation_log,
        "debug_snapshots": debug_snapshots,
    }

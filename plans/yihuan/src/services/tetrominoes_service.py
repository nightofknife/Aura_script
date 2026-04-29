"""Tetrominoes board recognition and move planning for the Yihuan plan."""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from packages.aura_core.api import service_info


Matrix = list[list[int]]
Cell = tuple[int, int]


def _normalize_cells(cells: list[Cell] | tuple[Cell, ...]) -> tuple[Cell, ...]:
    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    return tuple(sorted((int(row - min_row), int(col - min_col)) for row, col in cells))


def _shape_rotations() -> dict[str, tuple[tuple[Cell, ...], ...]]:
    raw: dict[str, list[list[Cell]]] = {
        "I": [
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(0, 0), (1, 0), (2, 0), (3, 0)],
        ],
        "O": [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        ],
        "T": [
            [(0, 0), (0, 1), (0, 2), (1, 1)],
            [(0, 1), (1, 0), (1, 1), (2, 1)],
            [(0, 1), (1, 0), (1, 1), (1, 2)],
            [(0, 0), (1, 0), (1, 1), (2, 0)],
        ],
        "S": [
            [(0, 1), (0, 2), (1, 0), (1, 1)],
            [(0, 0), (1, 0), (1, 1), (2, 1)],
        ],
        "Z": [
            [(0, 0), (0, 1), (1, 1), (1, 2)],
            [(0, 1), (1, 0), (1, 1), (2, 0)],
        ],
        "J": [
            [(0, 0), (1, 0), (1, 1), (1, 2)],
            [(0, 0), (0, 1), (1, 0), (2, 0)],
            [(0, 0), (0, 1), (0, 2), (1, 2)],
            [(0, 1), (1, 1), (2, 0), (2, 1)],
        ],
        "L": [
            [(0, 2), (1, 0), (1, 1), (1, 2)],
            [(0, 0), (1, 0), (2, 0), (2, 1)],
            [(0, 0), (0, 1), (0, 2), (1, 0)],
            [(0, 0), (0, 1), (1, 1), (2, 1)],
        ],
    }
    return {shape: tuple(_normalize_cells(rotation) for rotation in rotations) for shape, rotations in raw.items()}


@service_info(
    alias="yihuan_tetrominoes",
    public=True,
    singleton=True,
    description="Recognize and plan moves for the Yihuan Tetrominoes mini-game.",
)
class YihuanTetrominoesService:
    _DEFAULT_PROFILE = "default_1280x720_cn"
    SHAPES = _shape_rotations()
    SHAPE_BY_CELLS = {
        cells: (shape, rotation_index)
        for shape, rotations in SHAPES.items()
        for rotation_index, cells in enumerate(rotations)
    }

    def __init__(self) -> None:
        self._plan_root = Path(__file__).resolve().parents[2]
        self._profile_dir = self._plan_root / "data" / "tetrominoes"
        self._profile_cache: dict[str, dict[str, Any]] = {}
        self._tracker: dict[str, dict[str, Any]] = {}

    def load_profile(self, profile_name: str | None = None) -> dict[str, Any]:
        resolved_name = str(profile_name or self._DEFAULT_PROFILE).strip() or self._DEFAULT_PROFILE
        cached = self._profile_cache.get(resolved_name)
        if cached is not None:
            return dict(cached)

        profile_path = self._profile_dir / f"{resolved_name}.yaml"
        if not profile_path.is_file():
            raise FileNotFoundError(f"Tetrominoes profile not found: {profile_path}")

        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        normalized = {
            "profile_name": str(payload.get("profile_name") or resolved_name),
            "client_size": self._coerce_size(payload.get("client_size"), default=(1280, 720)),
            "board_origin": self._coerce_float_pair(payload.get("board_origin"), default=(473.0, 50.0)),
            "board_cell_size": self._coerce_float_pair(payload.get("board_cell_size"), default=(29.4, 29.4)),
            "board_rows": max(self._coerce_int(payload.get("board_rows"), 20), 1),
            "board_cols": max(self._coerce_int(payload.get("board_cols"), 10), 1),
            "sample_inner_ratio": min(max(self._coerce_float(payload.get("sample_inner_ratio"), 0.48), 0.1), 0.95),
            "occupied_s_min": max(self._coerce_int(payload.get("occupied_s_min"), 55), 0),
            "occupied_v_min": max(self._coerce_int(payload.get("occupied_v_min"), 150), 0),
            "occupied_min_ratio": min(max(self._coerce_float(payload.get("occupied_min_ratio"), 0.18), 0.01), 1.0),
            "board_confidence_min": min(max(self._coerce_float(payload.get("board_confidence_min"), 0.55), 0.0), 1.0),
            "solver": str(payload.get("solver") or "bcts"),
            "top_out_min_top_cells": max(self._coerce_int(payload.get("top_out_min_top_cells"), 4), 1),
            "max_seconds": max(self._coerce_float(payload.get("max_seconds"), 180.0), 0.0),
            "start_game_point": self._coerce_point(payload.get("start_game_point"), default=(1120, 670)),
            "start_game_delay_ms": max(self._coerce_int(payload.get("start_game_delay_ms"), 6000), 0),
            "start_timeout_sec": max(self._coerce_float(payload.get("start_timeout_sec"), 8.0), 0.1),
            "start_poll_ms": max(self._coerce_int(payload.get("start_poll_ms"), 120), 10),
            "start_piece_max_origin_row": max(self._coerce_int(payload.get("start_piece_max_origin_row"), 4), 0),
            "recognition_retry_count": max(self._coerce_int(payload.get("recognition_retry_count"), 8), 1),
            "recognition_retry_interval_ms": max(self._coerce_int(payload.get("recognition_retry_interval_ms"), 80), 0),
            "inter_key_delay_ms": max(self._coerce_int(payload.get("inter_key_delay_ms"), 300), 0),
            "key_press_ms": max(self._coerce_int(payload.get("key_press_ms"), 100), 0),
            "post_drop_delay_ms": max(self._coerce_int(payload.get("post_drop_delay_ms"), 250), 0),
            "debug_snapshot_dir": str(payload.get("debug_snapshot_dir") or "tmp/tetrominoes_debug"),
            "result_panel_region": self._coerce_region(payload.get("result_panel_region"), default=(420, 40, 435, 635)),
            "result_exit_button_region": self._coerce_region(
                payload.get("result_exit_button_region"),
                default=(540, 590, 210, 52),
            ),
            "result_exit_point": self._coerce_point(payload.get("result_exit_point"), default=(645, 616)),
            "result_exit_delay_ms": max(self._coerce_int(payload.get("result_exit_delay_ms"), 1000), 0),
            "recognition_timeout_result_wait_sec": max(
                self._coerce_float(payload.get("recognition_timeout_result_wait_sec"), 5.0),
                0.0,
            ),
            "recognition_timeout_result_poll_ms": max(
                self._coerce_int(payload.get("recognition_timeout_result_poll_ms"), 150),
                10,
            ),
            "result_purple_hsv_lower": self._coerce_hsv_triplet(
                payload.get("result_purple_hsv_lower"),
                default=(105, 35, 45),
            ),
            "result_purple_hsv_upper": self._coerce_hsv_triplet(
                payload.get("result_purple_hsv_upper"),
                default=(150, 255, 255),
            ),
            "result_panel_min_purple_ratio": min(
                max(self._coerce_float(payload.get("result_panel_min_purple_ratio"), 0.45), 0.0),
                1.0,
            ),
            "result_exit_white_s_max": max(self._coerce_int(payload.get("result_exit_white_s_max"), 60), 0),
            "result_exit_white_v_min": max(self._coerce_int(payload.get("result_exit_white_v_min"), 130), 0),
            "result_exit_min_white_ratio": min(
                max(self._coerce_float(payload.get("result_exit_min_white_ratio"), 0.35), 0.0),
                1.0,
            ),
            "next_regions": self._coerce_regions(payload.get("next_regions")),
        }
        self._profile_cache[resolved_name] = normalized
        return dict(normalized)

    def reset_tracker(self, profile_name: str | None = None) -> None:
        profile = self.load_profile(profile_name)
        self._tracker.pop(profile["profile_name"], None)

    def commit_settled_matrix(self, settled_matrix: Matrix, *, profile_name: str | None = None) -> None:
        profile = self.load_profile(profile_name)
        self._tracker[profile["profile_name"]] = {
            "settled": [[1 if value else 0 for value in row] for row in settled_matrix],
        }

    def analyze_state(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
        update_tracker: bool = True,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        origin_x, origin_y, cell_w, cell_h = self._scaled_grid(source_image, profile)
        cells, occupied = self._sample_board(source_image, profile)
        color_matrix = [[str(cell["color"]) for cell in row] for row in cells]
        occupied_matrix = occupied.astype(np.uint8).tolist()
        active_piece, piece_debug = self._infer_active_piece(occupied, color_matrix, profile)

        settled = occupied.copy()
        if active_piece["found"]:
            for row, col in active_piece["cells"]:
                settled[int(row), int(col)] = False

        if update_tracker and active_piece["found"]:
            self.commit_settled_matrix(settled.astype(np.uint8).tolist(), profile_name=profile["profile_name"])

        metrics = self.compute_metrics(settled)
        confidence_values = [float(cell["confidence"]) for row in cells for cell in row]
        board_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
        occupied_count = int(np.count_nonzero(occupied))
        settled_matrix = settled.astype(np.uint8).tolist()
        occupied_rows_text = self._matrix_to_rows(occupied)
        settled_rows_text = self._matrix_to_rows(settled)

        return {
            "profile_name": profile["profile_name"],
            "capture_size": [int(source_image.shape[1]), int(source_image.shape[0])],
            "phase": "playing" if active_piece["found"] else "unknown",
            "result_screen": self.analyze_result_screen(source_image, profile_name=profile["profile_name"]),
            "board": {
                "rows": int(profile["board_rows"]),
                "cols": int(profile["board_cols"]),
                "confidence": board_confidence,
                "occupied_count": occupied_count,
                "occupied_matrix": occupied_matrix,
                "settled_matrix": settled_matrix,
                "color_matrix": color_matrix,
                "rows_text": occupied_rows_text,
                "settled_rows_text": settled_rows_text,
            },
            "current_piece": active_piece,
            "next_queue": self._analyze_next_queue(source_image, profile),
            "metrics": metrics,
            "debug": {
                "grid": {
                    "scale_x": float(source_image.shape[1]) / float(profile["client_size"][0]),
                    "scale_y": float(source_image.shape[0]) / float(profile["client_size"][1]),
                    "origin_x": float(origin_x),
                    "origin_y": float(origin_y),
                    "cell_w": float(cell_w),
                    "cell_h": float(cell_h),
                    "rows": int(profile["board_rows"]),
                    "cols": int(profile["board_cols"]),
                    "sample_inner_ratio": float(profile["sample_inner_ratio"]),
                    "board_region": [
                        int(round(origin_x)),
                        int(round(origin_y)),
                        max(int(round(cell_w * int(profile["board_cols"]))), 1),
                        max(int(round(cell_h * int(profile["board_rows"]))), 1),
                    ],
                },
                "board": {
                    "occupied_rows_text": occupied_rows_text,
                    "settled_rows_text": settled_rows_text,
                    "occupied_matrix": occupied_matrix,
                    "settled_matrix": settled_matrix,
                    "cell_samples": [[dict(cell) for cell in row] for row in cells],
                },
                "piece_detection": piece_debug,
            },
        }

    def analyze_result_screen(
        self,
        source_image: np.ndarray,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        profile = self.load_profile(profile_name)
        scale_x, scale_y = self._scale_factors(source_image, profile)
        panel_region = self._scale_region(profile["result_panel_region"], scale_x, scale_y)
        exit_region = self._scale_region(profile["result_exit_button_region"], scale_x, scale_y)
        panel = self._crop_region(source_image, panel_region)
        exit_button = self._crop_region(source_image, exit_region)
        if panel.size == 0 or exit_button.size == 0:
            return {
                "found": False,
                "reason": "empty_region",
                "panel_region": list(panel_region),
                "exit_button_region": list(exit_region),
                "panel_purple_ratio": 0.0,
                "exit_white_ratio": 0.0,
            }

        panel_hsv = cv2.cvtColor(panel, cv2.COLOR_RGB2HSV)
        lower = np.array(profile["result_purple_hsv_lower"], dtype=np.uint8)
        upper = np.array(profile["result_purple_hsv_upper"], dtype=np.uint8)
        panel_mask = cv2.inRange(panel_hsv, lower, upper)
        panel_purple_ratio = float(np.count_nonzero(panel_mask)) / float(panel_mask.size)

        exit_hsv = cv2.cvtColor(exit_button, cv2.COLOR_RGB2HSV)
        exit_mask = (
            (exit_hsv[:, :, 1] <= int(profile["result_exit_white_s_max"]))
            & (exit_hsv[:, :, 2] >= int(profile["result_exit_white_v_min"]))
        )
        exit_white_ratio = float(np.count_nonzero(exit_mask)) / float(exit_mask.size)
        panel_pass = panel_purple_ratio >= float(profile["result_panel_min_purple_ratio"])
        exit_pass = exit_white_ratio >= float(profile["result_exit_min_white_ratio"])
        found = bool(panel_pass and exit_pass)
        if found:
            reason = "ok"
        elif not panel_pass:
            reason = "panel_purple_ratio_low"
        else:
            reason = "exit_white_ratio_low"

        return {
            "found": found,
            "reason": reason,
            "panel_region": list(panel_region),
            "exit_button_region": list(exit_region),
            "panel_purple_ratio": panel_purple_ratio,
            "panel_min_purple_ratio": float(profile["result_panel_min_purple_ratio"]),
            "exit_white_ratio": exit_white_ratio,
            "exit_min_white_ratio": float(profile["result_exit_min_white_ratio"]),
        }

    def choose_best_move(self, state: dict[str, Any]) -> dict[str, Any]:
        board = dict(state.get("board") or {})
        piece = dict(state.get("current_piece") or {})
        if not piece.get("found"):
            return {"found": False, "reason": "current_piece_missing"}

        shape = str(piece.get("shape") or "")
        if shape not in self.SHAPES:
            return {"found": False, "reason": "unsupported_shape", "shape": shape}

        settled = np.array(board.get("settled_matrix") or [], dtype=bool)
        if settled.ndim != 2 or settled.size == 0:
            return {"found": False, "reason": "invalid_board"}

        current_rotation = int(piece.get("rotation") or 0)
        current_col = int(piece.get("origin_col") or 0)
        rotations = self.SHAPES[shape]
        placements: list[dict[str, Any]] = []
        for rotation_index, rotation_cells in enumerate(rotations):
            width = max(col for _, col in rotation_cells) + 1
            for left_col in range(0, int(settled.shape[1]) - width + 1):
                final_row = self._drop_row(settled, rotation_cells, left_col)
                if final_row is None:
                    continue
                placed, lines_cleared, eroded_piece_cells = self._place_and_clear(
                    settled,
                    rotation_cells,
                    final_row,
                    left_col,
                )
                metrics = self.compute_metrics(placed)
                input_sequence = self.build_input_sequence(
                    current_rotation=current_rotation,
                    target_rotation=rotation_index,
                    rotation_count=len(rotations),
                    current_col=current_col,
                    target_col=left_col,
                )
                score, score_parts = self._score_placement_bcts(
                    placed,
                    rotation_cells=rotation_cells,
                    top_row=final_row,
                    lines_cleared=lines_cleared,
                    eroded_piece_cells=eroded_piece_cells,
                )
                placements.append(
                    {
                        "shape": shape,
                        "solver": "bcts",
                        "target_rotation": rotation_index,
                        "target_col": left_col,
                        "target_row": final_row,
                        "lines_cleared": lines_cleared,
                        "score": score,
                        "score_parts": score_parts,
                        "score_is_higher_better": True,
                        "input_sequence": input_sequence,
                        "projected_board": placed.astype(np.uint8).tolist(),
                        "projected_metrics": metrics,
                    }
                )

        if not placements:
            return {"found": False, "reason": "no_valid_placement", "shape": shape}

        sorted_placements = sorted(placements, key=lambda item: (float(item["score"]), -len(item["input_sequence"])), reverse=True)
        best = dict(sorted_placements[0])
        best["found"] = True
        best["reason"] = "ok"
        best["current_rotation"] = current_rotation
        best["current_col"] = current_col
        best["candidate_count"] = len(sorted_placements)
        best["top_candidates"] = [
            self._placement_summary(candidate, rank=index + 1) for index, candidate in enumerate(sorted_placements[:5])
        ]
        return best

    def build_input_sequence(
        self,
        *,
        current_rotation: int,
        target_rotation: int,
        rotation_count: int,
        current_col: int,
        target_col: int,
    ) -> list[str]:
        actions: list[str] = []
        rotations = max(int(rotation_count), 1)
        diff = (int(target_rotation) - int(current_rotation)) % rotations
        if rotations > 1:
            if diff == 1:
                actions.append("tetrominoes_rotate_cw")
            elif diff == 2:
                actions.extend(["tetrominoes_rotate_cw", "tetrominoes_rotate_cw"])
            elif diff == rotations - 1:
                actions.append("tetrominoes_rotate_ccw")
            elif diff > 0:
                actions.extend(["tetrominoes_rotate_cw"] * diff)

        horizontal_delta = int(target_col) - int(current_col)
        if horizontal_delta < 0:
            actions.extend(["tetrominoes_left"] * abs(horizontal_delta))
        elif horizontal_delta > 0:
            actions.extend(["tetrominoes_right"] * horizontal_delta)
        actions.append("tetrominoes_fast_drop")
        return actions

    def compute_metrics(self, board: np.ndarray | Matrix) -> dict[str, Any]:
        matrix = np.array(board, dtype=bool)
        if matrix.ndim != 2:
            return {
                "aggregate_height": 0,
                "max_height": 0,
                "holes": 0,
                "bumpiness": 0,
                "well_depth": 0,
                "top_occupied_count": 0,
            }

        rows, cols = matrix.shape
        heights: list[int] = []
        holes = 0
        for col in range(cols):
            filled_rows = np.flatnonzero(matrix[:, col])
            if filled_rows.size == 0:
                heights.append(0)
                continue
            first = int(filled_rows[0])
            height = rows - first
            heights.append(height)
            holes += int(np.count_nonzero(~matrix[first:, col]))

        bumpiness = sum(abs(heights[index] - heights[index + 1]) for index in range(max(len(heights) - 1, 0)))
        well_depth = 0
        for col, height in enumerate(heights):
            left = heights[col - 1] if col > 0 else rows
            right = heights[col + 1] if col < cols - 1 else rows
            well_depth += max(min(left, right) - height, 0)

        return {
            "aggregate_height": int(sum(heights)),
            "max_height": int(max(heights) if heights else 0),
            "holes": int(holes),
            "bumpiness": int(bumpiness),
            "well_depth": int(well_depth),
            "top_occupied_count": int(np.count_nonzero(matrix[:4, :])),
        }

    def _sample_board(self, source_image: np.ndarray, profile: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], np.ndarray]:
        rows = int(profile["board_rows"])
        cols = int(profile["board_cols"])
        origin_x, origin_y, cell_w, cell_h = self._scaled_grid(source_image, profile)
        sample_ratio = float(profile["sample_inner_ratio"])
        occupied = np.zeros((rows, cols), dtype=bool)
        cells: list[list[dict[str, Any]]] = []
        for row in range(rows):
            cell_row: list[dict[str, Any]] = []
            for col in range(cols):
                crop = self._crop_cell(source_image, origin_x, origin_y, cell_w, cell_h, row, col, sample_ratio)
                info = self._classify_cell(crop, profile)
                occupied[row, col] = bool(info["occupied"])
                cell_row.append(info)
            cells.append(cell_row)
        return cells, occupied

    def _classify_cell(self, crop: np.ndarray, profile: dict[str, Any]) -> dict[str, Any]:
        if crop.size == 0:
            return {"occupied": False, "color": "empty", "confidence": 0.0, "occupied_ratio": 0.0}

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        mask = (saturation >= int(profile["occupied_s_min"])) & (value >= int(profile["occupied_v_min"]))
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        occupied = ratio >= float(profile["occupied_min_ratio"])
        if not occupied:
            return {
                "occupied": False,
                "color": "empty",
                "confidence": max(0.0, min(1.0, 1.0 - ratio)),
                "occupied_ratio": ratio,
            }

        hue_values = hsv[:, :, 0][mask]
        hue = float(np.median(hue_values)) if hue_values.size else 0.0
        return {
            "occupied": True,
            "color": self._hue_to_color(hue),
            "confidence": max(0.0, min(1.0, ratio)),
            "occupied_ratio": ratio,
        }

    def _infer_active_piece(
        self,
        occupied: np.ndarray,
        color_matrix: list[list[str]],
        profile: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        debug: dict[str, Any] = {
            "tracker_available": False,
            "tracker_candidate": None,
            "tracker_previous_settled_rows_text": [],
            "tracker_previous_settled_matrix": [],
            "tracker_diff_rows_text": [],
            "tracker_diff_matrix": [],
            "tracker_diff_cells": [],
            "component_count": 0,
            "component_candidates": [],
            "selected_component_index": None,
            "selected_strategy": None,
        }
        tracker = self._tracker.get(str(profile["profile_name"]) or self._DEFAULT_PROFILE)
        if tracker and tracker.get("settled") is not None:
            settled = np.array(tracker["settled"], dtype=bool)
            if settled.shape == occupied.shape:
                debug["tracker_available"] = True
                debug["tracker_previous_settled_rows_text"] = self._matrix_to_rows(settled)
                debug["tracker_previous_settled_matrix"] = settled.astype(np.uint8).tolist()
                diff = occupied & ~settled
                debug["tracker_diff_rows_text"] = self._matrix_to_rows(diff)
                debug["tracker_diff_matrix"] = diff.astype(np.uint8).tolist()
                debug["tracker_diff_cells"] = [[int(row), int(col)] for row, col in np.argwhere(diff).tolist()]
                piece = self._piece_from_cells(np.argwhere(diff), color_matrix, source="tracker_diff")
                debug["tracker_candidate"] = self._piece_debug_summary(piece)
                if piece["found"]:
                    debug["selected_strategy"] = "tracker_diff"
                    return piece, debug

        candidates = []
        components = self._connected_components(occupied)
        debug["component_count"] = len(components)
        for component_index, component in enumerate(components):
            candidate_debug = {
                "component_index": int(component_index),
                "size": int(len(component)),
                "cells": [[int(row), int(col)] for row, col in component],
            }
            if len(component) != 4:
                candidate_debug["candidate"] = {"found": False, "reason": "component_size"}
                debug["component_candidates"].append(candidate_debug)
                continue
            piece = self._piece_from_cells(np.array(component), color_matrix, source="component")
            candidate_debug["candidate"] = self._piece_debug_summary(piece)
            if not piece["found"]:
                debug["component_candidates"].append(candidate_debug)
                continue
            min_row = min(row for row, _ in component)
            max_row = max(row for row, _ in component)
            color_score = float(piece.get("color_consistency") or 0.0)
            top_score = (int(profile["board_rows"]) - min_row) / float(profile["board_rows"])
            score = top_score + color_score - (max_row * 0.01)
            candidate_debug["score"] = float(score)
            debug["component_candidates"].append(candidate_debug)
            candidates.append((score, piece, component_index))

        if candidates:
            best_score, best_piece, best_component_index = max(candidates, key=lambda item: item[0])
            debug["selected_component_index"] = int(best_component_index)
            debug["selected_strategy"] = "component"
            debug["selected_score"] = float(best_score)
            return best_piece, debug

        return {
            "found": False,
            "reason": "no_legal_active_piece",
            "cells": [],
            "shape": None,
            "rotation": None,
            "origin_row": None,
            "origin_col": None,
            "confidence": 0.0,
        }, debug

    def _piece_from_cells(self, cells_array: np.ndarray, color_matrix: list[list[str]], *, source: str) -> dict[str, Any]:
        cells = [(int(row), int(col)) for row, col in cells_array.tolist()]
        if len(cells) != 4:
            return {"found": False, "reason": "piece_cell_count", "cells": cells}
        normalized = _normalize_cells(cells)
        shape_meta = self.SHAPE_BY_CELLS.get(normalized)
        if shape_meta is None:
            return {"found": False, "reason": "shape_not_matched", "cells": cells}

        shape, rotation = shape_meta
        colors = [color_matrix[row][col] for row, col in cells]
        color_counts = Counter(color for color in colors if color != "empty")
        color, count = color_counts.most_common(1)[0] if color_counts else ("unknown", 0)
        color_consistency = float(count) / 4.0
        min_row = min(row for row, _ in cells)
        min_col = min(col for _, col in cells)
        return {
            "found": True,
            "reason": "ok",
            "source": source,
            "shape": shape,
            "rotation": int(rotation),
            "cells": [[row, col] for row, col in sorted(cells)],
            "origin_row": int(min_row),
            "origin_col": int(min_col),
            "color": color,
            "color_consistency": color_consistency,
            "confidence": min(1.0, 0.55 + color_consistency * 0.45),
        }

    @staticmethod
    def _piece_debug_summary(piece: dict[str, Any]) -> dict[str, Any]:
        return {
            "found": bool(piece.get("found")),
            "reason": piece.get("reason"),
            "source": piece.get("source"),
            "shape": piece.get("shape"),
            "rotation": piece.get("rotation"),
            "origin_row": piece.get("origin_row"),
            "origin_col": piece.get("origin_col"),
            "color": piece.get("color"),
            "color_consistency": piece.get("color_consistency"),
            "confidence": piece.get("confidence"),
            "cells": [[int(row), int(col)] for row, col in piece.get("cells") or []],
        }

    @staticmethod
    def _placement_summary(candidate: dict[str, Any], *, rank: int) -> dict[str, Any]:
        return {
            "rank": int(rank),
            "shape": candidate.get("shape"),
            "target_rotation": candidate.get("target_rotation"),
            "target_col": candidate.get("target_col"),
            "target_row": candidate.get("target_row"),
            "lines_cleared": candidate.get("lines_cleared"),
            "score": candidate.get("score"),
            "score_parts": dict(candidate.get("score_parts") or {}),
            "input_sequence": list(candidate.get("input_sequence") or []),
            "projected_metrics": dict(candidate.get("projected_metrics") or {}),
        }

    def _connected_components(self, occupied: np.ndarray) -> list[list[Cell]]:
        rows, cols = occupied.shape
        visited = np.zeros_like(occupied, dtype=bool)
        components: list[list[Cell]] = []
        for start_row in range(rows):
            for start_col in range(cols):
                if not occupied[start_row, start_col] or visited[start_row, start_col]:
                    continue
                queue: deque[Cell] = deque([(start_row, start_col)])
                visited[start_row, start_col] = True
                component: list[Cell] = []
                while queue:
                    row, col = queue.popleft()
                    component.append((int(row), int(col)))
                    for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                        if 0 <= next_row < rows and 0 <= next_col < cols:
                            if occupied[next_row, next_col] and not visited[next_row, next_col]:
                                visited[next_row, next_col] = True
                                queue.append((next_row, next_col))
                components.append(component)
        return components

    def _drop_row(self, board: np.ndarray, rotation_cells: tuple[Cell, ...], left_col: int) -> int | None:
        row = -max(cell_row for cell_row, _ in rotation_cells)
        if self._collides(board, rotation_cells, row, left_col):
            return None
        while not self._collides(board, rotation_cells, row + 1, left_col):
            row += 1
        if any(row + cell_row < 0 for cell_row, _ in rotation_cells):
            return None
        return int(row)

    def _collides(self, board: np.ndarray, rotation_cells: tuple[Cell, ...], top_row: int, left_col: int) -> bool:
        rows, cols = board.shape
        for cell_row, cell_col in rotation_cells:
            row = int(top_row + cell_row)
            col = int(left_col + cell_col)
            if col < 0 or col >= cols or row >= rows:
                return True
            if row >= 0 and bool(board[row, col]):
                return True
        return False

    def _place_and_clear(
        self,
        board: np.ndarray,
        rotation_cells: tuple[Cell, ...],
        top_row: int,
        left_col: int,
    ) -> tuple[np.ndarray, int, int]:
        placed = board.copy()
        piece_rows: list[int] = []
        for cell_row, cell_col in rotation_cells:
            row = int(top_row + cell_row)
            placed[row, int(left_col + cell_col)] = True
            piece_rows.append(row)
        full_rows = np.all(placed, axis=1)
        lines_cleared = int(np.count_nonzero(full_rows))
        eroded_piece_cells = lines_cleared * sum(1 for row in piece_rows if bool(full_rows[row]))
        if lines_cleared:
            kept = placed[~full_rows, :]
            placed = np.vstack([np.zeros((lines_cleared, placed.shape[1]), dtype=bool), kept])
        return placed, lines_cleared, eroded_piece_cells

    def _score_placement_bcts(
        self,
        board: np.ndarray,
        *,
        rotation_cells: tuple[Cell, ...],
        top_row: int,
        lines_cleared: int,
        eroded_piece_cells: int,
    ) -> tuple[float, dict[str, float]]:
        features = self._bcts_features(
            board,
            rotation_cells=rotation_cells,
            top_row=top_row,
            lines_cleared=lines_cleared,
            eroded_piece_cells=eroded_piece_cells,
        )
        weights = {
            "rows_with_holes": -24.04,
            "column_transitions": -19.77,
            "holes": -13.08,
            "landing_height": -12.63,
            "cumulative_wells": -10.49,
            "row_transitions": -9.22,
            "eroded_piece_cells": 6.60,
            "hole_depth": -1.61,
        }
        contributions = {key: float(features[key]) * weight for key, weight in weights.items()}
        return float(sum(contributions.values())), {**features, "contributions": contributions}

    def _score_board(
        self,
        metrics: dict[str, Any],
        *,
        lines_cleared: int,
        input_cost: int,
    ) -> tuple[float, dict[str, float]]:
        parts = {
            "holes": float(metrics["holes"]) * 1000.0,
            "max_height": float(metrics["max_height"]) * 80.0,
            "aggregate_height": float(metrics["aggregate_height"]) * 8.0,
            "bumpiness": float(metrics["bumpiness"]) * 35.0,
            "well_depth": float(metrics["well_depth"]) * 20.0,
            "top_danger": float(metrics["top_occupied_count"]) * 2000.0,
            "line_clear": float(lines_cleared) * -650.0,
            "input_cost": float(input_cost) * 2.0,
        }
        return float(sum(parts.values())), parts

    def _bcts_features(
        self,
        board: np.ndarray,
        *,
        rotation_cells: tuple[Cell, ...],
        top_row: int,
        lines_cleared: int,
        eroded_piece_cells: int,
    ) -> dict[str, float]:
        matrix = np.array(board, dtype=bool)
        rows, cols = matrix.shape
        hole_mask = np.zeros_like(matrix, dtype=bool)
        hole_depth = 0
        for col in range(cols):
            filled_seen = 0
            for row in range(rows):
                if matrix[row, col]:
                    filled_seen += 1
                elif filled_seen > 0:
                    hole_mask[row, col] = True
                    hole_depth += filled_seen

        row_transitions = 0
        for row in range(rows):
            previous = True
            for col in range(cols):
                current = bool(matrix[row, col])
                if current != previous:
                    row_transitions += 1
                previous = current
            if previous is not True:
                row_transitions += 1

        column_transitions = 0
        for col in range(cols):
            previous = False
            for row in range(rows):
                current = bool(matrix[row, col])
                if current != previous:
                    column_transitions += 1
                previous = current
            if previous is not True:
                column_transitions += 1

        cumulative_wells = 0
        for col in range(cols):
            depth = 0
            for row in range(rows):
                left_filled = True if col == 0 else bool(matrix[row, col - 1])
                right_filled = True if col == cols - 1 else bool(matrix[row, col + 1])
                if not matrix[row, col] and left_filled and right_filled:
                    depth += 1
                    cumulative_wells += depth
                else:
                    depth = 0

        piece_heights = [rows - int(top_row + cell_row) for cell_row, _ in rotation_cells]
        landing_height = (float(min(piece_heights)) + float(max(piece_heights))) / 2.0 if piece_heights else 0.0
        return {
            "rows_with_holes": float(np.count_nonzero(np.any(hole_mask, axis=1))),
            "column_transitions": float(column_transitions),
            "holes": float(np.count_nonzero(hole_mask)),
            "landing_height": landing_height,
            "cumulative_wells": float(cumulative_wells),
            "row_transitions": float(row_transitions),
            "eroded_piece_cells": float(eroded_piece_cells),
            "hole_depth": float(hole_depth),
        }

    def _analyze_next_queue(self, source_image: np.ndarray, profile: dict[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        scale_x, scale_y = self._scale_factors(source_image, profile)
        for index, region in enumerate(profile.get("next_regions") or []):
            x, y, width, height = region
            crop = self._crop_region(
                source_image,
                (
                    int(round(x * scale_x)),
                    int(round(y * scale_y)),
                    int(round(width * scale_x)),
                    int(round(height * scale_y)),
                ),
            )
            info = self._classify_cell(crop, profile)
            result.append(
                {
                    "index": index,
                    "visible": bool(info["occupied"]),
                    "dominant_color": info["color"],
                    "confidence": info["confidence"],
                }
            )
        return result

    def _scaled_grid(self, source_image: np.ndarray, profile: dict[str, Any]) -> tuple[float, float, float, float]:
        scale_x, scale_y = self._scale_factors(source_image, profile)
        origin_x, origin_y = profile["board_origin"]
        cell_w, cell_h = profile["board_cell_size"]
        return float(origin_x * scale_x), float(origin_y * scale_y), float(cell_w * scale_x), float(cell_h * scale_y)

    def _scale_factors(self, source_image: np.ndarray, profile: dict[str, Any]) -> tuple[float, float]:
        client_w, client_h = profile["client_size"]
        image_h, image_w = source_image.shape[:2]
        return float(image_w) / float(client_w), float(image_h) / float(client_h)

    @staticmethod
    def _scale_region(
        region: tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
    ) -> tuple[int, int, int, int]:
        x, y, width, height = region
        return (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            max(int(round(width * scale_x)), 1),
            max(int(round(height * scale_y)), 1),
        )

    def _crop_cell(
        self,
        image: np.ndarray,
        origin_x: float,
        origin_y: float,
        cell_w: float,
        cell_h: float,
        row: int,
        col: int,
        sample_ratio: float,
    ) -> np.ndarray:
        cx = origin_x + (float(col) + 0.5) * cell_w
        cy = origin_y + (float(row) + 0.5) * cell_h
        half_w = max(int(round(cell_w * sample_ratio / 2.0)), 1)
        half_h = max(int(round(cell_h * sample_ratio / 2.0)), 1)
        x0 = int(round(cx)) - half_w
        y0 = int(round(cy)) - half_h
        return self._crop_region(image, (x0, y0, half_w * 2 + 1, half_h * 2 + 1))

    @staticmethod
    def _crop_region(image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = region
        image_h, image_w = image.shape[:2]
        left = max(int(x), 0)
        top = max(int(y), 0)
        right = min(max(left, int(x + width)), image_w)
        bottom = min(max(top, int(y + height)), image_h)
        return image[top:bottom, left:right]

    @staticmethod
    def _matrix_to_rows(matrix: np.ndarray) -> list[str]:
        return ["".join("#" if bool(value) else "." for value in row) for row in matrix]

    @staticmethod
    def _hue_to_color(hue: float) -> str:
        if hue < 8 or hue >= 170:
            return "red"
        if hue < 20:
            return "orange"
        if hue < 36:
            return "yellow"
        if hue < 85:
            return "green"
        if hue < 100:
            return "cyan"
        if hue < 130:
            return "blue"
        if hue < 160:
            return "purple"
        return "pink"

    @staticmethod
    def _coerce_size(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return max(int(value[0]), 1), max(int(value[1]), 1)
        return default

    @staticmethod
    def _coerce_float_pair(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
        return default

    @staticmethod
    def _coerce_point(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
        return default

    @staticmethod
    def _coerce_region(value: Any, *, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return (int(value[0]), int(value[1]), max(int(value[2]), 1), max(int(value[3]), 1))
        return default

    @staticmethod
    def _coerce_hsv_triplet(value: Any, *, default: tuple[int, int, int]) -> tuple[int, int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return (
                max(min(int(value[0]), 179), 0),
                max(min(int(value[1]), 255), 0),
                max(min(int(value[2]), 255), 0),
            )
        return default

    @staticmethod
    def _coerce_regions(value: Any) -> list[tuple[int, int, int, int]]:
        regions: list[tuple[int, int, int, int]] = []
        if not isinstance(value, list):
            return regions
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                regions.append((int(item[0]), int(item[1]), max(int(item[2]), 1), max(int(item[3]), 1)))
        return regions

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

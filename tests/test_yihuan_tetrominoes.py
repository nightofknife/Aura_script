from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from plans.aura_base.src.platform.contracts import CaptureResult
from plans.yihuan.src.actions import tetrominoes_actions
from plans.yihuan.src.services.tetrominoes_service import YihuanTetrominoesService


CELL_COLORS = {
    "I": (80, 220, 255),
    "O": (255, 214, 64),
    "T": (190, 95, 255),
    "S": (72, 220, 132),
    "Z": (255, 85, 95),
    "J": (92, 115, 255),
    "L": (255, 145, 65),
    "X": (120, 170, 255),
}


class _FakeApp:
    def __init__(
        self,
        image: np.ndarray | None = None,
        *,
        images: list[np.ndarray] | None = None,
        fail_capture: bool = False,
    ) -> None:
        self.image = image if image is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
        self.images = list(images or [])
        self.fail_capture = fail_capture
        self.release_all_calls = 0
        self.click_calls: list[tuple[int, int, str]] = []

    def capture(self):
        if self.fail_capture:
            return CaptureResult(success=False, image=None)
        if self.images:
            image = self.images.pop(0)
            self.image = image
            return CaptureResult(success=True, image=image.copy())
        return CaptureResult(success=True, image=self.image.copy())

    def click(self, x=None, y=None, button="left", clicks=1, interval=None):
        self.click_calls.append((int(x), int(y), str(button)))

    def release_all(self):
        self.release_all_calls += 1


class _FakeInputMapping:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def execute_action(self, action_name: str, *, phase: str, app, profile=None):
        self.calls.append((str(action_name), str(phase), None if profile is None else str(profile)))
        return {"ok": True, "action_name": action_name, "phase": phase}


class TestYihuanTetrominoesService(unittest.TestCase):
    def setUp(self) -> None:
        self.service = YihuanTetrominoesService()
        self.profile = self.service.load_profile()
        self.service.reset_tracker()

    def test_analyze_state_classifies_board_and_current_piece(self):
        image = _make_board_image(
            self.service,
            [((2, 4), "S"), ((2, 5), "S"), ((3, 3), "S"), ((3, 4), "S"), ((19, 0), "J")],
        )

        state = self.service.analyze_state(image, update_tracker=False)

        self.assertEqual(state["board"]["occupied_count"], 5)
        self.assertEqual(state["current_piece"]["shape"], "S")
        self.assertEqual(state["current_piece"]["rotation"], 0)
        self.assertGreaterEqual(state["board"]["confidence"], self.profile["board_confidence_min"])
        self.assertEqual(state["board"]["rows_text"][2], "....##....")
        self.assertEqual(state["board"]["color_matrix"][19][0], "blue")

    def test_analyze_state_scales_grid_to_capture_size(self):
        cells = [((1, 4), "O"), ((1, 5), "O"), ((2, 4), "O"), ((2, 5), "O")]
        baseline = self.service.analyze_state(_make_board_image(self.service, cells), update_tracker=False)
        scaled = self.service.analyze_state(_make_board_image(self.service, cells, scale=0.5), update_tracker=False)

        self.assertEqual(scaled["board"]["rows_text"], baseline["board"]["rows_text"])
        self.assertEqual(scaled["current_piece"]["shape"], "O")

    def test_all_tetromino_shapes_are_recognized(self):
        for shape, rotations in self.service.SHAPES.items():
            with self.subTest(shape=shape):
                cells = [((row + 2, col + 3), shape) for row, col in rotations[0]]
                image = _make_board_image(self.service, cells)
                state = self.service.analyze_state(image, update_tracker=False)
                self.assertTrue(state["current_piece"]["found"])
                self.assertEqual(state["current_piece"]["shape"], shape)
                self.assertEqual(state["current_piece"]["rotation"], 0)

    def test_tracker_diff_separates_active_piece_from_settled_board(self):
        settled = np.zeros((20, 10), dtype=np.uint8)
        settled[18, 4] = 1
        settled[18, 5] = 1
        settled[19, 4] = 1
        settled[19, 5] = 1
        self.service.commit_settled_matrix(settled.tolist())
        cells = [
            ((18, 4), "O"),
            ((18, 5), "O"),
            ((19, 4), "O"),
            ((19, 5), "O"),
            ((16, 4), "T"),
            ((17, 3), "T"),
            ((17, 4), "T"),
            ((17, 5), "T"),
        ]
        image = _make_board_image(self.service, cells)

        state = self.service.analyze_state(image, update_tracker=True)

        self.assertEqual(state["current_piece"]["source"], "tracker_diff")
        self.assertEqual(state["current_piece"]["shape"], "T")
        self.assertEqual(state["board"]["settled_matrix"][18][4], 1)
        self.assertEqual(state["board"]["settled_matrix"][17][4], 0)

    def test_solver_prefers_line_clear_for_i_piece_gap(self):
        cells = [((row, 0), "I") for row in range(1, 5)]
        cells.extend(((19, col), "X") for col in range(10) if col not in {4, 5, 6, 7})
        image = _make_board_image(self.service, cells)
        state = self.service.analyze_state(image, update_tracker=False)

        decision = self.service.choose_best_move(state)

        self.assertTrue(decision["found"])
        self.assertEqual(decision["shape"], "I")
        self.assertEqual(decision["target_rotation"], 0)
        self.assertEqual(decision["target_col"], 4)
        self.assertEqual(decision["lines_cleared"], 1)

    def test_solver_returns_input_sequence_in_rotation_move_drop_order(self):
        sequence = self.service.build_input_sequence(
            current_rotation=1,
            target_rotation=0,
            rotation_count=2,
            current_col=0,
            target_col=4,
        )

        self.assertEqual(
            sequence,
            [
                "tetrominoes_rotate_cw",
                "tetrominoes_right",
                "tetrominoes_right",
                "tetrominoes_right",
                "tetrominoes_right",
                "tetrominoes_fast_drop",
            ],
        )

    def test_metrics_reports_holes_and_top_danger(self):
        board = np.zeros((20, 10), dtype=np.uint8)
        board[0, 0] = 1
        board[18, 1] = 1
        board[19, 1] = 1
        board[17, 2] = 1
        board[19, 2] = 1

        metrics = self.service.compute_metrics(board)

        self.assertGreaterEqual(metrics["top_occupied_count"], 1)
        self.assertGreaterEqual(metrics["holes"], 1)
        self.assertGreaterEqual(metrics["max_height"], 20)

    def test_result_screen_detector_uses_visual_panel_and_exit_button(self):
        result_image = _make_result_screen_image(self.service)
        playing_image = _make_board_image(self.service, _line_clear_fixture_cells())

        result = self.service.analyze_result_screen(result_image)
        playing = self.service.analyze_result_screen(playing_image)

        self.assertTrue(result["found"])
        self.assertEqual(result["reason"], "ok")
        self.assertGreaterEqual(result["panel_purple_ratio"], result["panel_min_purple_ratio"])
        self.assertGreaterEqual(result["exit_white_ratio"], result["exit_min_white_ratio"])
        self.assertFalse(playing["found"])


class TestYihuanTetrominoesActions(unittest.TestCase):
    def setUp(self) -> None:
        self.service = YihuanTetrominoesService()
        self.service.reset_tracker()

    def test_analyze_screen_returns_decision_without_input(self):
        image = _make_board_image(self.service, _line_clear_fixture_cells())
        app = _FakeApp(image)

        result = tetrominoes_actions.yihuan_tetrominoes_analyze_screen(app, self.service)

        self.assertEqual(result["current_piece"]["shape"], "I")
        self.assertTrue(result["decision"]["found"])
        self.assertEqual(app.release_all_calls, 0)

    def test_analyze_screen_stops_on_visual_result_screen(self):
        app = _FakeApp(_make_result_screen_image(self.service))

        result = tetrominoes_actions.yihuan_tetrominoes_analyze_screen(app, self.service)

        self.assertEqual(result["phase"], "result")
        self.assertTrue(result["result_screen"]["found"])
        self.assertEqual(result["decision"]["reason"], "level_end")

    def test_run_session_dry_run_does_not_send_input(self):
        image = _make_board_image(self.service, _line_clear_fixture_cells())
        app = _FakeApp(image)
        input_mapping = _FakeInputMapping()

        result = tetrominoes_actions.yihuan_tetrominoes_run_session(
            app,
            input_mapping,
            self.service,
            max_seconds=5,
            max_pieces=1,
            dry_run=True,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "max_pieces")
        self.assertEqual(result["pieces_played"], 1)
        self.assertEqual(len(result["operation_log"]), 1)
        self.assertEqual(result["operation_log"][0]["detected_piece"]["shape"], "I")
        self.assertEqual(result["operation_log"][0]["executed_sequence"], [])
        self.assertFalse(result["start_clicked"])
        self.assertEqual(app.click_calls, [])
        self.assertEqual(input_mapping.calls, [])
        self.assertEqual(app.release_all_calls, 1)

    def test_run_session_clicks_start_then_executes_rotation_move_and_fast_drop(self):
        image = _make_board_image(self.service, _line_clear_fixture_cells())
        app = _FakeApp(image)
        input_mapping = _FakeInputMapping()

        with patch("plans.yihuan.src.actions.tetrominoes_actions.time.sleep") as sleep_mock:
            result = tetrominoes_actions.yihuan_tetrominoes_run_session(
                app,
                input_mapping,
                self.service,
                max_seconds=5,
                max_pieces=1,
                dry_run=False,
            )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["start_clicked"])
        self.assertEqual(app.click_calls, [(1120, 670, "left")])
        sleep_mock.assert_any_call(6.0)
        self.assertEqual(len(result["operation_log"]), 1)
        self.assertEqual(result["operation_log"][0]["detected_piece"]["shape"], "I")
        hold_calls = [call for call in input_mapping.calls if call[1] == "hold"]
        release_calls = [call for call in input_mapping.calls if call[1] == "release"]
        self.assertEqual([call[0] for call in hold_calls], result["decisions_tail"][0]["input_sequence"])
        self.assertEqual([call[0] for call in release_calls], result["decisions_tail"][0]["input_sequence"])
        self.assertEqual([call[0] for call in hold_calls], result["operation_log"][0]["executed_sequence"])
        self.assertEqual(input_mapping.calls[0][0], "tetrominoes_rotate_cw")
        self.assertEqual(input_mapping.calls[-1][0], "tetrominoes_fast_drop")
        self.assertTrue(all(call[1] in {"hold", "release"} for call in input_mapping.calls))
        self.assertTrue(all(call[2] == "default_1280x720_cn" for call in input_mapping.calls))
        self.assertEqual(app.release_all_calls, 1)

    def test_run_session_stops_when_start_leads_to_result_screen(self):
        app = _FakeApp(
            images=[
                _make_board_image(self.service, _line_clear_fixture_cells()),
                _make_result_screen_image(self.service),
            ]
        )
        input_mapping = _FakeInputMapping()

        with patch("plans.yihuan.src.actions.tetrominoes_actions.time.sleep") as sleep_mock:
            result = tetrominoes_actions.yihuan_tetrominoes_run_session(
                app,
                input_mapping,
                self.service,
                max_seconds=5,
                max_pieces=1,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "level_end")
        self.assertEqual(result["pieces_played"], 0)
        self.assertEqual(result["operation_log"], [])
        self.assertTrue(result["result_screen"]["found"])
        self.assertEqual(input_mapping.calls, [])
        self.assertEqual(app.click_calls, [(1120, 670, "left")])
        sleep_mock.assert_any_call(6.0)
        self.assertEqual(app.release_all_calls, 1)

    def test_run_session_clears_existing_result_screen_before_start(self):
        app = _FakeApp(
            images=[
                _make_result_screen_image(self.service),
                _make_board_image(self.service, _line_clear_fixture_cells()),
            ]
        )
        input_mapping = _FakeInputMapping()

        with patch("plans.yihuan.src.actions.tetrominoes_actions.time.sleep") as sleep_mock:
            result = tetrominoes_actions.yihuan_tetrominoes_run_session(
                app,
                input_mapping,
                self.service,
                max_seconds=5,
                max_pieces=1,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stopped_reason"], "max_pieces")
        self.assertTrue(result["result_screen_cleared_before_start"])
        self.assertEqual(app.click_calls, [(645, 616, "left"), (1120, 670, "left")])
        sleep_mock.assert_any_call(1.0)
        sleep_mock.assert_any_call(6.0)
        self.assertEqual(len(result["operation_log"]), 1)

    def test_run_session_capture_failure_returns_failed_without_input(self):
        app = _FakeApp(fail_capture=True)
        input_mapping = _FakeInputMapping()

        with patch("plans.yihuan.src.actions.tetrominoes_actions.time.sleep") as sleep_mock:
            result = tetrominoes_actions.yihuan_tetrominoes_run_session(
                app,
                input_mapping,
                self.service,
                max_seconds=5,
                max_pieces=1,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stopped_reason"], "capture_failed")
        self.assertEqual(input_mapping.calls, [])
        self.assertEqual(app.click_calls, [])
        self.assertEqual(app.release_all_calls, 1)


def _line_clear_fixture_cells() -> list[tuple[tuple[int, int], str]]:
    cells = [((row, 0), "I") for row in range(1, 5)]
    cells.extend(((19, col), "X") for col in range(10) if col not in {4, 5, 6, 7})
    return cells


def _make_board_image(
    service: YihuanTetrominoesService,
    cells: list[tuple[tuple[int, int], str]],
    *,
    scale: float = 1.0,
) -> np.ndarray:
    profile = service.load_profile()
    width = int(round(profile["client_size"][0] * scale))
    height = int(round(profile["client_size"][1] * scale))
    image = np.full((height, width, 3), 28, dtype=np.uint8)
    origin_x = float(profile["board_origin"][0]) * scale
    origin_y = float(profile["board_origin"][1]) * scale
    cell_w = float(profile["board_cell_size"][0]) * scale
    cell_h = float(profile["board_cell_size"][1]) * scale

    grid_color = np.array([52, 52, 52], dtype=np.uint8)
    for col in range(profile["board_cols"] + 1):
        x = int(round(origin_x + col * cell_w))
        image[:, max(x - 1, 0): min(x + 1, width)] = grid_color
    for row in range(profile["board_rows"] + 1):
        y = int(round(origin_y + row * cell_h))
        image[max(y - 1, 0): min(y + 1, height), :] = grid_color

    margin = max(int(round(3 * scale)), 1)
    for (row, col), shape in cells:
        color = np.array(CELL_COLORS[shape], dtype=np.uint8)
        x0 = int(round(origin_x + col * cell_w)) + margin
        y0 = int(round(origin_y + row * cell_h)) + margin
        x1 = int(round(origin_x + (col + 1) * cell_w)) - margin
        y1 = int(round(origin_y + (row + 1) * cell_h)) - margin
        cv2.rectangle(image, (x0, y0), (x1, y1), tuple(int(value) for value in color.tolist()), thickness=-1)
    return image


def _make_result_screen_image(service: YihuanTetrominoesService, *, scale: float = 1.0) -> np.ndarray:
    profile = service.load_profile()
    width = int(round(profile["client_size"][0] * scale))
    height = int(round(profile["client_size"][1] * scale))
    image = np.full((height, width, 3), 180, dtype=np.uint8)

    panel_x, panel_y, panel_w, panel_h = profile["result_panel_region"]
    exit_x, exit_y, exit_w, exit_h = profile["result_exit_button_region"]
    panel_rect = (
        int(round(panel_x * scale)),
        int(round(panel_y * scale)),
        int(round((panel_x + panel_w) * scale)),
        int(round((panel_y + panel_h) * scale)),
    )
    exit_rect = (
        int(round(exit_x * scale)),
        int(round(exit_y * scale)),
        int(round((exit_x + exit_w) * scale)),
        int(round((exit_y + exit_h) * scale)),
    )
    cv2.rectangle(image, panel_rect[:2], panel_rect[2:], (62, 58, 110), thickness=-1)
    cv2.rectangle(image, exit_rect[:2], exit_rect[2:], (220, 220, 225), thickness=-1)
    return image


if __name__ == "__main__":
    unittest.main()

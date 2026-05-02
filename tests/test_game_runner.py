from __future__ import annotations

import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from packages.aura_core.scheduler.cancellation import clear_task_cancel, is_task_cancel_requested
from packages.aura_core.scheduler.task_dispatcher import TaskDispatcher
from packages.aura_core.runtime import AdminPrivilegeRequiredError
from packages.aura_game import EmbeddedGameRunner, SubprocessGameRunner


class TestGameRunners(unittest.TestCase):
    def test_embedded_runner_start_stop_lifecycle(self):
        runner = EmbeddedGameRunner()
        try:
            self.assertFalse(runner.status()["ready"])
            self.assertTrue(runner.start()["ready"])
            self.assertFalse(runner.stop()["ready"])
        finally:
            runner.close()

    def test_embedded_runner_lists_games_and_tasks(self):
        runner = EmbeddedGameRunner()
        try:
            games = runner.list_games()
            names = {row["game_name"] for row in games}
            self.assertIn("aura_benchmark", names)

            tasks = runner.list_tasks("aura_benchmark")
            refs = {row["task_ref"] for row in tasks}
            self.assertIn("tasks:single_sleep.yaml", refs)
        finally:
            runner.close()

    def test_embedded_runner_can_execute_benchmark_task(self):
        runner = EmbeddedGameRunner()
        try:
            result = runner.run_task(
                game_name="aura_benchmark",
                task_ref="tasks:single_sleep.yaml",
                inputs={"duration_ms": 1, "scenario": "embedded_test"},
                wait=True,
                timeout_sec=60,
            )
            self.assertEqual(result["dispatch"]["game_name"], "aura_benchmark")
            self.assertEqual(result["run"]["detail"]["status"], "success")
            runs = runner.list_runs(limit=5, game_name="aura_benchmark")
            self.assertTrue(runs)
            self.assertEqual(runs[0]["game_name"], "aura_benchmark")
        finally:
            runner.close()

    def test_embedded_runner_cancel_task_delegates_to_runtime(self):
        runner = EmbeddedGameRunner()
        fake_runtime = Mock()
        fake_runtime.cancel_task.return_value = {"status": "success", "message": "cancelled"}
        with patch("packages.aura_game.runner.create_runtime", return_value=fake_runtime):
            result = runner.cancel_task("cid-123")

        self.assertEqual(result["status"], "success")
        fake_runtime.cancel_task.assert_called_once_with("cid-123")

    def test_embedded_runner_target_status_uses_runtime_target_service(self):
        runner = EmbeddedGameRunner()
        fake_service = Mock()
        fake_service.target_summary.return_value = {"title": "Yihuan", "client_rect_screen": [1, 2, 3, 4]}
        with (
            patch.object(runner, "_ensure_running_runtime", return_value=object()),
            patch("packages.aura_game.runner.service_registry.get_service_instance", return_value=fake_service) as get_service,
        ):
            result = runner.target_status(game_name="yihuan")

        self.assertTrue(result["ok"])
        self.assertEqual(result["game_name"], "yihuan")
        self.assertEqual(result["target"]["title"], "Yihuan")
        get_service.assert_called_once_with("target_runtime")

    def test_embedded_runner_target_snapshot_serializes_runtime_capture(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy is not installed")

        runner = EmbeddedGameRunner()
        fake_service = Mock()
        fake_service.capture.return_value = SimpleNamespace(
            success=True,
            image=np.zeros((2, 3, 3), dtype=np.uint8),
            backend="gdi",
            image_size=(3, 2),
            window_rect=(10, 20, 30, 40),
            relative_rect=(0, 0, 3, 2),
            quality_flags=["test"],
            error_message="",
        )
        fake_service.target_summary.return_value = {"title": "Yihuan"}
        with (
            patch.object(runner, "_ensure_running_runtime", return_value=object()),
            patch("packages.aura_game.runner.service_registry.get_service_instance", return_value=fake_service),
        ):
            result = runner.target_snapshot(game_name="yihuan", backend="gdi")

        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "gdi")
        self.assertEqual(result["image_size"], [3, 2])
        self.assertEqual(result["window_rect"], [10, 20, 30, 40])
        self.assertEqual(result["quality_flags"], ["test"])
        self.assertTrue(result["image_png"].startswith(b"\x89PNG"))

    def test_subprocess_runner_start_stop_lifecycle(self):
        runner = SubprocessGameRunner()
        try:
            self.assertFalse(runner.status()["ready"])
            self.assertTrue(runner.start()["ready"])
            self.assertFalse(runner.stop()["ready"])
        finally:
            runner.close()

    def test_subprocess_runner_lists_games(self):
        runner = SubprocessGameRunner()
        try:
            games = runner.list_games()
            names = {row["game_name"] for row in games}
            self.assertIn("aura_benchmark", names)
        finally:
            runner.close()

    def test_subprocess_runner_cancel_task_uses_request_channel(self):
        runner = SubprocessGameRunner()
        with patch.object(runner, "_request", return_value={"status": "success"}) as request:
            result = runner.cancel_task("cid-123")

        self.assertEqual(result["status"], "success")
        request.assert_called_once_with("cancel_task", cid="cid-123")

    def test_subprocess_runner_target_helpers_use_request_channel(self):
        runner = SubprocessGameRunner()
        with patch.object(runner, "_request", return_value={"ok": True}) as request:
            result = runner.target_status(game_name="yihuan")
            self.assertTrue(result["ok"])
            request.assert_called_once_with("target_status", game_name="yihuan")

        with patch.object(runner, "_request", return_value={"ok": True}) as request:
            result = runner.target_snapshot(game_name="yihuan", backend="gdi")
            self.assertTrue(result["ok"])
            request.assert_called_once_with("target_snapshot", game_name="yihuan", backend="gdi")

    def test_embedded_runner_requires_admin_startup(self):
        runner = EmbeddedGameRunner()
        try:
            with patch(
                "packages.aura_core.scheduler.core.ensure_admin_startup",
                side_effect=AdminPrivilegeRequiredError("Aura Scheduler"),
            ):
                with self.assertRaises(AdminPrivilegeRequiredError):
                    runner.list_games()
        finally:
            runner.close()

    def test_dispatcher_cancel_marks_cooperative_cancel_request(self):
        class FakeTask:
            def __init__(self):
                self.cancel_called = False

            def done(self):
                return False

            def cancel(self):
                self.cancel_called = True

        task = FakeTask()
        scheduler = SimpleNamespace(
            fallback_lock=threading.RLock(),
            running_tasks={"cid-123": task},
            _running_task_meta={},
            _loop=None,
        )
        clear_task_cancel("cid-123")

        try:
            result = TaskDispatcher(scheduler).cancel_task("cid-123")

            self.assertEqual(result["status"], "success")
            self.assertTrue(task.cancel_called)
            self.assertTrue(is_task_cancel_requested("cid-123"))
        finally:
            clear_task_cancel("cid-123")


if __name__ == "__main__":
    unittest.main()

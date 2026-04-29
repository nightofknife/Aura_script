from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

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


if __name__ == "__main__":
    unittest.main()

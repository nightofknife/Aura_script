from __future__ import annotations

import unittest

from plans.yihuan.src.actions.package_actions import yihuan_runtime_probe


class _FakeTargetRuntime:
    def __init__(self, payload=None, exc: Exception | None = None):
        self._payload = payload
        self._exc = exc

    def self_check(self):
        if self._exc is not None:
            raise self._exc
        return self._payload


class TestYihuanRuntimeProbe(unittest.TestCase):
    def test_runtime_probe_preserves_successful_window_probe(self):
        result = yihuan_runtime_probe(
            screen=_FakeTargetRuntime(
                {
                    "ok": True,
                    "provider": "windows",
                    "family": "windows_desktop",
                    "target": {"title": "异环"},
                    "capture": {"backend": "gdi"},
                    "input": {"backend": "sendinput"},
                    "warnings": [],
                }
            )
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "windows")
        self.assertEqual(result["target"]["title"], "异环")

    def test_runtime_probe_preserves_missing_window_probe(self):
        result = yihuan_runtime_probe(
            screen=_FakeTargetRuntime(
                {
                    "ok": False,
                    "provider": "windows",
                    "family": "windows_desktop",
                    "target": {},
                    "capture": {},
                    "input": {},
                    "warnings": ["window_not_found"],
                    "code": "window_not_found",
                    "message": "Could not bind the game window.",
                }
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "window_not_found")
        self.assertIn("window_not_found", result["warnings"])

    def test_runtime_probe_converts_unexpected_configuration_errors(self):
        result = yihuan_runtime_probe(
            screen=_FakeTargetRuntime(exc=RuntimeError("runtime config is invalid"))
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "RuntimeError")
        self.assertEqual(result["message"], "runtime config is invalid")
        self.assertEqual(result["target"], {})


if __name__ == "__main__":
    unittest.main()

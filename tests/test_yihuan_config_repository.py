from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from packages.yihuan_gui.config_repository import YihuanConfigRepository
from packages.yihuan_gui.logic import (
    CafeRunDefaults,
    FishingRunDefaults,
    GuiPreferences,
    MahjongRunDefaults,
    RuntimeSettings,
)


class _MemorySettingsStore:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default_value=None):
        return self.values.get(key, default_value)

    def setValue(self, key: str, value) -> None:  # noqa: N802
        self.values[key] = value


class TestYihuanConfigRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.plan_root = Path(self.temp_dir.name) / "yihuan"
        (self.plan_root / "data" / "input_profiles").mkdir(parents=True)
        (self.plan_root / "data" / "fishing").mkdir(parents=True)
        (self.plan_root / "data" / "cafe").mkdir(parents=True)
        (self.plan_root / "data" / "mahjong").mkdir(parents=True)
        (self.plan_root / "data" / "input_profiles" / "default_pc.yaml").write_text("actions: {}\n", encoding="utf-8")
        (self.plan_root / "data" / "input_profiles" / "gamepad.yaml").write_text("actions: {}\n", encoding="utf-8")
        (self.plan_root / "data" / "fishing" / "default_1280x720_cn.yaml").write_text(
            "profile_name: default_1280x720_cn\n",
            encoding="utf-8",
        )
        (self.plan_root / "data" / "fishing" / "custom_1280x720_cn.yaml").write_text(
            "profile_name: custom_1280x720_cn\n",
            encoding="utf-8",
        )
        (self.plan_root / "data" / "cafe" / "default_1280x720_cn.yaml").write_text(
            "\n".join(
                [
                    "profile_name: default_1280x720_cn",
                    "max_seconds: 130",
                    "min_order_interval_sec: 0.8",
                    "min_order_duration_sec: 0.2",
                    "fake_customer_enabled: true",
                    "fake_customer_hammer_debug_enabled: false",
                    "fake_customer_order_guard_enabled: true",
                    "fake_customer_order_guard_distance_px: 85",
                    "fake_customer_max_hammers_per_scan: 3",
                    "order_decision_debug_enabled: false",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (self.plan_root / "data" / "cafe" / "custom_cafe.yaml").write_text(
            "profile_name: custom_cafe\n",
            encoding="utf-8",
        )
        (self.plan_root / "data" / "mahjong" / "default_1280x720_cn.yaml").write_text(
            "profile_name: default_1280x720_cn\n",
            encoding="utf-8",
        )
        (self.plan_root / "data" / "mahjong" / "custom_mahjong.yaml").write_text(
            "profile_name: custom_mahjong\n",
            encoding="utf-8",
        )
        (self.plan_root / "config.yaml").write_text(
            "\n".join(
                [
                    "runtime:",
                    "  family: windows_desktop",
                    "  provider: windows",
                    "  target:",
                    '    mode: title',
                    '    title_regex: "(异环|Neverness to Everness)"',
                    "    exclude_titles:",
                    "      - Crash Reporter",
                    "      - Unreal Engine",
                    "    allow_borderless: true",
                    "  capture:",
                    "    backend: gdi",
                    "  input:",
                    "    backend: sendinput",
                    "  window_spec:",
                    "    mode: off",
                    "input:",
                    "  profile: default_pc",
                    "  actions: {}",
                    "extra:",
                    "  keep: true",
                ]
            ),
            encoding="utf-8",
        )
        self.store = _MemorySettingsStore()
        self.repo = YihuanConfigRepository(self.plan_root, self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_runtime_settings_round_trip_preserves_unedited_fields(self):
        self.repo.update_runtime_settings(
            RuntimeSettings(
                title_regex="(异环测试服)",
                exclude_titles=["Crash Reporter"],
                allow_borderless=False,
                capture_backend="dxgi",
                input_backend="window_message",
                input_profile="gamepad",
            )
        )

        payload = self.repo.load_config()
        self.assertEqual(payload["runtime"]["target"]["title_regex"], "(异环测试服)")
        self.assertEqual(payload["runtime"]["capture"]["backend"], "dxgi")
        self.assertEqual(payload["input"]["profile"], "gamepad")
        self.assertEqual(payload["extra"]["keep"], True)
        self.assertEqual(payload["input"]["actions"], {})

    def test_exclude_titles_text_helpers_round_trip(self):
        text = self.repo.exclude_titles_to_text(["Crash Reporter", "", "Unreal Engine"])
        values = self.repo.exclude_titles_from_text(text)

        self.assertEqual(text, "Crash Reporter\nUnreal Engine")
        self.assertEqual(values, ["Crash Reporter", "Unreal Engine"])

    def test_runtime_settings_validation_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "窗口标题匹配规则不能为空"):
            self.repo.validate_runtime_settings(
                RuntimeSettings(
                    title_regex="",
                    exclude_titles=[],
                    allow_borderless=True,
                    capture_backend="gdi",
                    input_backend="sendinput",
                    input_profile="default_pc",
                )
            )

        with self.assertRaisesRegex(ValueError, "不支持的截图后端"):
            self.repo.validate_runtime_settings(
                RuntimeSettings(
                    title_regex="(异环)",
                    exclude_titles=[],
                    allow_borderless=True,
                    capture_backend="invalid",
                    input_backend="sendinput",
                    input_profile="default_pc",
                )
            )

    def test_ui_preferences_round_trip(self):
        prefs = GuiPreferences(
            history_limit=80,
            auto_runtime_probe_on_startup=False,
            expand_developer_tools=True,
            task_start_delay_sec=7,
            quick_stop_hotkey="F9",
        )
        self.repo.save_ui_preferences(prefs)

        loaded = self.repo.get_ui_preferences()
        self.assertEqual(loaded.history_limit, 80)
        self.assertFalse(loaded.auto_runtime_probe_on_startup)
        self.assertTrue(loaded.expand_developer_tools)
        self.assertEqual(loaded.task_start_delay_sec, 7)
        self.assertEqual(loaded.quick_stop_hotkey, "F9")

    def test_ui_preferences_validation_rejects_non_positive_history_limit(self):
        with self.assertRaisesRegex(ValueError, "历史记录显示条数必须大于 0"):
            self.repo.validate_ui_preferences(
                GuiPreferences(history_limit=0, auto_runtime_probe_on_startup=True, expand_developer_tools=False)
            )

    def test_ui_preferences_validation_rejects_invalid_delay_and_hotkey(self):
        with self.assertRaisesRegex(ValueError, "任务启动延迟秒数必须在 0 到 60 之间"):
            self.repo.validate_ui_preferences(GuiPreferences(task_start_delay_sec=61))

        with self.assertRaisesRegex(ValueError, "快捷停止键必须是 F6 到 F12"):
            self.repo.validate_ui_preferences(GuiPreferences(quick_stop_hotkey="ESC"))

    def test_fishing_defaults_round_trip_preserves_unedited_fields(self):
        self.repo.update_fishing_defaults(FishingRunDefaults(profile_name="custom_1280x720_cn"))

        defaults = self.repo.get_fishing_defaults()
        payload = self.repo.load_config()
        self.assertEqual(defaults.profile_name, "custom_1280x720_cn")
        self.assertEqual(payload["fishing"]["profile_name"], "custom_1280x720_cn")
        self.assertEqual(payload["extra"]["keep"], True)

    def test_fishing_defaults_validation_rejects_unknown_profile(self):
        with self.assertRaisesRegex(ValueError, "钓鱼识别档案不存在"):
            self.repo.validate_fishing_defaults(FishingRunDefaults(profile_name="missing_profile"))

    def test_cafe_defaults_round_trip_preserves_unedited_fields(self):
        self.repo.update_cafe_defaults(CafeRunDefaults(profile_name="custom_cafe"))

        defaults = self.repo.get_cafe_defaults()
        payload = self.repo.load_config()
        self.assertEqual(defaults.profile_name, "custom_cafe")
        self.assertEqual(payload["cafe"]["profile_name"], "custom_cafe")
        self.assertEqual(payload["extra"]["keep"], True)

    def test_cafe_defaults_validation_rejects_unknown_profile(self):
        with self.assertRaisesRegex(ValueError, "沙威玛识别档案不存在"):
            self.repo.validate_cafe_defaults(CafeRunDefaults(profile_name="missing_profile"))

    def test_mahjong_defaults_round_trip_preserves_unedited_fields(self):
        self.repo.update_mahjong_defaults(MahjongRunDefaults(profile_name="custom_mahjong"))

        defaults = self.repo.get_mahjong_defaults()
        payload = self.repo.load_config()
        self.assertEqual(defaults.profile_name, "custom_mahjong")
        self.assertEqual(payload["mahjong"]["profile_name"], "custom_mahjong")
        self.assertEqual(payload["extra"]["keep"], True)

    def test_mahjong_defaults_validation_rejects_unknown_profile(self):
        with self.assertRaisesRegex(ValueError, "麻将识别档案不存在"):
            self.repo.validate_mahjong_defaults(MahjongRunDefaults(profile_name="missing_profile"))

    def test_cafe_profile_default_seconds_reads_profile_yaml(self):
        self.assertEqual(self.repo.get_cafe_profile_default_seconds("default_1280x720_cn"), 130.0)
        self.assertIsNone(self.repo.get_cafe_profile_default_seconds("missing_profile"))

    def test_cafe_defaults_hidden_pacing_uses_profile_yaml(self):
        defaults = self.repo.get_cafe_defaults(
            {
                "inputs": [
                    {"name": "profile_name", "type": "string", "default": "default_1280x720_cn"},
                    {"name": "min_order_interval_sec", "type": "number", "default": 0.5},
                    {"name": "min_order_duration_sec", "type": "number", "default": 0},
                ]
            }
        )

        self.assertEqual(defaults.min_order_interval_sec, 0.8)
        self.assertEqual(defaults.min_order_duration_sec, 0.2)

    def test_cafe_profile_runtime_defaults_include_pacing_and_fake_customer_flags(self):
        defaults = self.repo.get_cafe_profile_runtime_defaults("default_1280x720_cn")

        self.assertEqual(defaults["max_seconds"], 130.0)
        self.assertEqual(defaults["min_order_interval_sec"], 0.8)
        self.assertEqual(defaults["min_order_duration_sec"], 0.2)
        self.assertTrue(defaults["fake_customer_enabled"])
        self.assertFalse(defaults["fake_customer_hammer_debug_enabled"])
        self.assertTrue(defaults["fake_customer_order_guard_enabled"])
        self.assertEqual(defaults["fake_customer_order_guard_distance_px"], 85.0)
        self.assertEqual(defaults["fake_customer_max_hammers_per_scan"], 3.0)
        self.assertFalse(defaults["order_decision_debug_enabled"])


if __name__ == "__main__":
    unittest.main()

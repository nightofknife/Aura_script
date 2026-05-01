from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from plans.aura_base.src.platform.runtime_config import WINDOWS_CAPTURE_BACKENDS, WINDOWS_INPUT_BACKENDS

from .logic import (
    CafeRunDefaults,
    FishingRunDefaults,
    GuiPreferences,
    MahjongRunDefaults,
    OneCafeRunDefaults,
    RuntimeSettings,
    extract_auto_loop_defaults,
    extract_cafe_loop_defaults,
    extract_mahjong_loop_defaults,
    extract_one_cafe_defaults,
)


DEFAULT_HISTORY_LIMIT = 50
DEFAULT_AUTO_RUNTIME_PROBE = True
DEFAULT_EXPAND_DEVELOPER_TOOLS = False
DEFAULT_TASK_START_DELAY_SEC = 5
DEFAULT_QUICK_STOP_HOTKEY = "F8"
QUICK_STOP_HOTKEY_OPTIONS = ("F6", "F7", "F8", "F9", "F10", "F11", "F12")


class YihuanConfigRepository:
    """Read/write repository for Yihuan runtime settings and GUI preferences."""

    def __init__(self, plan_root: Path | str, settings_store: Any | None = None):
        self.plan_root = Path(plan_root).resolve()
        self.config_path = self.plan_root / "config.yaml"
        self.input_profiles_dir = self.plan_root / "data" / "input_profiles"
        self.fishing_profiles_dir = self.plan_root / "data" / "fishing"
        self.cafe_profiles_dir = self.plan_root / "data" / "cafe"
        self.one_cafe_profiles_dir = self.plan_root / "data" / "one_cafe"
        self.mahjong_profiles_dir = self.plan_root / "data" / "mahjong"
        self._settings_store = settings_store

    def load_config(self) -> dict[str, Any]:
        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("yihuan config.yaml must contain a mapping object.")
        return dict(payload)

    def save_config(self, updated: Mapping[str, Any]) -> None:
        payload = dict(updated)
        self.config_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def list_input_profiles(self) -> list[str]:
        if not self.input_profiles_dir.exists():
            return []
        names = sorted(path.stem for path in self.input_profiles_dir.glob("*.yaml"))
        return names

    def list_fishing_profiles(self) -> list[str]:
        if not self.fishing_profiles_dir.exists():
            return []
        names = sorted(path.stem for path in self.fishing_profiles_dir.glob("*.yaml"))
        return names

    def list_cafe_profiles(self) -> list[str]:
        if not self.cafe_profiles_dir.exists():
            return []
        names = sorted(path.stem for path in self.cafe_profiles_dir.glob("*.yaml"))
        return names

    def list_one_cafe_profiles(self) -> list[str]:
        if not self.one_cafe_profiles_dir.exists():
            return []
        names = sorted(path.name for path in self.one_cafe_profiles_dir.iterdir() if path.is_dir())
        names.extend(path.stem for path in self.one_cafe_profiles_dir.glob("*.yaml"))
        return sorted(set(names))

    def list_mahjong_profiles(self) -> list[str]:
        if not self.mahjong_profiles_dir.exists():
            return []
        names = sorted(path.name for path in self.mahjong_profiles_dir.iterdir() if path.is_dir())
        names.extend(path.stem for path in self.mahjong_profiles_dir.glob("*.yaml"))
        return sorted(set(names))

    def get_runtime_settings(self) -> RuntimeSettings:
        payload = self.load_config()
        runtime = dict(payload.get("runtime") or {})
        target = dict(runtime.get("target") or {})
        capture = dict(runtime.get("capture") or {})
        input_runtime = dict(runtime.get("input") or {})
        top_level_input = dict(payload.get("input") or {})
        return RuntimeSettings(
            title_regex=str(target.get("title_regex") or "").strip(),
            exclude_titles=[str(item).strip() for item in (target.get("exclude_titles") or []) if str(item).strip()],
            allow_borderless=bool(target.get("allow_borderless", True)),
            capture_backend=str(capture.get("backend") or "").strip(),
            input_backend=str(input_runtime.get("backend") or "").strip(),
            input_profile=str(top_level_input.get("profile") or "").strip(),
        )

    def update_runtime_settings(self, runtime_settings: RuntimeSettings) -> None:
        self.validate_runtime_settings(runtime_settings)
        payload = self.load_config()
        runtime = dict(payload.get("runtime") or {})
        target = dict(runtime.get("target") or {})
        capture = dict(runtime.get("capture") or {})
        input_runtime = dict(runtime.get("input") or {})
        top_level_input = dict(payload.get("input") or {})

        target["title_regex"] = runtime_settings.title_regex
        target["exclude_titles"] = list(runtime_settings.exclude_titles)
        target["allow_borderless"] = bool(runtime_settings.allow_borderless)
        capture["backend"] = runtime_settings.capture_backend
        input_runtime["backend"] = runtime_settings.input_backend
        top_level_input["profile"] = runtime_settings.input_profile

        runtime["target"] = target
        runtime["capture"] = capture
        runtime["input"] = input_runtime
        payload["runtime"] = runtime
        payload["input"] = top_level_input
        self.save_config(payload)

    def get_ui_preferences(self) -> GuiPreferences:
        return GuiPreferences(
            history_limit=self._coerce_int(self._settings_value("gui/history_limit", DEFAULT_HISTORY_LIMIT)),
            auto_runtime_probe_on_startup=self._coerce_bool(
                self._settings_value("gui/auto_runtime_probe_on_startup", DEFAULT_AUTO_RUNTIME_PROBE)
            ),
            expand_developer_tools=self._coerce_bool(
                self._settings_value("gui/expand_developer_tools", DEFAULT_EXPAND_DEVELOPER_TOOLS)
            ),
            task_start_delay_sec=self._coerce_int(
                self._settings_value("gui/task_start_delay_sec", DEFAULT_TASK_START_DELAY_SEC)
            ),
            quick_stop_hotkey=str(
                self._settings_value("gui/quick_stop_hotkey", DEFAULT_QUICK_STOP_HOTKEY) or DEFAULT_QUICK_STOP_HOTKEY
            ).strip()
            or DEFAULT_QUICK_STOP_HOTKEY,
        )

    def save_ui_preferences(self, preferences: GuiPreferences) -> None:
        self.validate_ui_preferences(preferences)
        self._settings_set_value("gui/history_limit", int(preferences.history_limit))
        self._settings_set_value(
            "gui/auto_runtime_probe_on_startup",
            bool(preferences.auto_runtime_probe_on_startup),
        )
        self._settings_set_value(
            "gui/expand_developer_tools",
            bool(preferences.expand_developer_tools),
        )
        self._settings_set_value("gui/task_start_delay_sec", int(preferences.task_start_delay_sec))
        self._settings_set_value("gui/quick_stop_hotkey", str(preferences.quick_stop_hotkey).strip().upper())

    def get_fishing_defaults(self, auto_loop_task: Mapping[str, Any] | None = None) -> FishingRunDefaults:
        defaults = extract_auto_loop_defaults(auto_loop_task)
        return FishingRunDefaults(
            profile_name=self._resolve_auto_loop_profile_name(auto_loop_task),
            sell_fish_every_rounds=defaults.sell_fish_every_rounds,
        )

    def update_fishing_defaults(self, defaults: FishingRunDefaults) -> None:
        self.validate_fishing_defaults(defaults)
        payload = self.load_config()
        fishing = dict(payload.get("fishing") or {})
        fishing["profile_name"] = defaults.profile_name
        payload["fishing"] = fishing
        self.save_config(payload)

    def get_cafe_defaults(self, cafe_task: Mapping[str, Any] | None = None) -> CafeRunDefaults:
        defaults = extract_cafe_loop_defaults(cafe_task)
        profile_name = self._resolve_cafe_profile_name(defaults.profile_name)
        profile_defaults = self.get_cafe_profile_runtime_defaults(profile_name)
        return CafeRunDefaults(
            profile_name=profile_name,
            max_seconds=defaults.max_seconds,
            max_orders=defaults.max_orders,
            start_game=defaults.start_game,
            wait_level_started=defaults.wait_level_started,
            full_assist_auto_hammer_mode=defaults.full_assist_auto_hammer_mode,
            min_order_interval_sec=float(
                profile_defaults.get("min_order_interval_sec", defaults.min_order_interval_sec)
            ),
            min_order_duration_sec=float(
                profile_defaults.get("min_order_duration_sec", defaults.min_order_duration_sec)
            ),
        )

    def update_cafe_defaults(self, defaults: CafeRunDefaults) -> None:
        self.validate_cafe_defaults(defaults)
        payload = self.load_config()
        cafe = dict(payload.get("cafe") or {})
        cafe["profile_name"] = defaults.profile_name
        payload["cafe"] = cafe
        self.save_config(payload)

    def get_one_cafe_defaults(self, one_cafe_task: Mapping[str, Any] | None = None) -> OneCafeRunDefaults:
        defaults = extract_one_cafe_defaults(one_cafe_task)
        return OneCafeRunDefaults(
            profile_name=self._resolve_one_cafe_profile_name(defaults.profile_name),
            withdraw_enabled=defaults.withdraw_enabled,
            restock_enabled=defaults.restock_enabled,
            restock_hours=defaults.restock_hours,
        )

    def update_one_cafe_defaults(self, defaults: OneCafeRunDefaults) -> None:
        self.validate_one_cafe_defaults(defaults)
        payload = self.load_config()
        one_cafe = dict(payload.get("one_cafe") or {})
        one_cafe["profile_name"] = defaults.profile_name
        payload["one_cafe"] = one_cafe
        self.save_config(payload)

    def get_mahjong_defaults(self, mahjong_task: Mapping[str, Any] | None = None) -> MahjongRunDefaults:
        defaults = extract_mahjong_loop_defaults(mahjong_task)
        return MahjongRunDefaults(
            profile_name=self._resolve_mahjong_profile_name(defaults.profile_name),
            max_seconds=defaults.max_seconds,
            start_game=defaults.start_game,
            auto_hu=defaults.auto_hu,
            auto_peng=defaults.auto_peng,
            auto_discard=defaults.auto_discard,
        )

    def update_mahjong_defaults(self, defaults: MahjongRunDefaults) -> None:
        self.validate_mahjong_defaults(defaults)
        payload = self.load_config()
        mahjong = dict(payload.get("mahjong") or {})
        mahjong["profile_name"] = defaults.profile_name
        payload["mahjong"] = mahjong
        self.save_config(payload)

    def get_cafe_profile_runtime_defaults(self, profile_name: str) -> dict[str, Any]:
        profile_name = str(profile_name or "").strip()
        if not profile_name:
            return {}

        profile_path = self.cafe_profiles_dir / f"{profile_name}.yaml"
        if not profile_path.is_file():
            return {}

        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            return {}

        result: dict[str, Any] = {}
        for key in (
            "max_seconds",
            "min_order_interval_sec",
            "min_order_duration_sec",
            "fake_customer_order_guard_distance_px",
            "fake_customer_max_hammers_per_scan",
        ):
            try:
                value = float(payload.get(key))
            except (TypeError, ValueError):
                continue
            if value >= 0:
                result[key] = value

        for key in (
            "fake_customer_enabled",
            "fake_customer_hammer_debug_enabled",
            "fake_customer_order_guard_enabled",
            "order_decision_debug_enabled",
        ):
            if key in payload:
                result[key] = self._coerce_bool(payload.get(key))
        return result

    def get_cafe_profile_default_seconds(self, profile_name: str) -> float | None:
        defaults = self.get_cafe_profile_runtime_defaults(profile_name)
        try:
            value = float(defaults.get("max_seconds"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def validate_runtime_settings(self, runtime_settings: RuntimeSettings) -> None:
        if not runtime_settings.title_regex.strip():
            raise ValueError("窗口标题匹配规则不能为空。")

        capture_backend = str(runtime_settings.capture_backend).strip().lower()
        if capture_backend not in WINDOWS_CAPTURE_BACKENDS:
            raise ValueError(f"不支持的截图后端：{runtime_settings.capture_backend}")

        input_backend = str(runtime_settings.input_backend).strip().lower()
        if input_backend not in WINDOWS_INPUT_BACKENDS:
            raise ValueError(f"不支持的输入后端：{runtime_settings.input_backend}")

        input_profile = str(runtime_settings.input_profile).strip()
        if not input_profile:
            raise ValueError("默认输入档案不能为空。")

        available_profiles = self.list_input_profiles()
        if available_profiles and input_profile not in available_profiles:
            raise ValueError(f"默认输入档案不存在：{input_profile}")

    def validate_ui_preferences(self, preferences: GuiPreferences) -> None:
        if int(preferences.history_limit) <= 0:
            raise ValueError("历史记录显示条数必须大于 0。")
        delay_sec = int(preferences.task_start_delay_sec)
        if delay_sec < 0 or delay_sec > 60:
            raise ValueError("任务启动延迟秒数必须在 0 到 60 之间。")
        hotkey = str(preferences.quick_stop_hotkey or "").strip().upper()
        if hotkey not in QUICK_STOP_HOTKEY_OPTIONS:
            raise ValueError("快捷停止键必须是 F6 到 F12。")

    def validate_fishing_defaults(self, defaults: FishingRunDefaults) -> None:
        profile_name = str(defaults.profile_name).strip()
        if not profile_name:
            raise ValueError("钓鱼识别档案不能为空。")
        available_profiles = self.list_fishing_profiles()
        if available_profiles and profile_name not in available_profiles:
            raise ValueError(f"钓鱼识别档案不存在：{profile_name}")

    def validate_cafe_defaults(self, defaults: CafeRunDefaults) -> None:
        profile_name = str(defaults.profile_name).strip()
        if not profile_name:
            raise ValueError("沙威玛识别档案不能为空。")
        available_profiles = self.list_cafe_profiles()
        if available_profiles and profile_name not in available_profiles:
            raise ValueError(f"沙威玛识别档案不存在：{profile_name}")

    def validate_one_cafe_defaults(self, defaults: OneCafeRunDefaults) -> None:
        profile_name = str(defaults.profile_name).strip()
        if not profile_name:
            raise ValueError("一咖舍识别档案不能为空。")
        available_profiles = self.list_one_cafe_profiles()
        if available_profiles and profile_name not in available_profiles:
            raise ValueError(f"一咖舍识别档案不存在：{profile_name}")

    def validate_mahjong_defaults(self, defaults: MahjongRunDefaults) -> None:
        profile_name = str(defaults.profile_name).strip()
        if not profile_name:
            raise ValueError("麻将识别档案不能为空。")
        available_profiles = self.list_mahjong_profiles()
        if available_profiles and profile_name not in available_profiles:
            raise ValueError(f"麻将识别档案不存在：{profile_name}")

    @staticmethod
    def exclude_titles_to_text(exclude_titles: Iterable[Any]) -> str:
        return "\n".join(str(item).strip() for item in exclude_titles if str(item).strip())

    @staticmethod
    def exclude_titles_from_text(text: str) -> list[str]:
        return [line.strip() for line in str(text).splitlines() if line.strip()]

    def _resolve_auto_loop_profile_name(self, auto_loop_task: Mapping[str, Any] | None) -> str:
        payload = self.load_config()
        fishing = dict(payload.get("fishing") or {})
        configured_profile = str(fishing.get("profile_name") or fishing.get("profile") or "").strip()
        if configured_profile:
            return configured_profile

        default_profile = FishingRunDefaults().profile_name
        for field in (auto_loop_task or {}).get("inputs") or []:
            if not isinstance(field, Mapping):
                continue
            if str(field.get("name") or "").strip() != "profile_name":
                continue
            resolved = str(field.get("default") or "").strip()
            if resolved:
                return resolved
        return default_profile

    def _resolve_cafe_profile_name(self, fallback_profile: str) -> str:
        payload = self.load_config()
        cafe = dict(payload.get("cafe") or {})
        configured_profile = str(cafe.get("profile_name") or cafe.get("profile") or "").strip()
        return configured_profile or str(fallback_profile or CafeRunDefaults().profile_name)

    def _resolve_one_cafe_profile_name(self, fallback_profile: str) -> str:
        payload = self.load_config()
        one_cafe = dict(payload.get("one_cafe") or {})
        configured_profile = str(one_cafe.get("profile_name") or one_cafe.get("profile") or "").strip()
        return configured_profile or str(fallback_profile or OneCafeRunDefaults().profile_name)

    def _resolve_mahjong_profile_name(self, fallback_profile: str) -> str:
        payload = self.load_config()
        mahjong = dict(payload.get("mahjong") or {})
        configured_profile = str(mahjong.get("profile_name") or mahjong.get("profile") or "").strip()
        return configured_profile or str(fallback_profile or MahjongRunDefaults().profile_name)

    def _settings_value(self, key: str, default_value: Any) -> Any:
        if self._settings_store is None:
            return default_value
        value = self._settings_store.value(key, default_value)
        return default_value if value is None else value

    def _settings_set_value(self, key: str, value: Any) -> None:
        if self._settings_store is None:
            return
        self._settings_store.setValue(key, value)

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
        return bool(value)

    @staticmethod
    def _coerce_int(value: Any) -> int:
        return int(value)

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from .models import MapCategory


@dataclass(frozen=True)
class MapOverlaySettings:
    enabled_categories: frozenset[str]
    search_text: str = ""
    show_labels: bool = False
    cluster_enabled: bool = True
    icon_size: int = 11
    label_size: int = 11
    viewport_only: bool = True
    auto_refresh_enabled: bool = False
    auto_detect_map_enabled: bool = True
    auto_rematch_on_map_change: bool = True
    map_watch_interval_ms: int = 700
    match_cooldown_ms: int = 1800
    auto_stop_seconds: int = -1
    debug_show_confidence: bool = False
    debug_show_match_bounds: bool = False
    debug_save_screenshot: bool = False

    def with_enabled_categories(self, category_ids: Iterable[str]) -> "MapOverlaySettings":
        return replace(self, enabled_categories=frozenset(str(item) for item in category_ids))


def default_enabled_categories(categories: Iterable[MapCategory]) -> frozenset[str]:
    return frozenset(category.id for category in categories if category.default_visible)


def load_map_overlay_settings(settings_store: Any | None, categories: Iterable[MapCategory]) -> MapOverlaySettings:
    categories = list(categories)
    default_categories = default_enabled_categories(categories)
    stored_categories = _settings_value(settings_store, "map_overlay/enabled_categories", None)
    if stored_categories is None:
        enabled_categories = default_categories
    else:
        enabled_categories = frozenset(_split_categories(stored_categories))

    return MapOverlaySettings(
        enabled_categories=enabled_categories,
        search_text=str(_settings_value(settings_store, "map_overlay/search_text", "") or ""),
        show_labels=_coerce_bool(_settings_value(settings_store, "map_overlay/show_labels", False)),
        cluster_enabled=_coerce_bool(_settings_value(settings_store, "map_overlay/cluster_enabled", True)),
        icon_size=_clamp_int(_settings_value(settings_store, "map_overlay/icon_size", 11), 6, 32, 11),
        label_size=_clamp_int(_settings_value(settings_store, "map_overlay/label_size", 11), 8, 24, 11),
        viewport_only=_coerce_bool(_settings_value(settings_store, "map_overlay/viewport_only", True)),
        auto_refresh_enabled=_coerce_bool(
            _settings_value(settings_store, "map_overlay/auto_refresh_enabled", False)
        ),
        auto_detect_map_enabled=_coerce_bool(
            _settings_value(settings_store, "map_overlay/auto_detect_map_enabled", True)
        ),
        auto_rematch_on_map_change=_coerce_bool(
            _settings_value(settings_store, "map_overlay/auto_rematch_on_map_change", True)
        ),
        map_watch_interval_ms=_clamp_int(
            _settings_value(settings_store, "map_overlay/map_watch_interval_ms", 700),
            300,
            5000,
            700,
        ),
        match_cooldown_ms=_clamp_int(
            _settings_value(settings_store, "map_overlay/match_cooldown_ms", 1800),
            500,
            15000,
            1800,
        ),
        auto_stop_seconds=_clamp_int(_settings_value(settings_store, "map_overlay/auto_stop_seconds", -1), -1, 3600, -1),
        debug_show_confidence=_coerce_bool(
            _settings_value(settings_store, "map_overlay/debug_show_confidence", False)
        ),
        debug_show_match_bounds=_coerce_bool(
            _settings_value(settings_store, "map_overlay/debug_show_match_bounds", False)
        ),
        debug_save_screenshot=_coerce_bool(
            _settings_value(settings_store, "map_overlay/debug_save_screenshot", False)
        ),
    )


def save_map_overlay_settings(settings_store: Any | None, settings: MapOverlaySettings) -> None:
    if settings_store is None:
        return
    settings_store.setValue("map_overlay/enabled_categories", ",".join(sorted(settings.enabled_categories)))
    settings_store.setValue("map_overlay/search_text", settings.search_text)
    settings_store.setValue("map_overlay/show_labels", bool(settings.show_labels))
    settings_store.setValue("map_overlay/cluster_enabled", bool(settings.cluster_enabled))
    settings_store.setValue("map_overlay/icon_size", int(settings.icon_size))
    settings_store.setValue("map_overlay/label_size", int(settings.label_size))
    settings_store.setValue("map_overlay/viewport_only", bool(settings.viewport_only))
    settings_store.setValue("map_overlay/auto_refresh_enabled", bool(settings.auto_refresh_enabled))
    settings_store.setValue("map_overlay/auto_detect_map_enabled", bool(settings.auto_detect_map_enabled))
    settings_store.setValue("map_overlay/auto_rematch_on_map_change", bool(settings.auto_rematch_on_map_change))
    settings_store.setValue("map_overlay/map_watch_interval_ms", int(settings.map_watch_interval_ms))
    settings_store.setValue("map_overlay/match_cooldown_ms", int(settings.match_cooldown_ms))
    settings_store.setValue("map_overlay/auto_stop_seconds", int(settings.auto_stop_seconds))
    settings_store.setValue("map_overlay/debug_show_confidence", bool(settings.debug_show_confidence))
    settings_store.setValue("map_overlay/debug_show_match_bounds", bool(settings.debug_show_match_bounds))
    settings_store.setValue("map_overlay/debug_save_screenshot", bool(settings.debug_save_screenshot))


def _settings_value(settings_store: Any | None, key: str, default_value: Any) -> Any:
    if settings_store is None:
        return default_value
    value = settings_store.value(key, default_value)
    return default_value if value is None else value


def _split_categories(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    return bool(value)


def _clamp_int(value: Any, minimum: int, maximum: int, default_value: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default_value
    return max(minimum, min(maximum, number))

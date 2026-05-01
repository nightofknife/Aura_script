from __future__ import annotations

import io
from pathlib import Path
import unittest

from PIL import Image

from packages.yihuan_gui.map_overlay.marker_projector import MarkerProjector
from packages.yihuan_gui.map_overlay.models import (
    CATEGORY_TRANSLATIONS_ZH,
    MapBounds,
    MapCategory,
    MapMarker,
    MapMatchResult,
    ZeroluckMapData,
    marker_display_name_zh,
)
from packages.yihuan_gui.map_overlay.settings import load_map_overlay_settings, save_map_overlay_settings
from packages.yihuan_gui.map_overlay.window_capture import AuraRuntimeCaptureClient
from packages.yihuan_gui.map_overlay.zeroluck_repository import ZeroluckRepository


class FakeSettingsStore:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default_value=None):
        return self.values.get(key, default_value)

    def setValue(self, key: str, value) -> None:  # noqa: N802
        self.values[key] = value


class TestYihuanMapOverlayData(unittest.TestCase):
    def test_map_overlay_does_not_keep_local_windows_capture_fallbacks(self):
        overlay_root = Path(__file__).resolve().parents[1] / "packages" / "yihuan_gui" / "map_overlay"

        for path in overlay_root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("win32gui", source, path.name)
            self.assertNotIn("ImageGrab", source, path.name)

    def test_map_overlay_window_is_lightweight_display_without_settings_drawer(self):
        overlay_window = (
            Path(__file__).resolve().parents[1]
            / "packages"
            / "yihuan_gui"
            / "map_overlay"
            / "overlay_window.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("settings_changed", overlay_window)
        self.assertNotIn("_drawer", overlay_window)
        self.assertNotIn("QGroupBox", overlay_window)

    def test_bundled_snapshot_contains_all_translated_zeroluck_categories(self):
        data = ZeroluckRepository().load()
        categories_by_id = data.categories_by_id

        self.assertEqual(set(CATEGORY_TRANSLATIONS_ZH), set(categories_by_id))
        for category_id, translated in CATEGORY_TRANSLATIONS_ZH.items():
            self.assertEqual(categories_by_id[category_id].name_zh, translated)
        self.assertEqual(
            {category.id for category in data.categories if category.default_visible},
            {"fast-travel", "oracle-stone", "currencies"},
        )
        self.assertGreaterEqual(data.marker_count, 2400)
        self.assertTrue(data.reference_image_path)
        self.assertEqual(data.reference_full_size, (10496, 11264))

    def test_marker_tile_pixel_conversion_uses_zeroluck_bounds_direction(self):
        bounds = MapBounds(min_x=-5632, max_x=4864, min_y=-5632, max_y=5632)
        marker = MapMarker(
            id="m1",
            category_id="fast-travel",
            title="Marker",
            region_id="area-001",
            map_x=-1830.6,
            map_y=50.4,
        )

        self.assertAlmostEqual(marker.tile_pixel(bounds)[0], 3801.4)
        self.assertAlmostEqual(marker.tile_pixel(bounds)[1], 5581.6)

    def test_marker_display_name_prefers_chinese_aliases(self):
        category = MapCategory(id="oracle-stone", name_en="Oracle Stones", name_zh="谕石")

        self.assertEqual(
            marker_display_name_zh(MapMarker("m1", "oracle-stone", "Oracle Stone", "", 0, 0), category),
            "谕石",
        )
        self.assertEqual(
            marker_display_name_zh(MapMarker("m2", "fast-travel", "ReroRero Phone Booth", "", 0, 0), None),
            "ReroRero 电话亭",
        )
        self.assertEqual(
            marker_display_name_zh(MapMarker("m3", "currencies", 'Gift from "21"', "", 0, 0), None),
            "21的赠礼",
        )

    def test_settings_roundtrip_preserves_filter_and_display_options(self):
        store = FakeSettingsStore()
        categories = [
            MapCategory(id="fast-travel", name_en="Fast Travel", name_zh="传送点", default_visible=True),
            MapCategory(id="locker", name_en="Storage Lockers", name_zh="储物柜", default_visible=False),
        ]
        settings = load_map_overlay_settings(store, categories)
        settings = settings.with_enabled_categories(["locker"])
        settings = type(settings)(
            enabled_categories=settings.enabled_categories,
            search_text="Oracle",
            show_labels=True,
            cluster_enabled=False,
            icon_size=30,
            label_size=14,
            viewport_only=False,
            auto_refresh_enabled=True,
            auto_detect_map_enabled=False,
            auto_rematch_on_map_change=False,
            map_watch_interval_ms=1200,
            match_cooldown_ms=2600,
            auto_stop_seconds=30,
            debug_show_confidence=True,
            debug_show_match_bounds=True,
            debug_save_screenshot=False,
        )

        save_map_overlay_settings(store, settings)
        loaded = load_map_overlay_settings(store, categories)

        self.assertEqual(loaded.enabled_categories, frozenset({"locker"}))
        self.assertEqual(loaded.search_text, "Oracle")
        self.assertTrue(loaded.show_labels)
        self.assertFalse(loaded.cluster_enabled)
        self.assertEqual(loaded.icon_size, 30)
        self.assertEqual(loaded.label_size, 14)
        self.assertFalse(loaded.viewport_only)
        self.assertTrue(loaded.auto_refresh_enabled)
        self.assertFalse(loaded.auto_detect_map_enabled)
        self.assertFalse(loaded.auto_rematch_on_map_change)
        self.assertEqual(loaded.map_watch_interval_ms, 1200)
        self.assertEqual(loaded.match_cooldown_ms, 2600)
        self.assertEqual(loaded.auto_stop_seconds, 30)
        self.assertTrue(loaded.debug_show_confidence)
        self.assertTrue(loaded.debug_show_match_bounds)

    def test_settings_defaults_enable_big_map_auto_detection(self):
        categories = [MapCategory(id="fast-travel", name_en="Fast Travel", name_zh="传送点", default_visible=True)]

        settings = load_map_overlay_settings(None, categories)

        self.assertTrue(settings.auto_detect_map_enabled)
        self.assertTrue(settings.auto_rematch_on_map_change)
        self.assertEqual(settings.map_watch_interval_ms, 700)
        self.assertEqual(settings.match_cooldown_ms, 1800)

    def test_settings_clamp_auto_detection_intervals(self):
        store = FakeSettingsStore()
        store.setValue("map_overlay/map_watch_interval_ms", 10)
        store.setValue("map_overlay/match_cooldown_ms", 99999)
        categories = [MapCategory(id="fast-travel", name_en="Fast Travel", name_zh="传送点", default_visible=True)]

        settings = load_map_overlay_settings(store, categories)

        self.assertEqual(settings.map_watch_interval_ms, 300)
        self.assertEqual(settings.match_cooldown_ms, 15000)


class TestYihuanMapOverlayProjection(unittest.TestCase):
    def test_projector_filters_search_and_clusters_close_points(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy is not installed")

        bounds = MapBounds(min_x=0, max_x=1000, min_y=0, max_y=1000)
        category = MapCategory(id="oracle-stone", name_en="Oracle Stones", name_zh="谕石", default_visible=True)
        data = ZeroluckMapData(
            categories=[category],
            regions=[],
            markers=[
                MapMarker("m1", "oracle-stone", "Oracle A", "", 100, 900),
                MapMarker("m2", "oracle-stone", "Oracle B", "", 112, 892),
                MapMarker("m3", "oracle-stone", "Hidden", "", 500, 500),
            ],
            bounds=bounds,
            tile_size=256,
            cols=1,
            rows=1,
            tiles=[],
        )
        match = MapMatchResult(
            success=True,
            confidence=1.0,
            transform=np.eye(3),
            screen_rect=(0, 0, 300, 300),
        )
        settings = load_map_overlay_settings(None, data.categories)
        settings = type(settings)(
            enabled_categories=frozenset({"oracle-stone"}),
            search_text="Oracle",
            show_labels=False,
            cluster_enabled=True,
            icon_size=22,
            label_size=11,
            viewport_only=True,
            auto_refresh_enabled=False,
            auto_detect_map_enabled=True,
            auto_rematch_on_map_change=True,
            map_watch_interval_ms=700,
            match_cooldown_ms=1800,
            auto_stop_seconds=-1,
            debug_show_confidence=False,
            debug_show_match_bounds=False,
            debug_save_screenshot=False,
        )

        items = MarkerProjector().project(data, match, settings)

        self.assertEqual(len(items), 1)
        self.assertEqual(getattr(items[0], "count", 0), 2)


class TestAuraRuntimeCaptureClient(unittest.TestCase):
    def test_snapshot_uses_runner_runtime_capture_payload(self):
        class FakeRunner:
            def target_snapshot(self, *, game_name: str, backend=None):
                buffer = io.BytesIO()
                Image.new("RGB", (4, 3), (1, 2, 3)).save(buffer, format="PNG")
                return {
                    "ok": True,
                    "game_name": game_name,
                    "backend": backend or "gdi",
                    "target": {
                        "title": "Yihuan",
                        "client_rect_screen": [10, 20, 1280, 720],
                    },
                    "image_png": buffer.getvalue(),
                    "quality_flags": ["test"],
                }

        client = AuraRuntimeCaptureClient(game_name="yihuan", runner=FakeRunner())

        snapshot = client.snapshot()

        self.assertEqual(snapshot.image.size, (4, 3))
        self.assertEqual(snapshot.geometry, (10, 20, 1280, 720))
        self.assertEqual(snapshot.info.title, "Yihuan")
        self.assertEqual(snapshot.backend, "gdi")
        self.assertEqual(snapshot.quality_flags, ("test",))


if __name__ == "__main__":
    unittest.main()

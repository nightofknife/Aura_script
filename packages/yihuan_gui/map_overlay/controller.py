from __future__ import annotations

from enum import Enum
import time
from typing import Any

from PIL import Image
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from .bigmap_locator import BigMapLocator
from .marker_projector import MarkerProjector, ProjectedItem
from .models import MapMatchResult, ZeroluckMapData
from .overlay_window import MapOverlayWindow
from .presence_detector import BigMapPresenceDetector, BigMapPresenceResult
from .settings import MapOverlaySettings, load_map_overlay_settings, save_map_overlay_settings
from .window_capture import AuraRuntimeCaptureClient, RuntimeTargetSnapshot
from .zeroluck_repository import ZeroluckDataError, ZeroluckRepository


class MapOverlayAutoState(str, Enum):
    WAITING_FOR_MAP = "waiting_for_map"
    MAP_CANDIDATE = "map_candidate"
    MATCHING = "matching"
    MAP_CONFIRMED = "map_confirmed"
    MAP_LOST = "map_lost"


class MapOverlayController(QObject):
    def __init__(
        self,
        settings_store: Any,
        *,
        game_name: str = "yihuan",
        capture_client: AuraRuntimeCaptureClient | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings_store = settings_store
        self._capture_client = capture_client or AuraRuntimeCaptureClient(game_name=game_name)
        self._repo = ZeroluckRepository()
        self._locator = BigMapLocator()
        self._presence_detector = BigMapPresenceDetector()
        self._projector = MarkerProjector()
        self._window: MapOverlayWindow | None = None
        self._data: ZeroluckMapData | None = None
        self._settings = MapOverlaySettings(enabled_categories=frozenset())
        self._last_match: MapMatchResult | None = None
        self._last_items: list[ProjectedItem] = []
        self._auto_state = MapOverlayAutoState.WAITING_FOR_MAP
        self._candidate_streak = 0
        self._missing_streak = 0
        self._matching = False
        self._last_full_match_at = 0.0
        self._next_auto_match_at = 0.0
        self._auto_match_failure_count = 0
        self._last_screen_hash: bytes | None = None
        self._last_presence_result: BigMapPresenceResult | None = None

        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setInterval(2000)
        self._auto_refresh_timer.timeout.connect(self.refresh)

        self._map_watch_timer = QTimer(self)
        self._map_watch_timer.timeout.connect(self._on_map_watch_timer)

        self._auto_stop_timer = QTimer(self)
        self._auto_stop_timer.setSingleShot(True)
        self._auto_stop_timer.timeout.connect(self.close)

        self._follow_timer = QTimer(self)
        self._follow_timer.setInterval(500)
        self._follow_timer.timeout.connect(self._sync_game_geometry)

    def show(self) -> None:
        try:
            self._data = self._repo.load()
        except ZeroluckDataError as exc:
            QMessageBox.warning(None, "异环大地图点位悬浮层", str(exc))
            return

        self._settings = load_map_overlay_settings(self._settings_store, self._data.categories)
        if self._window is None:
            self._window = MapOverlayWindow()
            self._window.refresh_requested.connect(self.refresh)
            self._window.settings_changed.connect(self._on_settings_changed)
            self._window.stop_requested.connect(self.close)
            self._window.closed.connect(self._on_window_closed)
        self._window.set_data(self._data, self._settings)
        if self._settings.auto_detect_map_enabled:
            self._auto_state = MapOverlayAutoState.WAITING_FOR_MAP
            self._window.set_status("等待打开异环大地图")
        else:
            self._window.set_status("打开异环大地图后点击刷新")
        self._sync_game_geometry()
        self._window.show_overlay()
        self._follow_timer.start()
        self._update_auto_refresh()
        self._update_map_watch()
        self._update_auto_stop()

    def close(self) -> None:
        self._auto_refresh_timer.stop()
        self._map_watch_timer.stop()
        self._auto_stop_timer.stop()
        self._follow_timer.stop()
        self._capture_client.close()
        if self._window is not None:
            self._window.close_overlay()
            self._window = None

    def refresh(self) -> None:
        if self._window is None or self._data is None:
            self.show()
            if self._window is None or self._data is None:
                return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._matching = True
        try:
            self._auto_match_failure_count = 0
            self._next_auto_match_at = 0.0
            self._window.set_status("正在定位异环窗口...")
            snapshot = self._capture_client.snapshot()
            self._window.set_game_geometry(snapshot.geometry)
            self._window.set_status("正在截图并匹配大地图...")
            self._refresh_from_screenshot(snapshot, source="manual")
        except Exception as exc:  # noqa: BLE001
            if self._window is not None:
                self._window.set_status(f"悬浮层刷新失败：{exc}")
        finally:
            self._matching = False
            QApplication.restoreOverrideCursor()

    def _refresh_from_screenshot(self, snapshot: RuntimeTargetSnapshot, *, source: str) -> None:
        screenshot = snapshot.image
        if self._window is None or self._data is None:
            return
        self._last_full_match_at = time.monotonic()
        if source == "auto":
            self._auto_state = MapOverlayAutoState.MATCHING
            self._window.set_status(self._presence_status("正在匹配 ZeroLuck 地图", self._last_presence_result))
        try:
            reference_path = self._repo.ensure_basemap_gray(self._data)
            match = self._locator.locate(
                screenshot,
                reference_path,
                reference_map_size=self._data.reference_full_size
                or (int(self._data.bounds.width), int(self._data.bounds.height)),
            )
            self._last_match = match
            if not match.success:
                self._last_items = []
                if source == "auto":
                    self._auto_state = MapOverlayAutoState.MAP_CANDIDATE
                    self._register_auto_match_failure()
                    status = self._presence_status(
                        "检测到疑似大地图，但匹配失败，请调整缩放/拖动地图后重试",
                        self._last_presence_result,
                    )
                else:
                    status = match.message or "未检测到大地图"
                self._window.set_projected_items([], self._settings, status=status)
                return

            self._last_items = self._projector.project(self._data, match, self._settings)
            self._auto_state = MapOverlayAutoState.MAP_CONFIRMED
            self._auto_match_failure_count = 0
            self._next_auto_match_at = 0.0
            self._last_screen_hash = _screen_hash(screenshot)
            cache_text = "，离线快照"
            if source == "auto":
                status = f"已自动标注 {len(self._last_items)} 个点位/聚合，置信度 {match.confidence:.0%}{cache_text}"
            else:
                status = f"已显示 {len(self._last_items)} 个点位/聚合，置信度 {match.confidence:.0%}{cache_text}"
            self._window.set_projected_items(
                self._last_items,
                self._settings,
                status=status,
                match_polygon=match.visible_polygon,
            )
        except Exception as exc:  # noqa: BLE001
            if source == "auto":
                self._register_auto_match_failure()
            self._window.set_status(f"悬浮层刷新失败：{exc}")

    def _sync_game_geometry(self) -> None:
        if self._window is None:
            return
        try:
            target = self._capture_client.target()
        except Exception:
            return
        if target.width > 0 and target.height > 0:
            self._window.set_game_geometry(target.geometry)

    def _on_settings_changed(self, settings: MapOverlaySettings) -> None:
        self._settings = settings
        save_map_overlay_settings(self._settings_store, settings)
        self._update_auto_refresh()
        self._update_map_watch()
        self._update_auto_stop()
        if self._window is None or self._data is None or self._last_match is None or not self._last_match.success:
            return
        self._last_items = self._projector.project(self._data, self._last_match, settings)
        self._window.set_projected_items(
            self._last_items,
            settings,
            status=f"已显示 {len(self._last_items)} 个点位/聚合，置信度 {self._last_match.confidence:.0%}，离线快照",
            match_polygon=self._last_match.visible_polygon,
        )

    def _update_auto_refresh(self) -> None:
        if self._settings.auto_refresh_enabled and self._window is not None:
            self._auto_refresh_timer.start()
        else:
            self._auto_refresh_timer.stop()

    def _update_map_watch(self) -> None:
        if self._settings.auto_detect_map_enabled and self._window is not None:
            self._map_watch_timer.setInterval(int(self._settings.map_watch_interval_ms))
            self._map_watch_timer.start()
        else:
            self._map_watch_timer.stop()

    def _update_auto_stop(self) -> None:
        seconds = int(self._settings.auto_stop_seconds)
        if seconds > 0 and self._window is not None:
            self._auto_stop_timer.start(seconds * 1000)
        else:
            self._auto_stop_timer.stop()

    def _on_window_closed(self) -> None:
        self._auto_refresh_timer.stop()
        self._map_watch_timer.stop()
        self._auto_stop_timer.stop()
        self._follow_timer.stop()
        self._window = None

    def _on_map_watch_timer(self) -> None:
        if self._window is None or self._data is None or not self._settings.auto_detect_map_enabled:
            return
        if self._matching:
            return

        try:
            snapshot = self._capture_client.snapshot()
        except Exception:
            self._candidate_streak = 0
            self._missing_streak += 1
            if self._missing_streak >= 2 and self._auto_state != MapOverlayAutoState.WAITING_FOR_MAP:
                self._clear_projected_items_for_map_closed("未找到异环窗口，等待打开大地图")
            return

        self._window.set_game_geometry(snapshot.geometry)
        screenshot = snapshot.image

        presence = self._presence_detector.detect(screenshot)
        self._last_presence_result = presence
        if presence.is_candidate:
            self._candidate_streak += 1
            self._missing_streak = 0
            if self._candidate_streak < 2:
                if self._auto_state not in {MapOverlayAutoState.MATCHING, MapOverlayAutoState.MAP_CONFIRMED}:
                    self._auto_state = MapOverlayAutoState.MAP_CANDIDATE
                    self._window.set_status(self._presence_status("检测到大地图，正在确认", presence))
                return

            if self._auto_state != MapOverlayAutoState.MAP_CONFIRMED:
                self._request_auto_match(snapshot, presence)
                return

            if self._settings.auto_rematch_on_map_change and self._screen_changed(screenshot):
                self._request_auto_match(snapshot, presence)
            return

        self._candidate_streak = 0
        self._missing_streak += 1
        missing_threshold = 5 if self._auto_state == MapOverlayAutoState.MAP_CONFIRMED and self._last_items else 2
        if self._missing_streak >= missing_threshold:
            if self._last_items or self._auto_state in {
                MapOverlayAutoState.MAP_CANDIDATE,
                MapOverlayAutoState.MATCHING,
                MapOverlayAutoState.MAP_CONFIRMED,
            }:
                self._clear_projected_items_for_map_closed("大地图已关闭，等待再次打开")
            elif self._auto_state != MapOverlayAutoState.WAITING_FOR_MAP:
                self._auto_state = MapOverlayAutoState.WAITING_FOR_MAP
                self._window.set_status(self._presence_status("等待打开异环大地图", presence))
        elif self._auto_state == MapOverlayAutoState.MAP_CONFIRMED and self._last_items:
            self._window.set_status(self._presence_status("已标注，等待确认大地图是否关闭", presence))

    def _request_auto_match(
        self,
        snapshot: RuntimeTargetSnapshot,
        presence: BigMapPresenceResult,
    ) -> None:
        now = time.monotonic()
        cooldown_sec = max(0.5, int(self._settings.match_cooldown_ms) / 1000.0)
        if self._matching or now < self._next_auto_match_at or now - self._last_full_match_at < cooldown_sec:
            return
        self._matching = True
        try:
            self._window.set_status(self._presence_status("正在匹配 ZeroLuck 地图", presence))
            self._refresh_from_screenshot(snapshot, source="auto")
        finally:
            self._matching = False

    def _clear_projected_items_for_map_closed(self, status: str) -> None:
        if self._window is None:
            return
        self._last_match = None
        self._last_items = []
        self._last_screen_hash = None
        self._candidate_streak = 0
        self._missing_streak = 0
        self._auto_state = MapOverlayAutoState.MAP_LOST
        self._window.set_projected_items([], self._settings, status=status)

    def _register_auto_match_failure(self) -> None:
        self._auto_match_failure_count += 1
        base_delay = max(0.5, int(self._settings.match_cooldown_ms) / 1000.0)
        if self._auto_match_failure_count >= 5:
            delay = 8.0
        elif self._auto_match_failure_count >= 3:
            delay = 4.0
        else:
            delay = base_delay
        self._next_auto_match_at = time.monotonic() + delay

    def _screen_changed(self, screenshot: Image.Image) -> bool:
        current_hash = _screen_hash(screenshot)
        previous_hash = self._last_screen_hash
        if previous_hash is None:
            self._last_screen_hash = current_hash
            return True
        diff = sum(abs(left - right) for left, right in zip(current_hash, previous_hash)) / max(1, len(current_hash))
        if diff >= 8.0:
            self._last_screen_hash = current_hash
            return True
        return False

    def _presence_status(self, prefix: str, presence: BigMapPresenceResult | None) -> str:
        if not self._settings.debug_show_confidence or presence is None:
            return prefix
        reasons = ", ".join(presence.reasons) if presence.reasons else "未命中"
        return f"{prefix}（地图检测 {presence.score:.0%}：{reasons}）"


def _screen_hash(image: Image.Image) -> bytes:
    try:
        resampling = Image.Resampling.BILINEAR
    except AttributeError:  # pragma: no cover - Pillow < 9 compatibility.
        resampling = Image.BILINEAR
    return image.convert("L").resize((32, 18), resampling).tobytes()

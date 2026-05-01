from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any

from PIL import Image
from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, QTimer, Signal, Slot

from .bigmap_locator import BigMapLocator
from .marker_projector import MarkerProjector, ProjectedItem
from .models import MapMatchResult, ZeroluckMapData
from .overlay_window import MapOverlayWindow
from .presence_detector import BigMapPresenceDetector, BigMapPresenceResult
from .settings import MapOverlaySettings
from .window_capture import AuraRuntimeCaptureClient, RuntimeTargetSnapshot
from .zeroluck_repository import ZeroluckDataError, ZeroluckRepository


class MapOverlayAutoState(str, Enum):
    STOPPED = "stopped"
    WAITING_FOR_MAP = "waiting_for_map"
    MAP_CANDIDATE = "map_candidate"
    MATCHING = "matching"
    MAP_CONFIRMED = "map_confirmed"
    MAP_LOST = "map_lost"
    ERROR = "error"


@dataclass(frozen=True)
class MapOverlayUiState:
    running: bool
    status: str
    auto_state: str = MapOverlayAutoState.STOPPED.value
    target_title: str = ""
    geometry: tuple[int, int, int, int] = (0, 0, 0, 0)
    backend: str = ""
    item_count: int = 0
    marker_count: int = 0
    category_count: int = 0
    confidence: float | None = None
    error: str = ""


class MapOverlayWorker(QObject):
    data_ready = Signal(object, object)
    geometry_changed = Signal(object)
    projection_ready = Signal(object)
    state_changed = Signal(object)
    log_message = Signal(str, str)

    def __init__(
        self,
        *,
        game_name: str,
        capture_client: AuraRuntimeCaptureClient | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._game_name = game_name
        self._capture_client = capture_client or AuraRuntimeCaptureClient(game_name=game_name)
        self._repo = ZeroluckRepository()
        self._locator = BigMapLocator()
        self._presence_detector = BigMapPresenceDetector()
        self._projector = MarkerProjector()
        self._data: ZeroluckMapData | None = None
        self._settings = MapOverlaySettings(enabled_categories=frozenset())
        self._last_match: MapMatchResult | None = None
        self._last_items: list[ProjectedItem] = []
        self._auto_state = MapOverlayAutoState.STOPPED
        self._candidate_streak = 0
        self._missing_streak = 0
        self._busy = False
        self._running = False
        self._last_full_match_at = 0.0
        self._next_auto_match_at = 0.0
        self._auto_match_failure_count = 0
        self._last_screen_hash: bytes | None = None
        self._last_presence_result: BigMapPresenceResult | None = None
        self._target_title = ""
        self._geometry: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._backend = ""
        self._confidence: float | None = None
        self._error = ""

    @Slot(object)
    def start(self, settings: MapOverlaySettings) -> None:
        self._settings = settings
        self._running = True
        self._auto_state = MapOverlayAutoState.WAITING_FOR_MAP
        self._error = ""
        try:
            if self._data is None:
                self._data = self._repo.load()
                self.data_ready.emit(self._data, _data_summary(self._data))
                self.log_message.emit(
                    f"ZeroLuck 离线快照已加载：{len(self._data.categories)} 个分类，{self._data.marker_count} 个点位。",
                    "info",
                )
            self._emit_state("等待打开异环大地图")
            self.sync_target()
        except ZeroluckDataError as exc:
            self._running = False
            self._auto_state = MapOverlayAutoState.ERROR
            self._error = str(exc)
            self.log_message.emit(f"地图数据加载失败：{exc}", "error")
            self._emit_state(f"地图数据加载失败：{exc}")
        except Exception as exc:  # noqa: BLE001
            self._running = False
            self._auto_state = MapOverlayAutoState.ERROR
            self._error = str(exc)
            self.log_message.emit(f"地图悬浮层启动失败：{exc}", "error")
            self._emit_state(f"地图悬浮层启动失败：{exc}")

    @Slot(object)
    def apply_settings(self, settings: MapOverlaySettings) -> None:
        self._settings = settings
        if not self._running:
            self._emit_state("设置已保存，辅助功能尚未启动")
            return
        if self._data is None or self._last_match is None or not self._last_match.success:
            self._emit_state("设置已保存，等待下一次地图匹配")
            return
        self._last_items = self._projector.project(self._data, self._last_match, settings)
        status = self._matched_status(auto=False)
        self._emit_projection(status, self._last_match.visible_polygon)

    @Slot()
    def refresh(self) -> None:
        if not self._running:
            self.log_message.emit("请先启动大地图点位悬浮层。", "warning")
            return
        if self._busy:
            self.log_message.emit("地图匹配仍在进行，本次刷新已跳过。", "warning")
            return
        self._busy = True
        self._auto_match_failure_count = 0
        self._next_auto_match_at = 0.0
        self._auto_state = MapOverlayAutoState.MATCHING
        self._emit_state("正在定位异环窗口...")
        try:
            snapshot = self._snapshot()
            self._emit_state("正在截图并匹配大地图...")
            self._refresh_from_screenshot(snapshot, source="manual")
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._auto_state = MapOverlayAutoState.ERROR
            self.log_message.emit(f"悬浮层刷新失败：{exc}", "error")
            self._emit_state(f"悬浮层刷新失败：{exc}")
        finally:
            self._busy = False

    @Slot()
    def sync_target(self) -> None:
        if not self._running:
            return
        try:
            target = self._capture_client.target()
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._emit_state("等待可用的异环窗口")
            return
        self._target_title = target.title
        self._geometry = target.geometry
        self.geometry_changed.emit(target.geometry)
        self._emit_state(self._status_for_current_auto_state())

    @Slot()
    def check_map(self) -> None:
        if not self._running or self._data is None or not self._settings.auto_detect_map_enabled:
            return
        if self._busy:
            return
        self._busy = True
        try:
            snapshot = self._snapshot()
        except Exception:
            self._candidate_streak = 0
            self._missing_streak += 1
            if self._missing_streak >= 2 and self._auto_state != MapOverlayAutoState.WAITING_FOR_MAP:
                self._clear_projected_items_for_map_closed("未找到异环窗口，等待打开大地图")
            else:
                self._emit_state("等待可用的异环窗口")
            self._busy = False
            return

        try:
            screenshot = snapshot.image
            presence = self._presence_detector.detect(screenshot)
            self._last_presence_result = presence
            if presence.is_candidate:
                self._candidate_streak += 1
                self._missing_streak = 0
                if self._candidate_streak < 2:
                    if self._auto_state not in {MapOverlayAutoState.MATCHING, MapOverlayAutoState.MAP_CONFIRMED}:
                        self._auto_state = MapOverlayAutoState.MAP_CANDIDATE
                        self._emit_state(self._presence_status("检测到大地图，正在确认", presence))
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
                    self._emit_state(self._presence_status("等待打开异环大地图", presence))
            elif self._auto_state == MapOverlayAutoState.MAP_CONFIRMED and self._last_items:
                self._emit_state(self._presence_status("已标注，等待确认大地图是否关闭", presence))
        finally:
            self._busy = False

    @Slot()
    def shutdown(self) -> None:
        self._running = False
        self._busy = False
        self._auto_state = MapOverlayAutoState.STOPPED
        self._capture_client.close()
        self._emit_state("辅助功能已关闭")

    def _snapshot(self) -> RuntimeTargetSnapshot:
        snapshot = self._capture_client.snapshot()
        self._target_title = snapshot.info.title
        self._geometry = snapshot.geometry
        self._backend = snapshot.backend
        self.geometry_changed.emit(snapshot.geometry)
        return snapshot

    def _refresh_from_screenshot(self, snapshot: RuntimeTargetSnapshot, *, source: str) -> None:
        screenshot = snapshot.image
        if self._data is None:
            return
        self._last_full_match_at = time.monotonic()
        if source == "auto":
            self._auto_state = MapOverlayAutoState.MATCHING
            self._emit_state(self._presence_status("正在匹配 ZeroLuck 地图", self._last_presence_result))
        reference_path = self._repo.ensure_basemap_gray(self._data)
        match = self._locator.locate(
            screenshot,
            reference_path,
            reference_map_size=self._data.reference_full_size
            or (int(self._data.bounds.width), int(self._data.bounds.height)),
        )
        self._last_match = match
        self._confidence = float(match.confidence)
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
            self.log_message.emit(status, "warning")
            self._emit_projection(status, [])
            return

        self._last_items = self._projector.project(self._data, match, self._settings)
        self._auto_state = MapOverlayAutoState.MAP_CONFIRMED
        self._auto_match_failure_count = 0
        self._next_auto_match_at = 0.0
        self._last_screen_hash = _screen_hash(screenshot)
        status = self._matched_status(auto=source == "auto")
        self.log_message.emit(status, "info")
        self._emit_projection(status, match.visible_polygon)

    def _request_auto_match(
        self,
        snapshot: RuntimeTargetSnapshot,
        presence: BigMapPresenceResult,
    ) -> None:
        now = time.monotonic()
        cooldown_sec = max(0.5, int(self._settings.match_cooldown_ms) / 1000.0)
        if now < self._next_auto_match_at or now - self._last_full_match_at < cooldown_sec:
            return
        self._auto_state = MapOverlayAutoState.MATCHING
        self._emit_state(self._presence_status("正在匹配 ZeroLuck 地图", presence))
        self._refresh_from_screenshot(snapshot, source="auto")

    def _clear_projected_items_for_map_closed(self, status: str) -> None:
        self._last_match = None
        self._last_items = []
        self._last_screen_hash = None
        self._candidate_streak = 0
        self._missing_streak = 0
        self._auto_state = MapOverlayAutoState.MAP_LOST
        self._confidence = None
        self.log_message.emit(status, "info")
        self._emit_projection(status, [])

    def _emit_projection(self, status: str, match_polygon: list[tuple[float, float]]) -> None:
        self.projection_ready.emit(
            {
                "items": list(self._last_items),
                "settings": self._settings,
                "status": status,
                "match_polygon": list(match_polygon),
            }
        )
        self._emit_state(status)

    def _emit_state(self, status: str) -> None:
        self.state_changed.emit(
            MapOverlayUiState(
                running=self._running,
                status=status,
                auto_state=self._auto_state.value,
                target_title=self._target_title,
                geometry=self._geometry,
                backend=self._backend,
                item_count=len(self._last_items),
                marker_count=int(self._data.marker_count) if self._data is not None else 0,
                category_count=len(self._data.categories) if self._data is not None else 0,
                confidence=self._confidence,
                error=self._error,
            )
        )

    def _matched_status(self, *, auto: bool) -> str:
        cache_text = "，离线快照"
        prefix = "已自动标注" if auto else "已显示"
        confidence = self._confidence or 0.0
        return f"{prefix} {len(self._last_items)} 个点位/聚合，置信度 {confidence:.0%}{cache_text}"

    def _status_for_current_auto_state(self) -> str:
        if self._auto_state == MapOverlayAutoState.MAP_CONFIRMED:
            return self._matched_status(auto=True)
        if self._auto_state == MapOverlayAutoState.MAP_CANDIDATE:
            return "检测到大地图，正在确认"
        if self._auto_state == MapOverlayAutoState.MATCHING:
            return "正在匹配 ZeroLuck 地图"
        if self._auto_state == MapOverlayAutoState.MAP_LOST:
            return "大地图已关闭，等待再次打开"
        if self._auto_state == MapOverlayAutoState.ERROR:
            return self._error or "地图悬浮层错误"
        if self._auto_state == MapOverlayAutoState.STOPPED:
            return "辅助功能已关闭"
        return "等待打开异环大地图"

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


class MapOverlayController(QObject):
    state_changed = Signal(object)
    data_ready = Signal(object, object)
    log_message = Signal(str, str)

    _request_start = Signal(object)
    _request_apply_settings = Signal(object)
    _request_refresh = Signal()
    _request_sync_target = Signal()
    _request_check_map = Signal()

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
        self._game_name = game_name
        self._capture_client = capture_client
        self._window: MapOverlayWindow | None = None
        self._worker_thread: QThread | None = None
        self._worker: MapOverlayWorker | None = None
        self._settings = MapOverlaySettings(enabled_categories=frozenset())
        self._data: ZeroluckMapData | None = None
        self._running = False
        self._closing_window = False

        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setInterval(2000)
        self._auto_refresh_timer.timeout.connect(self.refresh)

        self._map_watch_timer = QTimer(self)
        self._map_watch_timer.timeout.connect(self._request_check_map.emit)

        self._auto_stop_timer = QTimer(self)
        self._auto_stop_timer.setSingleShot(True)
        self._auto_stop_timer.timeout.connect(self.stop)

        self._follow_timer = QTimer(self)
        self._follow_timer.setInterval(500)
        self._follow_timer.timeout.connect(self._request_sync_target.emit)

    @property
    def running(self) -> bool:
        return self._running

    def show(self) -> None:
        self.start(self._settings)

    def start(self, settings: MapOverlaySettings | None = None) -> None:
        if settings is not None:
            self._settings = settings
        self._ensure_window()
        self._ensure_worker()
        if self._data is not None and self._window is not None:
            self._window.set_data(self._data, self._settings)
        self._window.set_status("正在启动大地图点位悬浮层...")
        self._window.show_overlay()
        self._running = True
        self._request_start.emit(self._settings)
        self._update_timers()
        self.log_message.emit("大地图点位悬浮层已启动。", "info")
        self.state_changed.emit(MapOverlayUiState(running=True, status="正在启动大地图点位悬浮层..."))

    def stop(self) -> None:
        self._auto_refresh_timer.stop()
        self._map_watch_timer.stop()
        self._auto_stop_timer.stop()
        self._follow_timer.stop()
        self._running = False
        if self._window is not None:
            self._closing_window = True
            try:
                self._window.close_overlay()
            finally:
                self._closing_window = False
                self._window = None
        self._shutdown_worker()
        self.log_message.emit("大地图点位悬浮层已关闭。", "info")
        self.state_changed.emit(MapOverlayUiState(running=False, status="辅助功能已关闭"))

    def close(self) -> None:
        self.stop()

    def refresh(self) -> None:
        if not self._running:
            self.log_message.emit("请先启动大地图点位悬浮层。", "warning")
            return
        self._request_refresh.emit()

    def apply_settings(self, settings: MapOverlaySettings) -> None:
        self._settings = settings
        if self._window is not None and self._data is not None:
            self._window.set_data(self._data, settings)
        self._update_timers()
        if self._worker is not None:
            self._request_apply_settings.emit(settings)

    def set_markers_visible(self, enabled: bool) -> None:
        if self._window is not None:
            self._window.set_markers_visible(enabled)
        self.log_message.emit("悬浮点位已显示。" if enabled else "悬浮点位已隐藏。", "info")

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        self._window = MapOverlayWindow()
        self._window.refresh_requested.connect(self.refresh)
        self._window.visibility_changed.connect(self.set_markers_visible)
        self._window.stop_requested.connect(self.stop)
        self._window.closed.connect(self._on_window_closed)

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker_thread is not None:
            return
        self._worker_thread = QThread(self)
        self._worker = MapOverlayWorker(game_name=self._game_name, capture_client=self._capture_client)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._request_start.connect(self._worker.start, Qt.QueuedConnection)
        self._request_apply_settings.connect(self._worker.apply_settings, Qt.QueuedConnection)
        self._request_refresh.connect(self._worker.refresh, Qt.QueuedConnection)
        self._request_sync_target.connect(self._worker.sync_target, Qt.QueuedConnection)
        self._request_check_map.connect(self._worker.check_map, Qt.QueuedConnection)
        self._worker.data_ready.connect(self._on_worker_data_ready)
        self._worker.geometry_changed.connect(self._on_geometry_changed)
        self._worker.projection_ready.connect(self._on_projection_ready)
        self._worker.state_changed.connect(self._on_worker_state_changed)
        self._worker.log_message.connect(self.log_message.emit)
        self._worker_thread.start()

    def _shutdown_worker(self) -> None:
        if self._worker is None or self._worker_thread is None:
            self._worker = None
            self._worker_thread = None
            return
        try:
            QMetaObject.invokeMethod(self._worker, "shutdown", Qt.BlockingQueuedConnection)
        except RuntimeError:
            pass
        self._worker_thread.quit()
        self._worker_thread.wait(3000)
        self._worker = None
        self._worker_thread = None

    def _update_timers(self) -> None:
        if not self._running:
            return
        if self._settings.auto_refresh_enabled:
            self._auto_refresh_timer.start()
        else:
            self._auto_refresh_timer.stop()
        if self._settings.auto_detect_map_enabled:
            self._map_watch_timer.setInterval(int(self._settings.map_watch_interval_ms))
            self._map_watch_timer.start()
        else:
            self._map_watch_timer.stop()
        seconds = int(self._settings.auto_stop_seconds)
        if seconds > 0:
            self._auto_stop_timer.start(seconds * 1000)
        else:
            self._auto_stop_timer.stop()
        self._follow_timer.start()

    def _on_worker_data_ready(self, data: ZeroluckMapData, summary: dict[str, Any]) -> None:
        self._data = data
        if self._window is not None:
            self._window.set_data(data, self._settings)
        self.data_ready.emit(data, summary)

    def _on_geometry_changed(self, geometry: tuple[int, int, int, int]) -> None:
        if self._window is not None:
            self._window.set_game_geometry(geometry)

    def _on_projection_ready(self, payload: dict[str, Any]) -> None:
        if self._window is None:
            return
        self._window.set_projected_items(
            list(payload.get("items") or []),
            payload.get("settings") or self._settings,
            status=str(payload.get("status") or ""),
            match_polygon=list(payload.get("match_polygon") or []),
        )

    def _on_worker_state_changed(self, state: MapOverlayUiState) -> None:
        self._running = bool(state.running)
        if not self._running:
            self._auto_refresh_timer.stop()
            self._map_watch_timer.stop()
            self._auto_stop_timer.stop()
            self._follow_timer.stop()
        if self._window is not None:
            self._window.set_status(state.status)
        self.state_changed.emit(state)

    def _on_window_closed(self) -> None:
        if self._closing_window:
            return
        self._window = None
        self._auto_refresh_timer.stop()
        self._map_watch_timer.stop()
        self._auto_stop_timer.stop()
        self._follow_timer.stop()
        self._running = False
        self._shutdown_worker()
        self.state_changed.emit(MapOverlayUiState(running=False, status="辅助功能已关闭"))


def _screen_hash(image: Image.Image) -> bytes:
    try:
        resampling = Image.Resampling.BILINEAR
    except AttributeError:  # pragma: no cover - Pillow < 9 compatibility.
        resampling = Image.BILINEAR
    return image.convert("L").resize((32, 18), resampling).tobytes()


def _data_summary(data: ZeroluckMapData) -> dict[str, Any]:
    return {
        "source": "ZeroLuck 离线快照",
        "fetched_at": data.fetched_at.isoformat() if data.fetched_at else "",
        "category_count": len(data.categories),
        "marker_count": data.marker_count,
    }

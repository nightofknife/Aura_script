from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .marker_projector import ProjectedItem
from .models import (
    CATEGORY_TRANSLATIONS_ZH,
    MapCategory,
    MarkerCluster,
    ProjectedMarker,
    ZeroluckMapData,
    marker_display_name_zh,
)
from .settings import MapOverlaySettings


CATEGORY_COLORS: dict[str, QColor] = {
    "fast-travel": QColor("#26b7ff"),
    "locker": QColor("#9fd86b"),
    "real-estate": QColor("#f4b44d"),
    "city-services": QColor("#4bd0c8"),
    "featured-business": QColor("#ff8b4a"),
    "oracle-stone": QColor("#b38cff"),
    "currencies": QColor("#ffd84d"),
    "collectible": QColor("#7ddf7a"),
    "stealable-loot": QColor("#ff6f91"),
    "quest-start": QColor("#f5f7ff"),
    "monsters": QColor("#ff5252"),
    "activities": QColor("#5ee7df"),
}


class _MapCanvasWindow(QWidget):
    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._items: list[ProjectedItem] = []
        self._settings = MapOverlaySettings(enabled_categories=frozenset())
        self._status = "等待刷新"
        self._match_polygon: list[tuple[float, float]] = []
        self._markers_visible = True
        self._categories_by_id: dict[str, MapCategory] = {}
        self._label_counts: dict[str, int] = {}
        self._placed_label_rects: list[QRect] = []

    @property
    def items(self) -> list[ProjectedItem]:
        return list(self._items)

    @property
    def match_polygon(self) -> list[tuple[float, float]]:
        return list(self._match_polygon)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        _set_click_through(self)

    def set_payload(
        self,
        items: list[ProjectedItem],
        settings: MapOverlaySettings,
        *,
        status: str,
        match_polygon: list[tuple[float, float]] | None = None,
        markers_visible: bool = True,
    ) -> None:
        self._items = list(items)
        self._settings = settings
        self._status = status
        self._match_polygon = list(match_polygon or [])
        self._markers_visible = markers_visible
        self._label_counts = _label_counts(self._items)
        self.update()

    def set_categories(self, categories: list[MapCategory]) -> None:
        self._categories_by_id = {category.id: category for category in categories}
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)

        if self._settings.debug_show_confidence and self._status:
            painter.setPen(QPen(QColor("#f8fbff"), 1))
            painter.setBrush(QColor(15, 18, 24, 185))
            painter.drawRoundedRect(QRect(16, 16, min(520, self.width() - 32), 34), 10, 10)
            painter.drawText(QRect(28, 16, min(500, self.width() - 56), 34), Qt.AlignVCenter, self._status)

        if self._settings.debug_show_match_bounds and len(self._match_polygon) >= 3:
            painter.setPen(QPen(QColor("#00e5ff"), 2, Qt.DashLine))
            points = [QPoint(round(x), round(y)) for x, y in self._match_polygon]
            painter.drawPolygon(points)

        if not self._markers_visible:
            painter.end()
            return

        self._placed_label_rects = []
        font = QFont()
        font.setPointSize(max(8, int(self._settings.label_size)))
        painter.setFont(font)
        for item in self._items:
            if isinstance(item, MarkerCluster):
                self._draw_cluster(painter, item)
            else:
                self._draw_marker(painter, item)
        self._draw_legend(painter)
        painter.end()

    def _draw_marker(self, painter: QPainter, item: ProjectedMarker) -> None:
        color = CATEGORY_COLORS.get(item.marker.category_id, QColor("#ffffff"))
        radius = max(3, int(self._settings.icon_size) // 2)
        x = round(item.screen_x)
        y = round(item.screen_y)
        painter.setPen(QPen(QColor(0, 0, 0, 190), 3))
        painter.setBrush(color)
        painter.drawEllipse(QPoint(x, y), radius, radius)
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawEllipse(QPoint(x, y), max(1, radius - 2), max(1, radius - 2))
        label = marker_display_name_zh(item.marker, item.category)
        if self._should_draw_label(label):
            self._draw_label(painter, x + radius + 7, y + 5, label)

    def _draw_cluster(self, painter: QPainter, item: MarkerCluster) -> None:
        radius = max(7, int(self._settings.icon_size))
        x = round(item.screen_x)
        y = round(item.screen_y)
        painter.setPen(QPen(QColor(0, 0, 0, 210), 3))
        painter.setBrush(QColor(255, 216, 77, 230))
        painter.drawEllipse(QPoint(x, y), radius, radius)
        painter.setPen(QPen(QColor("#111827"), 1))
        painter.drawText(QRect(x - radius, y - radius, radius * 2, radius * 2), Qt.AlignCenter, str(item.count))

    def _should_draw_label(self, label: str) -> bool:
        if self._settings.show_labels:
            return True
        return self._label_counts.get(label, 0) == 1 and label not in set(CATEGORY_TRANSLATIONS_ZH.values())

    def _draw_label(self, painter: QPainter, x: int, y: int, label: str) -> None:
        metrics = QFontMetrics(painter.font())
        width = metrics.horizontalAdvance(label)
        height = metrics.height()
        rect = QRect(x - 4, y - height + 2, width + 8, height + 4)
        if rect.right() > self.width() - 8 or rect.left() < 8 or rect.top() < 8 or rect.bottom() > self.height() - 8:
            return
        if any(rect.intersects(existing) for existing in self._placed_label_rects):
            return
        self._placed_label_rects.append(rect)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(QPen(QColor("#f8fbff"), 1))
        painter.drawText(x, y, label)

    def _draw_legend(self, painter: QPainter) -> None:
        entries = self._legend_entries()
        if not entries:
            return
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        width = max(metrics.horizontalAdvance(label) for _category_id, label, _color in entries) + 46
        row_height = 22
        x = 16
        y = 58 if self._settings.debug_show_confidence else 16
        height = 16 + row_height * len(entries)
        painter.setPen(QPen(QColor(255, 255, 255, 38), 1))
        painter.setBrush(QColor(10, 14, 22, 218))
        painter.drawRoundedRect(QRect(x, y, width, height), 10, 10)
        for index, (_category_id, label, color) in enumerate(entries):
            cy = y + 14 + index * row_height
            painter.setPen(QPen(QColor(0, 0, 0, 190), 2))
            painter.setBrush(color)
            painter.drawEllipse(QPoint(x + 16, cy), 6, 6)
            painter.setPen(QPen(QColor("#f8fbff"), 1))
            painter.drawText(x + 30, cy + 5, label)

    def _legend_entries(self) -> list[tuple[str, str, QColor]]:
        category_ids: set[str] = set()
        for item in self._items:
            if isinstance(item, MarkerCluster):
                category_ids.update(marker.marker.category_id for marker in item.markers)
            else:
                category_ids.add(item.marker.category_id)
        order = {category_id: index for index, category_id in enumerate(CATEGORY_TRANSLATIONS_ZH)}
        result: list[tuple[str, str, QColor]] = []
        for category_id in sorted(category_ids, key=lambda value: order.get(value, 999)):
            category = self._categories_by_id.get(category_id)
            label = category.display_name if category is not None else CATEGORY_TRANSLATIONS_ZH.get(category_id, category_id)
            result.append((category_id, label, CATEGORY_COLORS.get(category_id, QColor("#ffffff"))))
        return result


class MapOverlayWindow(QWidget):
    refresh_requested = Signal()
    visibility_changed = Signal(bool)
    stop_requested = Signal()
    closed = Signal()

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setObjectName("mapOverlayControl")
        self.setStyleSheet(
            """
            QWidget#mapOverlayControl {
                background: rgba(248, 250, 252, 246);
                border: 1px solid rgba(15, 23, 42, 42);
                border-radius: 14px;
                color: #0f172a;
            }
            QLabel {
                color: #0f172a;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid rgba(15, 23, 42, 46);
                border-radius: 9px;
                padding: 6px 10px;
                color: #0f172a;
            }
            QPushButton:hover {
                background: #e0f2fe;
            }
            """
        )
        self._canvas = _MapCanvasWindow()
        self._data: ZeroluckMapData | None = None
        self._settings = MapOverlaySettings(enabled_categories=frozenset())
        self._markers_visible = True
        self._status = "等待刷新"
        self._game_geometry = QRect(80, 80, 1280, 720)
        self._build_ui()

    @property
    def markers_visible(self) -> bool:
        return self._markers_visible

    def set_game_geometry(self, geometry: tuple[int, int, int, int]) -> None:
        x, y, width, height = geometry
        self._game_geometry = QRect(int(x), int(y), max(1, int(width)), max(1, int(height)))
        self._canvas.setGeometry(self._game_geometry)
        self._reposition_control()

    def set_data(self, data: ZeroluckMapData, settings: MapOverlaySettings) -> None:
        self._data = data
        self._settings = settings
        self._canvas.set_categories(data.categories)

    def set_status(self, status: str) -> None:
        self._status = status
        self._status_label.setText(status)

    def set_projected_items(
        self,
        items: list[ProjectedItem],
        settings: MapOverlaySettings,
        *,
        status: str,
        match_polygon: list[tuple[float, float]] | None = None,
    ) -> None:
        self._settings = settings
        self._status = status
        self._status_label.setText(status)
        self._canvas.set_payload(
            items,
            settings,
            status=status,
            match_polygon=match_polygon,
            markers_visible=self._markers_visible,
        )

    def set_markers_visible(self, visible: bool) -> None:
        self._markers_visible = bool(visible)
        self._toggle_visible_button.setText("隐藏" if self._markers_visible else "显示")
        self._canvas.set_payload(
            self._canvas.items,
            self._settings,
            status=self._status,
            match_polygon=self._canvas.match_polygon,
            markers_visible=self._markers_visible,
        )

    def show_overlay(self) -> None:
        self._canvas.show()
        self.show()
        self.raise_()
        self._reposition_control()

    def close_overlay(self) -> None:
        self._canvas.close()
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._canvas.close()
        self.closed.emit()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(10, 10, 10, 10)
        self._root_layout.setSpacing(8)

        control_row = QHBoxLayout()
        self._status_label = QLabel(self._status, self)
        self._status_label.setMinimumWidth(180)
        self._refresh_button = QPushButton("刷新", self)
        self._toggle_visible_button = QPushButton("隐藏", self)
        self._stop_button = QPushButton("关闭", self)
        self._refresh_button.clicked.connect(self.refresh_requested.emit)
        self._toggle_visible_button.clicked.connect(self._toggle_markers_visible)
        self._stop_button.clicked.connect(self.stop_requested.emit)
        control_row.addWidget(self._status_label, 1)
        control_row.addWidget(self._refresh_button)
        control_row.addWidget(self._toggle_visible_button)
        control_row.addWidget(self._stop_button)
        self._root_layout.addLayout(control_row)
        self._reposition_control()

    def _toggle_markers_visible(self) -> None:
        self.set_markers_visible(not self._markers_visible)
        self.visibility_changed.emit(self._markers_visible)

    def _reposition_control(self) -> None:
        width = min(480, max(360, self._game_geometry.width() - 32))
        height = 56
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        self.setFixedSize(width, height)
        x = self._game_geometry.right() - width - 16
        y = self._game_geometry.top() + 14
        self.move(max(self._game_geometry.left() + 8, x), y)


def _label_counts(items: list[ProjectedItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if isinstance(item, MarkerCluster):
            for projected in item.markers:
                label = marker_display_name_zh(projected.marker, projected.category)
                counts[label] = counts.get(label, 0) + 1
        else:
            label = marker_display_name_zh(item.marker, item.category)
            counts[label] = counts.get(label, 0) + 1
    return counts


def _set_click_through(widget: QWidget) -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        hwnd = int(widget.winId())
        user32 = ctypes.windll.user32
        ex_style = user32.GetWindowLongW(hwnd, -20)
        user32.SetWindowLongW(hwnd, -20, ex_style | 0x00000020 | 0x00080000)
    except Exception:
        return

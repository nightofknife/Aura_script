from __future__ import annotations

from collections.abc import Callable
import ctypes
import sys

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

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
        # Keep the legend above every marker/label so it remains readable in dense areas.
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
    settings_changed = Signal(object)
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
            QGroupBox {
                border: 1px solid rgba(15, 23, 42, 42);
                border-radius: 10px;
                margin-top: 12px;
                padding: 9px;
                font-weight: 700;
                color: #0f172a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #0f172a;
                background: #f8fafc;
            }
            QLabel, QCheckBox {
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
            QLineEdit, QSpinBox {
                background: #ffffff;
                border: 1px solid rgba(15, 23, 42, 46);
                border-radius: 8px;
                padding: 5px;
                color: #0f172a;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            """
        )
        self._canvas = _MapCanvasWindow()
        self._data: ZeroluckMapData | None = None
        self._settings = MapOverlaySettings(enabled_categories=frozenset())
        self._category_checks: dict[str, QCheckBox] = {}
        self._expanded = False
        self._markers_visible = True
        self._status = "等待刷新"
        self._game_geometry = QRect(80, 80, 1280, 720)
        self._build_ui()

    def set_game_geometry(self, geometry: tuple[int, int, int, int]) -> None:
        x, y, width, height = geometry
        self._game_geometry = QRect(int(x), int(y), max(1, int(width)), max(1, int(height)))
        self._canvas.setGeometry(self._game_geometry)
        self._reposition_control()

    def set_data(self, data: ZeroluckMapData, settings: MapOverlaySettings) -> None:
        self._data = data
        self._settings = settings
        self._canvas.set_categories(data.categories)
        self._rebuild_category_checks()
        self._sync_settings_widgets()

    def set_status(self, status: str) -> None:
        self._status = status
        self._status_label.setText(status)
        self._canvas.set_payload([], self._settings, status=status, markers_visible=False)

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
        self._sync_settings_widgets()

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
        self._settings_button = QPushButton("设置", self)
        self._refresh_button.clicked.connect(self.refresh_requested.emit)
        self._toggle_visible_button.clicked.connect(self._toggle_markers_visible)
        self._settings_button.clicked.connect(self._toggle_expanded)
        control_row.addWidget(self._status_label, 1)
        control_row.addWidget(self._refresh_button)
        control_row.addWidget(self._toggle_visible_button)
        control_row.addWidget(self._settings_button)
        self._root_layout.addLayout(control_row)

        self._drawer_scroll = QScrollArea(self)
        self._drawer_scroll.setWidgetResizable(True)
        self._drawer_scroll.setFrameShape(QFrame.NoFrame)
        self._drawer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._drawer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._drawer = QFrame(self._drawer_scroll)
        drawer_layout = QVBoxLayout(self._drawer)
        drawer_layout.setContentsMargins(0, 0, 0, 0)
        drawer_layout.setSpacing(8)
        self._drawer_scroll.setWidget(self._drawer)

        data_group = QGroupBox("数据源", self._drawer)
        data_layout = QVBoxLayout(data_group)
        self._data_label = QLabel("ZeroLuck 离线快照", data_group)
        self._data_label.setWordWrap(True)
        self._stop_button = QPushButton("停止任务", data_group)
        self._stop_button.setToolTip("关闭当前大地图点位悬浮层。")
        self._stop_button.clicked.connect(self.stop_requested.emit)
        data_layout.addWidget(self._data_label)
        data_layout.addWidget(self._stop_button)
        drawer_layout.addWidget(data_group)

        filter_group = QGroupBox("ZeroLuck 分类筛选", self._drawer)
        filter_layout = QVBoxLayout(filter_group)
        quick_row = QHBoxLayout()
        all_button = QPushButton("全选", filter_group)
        none_button = QPushButton("全不选", filter_group)
        default_button = QPushButton("恢复默认", filter_group)
        all_button.clicked.connect(lambda: self._set_all_categories(True))
        none_button.clicked.connect(lambda: self._set_all_categories(False))
        default_button.clicked.connect(self._restore_default_categories)
        quick_row.addWidget(all_button)
        quick_row.addWidget(none_button)
        quick_row.addWidget(default_button)
        filter_layout.addLayout(quick_row)
        self._search_edit = QLineEdit(filter_group)
        self._search_edit.setPlaceholderText("搜索点位标题、分类或 ID")
        self._search_edit.textChanged.connect(self._on_settings_widget_changed)
        filter_layout.addWidget(self._search_edit)
        self._category_scroll = QScrollArea(filter_group)
        self._category_scroll.setWidgetResizable(True)
        self._category_scroll.setMinimumHeight(120)
        self._category_scroll.setMaximumHeight(180)
        self._category_panel = QWidget(self._category_scroll)
        self._category_layout = QVBoxLayout(self._category_panel)
        self._category_layout.setContentsMargins(0, 0, 0, 0)
        self._category_scroll.setWidget(self._category_panel)
        filter_layout.addWidget(self._category_scroll, 1)
        drawer_layout.addWidget(filter_group, 1)

        display_group = QGroupBox("显示设置", self._drawer)
        grid = QGridLayout(display_group)
        self._show_labels_check = QCheckBox("显示标签", display_group)
        self._cluster_check = QCheckBox("启用聚合", display_group)
        self._viewport_only_check = QCheckBox("只显示当前视野点位", display_group)
        self._auto_refresh_check = QCheckBox("定时完整刷新（实验）", display_group)
        self._debug_confidence_check = QCheckBox("显示匹配置信度", display_group)
        self._debug_bounds_check = QCheckBox("显示匹配边界框", display_group)
        self._icon_size_spin = QSpinBox(display_group)
        self._icon_size_spin.setRange(6, 32)
        self._label_size_spin = QSpinBox(display_group)
        self._label_size_spin.setRange(8, 24)
        self._auto_stop_spin = QSpinBox(display_group)
        self._auto_stop_spin.setRange(-1, 3600)
        self._auto_stop_spin.setSpecialValueText("-1 不自动停止")
        checks: list[QAbstractButton] = [
            self._show_labels_check,
            self._cluster_check,
            self._viewport_only_check,
            self._auto_refresh_check,
            self._debug_confidence_check,
            self._debug_bounds_check,
        ]
        for button in checks:
            button.toggled.connect(self._on_settings_widget_changed)
        self._icon_size_spin.valueChanged.connect(self._on_settings_widget_changed)
        self._label_size_spin.valueChanged.connect(self._on_settings_widget_changed)
        self._auto_stop_spin.valueChanged.connect(self._on_settings_widget_changed)
        grid.addWidget(self._show_labels_check, 0, 0)
        grid.addWidget(self._cluster_check, 0, 1)
        grid.addWidget(self._viewport_only_check, 1, 0, 1, 2)
        grid.addWidget(QLabel("图标大小", display_group), 2, 0)
        grid.addWidget(self._icon_size_spin, 2, 1)
        grid.addWidget(QLabel("标签字号", display_group), 3, 0)
        grid.addWidget(self._label_size_spin, 3, 1)
        grid.addWidget(self._auto_refresh_check, 4, 0, 1, 2)
        grid.addWidget(QLabel("自动停止秒数", display_group), 5, 0)
        grid.addWidget(self._auto_stop_spin, 5, 1)
        grid.addWidget(self._debug_confidence_check, 6, 0, 1, 2)
        grid.addWidget(self._debug_bounds_check, 7, 0, 1, 2)
        drawer_layout.addWidget(display_group)

        auto_group = QGroupBox("自动标注", self._drawer)
        auto_grid = QGridLayout(auto_group)
        self._auto_detect_check = QCheckBox("打开大地图后自动标注", auto_group)
        self._auto_rematch_check = QCheckBox("地图变化后自动重新匹配", auto_group)
        self._map_watch_interval_spin = QSpinBox(auto_group)
        self._map_watch_interval_spin.setRange(300, 5000)
        self._map_watch_interval_spin.setSingleStep(100)
        self._map_watch_interval_spin.setSuffix(" ms")
        self._match_cooldown_spin = QSpinBox(auto_group)
        self._match_cooldown_spin.setRange(500, 15000)
        self._match_cooldown_spin.setSingleStep(100)
        self._match_cooldown_spin.setSuffix(" ms")
        self._auto_detect_check.toggled.connect(self._on_settings_widget_changed)
        self._auto_rematch_check.toggled.connect(self._on_settings_widget_changed)
        self._map_watch_interval_spin.valueChanged.connect(self._on_settings_widget_changed)
        self._match_cooldown_spin.valueChanged.connect(self._on_settings_widget_changed)
        auto_grid.addWidget(self._auto_detect_check, 0, 0, 1, 2)
        auto_grid.addWidget(self._auto_rematch_check, 1, 0, 1, 2)
        auto_grid.addWidget(QLabel("监听间隔", auto_group), 2, 0)
        auto_grid.addWidget(self._map_watch_interval_spin, 2, 1)
        auto_grid.addWidget(QLabel("匹配冷却", auto_group), 3, 0)
        auto_grid.addWidget(self._match_cooldown_spin, 3, 1)
        drawer_layout.addWidget(auto_group)

        self._root_layout.addWidget(self._drawer_scroll, 1)
        self._drawer_scroll.setVisible(False)
        self._drawer_scroll.setMaximumHeight(0)
        self._reposition_control()

    def _rebuild_category_checks(self) -> None:
        _clear_layout(self._category_layout)
        self._category_checks.clear()
        if self._data is None:
            return
        for category in self._data.categories:
            checkbox = QCheckBox(
                f"{category.display_name}  ({category.total or _count_category(self._data, category.id)})",
                self._category_panel,
            )
            checkbox.setToolTip(f"{category.name_en}\n{category.id}")
            checkbox.setChecked(category.id in self._settings.enabled_categories)
            checkbox.toggled.connect(self._on_category_toggled)
            self._category_checks[category.id] = checkbox
            self._category_layout.addWidget(checkbox)
        self._category_layout.addStretch(1)
        snapshot_text = "ZeroLuck 离线快照"
        if self._data.fetched_at:
            snapshot_text += f"\n生成时间：{self._data.fetched_at.isoformat()}"
        snapshot_text += f"\n分类：{len(self._data.categories)}，点位：{self._data.marker_count}"
        self._data_label.setText(snapshot_text)

    def _sync_settings_widgets(self) -> None:
        blockers: list[Callable[[], None]] = []
        for widget in (
            self._search_edit,
            self._show_labels_check,
            self._cluster_check,
            self._viewport_only_check,
            self._auto_refresh_check,
            self._auto_detect_check,
            self._auto_rematch_check,
            self._debug_confidence_check,
            self._debug_bounds_check,
            self._icon_size_spin,
            self._label_size_spin,
            self._map_watch_interval_spin,
            self._match_cooldown_spin,
            self._auto_stop_spin,
        ):
            widget.blockSignals(True)
            blockers.append(lambda widget=widget: widget.blockSignals(False))
        self._search_edit.setText(self._settings.search_text)
        self._show_labels_check.setChecked(self._settings.show_labels)
        self._cluster_check.setChecked(self._settings.cluster_enabled)
        self._viewport_only_check.setChecked(self._settings.viewport_only)
        self._auto_refresh_check.setChecked(self._settings.auto_refresh_enabled)
        self._auto_detect_check.setChecked(self._settings.auto_detect_map_enabled)
        self._auto_rematch_check.setChecked(self._settings.auto_rematch_on_map_change)
        self._debug_confidence_check.setChecked(self._settings.debug_show_confidence)
        self._debug_bounds_check.setChecked(self._settings.debug_show_match_bounds)
        self._icon_size_spin.setValue(self._settings.icon_size)
        self._label_size_spin.setValue(self._settings.label_size)
        self._map_watch_interval_spin.setValue(self._settings.map_watch_interval_ms)
        self._match_cooldown_spin.setValue(self._settings.match_cooldown_ms)
        self._auto_stop_spin.setValue(self._settings.auto_stop_seconds)
        for category_id, checkbox in self._category_checks.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(category_id in self._settings.enabled_categories)
            checkbox.blockSignals(False)
        for unblock in blockers:
            unblock()

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._drawer_scroll.setVisible(self._expanded)
        self._drawer_scroll.setMaximumHeight(16777215 if self._expanded else 0)
        self._settings_button.setText("设置")
        self._reposition_control()

    def _toggle_markers_visible(self) -> None:
        self._markers_visible = not self._markers_visible
        self._toggle_visible_button.setText("隐藏" if self._markers_visible else "显示")
        self._canvas.set_payload(
            self._canvas._items,  # noqa: SLF001 - same UI component, avoids extra state duplication.
            self._settings,
            status=self._status,
            match_polygon=self._canvas._match_polygon,  # noqa: SLF001
            markers_visible=self._markers_visible,
        )

    def _reposition_control(self) -> None:
        width = min(480, max(360, self._game_geometry.width() - 32)) if self._expanded else 360
        height = 720 if self._expanded else 56
        height = min(height, max(56, self._game_geometry.height() - 28))
        if self._expanded:
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)
            self.setFixedSize(width, height)
            self._drawer_scroll.setVisible(True)
            self._drawer_scroll.setMaximumHeight(max(1, height - 74))
        else:
            self._drawer_scroll.setVisible(False)
            self._drawer_scroll.setMaximumHeight(0)
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)
            self.setFixedSize(width, height)
        x = self._game_geometry.right() - width - 16
        y = self._game_geometry.top() + 14
        self.move(max(self._game_geometry.left() + 8, x), y)

    def _current_settings_from_widgets(self) -> MapOverlaySettings:
        enabled = frozenset(category_id for category_id, checkbox in self._category_checks.items() if checkbox.isChecked())
        return MapOverlaySettings(
            enabled_categories=enabled,
            search_text=self._search_edit.text(),
            show_labels=self._show_labels_check.isChecked(),
            cluster_enabled=self._cluster_check.isChecked(),
            icon_size=self._icon_size_spin.value(),
            label_size=self._label_size_spin.value(),
            viewport_only=self._viewport_only_check.isChecked(),
            auto_refresh_enabled=self._auto_refresh_check.isChecked(),
            auto_detect_map_enabled=self._auto_detect_check.isChecked(),
            auto_rematch_on_map_change=self._auto_rematch_check.isChecked(),
            map_watch_interval_ms=self._map_watch_interval_spin.value(),
            match_cooldown_ms=self._match_cooldown_spin.value(),
            auto_stop_seconds=self._auto_stop_spin.value(),
            debug_show_confidence=self._debug_confidence_check.isChecked(),
            debug_show_match_bounds=self._debug_bounds_check.isChecked(),
            debug_save_screenshot=False,
        )

    def _on_category_toggled(self, *_args: object) -> None:
        self._emit_settings_changed()

    def _on_settings_widget_changed(self, *_args: object) -> None:
        self._emit_settings_changed()

    def _emit_settings_changed(self) -> None:
        self._settings = self._current_settings_from_widgets()
        self.settings_changed.emit(self._settings)

    def _set_all_categories(self, checked: bool) -> None:
        for checkbox in self._category_checks.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
        self._emit_settings_changed()

    def _restore_default_categories(self) -> None:
        if self._data is None:
            return
        defaults = {category.id for category in self._data.categories if category.default_visible}
        for category_id, checkbox in self._category_checks.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(category_id in defaults)
            checkbox.blockSignals(False)
        self._emit_settings_changed()


def _clear_layout(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def _count_category(data: ZeroluckMapData, category_id: str) -> int:
    return sum(1 for marker in data.markers if marker.category_id == category_id)


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

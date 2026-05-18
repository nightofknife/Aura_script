from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..map_overlay.models import ZeroluckMapData
from ..map_overlay.settings import MapOverlaySettings, load_map_overlay_settings, save_map_overlay_settings
from ..map_overlay.zeroluck_repository import ZeroluckDataError, ZeroluckRepository
from ..logic import GAME_NAME
from ..task_specs import AUXILIARY_TOOLS


class AuxiliaryPageMixin:
    def _build_auxiliary_page(self) -> None:
        layout = QVBoxLayout(self._auxiliary_page)
        splitter = QSplitter(Qt.Horizontal, self._auxiliary_page)
        layout.addWidget(splitter, 1)

        splitter.addWidget(self._build_auxiliary_tool_column(splitter))
        splitter.addWidget(self._build_auxiliary_settings_column(splitter))
        splitter.addWidget(self._build_auxiliary_log_column(splitter))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([260, 560, 520])
        if self._auxiliary_tool_list.count():
            self._auxiliary_tool_list.setCurrentRow(0)

    def _build_auxiliary_tool_column(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)

        title = QLabel("辅助功能", panel)
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)

        self._auxiliary_tool_list = QListWidget(panel)
        for tool_id, spec in AUXILIARY_TOOLS.items():
            item = QListWidgetItem(spec.label)
            item.setData(Qt.UserRole, tool_id)
            self._auxiliary_tool_list.addItem(item)
        self._auxiliary_tool_list.currentItemChanged.connect(self._on_auxiliary_tool_changed)
        layout.addWidget(self._auxiliary_tool_list, 1)

        self._selected_auxiliary_status_label = QLabel("空闲", panel)
        self._selected_auxiliary_status_label.setWordWrap(True)
        layout.addWidget(self._selected_auxiliary_status_label)
        layout.addStretch(1)

        self._start_auxiliary_button = QPushButton("启动辅助", panel)
        self._start_auxiliary_button.clicked.connect(self._start_selected_auxiliary_tool)
        self._stop_auxiliary_button = QPushButton("关闭辅助", panel)
        self._stop_auxiliary_button.clicked.connect(self._stop_selected_auxiliary_tool)
        layout.addWidget(self._start_auxiliary_button)
        layout.addWidget(self._stop_auxiliary_button)
        return panel

    def _build_auxiliary_settings_column(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        self._auxiliary_title_label = QLabel("大地图点位悬浮层", panel)
        self._auxiliary_title_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        self._auxiliary_description_label = QLabel(AUXILIARY_TOOLS["map_overlay"].description, panel)
        self._auxiliary_description_label.setWordWrap(True)
        layout.addWidget(self._auxiliary_title_label)
        layout.addWidget(self._auxiliary_description_label)

        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self._build_map_overlay_settings_panel(body, body_layout)
        return panel

    def _build_map_overlay_settings_panel(self, body: QWidget, layout: QVBoxLayout) -> None:
        self._load_map_overlay_data_for_ui()

        data_group = QGroupBox("数据源", body)
        data_layout = QVBoxLayout(data_group)
        self._map_data_label = QLabel(self._map_overlay_data_summary_text(), data_group)
        self._map_data_label.setWordWrap(True)
        data_layout.addWidget(self._map_data_label)
        layout.addWidget(data_group)

        filter_group = QGroupBox("ZeroLuck 分类筛选", body)
        filter_layout = QVBoxLayout(filter_group)
        quick_row = QHBoxLayout()
        all_button = QPushButton("全选", filter_group)
        none_button = QPushButton("全不选", filter_group)
        default_button = QPushButton("恢复默认", filter_group)
        all_button.clicked.connect(lambda: self._set_map_categories_checked(True))
        none_button.clicked.connect(lambda: self._set_map_categories_checked(False))
        default_button.clicked.connect(self._restore_default_map_categories)
        quick_row.addWidget(all_button)
        quick_row.addWidget(none_button)
        quick_row.addWidget(default_button)
        filter_layout.addLayout(quick_row)

        self._map_search_edit = QLineEdit(filter_group)
        self._map_search_edit.setPlaceholderText("搜索点位标题、分类或 ID")
        self._map_search_edit.textChanged.connect(self._on_map_overlay_settings_widget_changed)
        filter_layout.addWidget(self._map_search_edit)

        self._map_category_scroll = QScrollArea(filter_group)
        self._map_category_scroll.setWidgetResizable(True)
        self._map_category_scroll.setMinimumHeight(180)
        self._map_category_panel = QWidget(self._map_category_scroll)
        self._map_category_layout = QVBoxLayout(self._map_category_panel)
        self._map_category_layout.setContentsMargins(0, 0, 0, 0)
        self._map_category_scroll.setWidget(self._map_category_panel)
        filter_layout.addWidget(self._map_category_scroll, 1)
        layout.addWidget(filter_group, 1)
        self._rebuild_map_category_checks()

        display_group = QGroupBox("显示设置", body)
        display_grid = QGridLayout(display_group)
        self._map_show_labels_check = QCheckBox("显示标签", display_group)
        self._map_cluster_check = QCheckBox("启用聚合", display_group)
        self._map_viewport_only_check = QCheckBox("只显示当前视野点位", display_group)
        self._map_auto_refresh_check = QCheckBox("定时完整刷新（实验）", display_group)
        self._map_debug_confidence_check = QCheckBox("显示匹配置信度", display_group)
        self._map_debug_bounds_check = QCheckBox("显示匹配边界框", display_group)
        self._map_icon_size_spin = QSpinBox(display_group)
        self._map_icon_size_spin.setRange(6, 32)
        self._map_label_size_spin = QSpinBox(display_group)
        self._map_label_size_spin.setRange(8, 24)
        self._map_auto_stop_spin = QSpinBox(display_group)
        self._map_auto_stop_spin.setRange(-1, 3600)
        self._map_auto_stop_spin.setSpecialValueText("-1 不自动停止")
        display_grid.addWidget(self._map_show_labels_check, 0, 0)
        display_grid.addWidget(self._map_cluster_check, 0, 1)
        display_grid.addWidget(self._map_viewport_only_check, 1, 0, 1, 2)
        display_grid.addWidget(QLabel("图标大小", display_group), 2, 0)
        display_grid.addWidget(self._map_icon_size_spin, 2, 1)
        display_grid.addWidget(QLabel("标签字号", display_group), 3, 0)
        display_grid.addWidget(self._map_label_size_spin, 3, 1)
        display_grid.addWidget(self._map_auto_refresh_check, 4, 0, 1, 2)
        display_grid.addWidget(QLabel("自动停止秒数", display_group), 5, 0)
        display_grid.addWidget(self._map_auto_stop_spin, 5, 1)
        display_grid.addWidget(self._map_debug_confidence_check, 6, 0, 1, 2)
        display_grid.addWidget(self._map_debug_bounds_check, 7, 0, 1, 2)
        layout.addWidget(display_group)

        auto_group = QGroupBox("自动标注", body)
        auto_grid = QGridLayout(auto_group)
        self._map_auto_detect_check = QCheckBox("打开大地图后自动标注", auto_group)
        self._map_auto_rematch_check = QCheckBox("地图变化后自动重新匹配", auto_group)
        self._map_watch_interval_spin = QSpinBox(auto_group)
        self._map_watch_interval_spin.setRange(300, 5000)
        self._map_watch_interval_spin.setSingleStep(100)
        self._map_watch_interval_spin.setSuffix(" ms")
        self._map_match_cooldown_spin = QSpinBox(auto_group)
        self._map_match_cooldown_spin.setRange(500, 15000)
        self._map_match_cooldown_spin.setSingleStep(100)
        self._map_match_cooldown_spin.setSuffix(" ms")
        auto_grid.addWidget(self._map_auto_detect_check, 0, 0, 1, 2)
        auto_grid.addWidget(self._map_auto_rematch_check, 1, 0, 1, 2)
        auto_grid.addWidget(QLabel("监听间隔", auto_group), 2, 0)
        auto_grid.addWidget(self._map_watch_interval_spin, 2, 1)
        auto_grid.addWidget(QLabel("匹配冷却", auto_group), 3, 0)
        auto_grid.addWidget(self._map_match_cooldown_spin, 3, 1)
        layout.addWidget(auto_group)

        for widget in (
            self._map_show_labels_check,
            self._map_cluster_check,
            self._map_viewport_only_check,
            self._map_auto_refresh_check,
            self._map_debug_confidence_check,
            self._map_debug_bounds_check,
            self._map_auto_detect_check,
            self._map_auto_rematch_check,
            self._map_icon_size_spin,
            self._map_label_size_spin,
            self._map_auto_stop_spin,
            self._map_watch_interval_spin,
            self._map_match_cooldown_spin,
        ):
            if isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_map_overlay_settings_widget_changed)
            else:
                widget.valueChanged.connect(self._on_map_overlay_settings_widget_changed)

        hint = QLabel("这些设置只影响大地图点位悬浮层；辅助功能不会进入任务历史，也不会占用任务停止快捷键。", body)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        self._sync_map_overlay_settings_widgets()

    def _build_auxiliary_log_column(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        header_row = QHBoxLayout()
        title = QLabel("辅助状态与日志", panel)
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        self._map_refresh_button = QPushButton("手动刷新", panel)
        self._map_refresh_button.clicked.connect(self._refresh_map_overlay)
        self._map_visibility_button = QPushButton("隐藏点位", panel)
        self._map_visibility_button.clicked.connect(self._toggle_map_overlay_visibility)
        header_row.addWidget(title)
        header_row.addStretch(1)
        header_row.addWidget(self._map_refresh_button)
        header_row.addWidget(self._map_visibility_button)
        layout.addLayout(header_row)

        status_group = QGroupBox("大地图点位悬浮层", panel)
        status_form = QFormLayout(status_group)
        self._map_overlay_status_label = QLabel("未启动", status_group)
        self._map_overlay_target_label = QLabel("-", status_group)
        self._map_overlay_backend_label = QLabel("-", status_group)
        self._map_overlay_count_label = QLabel("-", status_group)
        self._map_overlay_confidence_label = QLabel("-", status_group)
        self._map_overlay_error_label = QLabel("-", status_group)
        for label in (
            self._map_overlay_status_label,
            self._map_overlay_target_label,
            self._map_overlay_backend_label,
            self._map_overlay_count_label,
            self._map_overlay_confidence_label,
            self._map_overlay_error_label,
        ):
            label.setWordWrap(True)
        status_form.addRow("状态", self._map_overlay_status_label)
        status_form.addRow("目标窗口", self._map_overlay_target_label)
        status_form.addRow("截图后端", self._map_overlay_backend_label)
        status_form.addRow("点位数量", self._map_overlay_count_label)
        status_form.addRow("匹配置信度", self._map_overlay_confidence_label)
        status_form.addRow("最近错误", self._map_overlay_error_label)
        layout.addWidget(status_group)

        self._auxiliary_log_view = QPlainTextEdit(panel)
        self._auxiliary_log_view.setReadOnly(True)
        layout.addWidget(self._auxiliary_log_view, 1)
        self._append_auxiliary_log("辅助功能界面已就绪。")
        return panel

    def _load_map_overlay_data_for_ui(self) -> None:
        try:
            self._map_overlay_data = ZeroluckRepository().load()
            self._map_overlay_data_error = ""
            self._map_overlay_settings = load_map_overlay_settings(
                self._settings_store,
                self._map_overlay_data.categories,
            )
        except ZeroluckDataError as exc:
            self._map_overlay_data = None
            self._map_overlay_data_error = str(exc)
            self._map_overlay_settings = load_map_overlay_settings(self._settings_store, [])

    def _map_overlay_data_summary_text(self) -> str:
        if self._map_overlay_data_error:
            return f"地图数据加载失败：{self._map_overlay_data_error}"
        if self._map_overlay_data is None:
            return "ZeroLuck 离线快照尚未加载。"
        text = "ZeroLuck 离线快照"
        if self._map_overlay_data.fetched_at:
            text += f"\n生成时间：{self._map_overlay_data.fetched_at.isoformat()}"
        text += f"\n分类：{len(self._map_overlay_data.categories)}，点位：{self._map_overlay_data.marker_count}"
        return text

    def _rebuild_map_category_checks(self) -> None:
        while self._map_category_layout.count():
            item = self._map_category_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._map_category_checks.clear()
        if self._map_overlay_data is None:
            self._map_category_layout.addWidget(QLabel("地图数据未加载，无法显示分类。", self._map_category_panel))
            self._map_category_layout.addStretch(1)
            return
        for category in self._map_overlay_data.categories:
            count = category.total or sum(1 for marker in self._map_overlay_data.markers if marker.category_id == category.id)
            checkbox = QCheckBox(f"{category.display_name}  ({count})", self._map_category_panel)
            checkbox.setToolTip(f"{category.name_en}\n{category.id}")
            checkbox.toggled.connect(self._on_map_overlay_settings_widget_changed)
            self._map_category_checks[category.id] = checkbox
            self._map_category_layout.addWidget(checkbox)
        self._map_category_layout.addStretch(1)

    def _sync_map_overlay_settings_widgets(self) -> None:
        self._map_settings_syncing = True
        try:
            self._map_search_edit.setText(self._map_overlay_settings.search_text)
            self._map_show_labels_check.setChecked(self._map_overlay_settings.show_labels)
            self._map_cluster_check.setChecked(self._map_overlay_settings.cluster_enabled)
            self._map_viewport_only_check.setChecked(self._map_overlay_settings.viewport_only)
            self._map_auto_refresh_check.setChecked(self._map_overlay_settings.auto_refresh_enabled)
            self._map_debug_confidence_check.setChecked(self._map_overlay_settings.debug_show_confidence)
            self._map_debug_bounds_check.setChecked(self._map_overlay_settings.debug_show_match_bounds)
            self._map_auto_detect_check.setChecked(self._map_overlay_settings.auto_detect_map_enabled)
            self._map_auto_rematch_check.setChecked(self._map_overlay_settings.auto_rematch_on_map_change)
            self._map_icon_size_spin.setValue(self._map_overlay_settings.icon_size)
            self._map_label_size_spin.setValue(self._map_overlay_settings.label_size)
            self._map_auto_stop_spin.setValue(self._map_overlay_settings.auto_stop_seconds)
            self._map_watch_interval_spin.setValue(self._map_overlay_settings.map_watch_interval_ms)
            self._map_match_cooldown_spin.setValue(self._map_overlay_settings.match_cooldown_ms)
            for category_id, checkbox in self._map_category_checks.items():
                checkbox.setChecked(category_id in self._map_overlay_settings.enabled_categories)
        finally:
            self._map_settings_syncing = False

    def _current_map_overlay_settings_from_widgets(self) -> MapOverlaySettings:
        enabled = frozenset(
            category_id for category_id, checkbox in self._map_category_checks.items() if checkbox.isChecked()
        )
        return MapOverlaySettings(
            enabled_categories=enabled,
            search_text=self._map_search_edit.text(),
            show_labels=self._map_show_labels_check.isChecked(),
            cluster_enabled=self._map_cluster_check.isChecked(),
            icon_size=self._map_icon_size_spin.value(),
            label_size=self._map_label_size_spin.value(),
            viewport_only=self._map_viewport_only_check.isChecked(),
            auto_refresh_enabled=self._map_auto_refresh_check.isChecked(),
            auto_detect_map_enabled=self._map_auto_detect_check.isChecked(),
            auto_rematch_on_map_change=self._map_auto_rematch_check.isChecked(),
            map_watch_interval_ms=self._map_watch_interval_spin.value(),
            match_cooldown_ms=self._map_match_cooldown_spin.value(),
            auto_stop_seconds=self._map_auto_stop_spin.value(),
            debug_show_confidence=self._map_debug_confidence_check.isChecked(),
            debug_show_match_bounds=self._map_debug_bounds_check.isChecked(),
            debug_save_screenshot=False,
        )

    def _on_map_overlay_settings_widget_changed(self, *_args: object) -> None:
        if self._map_settings_syncing:
            return
        self._map_overlay_settings = self._current_map_overlay_settings_from_widgets()
        save_map_overlay_settings(self._settings_store, self._map_overlay_settings)
        if self._map_overlay_controller is not None:
            self._map_overlay_controller.apply_settings(self._map_overlay_settings)
        self._append_auxiliary_log("地图悬浮层设置已保存。")

    def _set_map_categories_checked(self, checked: bool) -> None:
        self._map_settings_syncing = True
        try:
            for checkbox in self._map_category_checks.values():
                checkbox.setChecked(checked)
        finally:
            self._map_settings_syncing = False
        self._on_map_overlay_settings_widget_changed()

    def _restore_default_map_categories(self) -> None:
        if self._map_overlay_data is None:
            return
        defaults = {category.id for category in self._map_overlay_data.categories if category.default_visible}
        self._map_settings_syncing = True
        try:
            for category_id, checkbox in self._map_category_checks.items():
                checkbox.setChecked(category_id in defaults)
        finally:
            self._map_settings_syncing = False
        self._on_map_overlay_settings_widget_changed()

    def _on_auxiliary_tool_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        tool_id = str(current.data(Qt.UserRole) or "map_overlay")
        if tool_id not in AUXILIARY_TOOLS:
            tool_id = "map_overlay"
        self._selected_auxiliary_tool_id = tool_id
        spec = AUXILIARY_TOOLS[tool_id]
        self._auxiliary_title_label.setText(spec.label)
        self._auxiliary_description_label.setText(spec.description)
        self._apply_auxiliary_guard()

    def _select_auxiliary_tool_id(self, tool_id: str) -> None:
        for index in range(self._auxiliary_tool_list.count()):
            item = self._auxiliary_tool_list.item(index)
            if str(item.data(Qt.UserRole) or "") == tool_id:
                self._auxiliary_tool_list.setCurrentItem(item)
                return

    def _ensure_map_overlay_controller(self) -> MapOverlayController:
        if self._map_overlay_controller is not None:
            return self._map_overlay_controller
        from .. import app as app_module

        self._map_overlay_controller = app_module.MapOverlayController(
            self._settings_store,
            game_name=GAME_NAME,
            parent=self,
        )
        self._map_overlay_controller.state_changed.connect(self._on_map_overlay_state_changed)
        self._map_overlay_controller.data_ready.connect(self._on_map_overlay_data_ready)
        self._map_overlay_controller.log_message.connect(self._on_map_overlay_log_message)
        return self._map_overlay_controller

    def _start_selected_auxiliary_tool(self) -> None:
        if self._selected_auxiliary_tool_id != "map_overlay":
            self._append_auxiliary_log("未知辅助功能，无法启动。", level="warning")
            return
        settings = self._current_map_overlay_settings_from_widgets()
        save_map_overlay_settings(self._settings_store, settings)
        self._map_overlay_settings = settings
        controller = self._ensure_map_overlay_controller()
        controller.start(settings)
        self._append_auxiliary_log("正在启动大地图点位悬浮层。")
        self._apply_auxiliary_guard()

    def _stop_selected_auxiliary_tool(self) -> None:
        if self._map_overlay_controller is None:
            self._append_auxiliary_log("当前没有正在运行的辅助功能。", level="warning")
            return
        self._map_overlay_controller.stop()
        self._map_overlay_controller = None
        self._map_overlay_running = False
        self._append_auxiliary_log("已关闭大地图点位悬浮层。")
        self._apply_auxiliary_guard()

    def _refresh_map_overlay(self) -> None:
        if self._map_overlay_controller is None or not self._map_overlay_running:
            self._append_auxiliary_log("请先启动大地图点位悬浮层。", level="warning")
            return
        self._map_overlay_controller.refresh()
        self._append_auxiliary_log("已请求手动刷新大地图点位。")

    def _toggle_map_overlay_visibility(self) -> None:
        if self._map_overlay_controller is None:
            self._append_auxiliary_log("请先启动大地图点位悬浮层。", level="warning")
            return
        self._map_markers_visible = not self._map_markers_visible
        self._map_overlay_controller.set_markers_visible(self._map_markers_visible)
        self._map_visibility_button.setText("隐藏点位" if self._map_markers_visible else "显示点位")

    def _on_map_overlay_state_changed(self, state: MapOverlayUiState) -> None:
        self._map_overlay_running = bool(state.running)
        self._selected_auxiliary_status_label.setText(state.status)
        self._map_overlay_status_label.setText(state.status)
        self._map_overlay_target_label.setText(state.target_title or "-")
        self._map_overlay_backend_label.setText(state.backend or "-")
        self._map_overlay_count_label.setText(
            f"当前显示 {state.item_count} 个点位/聚合；数据源 {state.category_count} 类 / {state.marker_count} 点"
        )
        self._map_overlay_confidence_label.setText("-" if state.confidence is None else f"{state.confidence:.0%}")
        self._map_overlay_error_label.setText(state.error or "-")
        self._apply_auxiliary_guard()

    def _on_map_overlay_data_ready(self, data: ZeroluckMapData, _summary: dict[str, Any]) -> None:
        self._map_overlay_data = data
        self._map_overlay_data_error = ""
        self._map_data_label.setText(self._map_overlay_data_summary_text())
        if not self._map_category_checks:
            self._map_overlay_settings = load_map_overlay_settings(self._settings_store, data.categories)
            self._rebuild_map_category_checks()
            self._sync_map_overlay_settings_widgets()

    def _on_map_overlay_log_message(self, message: str, level: str = "info") -> None:
        self._append_auxiliary_log(message, level=level)

    def _append_auxiliary_log(self, message: str, *, level: str = "info") -> None:
        if not hasattr(self, "_auxiliary_log_view"):
            return
        prefix = {
            "warning": "警告",
            "error": "错误",
            "info": "信息",
        }.get(level, "信息")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._auxiliary_log_view.appendPlainText(f"[{timestamp}] [{prefix}] {message}")

    def _apply_auxiliary_guard(self) -> None:
        if not hasattr(self, "_start_auxiliary_button"):
            return
        self._start_auxiliary_button.setEnabled(not self._map_overlay_running)
        self._stop_auxiliary_button.setEnabled(self._map_overlay_running)
        self._map_refresh_button.setEnabled(self._map_overlay_running)
        self._map_visibility_button.setEnabled(self._map_overlay_running)

    def _open_map_overlay(self) -> None:
        self._pages.setCurrentWidget(self._auxiliary_page)
        self._start_selected_auxiliary_tool()

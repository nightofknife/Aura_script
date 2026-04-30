from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from PySide6.QtCore import QMetaObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from packages.aura_core.runtime.privilege import (
    AdminPrivilegeRequiredError,
    ensure_admin_startup,
    is_running_as_admin,
)

from .bridge import RunnerBridge
from .config_repository import QUICK_STOP_HOTKEY_OPTIONS, YihuanConfigRepository
from .logic import (
    CafeRunDefaults,
    FishingRunDefaults,
    GAME_NAME,
    GuiPreferences,
    RuntimeSettings,
    TASK_AUTO_LOOP,
    TASK_CAFE_AUTO_LOOP,
    TASK_PLAN_LOADED,
    TASK_PLAN_READY,
    TASK_RUNTIME_PROBE,
    VISIBLE_HISTORY_TASK_REFS,
    auto_loop_business_status,
    build_auto_loop_inputs,
    build_cafe_loop_inputs,
    build_settings_sections,
    cafe_loop_business_status,
    event_display_name,
    format_event_stream,
    format_nodes_timeline,
    history_row_label,
    is_runtime_interacting_task,
    render_auto_loop_brief_text,
    render_cafe_loop_brief_text,
    render_history_summary_html,
    render_json,
    render_overview_plan_info_html,
    render_runtime_probe_html,
    status_display_name,
    task_display_name,
    task_is_enabled,
)
from .map_overlay.controller import MapOverlayController


WORKBENCH_TASKS: dict[str, dict[str, str]] = {
    "fishing": {
        "label": "钓鱼",
        "task_ref": TASK_AUTO_LOOP,
        "description": "自动循环钓鱼，只保留最大轮数快捷输入。",
    },
    "cafe": {
        "label": "沙威玛",
        "task_ref": TASK_CAFE_AUTO_LOOP,
        "description": "自动执行沙威玛小游戏，识别订单、补货并制作餐品。",
    },
}
WORKBENCH_TASK_REFS = tuple(item["task_ref"] for item in WORKBENCH_TASKS.values())


@dataclass(frozen=True)
class PendingLaunch:
    task_ref: str
    inputs: dict[str, Any]
    selected_task_id: str
    remaining_sec: int
    total_sec: int


class RunDetailViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._title_label = QLabel("尚未选择运行记录。", self)
        self._title_label.setWordWrap(True)
        layout.addWidget(self._title_label)

        self._tabs = QTabWidget(self)
        self._summary_browser = QTextBrowser(self)
        self._nodes_view = QPlainTextEdit(self)
        self._nodes_view.setReadOnly(True)
        self._raw_view = QPlainTextEdit(self)
        self._raw_view.setReadOnly(True)
        self._events_view = QPlainTextEdit(self)
        self._events_view.setReadOnly(True)
        self._tabs.addTab(self._summary_browser, "摘要")
        self._tabs.addTab(self._nodes_view, "时间线")
        self._tabs.addTab(self._raw_view, "原始 JSON")
        self._tabs.addTab(self._events_view, "事件流")
        layout.addWidget(self._tabs)

    def show_detail(self, detail: dict[str, Any] | None) -> None:
        payload = dict(detail or {})
        if not payload:
            self._title_label.setText("尚未选择运行记录。")
            self._summary_browser.setHtml("<p>暂无可显示的详情。</p>")
            self._nodes_view.setPlainText("")
            self._raw_view.setPlainText("")
            self._events_view.setPlainText("")
            return

        cid = str(payload.get("cid") or "-")
        task_name = task_display_name(payload.get("task_name"))
        status = status_display_name(payload.get("status"))
        self._title_label.setText(f"{status}  {task_name}  {cid}")
        self._summary_browser.setHtml(render_history_summary_html(payload))
        self._nodes_view.setPlainText(format_nodes_timeline(payload.get("nodes") or []))
        self._raw_view.setPlainText(render_json(payload))
        self._events_view.setPlainText(format_event_stream(payload.get("_live_events") or []))


class _WindowsMsg(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", getattr(wintypes, "WPARAM", ctypes.c_size_t)),
        ("lParam", getattr(wintypes, "LPARAM", ctypes.c_ssize_t)),
        ("time", wintypes.DWORD),
        ("pt", wintypes.POINT),
    ]


class WindowsGlobalHotkeyManager:
    WM_HOTKEY = 0x0312
    HOTKEY_ID = 0xA0A8
    VK_BY_KEY = {f"F{index}": 0x6F + index for index in range(1, 13)}

    def __init__(self, window: QWidget) -> None:
        self._window = window
        self._registered_key: str | None = None

    @property
    def registered_key(self) -> str | None:
        return self._registered_key

    def register(self, key: str) -> tuple[bool, str]:
        self.unregister()
        normalized = str(key or "").strip().upper()
        if normalized not in QUICK_STOP_HOTKEY_OPTIONS:
            return False, "快捷停止键必须是 F6 到 F12。"
        if not sys.platform.startswith("win"):
            return False, "全局快捷键仅在 Windows 下可用。"

        hwnd = int(self._window.winId())
        vk = self.VK_BY_KEY.get(normalized)
        if not vk:
            return False, f"不支持的快捷键：{normalized}"
        ok = bool(ctypes.windll.user32.RegisterHotKey(hwnd, self.HOTKEY_ID, 0, vk))
        if not ok:
            return False, "全局停止键注册失败，可能已被其他程序占用。"
        self._registered_key = normalized
        return True, f"全局停止键已注册：{normalized}"

    def unregister(self) -> None:
        if not self._registered_key or not sys.platform.startswith("win"):
            self._registered_key = None
            return
        try:
            ctypes.windll.user32.UnregisterHotKey(int(self._window.winId()), self.HOTKEY_ID)
        finally:
            self._registered_key = None

    def matches_native_message(self, message: int) -> bool:
        if not sys.platform.startswith("win"):
            return False
        try:
            msg = _WindowsMsg.from_address(int(message))
        except (TypeError, ValueError):
            return False
        return msg.message == self.WM_HOTKEY and int(msg.wParam) == self.HOTKEY_ID


class YihuanMainWindow(QMainWindow):
    request_initialize = Signal()
    request_run_task = Signal(str, object)
    request_cancel_task = Signal(str)
    request_fetch_run_detail = Signal(str)
    request_refresh_history = Signal()
    request_refresh_plan_info = Signal()
    request_refresh_runtime_probe = Signal()
    request_refresh_doctor = Signal()
    request_apply_preferences = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AURA控制台")
        self.resize(1500, 940)

        self._settings_store = QSettings("Aura", "YihuanGui")
        self._repo = YihuanConfigRepository(self._resolve_plan_root(), self._settings_store)
        self._ui_preferences = self._repo.get_ui_preferences()
        self._runtime_settings = self._repo.get_runtime_settings()
        self._fishing_defaults = self._repo.get_fishing_defaults()
        self._cafe_defaults = self._repo.get_cafe_defaults()

        self._task_rows: dict[str, dict[str, Any]] = {}
        self._history_rows: dict[str, dict[str, Any]] = {}
        self._detail_cache: dict[str, dict[str, Any]] = {}
        self._cid_to_task_ref: dict[str, str] = {}
        self._live_state: dict[str, Any] = {"active_runs": {}}
        self._runner_ready = False
        self._tasks_ready = False
        self._selected_task_id = "fishing"
        self._current_active_cid: str | None = None
        self._current_history_cid: str | None = None
        self._last_business_cids: dict[str, str] = {}
        self._pending_launch: PendingLaunch | None = None
        self._hotkey_warning_shown = False
        self._map_overlay_controller: MapOverlayController | None = None

        self._pending_timer = QTimer(self)
        self._pending_timer.setInterval(1000)
        self._pending_timer.timeout.connect(self._on_pending_timer_tick)
        self._quick_stop_shortcut: QShortcut | None = None
        self._hotkey_manager = WindowsGlobalHotkeyManager(self)

        self.setStatusBar(QStatusBar(self))
        self._build_menu()
        self._build_central_pages()
        self._restore_window_settings()
        self._load_settings_widgets()
        self._install_quick_stop_hotkey(self._ui_preferences.quick_stop_hotkey, show_warning=False)
        self._setup_bridge()
        self._append_log("GUI 已启动，正在后台预热 Aura 框架。")
        self._apply_task_guard()

    @staticmethod
    def _resolve_plan_root() -> Path:
        return Path(__file__).resolve().parents[2] / "plans" / "yihuan"

    def _build_menu(self) -> None:
        task_action = QAction("任务界面", self)
        task_action.triggered.connect(lambda: self._pages.setCurrentWidget(self._workbench_page))
        settings_action = QAction("设置界面", self)
        settings_action.triggered.connect(lambda: self._pages.setCurrentWidget(self._settings_page))
        self.menuBar().addAction(task_action)
        self.menuBar().addAction(settings_action)

        tools_menu = self.menuBar().addMenu("辅助")
        self._refresh_probe_action = QAction("刷新运行时探针", self)
        self._refresh_probe_action.triggered.connect(self._request_refresh_runtime_probe_from_ui)
        refresh_plan_action = QAction("刷新方案包信息", self)
        refresh_plan_action.triggered.connect(self.request_refresh_plan_info.emit)
        refresh_history_action = QAction("刷新最近记录", self)
        refresh_history_action.triggered.connect(self.request_refresh_history.emit)
        map_overlay_action = QAction("异环大地图点位悬浮层", self)
        map_overlay_action.triggered.connect(self._open_map_overlay)
        tools_menu.addAction(self._refresh_probe_action)
        tools_menu.addAction(refresh_plan_action)
        tools_menu.addAction(refresh_history_action)
        tools_menu.addSeparator()
        tools_menu.addAction(map_overlay_action)

    def _build_central_pages(self) -> None:
        self._pages = QStackedWidget(self)
        self.setCentralWidget(self._pages)
        self._workbench_page = QWidget(self)
        self._settings_page = QWidget(self)
        self._pages.addWidget(self._workbench_page)
        self._pages.addWidget(self._settings_page)
        self._build_workbench_page()
        self._build_settings_page()

    def _build_workbench_page(self) -> None:
        layout = QVBoxLayout(self._workbench_page)
        layout.addWidget(self._build_status_bar_widget())

        splitter = QSplitter(Qt.Horizontal, self._workbench_page)
        layout.addWidget(splitter, 1)

        splitter.addWidget(self._build_task_column(splitter))
        splitter.addWidget(self._build_parameter_column(splitter))
        splitter.addWidget(self._build_log_column(splitter))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([260, 420, 760])
        if self._task_list.count():
            self._task_list.setCurrentRow(0)

    def _build_status_bar_widget(self) -> QWidget:
        group = QGroupBox("运行状态", self._workbench_page)
        row = QHBoxLayout(group)
        self._admin_label = QLabel("-", group)
        self._runner_label = QLabel("框架启动中", group)
        self._runtime_summary_label = QLabel("运行时探针：等待探测", group)
        self._last_event_label = QLabel("最近事件：-", group)
        self._last_error_label = QLabel("最近错误：-", group)
        self._runtime_summary_label.setWordWrap(True)
        self._last_error_label.setWordWrap(True)
        row.addWidget(QLabel("管理员", group))
        row.addWidget(self._admin_label)
        row.addSpacing(18)
        row.addWidget(QLabel("框架", group))
        row.addWidget(self._runner_label)
        row.addSpacing(18)
        row.addWidget(self._runtime_summary_label, 1)
        row.addWidget(self._last_event_label, 1)
        row.addWidget(self._last_error_label, 1)
        return group

    def _build_task_column(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)

        title = QLabel("任务", panel)
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)

        self._task_list = QListWidget(panel)
        for task_id, spec in WORKBENCH_TASKS.items():
            item = QListWidgetItem(spec["label"])
            item.setData(Qt.UserRole, task_id)
            self._task_list.addItem(item)
        self._task_list.currentItemChanged.connect(self._on_task_item_changed)
        layout.addWidget(self._task_list, 1)

        self._selected_task_status_label = QLabel("等待任务列表加载。", panel)
        self._selected_task_status_label.setWordWrap(True)
        layout.addWidget(self._selected_task_status_label)

        layout.addStretch(1)

        self._start_button = QPushButton("开始执行", panel)
        self._start_button.clicked.connect(self._request_start_selected_task)
        self._stop_button = QPushButton(f"停止任务（{self._ui_preferences.quick_stop_hotkey}）", panel)
        self._stop_button.clicked.connect(self._request_stop_current_task)
        layout.addWidget(self._start_button)
        layout.addWidget(self._stop_button)
        return panel

    def _build_parameter_column(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        self._task_title_label = QLabel("钓鱼", panel)
        self._task_title_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        self._task_description_label = QLabel("", panel)
        self._task_description_label.setWordWrap(True)
        layout.addWidget(self._task_title_label)
        layout.addWidget(self._task_description_label)

        self._parameter_stack = QStackedWidget(panel)
        self._parameter_pages: dict[str, QWidget] = {}
        self._parameter_pages["fishing"] = self._build_fishing_parameters(self._parameter_stack)
        self._parameter_pages["cafe"] = self._build_cafe_parameters(self._parameter_stack)
        for task_id in WORKBENCH_TASKS:
            self._parameter_stack.addWidget(self._parameter_pages[task_id])
        layout.addWidget(self._parameter_stack, 1)
        return panel

    def _build_fishing_parameters(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)

        group = QGroupBox("钓鱼参数", page)
        form = QFormLayout(group)
        self._fishing_profile_label = QLabel("-", group)
        self._max_rounds_spin = QSpinBox(group)
        self._max_rounds_spin.setMinimum(0)
        self._max_rounds_spin.setMaximum(99999)
        self._max_rounds_spin.setValue(0)
        self._max_rounds_spin.setToolTip("0 表示不限制轮数，直到任务自身停止。")
        form.addRow("钓鱼识别档案", self._fishing_profile_label)
        form.addRow("最大轮数（0 = 不限制轮数）", self._max_rounds_spin)
        layout.addWidget(group)

        hint = QLabel("只在这里放本次运行最常用的快捷参数；识别档案请到设置界面调整。", page)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_cafe_parameters(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)

        group = QGroupBox("沙威玛参数", page)
        form = QFormLayout(group)
        self._cafe_profile_label = QLabel("-", group)
        self._cafe_max_seconds_spin = QSpinBox(group)
        self._cafe_max_seconds_spin.setMinimum(0)
        self._cafe_max_seconds_spin.setMaximum(99999)
        self._cafe_max_seconds_spin.setToolTip("0 表示不按运行时间停止。")
        self._cafe_max_orders_spin = QSpinBox(group)
        self._cafe_max_orders_spin.setMinimum(0)
        self._cafe_max_orders_spin.setMaximum(99999)
        self._cafe_max_orders_spin.setToolTip("0 表示不按订单数量停止。")
        self._cafe_start_game_check = QCheckBox("自动点击开始游戏", group)
        self._cafe_wait_level_started_check = QCheckBox("等待关卡开始", group)
        self._cafe_full_assist_auto_hammer_check = QCheckBox("全辅助自动锤子模式", group)
        self._cafe_limit_hint_label = QLabel("", group)
        self._cafe_limit_hint_label.setWordWrap(True)
        form.addRow("沙威玛识别档案", self._cafe_profile_label)
        form.addRow("最大运行秒数（0 = 不按时间停止）", self._cafe_max_seconds_spin)
        form.addRow("最大订单数（0 = 不按订单数停止）", self._cafe_max_orders_spin)
        form.addRow("", self._cafe_start_game_check)
        form.addRow("", self._cafe_wait_level_started_check)
        form.addRow("", self._cafe_full_assist_auto_hammer_check)
        form.addRow("档案说明", self._cafe_limit_hint_label)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _build_log_column(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)

        header_row = QHBoxLayout()
        title = QLabel("运行日志与详情", panel)
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        refresh_history_button = QPushButton("刷新最近记录", panel)
        refresh_history_button.clicked.connect(self.request_refresh_history.emit)
        header_row.addWidget(title)
        header_row.addStretch(1)
        header_row.addWidget(refresh_history_button)
        layout.addLayout(header_row)

        splitter = QSplitter(Qt.Vertical, panel)
        layout.addWidget(splitter, 1)

        self._log_view = QPlainTextEdit(splitter)
        self._log_view.setReadOnly(True)
        splitter.addWidget(self._log_view)

        detail_splitter = QSplitter(Qt.Horizontal, splitter)
        self._history_list = QListWidget(detail_splitter)
        self._history_list.currentItemChanged.connect(self._on_history_item_changed)
        detail_splitter.addWidget(self._history_list)
        self._detail_viewer = RunDetailViewer(detail_splitter)
        detail_splitter.addWidget(self._detail_viewer)
        detail_splitter.setStretchFactor(0, 0)
        detail_splitter.setStretchFactor(1, 1)
        splitter.addWidget(detail_splitter)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        return panel

    def _build_settings_page(self) -> None:
        layout = QVBoxLayout(self._settings_page)
        scroll = QScrollArea(self._settings_page)
        scroll.setWidgetResizable(True)
        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        runtime_group = QGroupBox("运行环境", body)
        runtime_form = QFormLayout(runtime_group)
        self._title_regex_edit = QLineEdit(runtime_group)
        self._exclude_titles_edit = QPlainTextEdit(runtime_group)
        self._exclude_titles_edit.setPlaceholderText("每行一个排除标题")
        self._allow_borderless_check = QCheckBox("允许匹配无边框窗口", runtime_group)
        self._capture_backend_combo = QComboBox(runtime_group)
        self._input_backend_combo = QComboBox(runtime_group)
        self._input_profile_combo = QComboBox(runtime_group)
        runtime_form.addRow("窗口标题匹配规则", self._title_regex_edit)
        runtime_form.addRow("排除窗口标题", self._exclude_titles_edit)
        runtime_form.addRow("", self._allow_borderless_check)
        runtime_form.addRow("截图后端", self._capture_backend_combo)
        runtime_form.addRow("输入后端", self._input_backend_combo)
        runtime_form.addRow("默认输入档案", self._input_profile_combo)
        body_layout.addWidget(runtime_group)

        defaults_group = QGroupBox("任务默认档案", body)
        defaults_form = QFormLayout(defaults_group)
        self._fishing_profile_combo = QComboBox(defaults_group)
        self._cafe_profile_combo = QComboBox(defaults_group)
        defaults_form.addRow("钓鱼识别档案", self._fishing_profile_combo)
        defaults_form.addRow("沙威玛识别档案", self._cafe_profile_combo)
        body_layout.addWidget(defaults_group)

        ui_group = QGroupBox("界面偏好", body)
        ui_form = QFormLayout(ui_group)
        self._history_limit_spin = QSpinBox(ui_group)
        self._history_limit_spin.setMinimum(1)
        self._history_limit_spin.setMaximum(500)
        self._auto_probe_check = QCheckBox("启动时自动执行运行时探针", ui_group)
        self._task_start_delay_spin = QSpinBox(ui_group)
        self._task_start_delay_spin.setMinimum(0)
        self._task_start_delay_spin.setMaximum(60)
        self._task_start_delay_spin.setToolTip("0 表示点击开始后立即提交任务。")
        self._quick_stop_hotkey_combo = QComboBox(ui_group)
        ui_form.addRow("历史记录显示条数", self._history_limit_spin)
        ui_form.addRow("", self._auto_probe_check)
        ui_form.addRow("任务启动延迟秒数（0 = 立即执行）", self._task_start_delay_spin)
        ui_form.addRow("快捷停止键", self._quick_stop_hotkey_combo)
        body_layout.addWidget(ui_group)

        action_row = QHBoxLayout()
        self._reload_settings_button = QPushButton("重新加载设置", body)
        self._reload_settings_button.clicked.connect(self._load_settings_widgets)
        self._save_settings_button = QPushButton("保存设置", body)
        self._save_settings_button.clicked.connect(self._save_settings)
        action_row.addWidget(self._reload_settings_button)
        action_row.addWidget(self._save_settings_button)
        action_row.addStretch(1)
        body_layout.addLayout(action_row)

        hint = QLabel("设置页只负责运行环境、任务默认档案和界面偏好；玩法识别阈值、区域坐标与时序参数仍在各识别档案中维护。", body)
        hint.setWordWrap(True)
        body_layout.addWidget(hint)
        body_layout.addStretch(1)

    def _load_settings_widgets(self) -> None:
        self._runtime_settings = self._repo.get_runtime_settings()
        self._ui_preferences = self._repo.get_ui_preferences()
        available_profiles = self._repo.list_input_profiles()
        sections = build_settings_sections(self._runtime_settings, self._ui_preferences, available_profiles)
        runtime_section = next(section for section in sections if section.section_id == "runtime")
        ui_section = next(section for section in sections if section.section_id == "ui")
        runtime_map = {field.key: field for field in runtime_section.fields}
        ui_map = {field.key: field for field in ui_section.fields}

        self._title_regex_edit.setText(str(runtime_map["runtime.target.title_regex"].value))
        self._exclude_titles_edit.setPlainText(str(runtime_map["runtime.target.exclude_titles"].value))
        self._allow_borderless_check.setChecked(bool(runtime_map["runtime.target.allow_borderless"].value))
        self._set_combo_items(
            self._capture_backend_combo,
            runtime_map["runtime.capture.backend"].value,
            ["gdi", "dxgi", "wgc", "printwindow"],
        )
        self._set_combo_items(
            self._input_backend_combo,
            runtime_map["runtime.input.backend"].value,
            ["sendinput", "window_message"],
        )
        profile_field = runtime_map["input.profile"]
        self._set_combo_items(self._input_profile_combo, profile_field.value, list(profile_field.options))

        self._fishing_defaults = self._repo.get_fishing_defaults(self._task_rows.get(TASK_AUTO_LOOP))
        self._set_combo_items(
            self._fishing_profile_combo,
            self._fishing_defaults.profile_name,
            self._repo.list_fishing_profiles(),
        )
        self._fishing_profile_label.setText(self._fishing_defaults.profile_name)

        self._cafe_defaults = self._repo.get_cafe_defaults(self._task_rows.get(TASK_CAFE_AUTO_LOOP))
        self._set_combo_items(
            self._cafe_profile_combo,
            self._cafe_defaults.profile_name,
            self._repo.list_cafe_profiles(),
        )
        self._cafe_profile_label.setText(self._cafe_defaults.profile_name)
        self._cafe_max_seconds_spin.setValue(int(self._cafe_defaults.max_seconds))
        self._cafe_max_orders_spin.setValue(int(self._cafe_defaults.max_orders))
        self._cafe_start_game_check.setChecked(bool(self._cafe_defaults.start_game))
        self._cafe_wait_level_started_check.setChecked(bool(self._cafe_defaults.wait_level_started))
        self._cafe_full_assist_auto_hammer_check.setChecked(bool(self._cafe_defaults.full_assist_auto_hammer_mode))
        self._refresh_cafe_limit_hint()

        self._history_limit_spin.setValue(int(ui_map["gui.history_limit"].value))
        self._auto_probe_check.setChecked(bool(ui_map["gui.auto_runtime_probe_on_startup"].value))
        self._task_start_delay_spin.setValue(int(ui_map["gui.task_start_delay_sec"].value))
        self._set_combo_items(
            self._quick_stop_hotkey_combo,
            str(ui_map["gui.quick_stop_hotkey"].value).strip().upper(),
            list(QUICK_STOP_HOTKEY_OPTIONS),
        )
        self._stop_button.setText(f"停止任务（{self._ui_preferences.quick_stop_hotkey}）")

    @staticmethod
    def _set_combo_items(combo: QComboBox, current_value: Any, options: list[str]) -> None:
        combo.clear()
        seen: set[str] = set()
        for option in [str(current_value or "")] + [str(item) for item in options]:
            if not option or option in seen:
                continue
            combo.addItem(option, option)
            seen.add(option)
        index = combo.findData(str(current_value or ""))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _save_settings(self) -> None:
        runtime_settings = RuntimeSettings(
            title_regex=self._title_regex_edit.text().strip(),
            exclude_titles=self._repo.exclude_titles_from_text(self._exclude_titles_edit.toPlainText()),
            allow_borderless=self._allow_borderless_check.isChecked(),
            capture_backend=str(self._capture_backend_combo.currentData() or self._capture_backend_combo.currentText()),
            input_backend=str(self._input_backend_combo.currentData() or self._input_backend_combo.currentText()),
            input_profile=str(self._input_profile_combo.currentData() or self._input_profile_combo.currentText()),
        )
        ui_preferences = GuiPreferences(
            history_limit=int(self._history_limit_spin.value()),
            auto_runtime_probe_on_startup=self._auto_probe_check.isChecked(),
            expand_developer_tools=self._ui_preferences.expand_developer_tools,
            task_start_delay_sec=int(self._task_start_delay_spin.value()),
            quick_stop_hotkey=str(
                self._quick_stop_hotkey_combo.currentData() or self._quick_stop_hotkey_combo.currentText()
            ).strip()
            .upper(),
        )
        fishing_defaults = FishingRunDefaults(
            profile_name=str(self._fishing_profile_combo.currentData() or self._fishing_profile_combo.currentText())
        )
        cafe_defaults = CafeRunDefaults(
            profile_name=str(self._cafe_profile_combo.currentData() or self._cafe_profile_combo.currentText()),
            max_seconds=self._cafe_defaults.max_seconds,
            max_orders=self._cafe_defaults.max_orders,
            start_game=self._cafe_defaults.start_game,
            wait_level_started=self._cafe_defaults.wait_level_started,
            full_assist_auto_hammer_mode=self._cafe_defaults.full_assist_auto_hammer_mode,
            min_order_interval_sec=self._cafe_defaults.min_order_interval_sec,
            min_order_duration_sec=self._cafe_defaults.min_order_duration_sec,
        )

        try:
            self._repo.update_runtime_settings(runtime_settings)
            self._repo.update_fishing_defaults(fishing_defaults)
            self._repo.update_cafe_defaults(cafe_defaults)
            self._repo.save_ui_preferences(ui_preferences)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存设置失败", str(exc))
            return

        self._runtime_settings = runtime_settings
        self._ui_preferences = ui_preferences
        self._fishing_defaults = self._repo.get_fishing_defaults(self._task_rows.get(TASK_AUTO_LOOP))
        self._cafe_defaults = self._repo.get_cafe_defaults(self._task_rows.get(TASK_CAFE_AUTO_LOOP))
        self._fishing_profile_label.setText(self._fishing_defaults.profile_name)
        self._cafe_profile_label.setText(self._cafe_defaults.profile_name)
        self._refresh_cafe_limit_hint()
        self._install_quick_stop_hotkey(ui_preferences.quick_stop_hotkey, show_warning=True)
        self.request_apply_preferences.emit(ui_preferences)
        self.request_refresh_history.emit()
        self.statusBar().showMessage("设置已保存，新配置将在后续任务和探针中生效。", 6000)
        self._append_log("设置已保存，新配置将在后续任务和探针中生效。")
        self._apply_task_guard()

    def _refresh_cafe_limit_hint(self) -> None:
        runtime_defaults = self._repo.get_cafe_profile_runtime_defaults(self._cafe_defaults.profile_name)
        default_seconds = runtime_defaults.get("max_seconds")
        if default_seconds is None:
            default_text = "当前档案默认运行时长未知"
        else:
            seconds_text = int(default_seconds) if float(default_seconds).is_integer() else round(default_seconds, 1)
            default_text = f"当前档案默认运行时长：{seconds_text} 秒"

        interval_text = self._format_optional_seconds(runtime_defaults.get("min_order_interval_sec"))
        duration_text = self._format_optional_seconds(runtime_defaults.get("min_order_duration_sec"))
        fake_customer_text = "开启" if bool(runtime_defaults.get("fake_customer_enabled", True)) else "关闭"
        order_guard_text = "开启" if bool(runtime_defaults.get("fake_customer_order_guard_enabled", True)) else "关闭"
        self._cafe_limit_hint_label.setText(
            f"{default_text}；订单间隔：{interval_text}；单订单最短耗时：{duration_text}；"
            f"假顾客驱赶：{fake_customer_text}；订单守卫：{order_guard_text}。"
            "最大运行秒数填 0 表示不按时间停止；最大订单数填 0 表示不按订单数停止。"
        )

    @staticmethod
    def _format_optional_seconds(value: Any) -> str:
        if value is None:
            return "未配置"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number <= 0:
            return "不限制"
        return f"{number:.3f} 秒"

    def _setup_bridge(self) -> None:
        self._bridge_thread = QThread(self)
        self._bridge = RunnerBridge(self._ui_preferences)
        self._bridge.moveToThread(self._bridge_thread)
        self._bridge_thread.finished.connect(self._bridge.deleteLater)

        self.request_initialize.connect(self._bridge.initialize, Qt.QueuedConnection)
        self.request_run_task.connect(self._bridge.run_task, Qt.QueuedConnection)
        self.request_cancel_task.connect(self._bridge.cancel_task, Qt.QueuedConnection)
        self.request_fetch_run_detail.connect(self._bridge.fetch_run_detail, Qt.QueuedConnection)
        self.request_refresh_history.connect(self._bridge.refresh_history, Qt.QueuedConnection)
        self.request_refresh_plan_info.connect(self._bridge.refresh_plan_info, Qt.QueuedConnection)
        self.request_refresh_runtime_probe.connect(self._bridge.refresh_runtime_probe, Qt.QueuedConnection)
        self.request_refresh_doctor.connect(self._bridge.refresh_doctor, Qt.QueuedConnection)
        self.request_apply_preferences.connect(self._bridge.apply_preferences, Qt.QueuedConnection)

        self._bridge.status_changed.connect(self._on_runner_status_changed)
        self._bridge.tasks_loaded.connect(self._on_tasks_loaded)
        self._bridge.history_loaded.connect(self._on_history_loaded)
        self._bridge.event_batch_received.connect(self._on_event_batch_received)
        self._bridge.task_dispatched.connect(self._on_task_dispatched)
        self._bridge.run_detail_ready.connect(self._on_run_detail_ready)
        self._bridge.plan_info_ready.connect(self._on_plan_info_ready)
        self._bridge.runtime_probe_ready.connect(self._on_runtime_probe_ready)
        self._bridge.doctor_ready.connect(self._on_doctor_ready)
        self._bridge.live_state_changed.connect(self._on_live_state_changed)
        self._bridge.control_message.connect(self._on_bridge_control_message)
        self._bridge.error_occurred.connect(self._on_bridge_error)

        self._bridge_thread.start()
        self.request_initialize.emit()

    def _open_map_overlay(self) -> None:
        if self._map_overlay_controller is None:
            self._map_overlay_controller = MapOverlayController(
                self._settings_store,
                game_name=GAME_NAME,
                parent=self,
            )
        self._map_overlay_controller.show()

    def _restore_window_settings(self) -> None:
        geometry = self._settings_store.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = self._settings_store.value("window/state")
        if state:
            self.restoreState(state)
        page_index = int(self._settings_store.value("window/page_index", 0))
        if 0 <= page_index < self._pages.count():
            self._pages.setCurrentIndex(page_index)
        task_id = str(self._settings_store.value("window/selected_task_id", "fishing") or "fishing")
        self._select_task_id(task_id if task_id in WORKBENCH_TASKS else "fishing")

    def _persist_window_settings(self) -> None:
        self._settings_store.setValue("window/geometry", self.saveGeometry())
        self._settings_store.setValue("window/state", self.saveState())
        self._settings_store.setValue("window/page_index", self._pages.currentIndex())
        self._settings_store.setValue("window/selected_task_id", self._selected_task_id)

    def _select_task_id(self, task_id: str) -> None:
        for index in range(self._task_list.count()):
            item = self._task_list.item(index)
            if str(item.data(Qt.UserRole) or "") == task_id:
                self._task_list.setCurrentItem(item)
                return

    def _on_task_item_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        task_id = str(current.data(Qt.UserRole) or "fishing")
        if task_id not in WORKBENCH_TASKS:
            task_id = "fishing"
        self._selected_task_id = task_id
        spec = WORKBENCH_TASKS[task_id]
        self._task_title_label.setText(spec["label"])
        self._task_description_label.setText(spec["description"])
        self._parameter_stack.setCurrentWidget(self._parameter_pages[task_id])
        self._apply_task_guard()

    def _request_start_selected_task(self) -> None:
        if self._pending_launch is not None:
            self._append_log("已有任务处于倒计时等待中，请先停止或等待其开始。", level="warning")
            return
        active_cid, _active_run = self._active_runtime_run()
        if active_cid:
            self._append_log("已有会操作游戏窗口的任务正在运行，无法同时启动新任务。", level="warning")
            return
        if not self._runner_ready:
            QMessageBox.warning(self, "框架尚未就绪", "Aura 框架仍在后台启动，请稍后再执行任务。")
            return

        task_ref = self._selected_task_ref()
        if not self._task_available(task_ref):
            QMessageBox.warning(self, "任务缺失", f"当前工作区没有找到任务：{task_display_name(task_ref)}")
            return

        inputs = self._collect_selected_task_inputs()
        delay_sec = int(self._ui_preferences.task_start_delay_sec)
        if delay_sec <= 0:
            self._append_log(f"正在提交任务：{task_display_name(task_ref)}")
            self._dispatch_task(task_ref, inputs)
            return

        self._pending_launch = PendingLaunch(
            task_ref=task_ref,
            inputs=inputs,
            selected_task_id=self._selected_task_id,
            remaining_sec=delay_sec,
            total_sec=delay_sec,
        )
        self._pending_timer.start()
        self._append_log(
            f"{task_display_name(task_ref)} 将在 {delay_sec} 秒后开始，可按 "
            f"{self._ui_preferences.quick_stop_hotkey} 或点击停止取消。"
        )
        self.statusBar().showMessage(f"任务将在 {delay_sec} 秒后开始。", 5000)
        self._apply_task_guard()

    def _request_refresh_runtime_probe_from_ui(self) -> None:
        if self._pending_launch is not None or self._active_runtime_run()[0]:
            self._append_log("已有会操作游戏窗口的任务正在运行或等待启动，暂不能刷新运行时探针。", level="warning")
            return
        self.request_refresh_runtime_probe.emit()

    def _on_pending_timer_tick(self) -> None:
        if self._pending_launch is None:
            self._pending_timer.stop()
            return
        remaining = self._pending_launch.remaining_sec - 1
        self._pending_launch = PendingLaunch(
            task_ref=self._pending_launch.task_ref,
            inputs=self._pending_launch.inputs,
            selected_task_id=self._pending_launch.selected_task_id,
            remaining_sec=remaining,
            total_sec=self._pending_launch.total_sec,
        )
        if remaining <= 0:
            self._dispatch_pending_launch()
            return
        self._append_log(f"倒计时：{remaining} 秒后开始执行 {task_display_name(self._pending_launch.task_ref)}。")
        self.statusBar().showMessage(f"倒计时：{remaining} 秒后开始。", 1500)
        self._apply_task_guard()

    def _dispatch_pending_launch(self) -> None:
        if self._pending_launch is None:
            return
        pending = self._pending_launch
        self._pending_launch = None
        self._pending_timer.stop()
        self._append_log(f"倒计时结束，正在提交任务：{task_display_name(pending.task_ref)}")
        self._dispatch_task(pending.task_ref, pending.inputs)
        self._apply_task_guard()

    def _dispatch_task(self, task_ref: str, inputs: dict[str, Any]) -> None:
        self._settings_store.setValue("window/last_task_ref", task_ref)
        self.statusBar().showMessage(f"正在提交任务：{task_display_name(task_ref)}", 5000)
        self.request_run_task.emit(task_ref, inputs)

    def _request_stop_current_task(self) -> None:
        if self._pending_launch is not None:
            task_name = task_display_name(self._pending_launch.task_ref)
            self._pending_timer.stop()
            self._pending_launch = None
            self._append_log(f"已取消待启动任务：{task_name}")
            self.statusBar().showMessage("已取消待启动任务。", 4000)
            self._apply_task_guard()
            return

        active_cid, active_run = self._active_runtime_run()
        cid = active_cid or self._current_active_cid
        if cid:
            task_name = task_display_name((active_run or {}).get("task_name") or self._cid_to_task_ref.get(cid))
            self._append_log(f"正在请求停止任务：{task_name} / {cid}")
            self.statusBar().showMessage("正在请求停止任务。", 5000)
            self.request_cancel_task.emit(cid)
            return

        self._append_log("当前没有可停止的任务。", level="warning")
        self.statusBar().showMessage("当前没有可停止的任务。", 4000)

    def _selected_task_ref(self) -> str:
        return WORKBENCH_TASKS[self._selected_task_id]["task_ref"]

    def _task_available(self, task_ref: str) -> bool:
        return task_ref in self._task_rows

    def _collect_selected_task_inputs(self) -> dict[str, Any]:
        if self._selected_task_id == "fishing":
            return build_auto_loop_inputs(self._max_rounds_spin.value(), self._fishing_defaults)
        if self._selected_task_id == "cafe":
            return build_cafe_loop_inputs(
                self._cafe_max_seconds_spin.value(),
                self._cafe_max_orders_spin.value(),
                self._cafe_start_game_check.isChecked(),
                self._cafe_wait_level_started_check.isChecked(),
                self._cafe_full_assist_auto_hammer_check.isChecked(),
                self._cafe_defaults,
            )
        raise ValueError(f"未知任务：{self._selected_task_id}")

    def _active_runtime_run(self) -> tuple[str | None, dict[str, Any] | None]:
        active_runs = self._live_state.get("active_runs") or {}
        for cid, run in active_runs.items():
            task_name = str((run or {}).get("task_name") or "").strip()
            if is_runtime_interacting_task(task_name):
                return str(cid), dict(run)
        return None, None

    def _apply_task_guard(self) -> None:
        active_cid, active_run = self._active_runtime_run()
        active_runs = self._live_state.get("active_runs") or {}
        task_ref = self._selected_task_ref()
        task_available = self._task_available(task_ref)
        start_allowed = (
            self._runner_ready
            and self._tasks_ready
            and task_available
            and self._pending_launch is None
            and not active_cid
            and task_is_enabled(task_ref, active_runs)
        )
        self._start_button.setEnabled(start_allowed)
        self._stop_button.setEnabled(self._pending_launch is not None or bool(active_cid or self._current_active_cid))
        if hasattr(self, "_refresh_probe_action"):
            self._refresh_probe_action.setEnabled(self._pending_launch is None and not active_cid)

        if self._pending_launch is not None:
            self._start_button.setText(f"等待启动（{self._pending_launch.remaining_sec}s）")
            status = f"倒计时中：{self._pending_launch.remaining_sec} 秒后执行"
        else:
            self._start_button.setText("开始执行")
            if not self._runner_ready:
                status = "框架启动中"
            elif not self._tasks_ready:
                status = "正在加载任务列表"
            elif not task_available:
                status = "任务缺失"
            elif active_cid:
                status = f"运行中：{task_display_name((active_run or {}).get('task_name'))}"
            else:
                status = "空闲"
        self._selected_task_status_label.setText(status)
        self._refresh_task_list_labels()

    def _refresh_task_list_labels(self) -> None:
        active_cid, active_run = self._active_runtime_run()
        active_task = str((active_run or {}).get("task_name") or "")
        for index in range(self._task_list.count()):
            item = self._task_list.item(index)
            task_id = str(item.data(Qt.UserRole) or "")
            spec = WORKBENCH_TASKS.get(task_id)
            if not spec:
                continue
            task_ref = spec["task_ref"]
            if self._pending_launch and self._pending_launch.task_ref == task_ref:
                state = "等待启动"
            elif active_cid and active_task == task_ref:
                state = "运行中"
            elif task_ref not in self._task_rows:
                state = "任务缺失"
            else:
                state = "可用"
            item.setText(f"{spec['label']}\n{state}")

    def _on_runner_status_changed(self, status: dict[str, Any]) -> None:
        ready = bool((status or {}).get("ready"))
        self._runner_ready = ready
        message = str((status or {}).get("message") or "")
        if ready:
            self._runner_label.setText("框架就绪")
        elif message:
            self._runner_label.setText(message)
        else:
            self._runner_label.setText("框架已停止")
        self._admin_label.setText("是" if is_running_as_admin() else "否")
        self._apply_task_guard()

    def _on_tasks_loaded(self, tasks: list[dict[str, Any]]) -> None:
        self._task_rows = {str(task.get("task_ref") or ""): dict(task) for task in tasks}
        self._tasks_ready = True
        self._fishing_defaults = self._repo.get_fishing_defaults(self._task_rows.get(TASK_AUTO_LOOP))
        self._cafe_defaults = self._repo.get_cafe_defaults(self._task_rows.get(TASK_CAFE_AUTO_LOOP))
        self._fishing_profile_label.setText(self._fishing_defaults.profile_name)
        self._cafe_profile_label.setText(self._cafe_defaults.profile_name)
        self._cafe_max_seconds_spin.setValue(int(self._cafe_defaults.max_seconds))
        self._cafe_max_orders_spin.setValue(int(self._cafe_defaults.max_orders))
        self._cafe_start_game_check.setChecked(bool(self._cafe_defaults.start_game))
        self._cafe_wait_level_started_check.setChecked(bool(self._cafe_defaults.wait_level_started))
        self._cafe_full_assist_auto_hammer_check.setChecked(bool(self._cafe_defaults.full_assist_auto_hammer_mode))
        self._refresh_cafe_limit_hint()
        self._append_log(f"任务列表已加载：{len(self._task_rows)} 个异环任务。")
        self._apply_task_guard()

    def _on_history_loaded(self, rows: list[dict[str, Any]]) -> None:
        previous_cid = self._current_history_cid
        self._history_list.clear()
        filtered_rows = [
            dict(row)
            for row in rows
            if str(row.get("task_name") or "").strip() in VISIBLE_HISTORY_TASK_REFS
        ]
        self._history_rows = {str(row.get("cid") or ""): dict(row) for row in filtered_rows}
        for row in filtered_rows:
            cid = str(row.get("cid") or "")
            if not cid:
                continue
            item = QListWidgetItem(history_row_label(row))
            item.setData(Qt.UserRole, cid)
            self._history_list.addItem(item)
            if previous_cid and previous_cid == cid:
                self._history_list.setCurrentItem(item)
        if self._history_list.count() and self._history_list.currentItem() is None:
            self._history_list.setCurrentRow(0)

    def _on_history_item_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            self._current_history_cid = None
            return
        cid = str(current.data(Qt.UserRole) or "").strip()
        self._current_history_cid = cid
        cached = self._detail_cache.get(cid) or self._history_rows.get(cid)
        if cached and "final_result" in cached:
            self._detail_viewer.show_detail(cached)
            return
        self.request_fetch_run_detail.emit(cid)

    def _on_event_batch_received(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            name = str(event.get("name") or "")
            payload = dict(event.get("payload") or {})
            cid = str(payload.get("cid") or "").strip()
            task_name = str(payload.get("task_name") or "").strip()
            pieces = [event_display_name(name)]
            if task_name:
                pieces.append(task_display_name(task_name))
            if cid:
                pieces.append(cid)
            if payload.get("final_status") or payload.get("status"):
                pieces.append(status_display_name(payload.get("final_status") or payload.get("status")))
            if payload.get("node_id"):
                pieces.append(str(payload["node_id"]))
            self._append_log(" / ".join(pieces))

    def _on_task_dispatched(self, dispatch: dict[str, Any]) -> None:
        cid = str(dispatch.get("cid") or "").strip()
        task_ref = str(dispatch.get("_task_ref") or "").strip()
        if cid and task_ref:
            self._cid_to_task_ref[cid] = task_ref
            if is_runtime_interacting_task(task_ref):
                self._current_active_cid = cid
            if task_ref in WORKBENCH_TASK_REFS:
                self._last_business_cids[task_ref] = cid
        self._append_log(f"{task_display_name(task_ref)} 已加入执行队列：{cid or '-'}")
        self.statusBar().showMessage(f"{task_display_name(task_ref)} 已加入执行队列。", 5000)
        self._apply_task_guard()

    def _on_run_detail_ready(self, cid: str, detail: dict[str, Any]) -> None:
        payload = dict(detail)
        task_ref = str(self._cid_to_task_ref.get(cid) or payload.get("task_name") or "")
        if task_ref:
            payload["task_name"] = task_ref
        self._detail_cache[cid] = payload
        self._history_rows[cid] = payload
        if task_ref in WORKBENCH_TASK_REFS or self._current_history_cid == cid:
            self._detail_viewer.show_detail(payload)

        if self._current_history_cid == cid:
            self._detail_viewer.show_detail(payload)

        if task_ref == TASK_AUTO_LOOP:
            self._append_log(f"钓鱼结果：{render_auto_loop_brief_text(payload)}")
        elif task_ref == TASK_CAFE_AUTO_LOOP:
            self._append_log(f"沙威玛结果：{render_cafe_loop_brief_text(payload)}")

        final_status = str(
            auto_loop_business_status(payload)
            or cafe_loop_business_status(payload)
            or payload.get("status")
            or ""
        ).lower()
        if final_status and final_status != "success":
            self._last_error_label.setText(f"最近错误：{task_display_name(task_ref)}：{status_display_name(final_status)}")
        if cid == self._current_active_cid:
            self._current_active_cid = None
        self._apply_task_guard()

    def _on_plan_info_ready(self, info: dict[str, Any]) -> None:
        self._append_log("异环方案包信息已刷新。")
        if not self._detail_cache and not self._history_rows:
            self._detail_viewer.show_detail(
                {
                    "cid": "-",
                    "task_name": TASK_PLAN_READY,
                    "status": "success",
                    "final_result": {"user_data": {"info": info}},
                }
            )

    def _on_runtime_probe_ready(self, probe: dict[str, Any]) -> None:
        status_text = "正常" if probe.get("ok") else "未就绪"
        self._runtime_summary_label.setText(f"运行时探针：{status_text} / {probe.get('provider') or '-'}")
        self._append_log(f"运行时探针：{status_text}。")

    def _on_doctor_ready(self, doctor: dict[str, Any]) -> None:
        self._append_log("全局诊断数据已刷新。")
        self._detail_viewer.show_detail(
            {
                "cid": "-",
                "task_name": "全局诊断",
                "status": "success",
                "final_result": {"user_data": doctor},
            }
        )

    def _on_live_state_changed(self, state: dict[str, Any]) -> None:
        self._live_state = dict(state)
        self._last_event_label.setText(f"最近事件：{event_display_name(state.get('last_event_name') or '-')}")
        if state.get("last_error"):
            self._last_error_label.setText(f"最近错误：{state['last_error']}")
        self._apply_task_guard()

    def _on_bridge_control_message(self, payload: dict[str, Any]) -> None:
        message = str((payload or {}).get("message") or payload or "")
        if message:
            self._append_log(message, level=str((payload or {}).get("level") or "info"))
            self.statusBar().showMessage(message, 5000)

    def _on_bridge_error(self, payload: dict[str, Any]) -> None:
        title = str(payload.get("title") or "异环 GUI 错误")
        message = str(payload.get("message") or "发生未知错误。")
        self._last_error_label.setText(f"最近错误：{message}")
        self._append_log(f"{title}：{message}", level="error")
        self.statusBar().showMessage(message, 8000)
        if payload.get("kind") == "startup":
            self._runner_label.setText("框架启动失败")
            QMessageBox.critical(self, title, message)
        self._apply_task_guard()

    def _append_log(self, message: str, *, level: str = "info") -> None:
        prefix = {
            "warning": "警告",
            "error": "错误",
            "info": "信息",
        }.get(level, "信息")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{timestamp}] [{prefix}] {message}")

    def _install_quick_stop_hotkey(self, key: str, *, show_warning: bool) -> None:
        normalized = str(key or "F8").strip().upper()
        if self._quick_stop_shortcut is None:
            self._quick_stop_shortcut = QShortcut(QKeySequence(normalized), self)
            self._quick_stop_shortcut.setContext(Qt.ApplicationShortcut)
            self._quick_stop_shortcut.activated.connect(self._request_stop_current_task)
        else:
            self._quick_stop_shortcut.setKey(QKeySequence(normalized))

        ok, message = self._hotkey_manager.register(normalized)
        self._stop_button.setText(f"停止任务（{normalized}）")
        if ok:
            self._append_log(message)
            self._hotkey_warning_shown = False
            return
        self._append_log(f"{message} 已启用窗口内快捷键作为降级方案。", level="warning")
        if show_warning or not self._hotkey_warning_shown:
            self._hotkey_warning_shown = True
            QMessageBox.warning(self, "快捷停止键", f"{message}\n\n已启用窗口内快捷键作为降级方案。")

    def nativeEvent(self, event_type, message):  # noqa: N802
        try:
            message_ptr = int(message)
        except (TypeError, ValueError):
            return super().nativeEvent(event_type, message)
        if self._hotkey_manager.matches_native_message(message_ptr):
            self._request_stop_current_task()
            return True, 0
        return super().nativeEvent(event_type, message)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._pending_launch is not None:
            self._pending_timer.stop()
            self._pending_launch = None
        active_cid, _active_run = self._active_runtime_run()
        if active_cid:
            answer = QMessageBox.question(
                self,
                "关闭 AURA 控制台",
                (
                    "当前仍有会操作游戏窗口的任务在运行。\n\n"
                    "建议先按快捷停止键或点击“停止任务”。仍然要关闭控制台吗？"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return

        self._persist_window_settings()
        self._hotkey_manager.unregister()
        if self._map_overlay_controller is not None:
            self._map_overlay_controller.close()
            self._map_overlay_controller = None
        if hasattr(self, "_bridge") and hasattr(self, "_bridge_thread"):
            try:
                QMetaObject.invokeMethod(self._bridge, "shutdown", Qt.BlockingQueuedConnection)
            except Exception:
                pass
            self._bridge_thread.quit()
            self._bridge_thread.wait(3000)
        super().closeEvent(event)


def launch_yihuan_gui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("Aura")
    app.setApplicationName("YihuanGui")
    app.setStyle("Fusion")

    try:
        ensure_admin_startup("异环 GUI")
    except AdminPrivilegeRequiredError:
        QMessageBox.critical(
            None,
            "需要管理员权限",
            (
                "异环 GUI 必须在 Windows 管理员权限下启动。\n"
                "请使用“以管理员身份运行”重新启动 Codex、Python 或 Aura CLI。"
            ),
        )
        return 1

    window = YihuanMainWindow()
    window.show()
    return app.exec()

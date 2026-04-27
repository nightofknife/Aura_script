from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from PySide6.QtCore import QMetaObject, QSettings, Qt, QThread, Signal
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
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
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
from .config_repository import YihuanConfigRepository
from .logic import (
    CafeRunDefaults,
    DEVELOPER_TASK_REFS,
    FishingRunDefaults,
    GAME_NAME,
    GuiPreferences,
    RuntimeSettings,
    TASK_AUTO_LOOP,
    TASK_CAFE_AUTO_LOOP,
    TASK_PLAN_LOADED,
    TASK_PLAN_READY,
    TASK_RUNTIME_PROBE,
    build_auto_loop_inputs,
    build_cafe_loop_inputs,
    build_settings_sections,
    auto_loop_business_status,
    cafe_loop_business_status,
    event_display_name,
    extract_auto_loop_defaults,
    format_event_stream,
    format_nodes_timeline,
    history_row_label,
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


class YihuanMainWindow(QMainWindow):
    request_initialize = Signal()
    request_run_task = Signal(str, object)
    request_fetch_run_detail = Signal(str)
    request_refresh_history = Signal()
    request_refresh_plan_info = Signal()
    request_refresh_runtime_probe = Signal()
    request_refresh_doctor = Signal()
    request_apply_preferences = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AURA控制台")
        self.resize(1440, 960)

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
        self._latest_plan_info: dict[str, Any] = {}
        self._latest_runtime_probe: dict[str, Any] = {}
        self._latest_doctor: dict[str, Any] = {}
        self._live_state: dict[str, Any] = {"active_runs": {}}
        self._current_fishing_cid: str | None = None
        self._current_cafe_cid: str | None = None
        self._current_history_cid: str | None = None
        self._auto_loop_available = False
        self._cafe_auto_loop_available = False

        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        self._tabs = QTabWidget(central)
        central_layout.addWidget(self._tabs)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))

        self._overview_page = QWidget(self)
        self._fishing_page = QWidget(self)
        self._cafe_page = QWidget(self)
        self._history_page = QWidget(self)
        self._settings_page = QWidget(self)
        self._tabs.addTab(self._overview_page, "总览")
        self._tabs.addTab(self._fishing_page, "钓鱼")
        self._tabs.addTab(self._cafe_page, "沙威玛")
        self._tabs.addTab(self._history_page, "历史")
        self._tabs.addTab(self._settings_page, "设置")

        self._build_overview_page()
        self._build_fishing_page()
        self._build_cafe_page()
        self._build_history_page()
        self._build_settings_page()
        self._restore_window_settings()
        self._load_settings_widgets()
        self._setup_bridge()

    @staticmethod
    def _resolve_plan_root() -> Path:
        return Path(__file__).resolve().parents[2] / "plans" / "yihuan"

    def _build_overview_page(self) -> None:
        layout = QVBoxLayout(self._overview_page)

        status_group = QGroupBox("运行状态", self._overview_page)
        status_form = QFormLayout(status_group)
        self._admin_label = QLabel("-", status_group)
        self._runner_label = QLabel("启动中...", status_group)
        self._runtime_summary_label = QLabel("等待探测", status_group)
        self._runtime_summary_label.setWordWrap(True)
        self._last_error_label = QLabel("-", status_group)
        self._last_error_label.setWordWrap(True)
        self._last_event_label = QLabel("-", status_group)
        status_form.addRow("管理员权限", self._admin_label)
        status_form.addRow("运行器", self._runner_label)
        status_form.addRow("运行时探针", self._runtime_summary_label)
        status_form.addRow("最近错误", self._last_error_label)
        status_form.addRow("最近事件", self._last_event_label)
        layout.addWidget(status_group)

        task_group = QGroupBox("任务概览", self._overview_page)
        task_form = QFormLayout(task_group)
        self._task_count_label = QLabel("0", task_group)
        self._recent_status_label = QLabel("-", task_group)
        self._recent_cid_label = QLabel("-", task_group)
        task_form.addRow("任务数量", self._task_count_label)
        task_form.addRow("最近状态", self._recent_status_label)
        task_form.addRow("最近运行编号", self._recent_cid_label)
        layout.addWidget(task_group)

        plan_group = QGroupBox("异环方案包信息", self._overview_page)
        plan_layout = QVBoxLayout(plan_group)
        self._plan_info_browser = QTextBrowser(plan_group)
        plan_layout.addWidget(self._plan_info_browser)
        layout.addWidget(plan_group)

        runtime_group = QGroupBox("运行时探针", self._overview_page)
        runtime_layout = QVBoxLayout(runtime_group)
        self._runtime_probe_browser = QTextBrowser(runtime_group)
        runtime_layout.addWidget(self._runtime_probe_browser)
        refresh_probe_button = QPushButton("刷新运行时探针", runtime_group)
        refresh_probe_button.clicked.connect(self.request_refresh_runtime_probe.emit)
        runtime_layout.addWidget(refresh_probe_button)
        layout.addWidget(runtime_group)

        self._developer_group = QGroupBox("开发者工具", self._overview_page)
        self._developer_group.setCheckable(True)
        self._developer_group.setChecked(self._ui_preferences.expand_developer_tools)
        developer_layout = QVBoxLayout(self._developer_group)
        self._developer_body = QWidget(self._developer_group)
        developer_body_layout = QVBoxLayout(self._developer_body)
        plan_info_button = QPushButton("刷新方案包信息", self._developer_body)
        plan_info_button.clicked.connect(self.request_refresh_plan_info.emit)
        plan_loaded_button = QPushButton("执行方案包加载检查", self._developer_body)
        plan_loaded_button.clicked.connect(lambda: self._dispatch_task(TASK_PLAN_LOADED, {}))
        history_button = QPushButton("刷新历史记录", self._developer_body)
        history_button.clicked.connect(self.request_refresh_history.emit)
        doctor_button = QPushButton("刷新全局诊断", self._developer_body)
        doctor_button.clicked.connect(self.request_refresh_doctor.emit)
        developer_body_layout.addWidget(plan_info_button)
        developer_body_layout.addWidget(plan_loaded_button)
        developer_body_layout.addWidget(history_button)
        developer_body_layout.addWidget(doctor_button)
        self._doctor_view = QPlainTextEdit(self._developer_body)
        self._doctor_view.setReadOnly(True)
        developer_body_layout.addWidget(self._doctor_view)
        self._developer_group.toggled.connect(self._developer_body.setVisible)
        self._developer_body.setVisible(self._developer_group.isChecked())
        developer_layout.addWidget(self._developer_body)
        layout.addWidget(self._developer_group)
        layout.addStretch(1)

    def _build_fishing_page(self) -> None:
        layout = QVBoxLayout(self._fishing_page)
        splitter = QSplitter(Qt.Horizontal, self._fishing_page)
        layout.addWidget(splitter)

        left_scroll = QScrollArea(splitter)
        left_scroll.setWidgetResizable(True)
        left_body = QWidget(left_scroll)
        left_layout = QVBoxLayout(left_body)
        left_scroll.setWidget(left_body)
        splitter.addWidget(left_scroll)

        control_group = QGroupBox("自动钓鱼", left_body)
        control_form = QFormLayout(control_group)
        self._fishing_profile_label = QLabel("-", control_group)
        self._max_rounds_spin = QSpinBox(control_group)
        self._max_rounds_spin.setMinimum(0)
        self._max_rounds_spin.setMaximum(99999)
        self._max_rounds_spin.setValue(0)
        self._start_auto_loop_button = QPushButton("开始自动钓鱼", control_group)
        self._start_auto_loop_button.clicked.connect(self._start_auto_loop)
        control_form.addRow("钓鱼配置档案", self._fishing_profile_label)
        control_form.addRow("最大轮数", self._max_rounds_spin)
        control_form.addRow("", self._start_auto_loop_button)
        left_layout.addWidget(control_group)

        state_group = QGroupBox("当前状态", left_body)
        state_form = QFormLayout(state_group)
        self._fishing_task_label = QLabel("等待执行", state_group)
        self._fishing_guard_label = QLabel("空闲", state_group)
        self._fishing_last_result_label = QLabel("暂无结果", state_group)
        self._fishing_last_result_label.setWordWrap(True)
        state_form.addRow("当前任务", self._fishing_task_label)
        state_form.addRow("运行守卫", self._fishing_guard_label)
        state_form.addRow("最近结果", self._fishing_last_result_label)
        left_layout.addWidget(state_group)

        summary_group = QGroupBox("最近一次自动钓鱼摘要", left_body)
        summary_layout = QVBoxLayout(summary_group)
        self._fishing_summary_browser = QTextBrowser(summary_group)
        self._fishing_summary_browser.setHtml("<p>尚未执行自动钓鱼。</p>")
        summary_layout.addWidget(self._fishing_summary_browser)
        view_history_button = QPushButton("跳转到历史记录", summary_group)
        view_history_button.clicked.connect(self._jump_to_history)
        summary_layout.addWidget(view_history_button)
        left_layout.addWidget(summary_group)
        left_layout.addStretch(1)

        self._fishing_detail_viewer = RunDetailViewer(splitter)
        splitter.addWidget(self._fishing_detail_viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _build_cafe_page(self) -> None:
        layout = QVBoxLayout(self._cafe_page)
        splitter = QSplitter(Qt.Horizontal, self._cafe_page)
        layout.addWidget(splitter)

        left_scroll = QScrollArea(splitter)
        left_scroll.setWidgetResizable(True)
        left_body = QWidget(left_scroll)
        left_layout = QVBoxLayout(left_body)
        left_scroll.setWidget(left_body)
        splitter.addWidget(left_scroll)

        control_group = QGroupBox("沙威玛", left_body)
        control_form = QFormLayout(control_group)
        self._cafe_profile_label = QLabel("-", control_group)
        self._cafe_max_seconds_spin = QSpinBox(control_group)
        self._cafe_max_seconds_spin.setMinimum(0)
        self._cafe_max_seconds_spin.setMaximum(99999)
        self._cafe_max_seconds_spin.setToolTip("填 0 时使用当前沙威玛识别档案里的默认运行时长。")
        self._cafe_max_orders_spin = QSpinBox(control_group)
        self._cafe_max_orders_spin.setMinimum(0)
        self._cafe_max_orders_spin.setMaximum(99999)
        self._cafe_max_orders_spin.setToolTip("填 0 时不按订单数量停止。")
        self._cafe_start_game_check = QCheckBox("自动点击开始游戏", control_group)
        self._cafe_wait_level_started_check = QCheckBox("等待关卡开始", control_group)
        self._start_cafe_button = QPushButton("开始沙威玛", control_group)
        self._start_cafe_button.clicked.connect(self._start_cafe_loop)
        self._cafe_limit_hint_label = QLabel("", control_group)
        self._cafe_limit_hint_label.setWordWrap(True)
        control_form.addRow("沙威玛识别档案", self._cafe_profile_label)
        control_form.addRow("最大运行秒数（0 = 使用档案默认）", self._cafe_max_seconds_spin)
        control_form.addRow("最大订单数（0 = 不限制订单数）", self._cafe_max_orders_spin)
        control_form.addRow("说明", self._cafe_limit_hint_label)
        control_form.addRow("", self._cafe_start_game_check)
        control_form.addRow("", self._cafe_wait_level_started_check)
        control_form.addRow("", self._start_cafe_button)
        left_layout.addWidget(control_group)

        state_group = QGroupBox("当前状态", left_body)
        state_form = QFormLayout(state_group)
        self._cafe_task_label = QLabel("等待执行", state_group)
        self._cafe_guard_label = QLabel("空闲", state_group)
        self._cafe_last_result_label = QLabel("暂无结果", state_group)
        self._cafe_last_result_label.setWordWrap(True)
        state_form.addRow("当前任务", self._cafe_task_label)
        state_form.addRow("运行守卫", self._cafe_guard_label)
        state_form.addRow("最近结果", self._cafe_last_result_label)
        left_layout.addWidget(state_group)

        summary_group = QGroupBox("最近一次沙威玛摘要", left_body)
        summary_layout = QVBoxLayout(summary_group)
        self._cafe_summary_browser = QTextBrowser(summary_group)
        self._cafe_summary_browser.setHtml("<p>尚未执行沙威玛。</p>")
        summary_layout.addWidget(self._cafe_summary_browser)
        view_history_button = QPushButton("跳转到历史记录", summary_group)
        view_history_button.clicked.connect(self._jump_cafe_to_history)
        summary_layout.addWidget(view_history_button)
        left_layout.addWidget(summary_group)
        left_layout.addStretch(1)

        self._cafe_detail_viewer = RunDetailViewer(splitter)
        splitter.addWidget(self._cafe_detail_viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _build_history_page(self) -> None:
        layout = QVBoxLayout(self._history_page)
        button_row = QHBoxLayout()
        refresh_button = QPushButton("刷新历史记录", self._history_page)
        refresh_button.clicked.connect(self.request_refresh_history.emit)
        button_row.addWidget(refresh_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        splitter = QSplitter(Qt.Horizontal, self._history_page)
        layout.addWidget(splitter)
        self._history_list = QListWidget(splitter)
        self._history_list.currentItemChanged.connect(self._on_history_item_changed)
        splitter.addWidget(self._history_list)
        self._history_detail_viewer = RunDetailViewer(splitter)
        splitter.addWidget(self._history_detail_viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _build_settings_page(self) -> None:
        layout = QVBoxLayout(self._settings_page)

        runtime_group = QGroupBox("运行环境", self._settings_page)
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
        layout.addWidget(runtime_group)

        fishing_group = QGroupBox("钓鱼设置", self._settings_page)
        fishing_form = QFormLayout(fishing_group)
        self._fishing_profile_combo = QComboBox(fishing_group)
        fishing_form.addRow("钓鱼识别档案", self._fishing_profile_combo)
        layout.addWidget(fishing_group)

        cafe_group = QGroupBox("沙威玛设置", self._settings_page)
        cafe_form = QFormLayout(cafe_group)
        self._cafe_profile_combo = QComboBox(cafe_group)
        cafe_form.addRow("沙威玛识别档案", self._cafe_profile_combo)
        layout.addWidget(cafe_group)

        ui_group = QGroupBox("界面偏好", self._settings_page)
        ui_form = QFormLayout(ui_group)
        self._history_limit_spin = QSpinBox(ui_group)
        self._history_limit_spin.setMinimum(1)
        self._history_limit_spin.setMaximum(500)
        self._auto_probe_check = QCheckBox("启动时自动执行运行时探针", ui_group)
        self._expand_developer_tools_check = QCheckBox("默认展开开发者工具", ui_group)
        ui_form.addRow("历史记录显示条数", self._history_limit_spin)
        ui_form.addRow("", self._auto_probe_check)
        ui_form.addRow("", self._expand_developer_tools_check)
        layout.addWidget(ui_group)

        action_row = QHBoxLayout()
        self._reload_settings_button = QPushButton("重新加载设置", self._settings_page)
        self._reload_settings_button.clicked.connect(self._load_settings_widgets)
        self._save_settings_button = QPushButton("保存设置", self._settings_page)
        self._save_settings_button.clicked.connect(self._save_settings)
        action_row.addWidget(self._reload_settings_button)
        action_row.addWidget(self._save_settings_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self._settings_hint_label = QLabel(
            "设置页只负责运行环境和界面偏好；钓鱼任务的具体执行参数不在这里维护。",
            self._settings_page,
        )
        self._settings_hint_label.setWordWrap(True)
        layout.addWidget(self._settings_hint_label)
        layout.addStretch(1)

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
            [field.value for field in runtime_section.fields if field.key == "runtime.capture.backend"][0],
            ["gdi", "dxgi", "wgc", "printwindow"],
        )
        self._set_combo_items(
            self._input_backend_combo,
            [field.value for field in runtime_section.fields if field.key == "runtime.input.backend"][0],
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
        self._refresh_cafe_limit_hint()

        self._history_limit_spin.setValue(int(ui_map["gui.history_limit"].value))
        self._auto_probe_check.setChecked(bool(ui_map["gui.auto_runtime_probe_on_startup"].value))
        self._expand_developer_tools_check.setChecked(bool(ui_map["gui.expand_developer_tools"].value))

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
        hammer_debug_text = "开启" if bool(runtime_defaults.get("fake_customer_hammer_debug_enabled", True)) else "关闭"
        self._cafe_limit_hint_label.setText(
            f"{default_text}；订单间隔：{interval_text}；单订单最短耗时：{duration_text}；"
            f"假顾客驱赶：{fake_customer_text}；锤击调试图：{hammer_debug_text}。"
            "最大运行秒数填 0 会使用档案默认值；最大订单数填 0 表示不限制订单数。"
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
        self.request_fetch_run_detail.connect(self._bridge.fetch_run_detail, Qt.QueuedConnection)
        self.request_refresh_history.connect(self._bridge.refresh_history, Qt.QueuedConnection)
        self.request_refresh_plan_info.connect(self._bridge.refresh_plan_info, Qt.QueuedConnection)
        self.request_refresh_runtime_probe.connect(self._bridge.refresh_runtime_probe, Qt.QueuedConnection)
        self.request_refresh_doctor.connect(self._bridge.refresh_doctor, Qt.QueuedConnection)
        self.request_apply_preferences.connect(self._bridge.apply_preferences, Qt.QueuedConnection)

        self._bridge.status_changed.connect(self._on_runner_status_changed)
        self._bridge.tasks_loaded.connect(self._on_tasks_loaded)
        self._bridge.history_loaded.connect(self._on_history_loaded)
        self._bridge.task_dispatched.connect(self._on_task_dispatched)
        self._bridge.run_detail_ready.connect(self._on_run_detail_ready)
        self._bridge.plan_info_ready.connect(self._on_plan_info_ready)
        self._bridge.runtime_probe_ready.connect(self._on_runtime_probe_ready)
        self._bridge.doctor_ready.connect(self._on_doctor_ready)
        self._bridge.live_state_changed.connect(self._on_live_state_changed)
        self._bridge.error_occurred.connect(self._on_bridge_error)

        self._bridge_thread.start()
        self.request_initialize.emit()

    def _restore_window_settings(self) -> None:
        geometry = self._settings_store.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = self._settings_store.value("window/state")
        if state:
            self.restoreState(state)
        tab_index = int(self._settings_store.value("window/tab_index", 0))
        if 0 <= tab_index < self._tabs.count():
            self._tabs.setCurrentIndex(tab_index)

    def _persist_window_settings(self) -> None:
        self._settings_store.setValue("window/geometry", self.saveGeometry())
        self._settings_store.setValue("window/state", self.saveState())
        self._settings_store.setValue("window/tab_index", self._tabs.currentIndex())

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
            expand_developer_tools=self._expand_developer_tools_check.isChecked(),
        )
        fishing_defaults = FishingRunDefaults(
            profile_name=str(self._fishing_profile_combo.currentData() or self._fishing_profile_combo.currentText())
        )
        cafe_defaults = CafeRunDefaults(
            profile_name=str(self._cafe_profile_combo.currentData() or self._cafe_profile_combo.currentText()),
            max_seconds=int(self._cafe_max_seconds_spin.value()),
            max_orders=int(self._cafe_max_orders_spin.value()),
            start_game=self._cafe_start_game_check.isChecked(),
            wait_level_started=self._cafe_wait_level_started_check.isChecked(),
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
        self._fishing_defaults = fishing_defaults
        self._cafe_defaults = cafe_defaults
        self._fishing_profile_label.setText(self._fishing_defaults.profile_name)
        self._cafe_profile_label.setText(self._cafe_defaults.profile_name)
        self._refresh_cafe_limit_hint()
        self._developer_group.setChecked(ui_preferences.expand_developer_tools)
        self.request_apply_preferences.emit(ui_preferences)
        self.request_refresh_history.emit()
        self.statusBar().showMessage("设置已保存，新配置将在后续任务和探针中生效。", 6000)

    def _on_runner_status_changed(self, status: dict[str, Any]) -> None:
        ready = bool((status or {}).get("ready"))
        self._runner_label.setText("就绪" if ready else "已停止")
        self._admin_label.setText("是" if is_running_as_admin() else "否")

    def _on_tasks_loaded(self, tasks: list[dict[str, Any]]) -> None:
        self._task_rows = {str(task.get("task_ref") or ""): dict(task) for task in tasks}
        self._task_count_label.setText(str(len(self._task_rows)))
        self._auto_loop_available = TASK_AUTO_LOOP in self._task_rows
        self._cafe_auto_loop_available = TASK_CAFE_AUTO_LOOP in self._task_rows
        self._fishing_defaults = self._repo.get_fishing_defaults(self._task_rows.get(TASK_AUTO_LOOP))
        self._cafe_defaults = self._repo.get_cafe_defaults(self._task_rows.get(TASK_CAFE_AUTO_LOOP))
        self._fishing_profile_label.setText(self._fishing_defaults.profile_name)
        self._cafe_profile_label.setText(self._cafe_defaults.profile_name)
        self._cafe_max_seconds_spin.setValue(int(self._cafe_defaults.max_seconds))
        self._cafe_max_orders_spin.setValue(int(self._cafe_defaults.max_orders))
        self._cafe_start_game_check.setChecked(bool(self._cafe_defaults.start_game))
        self._cafe_wait_level_started_check.setChecked(bool(self._cafe_defaults.wait_level_started))
        self._refresh_cafe_limit_hint()
        if not self._auto_loop_available:
            self._start_auto_loop_button.setEnabled(False)
            self._fishing_guard_label.setText("未找到自动循环钓鱼任务。")
        if not self._cafe_auto_loop_available:
            self._start_cafe_button.setEnabled(False)
            self._cafe_guard_label.setText("未找到沙威玛任务。")
        self._apply_task_guard()

    def _dispatch_task(self, task_ref: str, inputs: dict[str, Any]) -> None:
        self._settings_store.setValue("window/last_task_ref", task_ref)
        self.statusBar().showMessage(f"正在提交任务：{task_display_name(task_ref)}", 5000)
        self.request_run_task.emit(task_ref, inputs)

    def _start_auto_loop(self) -> None:
        if not self._auto_loop_available:
            QMessageBox.warning(self, "无法启动自动钓鱼", "当前工作区中没有找到自动循环钓鱼任务。")
            return
        payload = build_auto_loop_inputs(self._max_rounds_spin.value(), self._fishing_defaults)
        self._dispatch_task(TASK_AUTO_LOOP, payload)

    def _start_cafe_loop(self) -> None:
        if not self._cafe_auto_loop_available:
            QMessageBox.warning(self, "无法启动沙威玛", "当前工作区中没有找到沙威玛任务。")
            return
        payload = build_cafe_loop_inputs(
            self._cafe_max_seconds_spin.value(),
            self._cafe_max_orders_spin.value(),
            self._cafe_start_game_check.isChecked(),
            self._cafe_wait_level_started_check.isChecked(),
            self._cafe_defaults,
        )
        self._dispatch_task(TASK_CAFE_AUTO_LOOP, payload)

    def _jump_to_history(self) -> None:
        self._tabs.setCurrentWidget(self._history_page)
        if self._current_fishing_cid and self._current_fishing_cid in self._history_rows:
            for index in range(self._history_list.count()):
                item = self._history_list.item(index)
                if str(item.data(Qt.UserRole) or "") == self._current_fishing_cid:
                    self._history_list.setCurrentItem(item)
                    break

    def _jump_cafe_to_history(self) -> None:
        self._tabs.setCurrentWidget(self._history_page)
        if self._current_cafe_cid and self._current_cafe_cid in self._history_rows:
            for index in range(self._history_list.count()):
                item = self._history_list.item(index)
                if str(item.data(Qt.UserRole) or "") == self._current_cafe_cid:
                    self._history_list.setCurrentItem(item)
                    break

    def _on_task_dispatched(self, dispatch: dict[str, Any]) -> None:
        cid = str(dispatch.get("cid") or "").strip()
        task_ref = str(dispatch.get("_task_ref") or "").strip()
        if cid and task_ref:
            self._cid_to_task_ref[cid] = task_ref
            if task_ref == TASK_AUTO_LOOP:
                self._current_fishing_cid = cid
            elif task_ref == TASK_CAFE_AUTO_LOOP:
                self._current_cafe_cid = cid
        self._recent_cid_label.setText(cid or "-")
        self._recent_status_label.setText(status_display_name(dispatch.get("status") or "-"))
        self.statusBar().showMessage(f"{task_display_name(task_ref)} 已加入执行队列。", 5000)

    def _on_run_detail_ready(self, cid: str, detail: dict[str, Any]) -> None:
        payload = dict(detail)
        task_ref = str(self._cid_to_task_ref.get(cid) or payload.get("task_name") or "")
        if task_ref:
            payload["task_name"] = task_ref
        self._detail_cache[cid] = payload
        self._history_rows[cid] = payload

        if self._current_history_cid == cid:
            self._history_detail_viewer.show_detail(payload)
        if self._current_fishing_cid == cid:
            self._fishing_detail_viewer.show_detail(payload)
        if self._current_cafe_cid == cid:
            self._cafe_detail_viewer.show_detail(payload)

        self._recent_status_label.setText(status_display_name(payload.get("status")))
        self._recent_cid_label.setText(cid)

        if str(payload.get("task_name") or "") == TASK_AUTO_LOOP:
            business_status = auto_loop_business_status(payload)
            if business_status:
                self._recent_status_label.setText(status_display_name(business_status))
            self._fishing_last_result_label.setText(render_auto_loop_brief_text(payload))
            self._fishing_summary_browser.setHtml(render_history_summary_html(payload))

        if str(payload.get("task_name") or "") == TASK_CAFE_AUTO_LOOP:
            business_status = cafe_loop_business_status(payload)
            if business_status:
                self._recent_status_label.setText(status_display_name(business_status))
            self._cafe_last_result_label.setText(render_cafe_loop_brief_text(payload))
            self._cafe_summary_browser.setHtml(render_history_summary_html(payload))

        final_status = str(
            auto_loop_business_status(payload)
            or cafe_loop_business_status(payload)
            or payload.get("status")
            or ""
        ).lower()
        if final_status != "success":
            self._last_error_label.setText(
                f"{task_display_name(payload.get('task_name'))}: {status_display_name(final_status)}"
            )

    def _on_plan_info_ready(self, info: dict[str, Any]) -> None:
        self._latest_plan_info = dict(info)
        self._plan_info_browser.setHtml(render_overview_plan_info_html(info))

    def _on_runtime_probe_ready(self, probe: dict[str, Any]) -> None:
        self._latest_runtime_probe = dict(probe)
        self._runtime_probe_browser.setHtml(render_runtime_probe_html(probe))
        status_text = "正常" if probe.get("ok") else "未就绪"
        self._runtime_summary_label.setText(f"{status_text} / {probe.get('provider') or '-'}")

    def _on_doctor_ready(self, doctor: dict[str, Any]) -> None:
        self._latest_doctor = dict(doctor)
        self._doctor_view.setPlainText(render_json(doctor))

    def _on_live_state_changed(self, state: dict[str, Any]) -> None:
        self._live_state = dict(state)
        self._last_event_label.setText(event_display_name(state.get("last_event_name") or "-"))
        if state.get("last_error"):
            self._last_error_label.setText(str(state["last_error"]))
        latest_cid = str(state.get("latest_cid") or "").strip()
        if latest_cid:
            self._recent_cid_label.setText(latest_cid)
        if state.get("latest_status"):
            self._recent_status_label.setText(status_display_name(state["latest_status"]))
        self._apply_task_guard()

    def _apply_task_guard(self) -> None:
        active_runs = self._live_state.get("active_runs") or {}
        fishing_enabled = self._auto_loop_available and task_is_enabled(TASK_AUTO_LOOP, active_runs)
        cafe_enabled = self._cafe_auto_loop_available and task_is_enabled(TASK_CAFE_AUTO_LOOP, active_runs)
        self._start_auto_loop_button.setEnabled(fishing_enabled)
        self._start_cafe_button.setEnabled(cafe_enabled)

        active_runtime_task = None
        for run in active_runs.values():
            task_name = str(run.get("task_name") or "").strip()
            if task_name.startswith(("tasks:fishing:", "tasks:cafe:")):
                active_runtime_task = task_name
                break

        if active_runtime_task:
            self._fishing_task_label.setText(task_display_name(active_runtime_task))
            self._fishing_guard_label.setText("已有会操作游戏窗口的任务在运行。")
            self._cafe_task_label.setText(task_display_name(active_runtime_task))
            self._cafe_guard_label.setText("已有会操作游戏窗口的任务在运行。")
        elif not self._auto_loop_available:
            self._fishing_task_label.setText("自动循环钓鱼任务缺失")
            self._fishing_guard_label.setText("无法启动")
        else:
            self._fishing_task_label.setText("等待执行")
            self._fishing_guard_label.setText("空闲")

        if not active_runtime_task:
            if not self._cafe_auto_loop_available:
                self._cafe_task_label.setText("沙威玛任务缺失")
                self._cafe_guard_label.setText("无法启动")
            else:
                self._cafe_task_label.setText("等待执行")
                self._cafe_guard_label.setText("空闲")

    def _on_history_loaded(self, rows: list[dict[str, Any]]) -> None:
        previous_cid = self._current_history_cid
        self._history_list.clear()
        self._history_rows = {str(row.get("cid") or ""): dict(row) for row in rows}
        for row in rows:
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
            self._history_detail_viewer.show_detail(None)
            return
        cid = str(current.data(Qt.UserRole) or "").strip()
        self._current_history_cid = cid
        cached = self._detail_cache.get(cid) or self._history_rows.get(cid)
        if cached and "final_result" in cached:
            self._history_detail_viewer.show_detail(cached)
            return
        self.request_fetch_run_detail.emit(cid)

    def _on_bridge_error(self, payload: dict[str, Any]) -> None:
        title = str(payload.get("title") or "异环 GUI 错误")
        message = str(payload.get("message") or "发生未知错误。")
        self._last_error_label.setText(message)
        self.statusBar().showMessage(message, 8000)
        if payload.get("kind") == "startup":
            QMessageBox.critical(self, title, message)

    def closeEvent(self, event) -> None:  # noqa: N802
        active_runs = self._live_state.get("active_runs") or {}
        if any(active_runs):
            answer = QMessageBox.question(
                self,
                "关闭异环控制台",
                (
                    "当前仍有异环任务在运行。\n\n"
                    "关闭 GUI 会同时终止后台 runner 进程，并且可能中断任务执行。\n\n"
                    "仍然要关闭控制台吗？"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return

        self._persist_window_settings()
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

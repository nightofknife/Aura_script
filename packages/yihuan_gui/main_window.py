from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QStatusBar, QStackedWidget, QWidget

from packages.aura_core.runtime.privilege import AdminPrivilegeRequiredError, ensure_admin_startup, is_running_as_admin

from .config_repository import YihuanConfigRepository
from .logic import (
    CafeRunDefaults,
    CombatRunDefaults,
    FishingRunDefaults,
    MahjongRunDefaults,
    OneCafeRunDefaults,
    PianoRunDefaults,
    RhythmRunDefaults,
    RuntimeSettings,
    TetrominoesRunDefaults,
)
from .map_overlay.models import ZeroluckMapData
from .map_overlay.settings import MapOverlaySettings
from .models import PendingLaunch
from .pages.auxiliary import AuxiliaryPageMixin
from .pages.bridge_handlers import BridgeHandlersMixin
from .pages.defaults import DefaultsSyncMixin
from .pages.settings import SettingsPageMixin
from .pages.window_state import WindowStateMixin
from .pages.workbench import WorkbenchPageMixin
from .theme import APP_STYLESHEET


class YihuanMainWindow(
    QMainWindow,
    WorkbenchPageMixin,
    AuxiliaryPageMixin,
    SettingsPageMixin,
    DefaultsSyncMixin,
    BridgeHandlersMixin,
    WindowStateMixin,
):
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
        self.setStyleSheet(APP_STYLESHEET)

        self._settings_store = QSettings("Aura", "YihuanGui")
        self._repo = YihuanConfigRepository(self._resolve_plan_root(), self._settings_store)
        self._ui_preferences = self._repo.get_ui_preferences()
        self._runtime_settings = self._repo.get_runtime_settings()
        self._fishing_defaults = self._repo.get_fishing_defaults()
        self._cafe_defaults = self._repo.get_cafe_defaults()
        self._one_cafe_defaults = self._repo.get_one_cafe_defaults()
        self._mahjong_defaults = self._repo.get_mahjong_defaults()
        self._combat_defaults = self._repo.get_combat_defaults()
        self._tetrominoes_defaults = self._repo.get_tetrominoes_defaults()
        self._rhythm_defaults = self._repo.get_rhythm_defaults()
        self._piano_defaults = self._repo.get_piano_defaults()

        self._task_rows: dict[str, dict[str, Any]] = {}
        self._history_rows: dict[str, dict[str, Any]] = {}
        self._detail_cache: dict[str, dict[str, Any]] = {}
        self._cid_to_task_ref: dict[str, str] = {}
        self._live_state: dict[str, Any] = {"active_runs": {}}
        self._runner_ready = False
        self._tasks_ready = False
        self._selected_task_id = "fishing"
        self._selected_auxiliary_tool_id = "map_overlay"
        self._current_active_cid: str | None = None
        self._current_history_cid: str | None = None
        self._last_business_cids: dict[str, str] = {}
        self._pending_launch: PendingLaunch | None = None
        self._hotkey_warning_shown = False
        self._map_overlay_controller: MapOverlayController | None = None
        self._map_overlay_running = False
        self._map_overlay_data: ZeroluckMapData | None = None
        self._map_overlay_data_error = ""
        self._map_overlay_settings = MapOverlaySettings(enabled_categories=frozenset())
        self._map_category_checks: dict[str, QCheckBox] = {}
        self._map_settings_syncing = False
        self._map_markers_visible = True

        self._pending_timer = QTimer(self)
        self._pending_timer.setInterval(1000)
        self._pending_timer.timeout.connect(self._on_pending_timer_tick)
        self._quick_stop_shortcut: QShortcut | None = None
        from . import app as app_module

        self._hotkey_manager = app_module.WindowsGlobalHotkeyManager(self)

        self.setStatusBar(QStatusBar(self))
        self._build_menu()
        self._build_central_pages()
        self._restore_window_settings()
        self._load_settings_widgets()
        self._install_quick_stop_hotkey(self._ui_preferences.quick_stop_hotkey, show_warning=False)
        self._setup_bridge()
        self._append_log("GUI 已启动，正在后台预热 Aura 框架。")
        self._apply_task_guard()
        self._apply_auxiliary_guard()

    @staticmethod
    def _resolve_plan_root() -> Path:
        base_path = os.environ.get("AURA_BASE_PATH")
        if base_path:
            return Path(base_path).resolve() / "plans" / "yihuan"
        return Path(__file__).resolve().parents[2] / "plans" / "yihuan"

    def _build_menu(self) -> None:
        task_action = QAction("任务界面", self)
        task_action.triggered.connect(lambda: self._pages.setCurrentWidget(self._workbench_page))
        auxiliary_action = QAction("辅助功能", self)
        auxiliary_action.triggered.connect(lambda: self._pages.setCurrentWidget(self._auxiliary_page))
        settings_action = QAction("设置界面", self)
        settings_action.triggered.connect(lambda: self._pages.setCurrentWidget(self._settings_page))
        self.menuBar().addAction(task_action)
        self.menuBar().addAction(auxiliary_action)
        self.menuBar().addAction(settings_action)

    def _build_central_pages(self) -> None:
        self._pages = QStackedWidget(self)
        self.setCentralWidget(self._pages)
        self._workbench_page = QWidget(self)
        self._auxiliary_page = QWidget(self)
        self._settings_page = QWidget(self)
        self._pages.addWidget(self._workbench_page)
        self._pages.addWidget(self._auxiliary_page)
        self._pages.addWidget(self._settings_page)
        self._build_workbench_page()
        self._build_auxiliary_page()
        self._build_settings_page()


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

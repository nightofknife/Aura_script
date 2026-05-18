from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QMessageBox

from packages.aura_core.runtime.privilege import is_running_as_admin

from ..config_repository import QUICK_STOP_HOTKEY_OPTIONS
from ..logic import is_runtime_interacting_task, task_display_name
from ..task_specs import AUXILIARY_TOOLS, WORKBENCH_TASKS


class WindowStateMixin:
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
        aux_id = str(self._settings_store.value("window/selected_auxiliary_tool_id", "map_overlay") or "map_overlay")
        self._select_auxiliary_tool_id(aux_id if aux_id in AUXILIARY_TOOLS else "map_overlay")

    def _persist_window_settings(self) -> None:
        self._settings_store.setValue("window/geometry", self.saveGeometry())
        self._settings_store.setValue("window/state", self.saveState())
        self._settings_store.setValue("window/page_index", self._pages.currentIndex())
        self._settings_store.setValue("window/selected_task_id", self._selected_task_id)
        self._settings_store.setValue("window/selected_auxiliary_tool_id", self._selected_auxiliary_tool_id)

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

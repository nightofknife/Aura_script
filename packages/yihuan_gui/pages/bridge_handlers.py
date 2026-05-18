from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import QMetaObject, Qt, QThread
from PySide6.QtWidgets import QListWidgetItem

from packages.aura_core.runtime.privilege import is_running_as_admin

from ..logic import (
    TASK_AUTO_LOOP,
    TASK_CAFE_AUTO_LOOP,
    TASK_COMBAT_AUTO_LOOP,
    TASK_MAHJONG_AUTO_LOOP,
    TASK_ONE_CAFE_REVENUE_RESTOCK,
    TASK_PIANO_PLAY_MIDI,
    TASK_PLAN_READY,
    TASK_RHYTHM_AUTO_LOOP,
    TASK_RUNTIME_PROBE,
    TASK_TETROMINOES_AUTO_LOOP,
    VISIBLE_HISTORY_TASK_REFS,
    auto_loop_business_status,
    cafe_loop_business_status,
    combat_loop_business_status,
    event_display_name,
    history_row_label,
    is_runtime_interacting_task,
    mahjong_loop_business_status,
    one_cafe_business_status,
    piano_play_midi_business_status,
    render_auto_loop_brief_text,
    render_cafe_loop_brief_text,
    render_combat_loop_brief_text,
    render_mahjong_loop_brief_text,
    render_one_cafe_brief_text,
    render_piano_play_midi_brief_text,
    render_rhythm_loop_brief_text,
    render_tetrominoes_loop_brief_text,
    rhythm_loop_business_status,
    status_display_name,
    task_display_name,
    tetrominoes_loop_business_status,
)
from ..task_specs import WORKBENCH_TASK_REFS


class BridgeHandlersMixin:
    def _setup_bridge(self) -> None:
        self._bridge_thread = QThread(self)
        from .. import app as app_module

        self._bridge = app_module.RunnerBridge(self._ui_preferences)
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
        self._one_cafe_defaults = self._repo.get_one_cafe_defaults(
            self._task_rows.get(TASK_ONE_CAFE_REVENUE_RESTOCK)
        )
        self._mahjong_defaults = self._repo.get_mahjong_defaults(self._task_rows.get(TASK_MAHJONG_AUTO_LOOP))
        self._combat_defaults = self._repo.get_combat_defaults(self._task_rows.get(TASK_COMBAT_AUTO_LOOP))
        self._tetrominoes_defaults = self._repo.get_tetrominoes_defaults(self._task_rows.get(TASK_TETROMINOES_AUTO_LOOP))
        self._rhythm_defaults = self._repo.get_rhythm_defaults(self._task_rows.get(TASK_RHYTHM_AUTO_LOOP))
        self._piano_defaults = self._repo.get_piano_defaults(self._task_rows.get(TASK_PIANO_PLAY_MIDI))
        self._fishing_profile_label.setText(self._fishing_defaults.profile_name)
        self._cafe_profile_label.setText(self._cafe_defaults.profile_name)
        self._cafe_max_seconds_spin.setValue(int(self._cafe_defaults.max_seconds))
        self._cafe_max_orders_spin.setValue(int(self._cafe_defaults.max_orders))
        self._cafe_start_game_check.setChecked(bool(self._cafe_defaults.start_game))
        self._cafe_wait_level_started_check.setChecked(bool(self._cafe_defaults.wait_level_started))
        self._cafe_full_assist_auto_hammer_check.setChecked(bool(self._cafe_defaults.full_assist_auto_hammer_mode))
        self._refresh_cafe_limit_hint()
        self._sync_one_cafe_widgets_from_defaults()
        self._sync_mahjong_widgets_from_defaults()
        self._sync_combat_widgets_from_defaults()
        self._sync_tetrominoes_widgets_from_defaults()
        self._sync_rhythm_widgets_from_defaults()
        self._sync_piano_widgets_from_defaults()
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
            getattr(self, "_history_detail_viewer", self._detail_viewer).show_detail(cached)
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
            getattr(self, "_history_detail_viewer", self._detail_viewer).show_detail(payload)

        if task_ref == TASK_AUTO_LOOP:
            self._append_log(f"钓鱼结果：{render_auto_loop_brief_text(payload)}")
        elif task_ref == TASK_CAFE_AUTO_LOOP:
            self._append_log(f"沙威玛结果：{render_cafe_loop_brief_text(payload)}")
        elif task_ref == TASK_ONE_CAFE_REVENUE_RESTOCK:
            self._append_log(f"一咖舍结果：{render_one_cafe_brief_text(payload)}")
        elif task_ref == TASK_MAHJONG_AUTO_LOOP:
            self._append_log(f"麻将结果：{render_mahjong_loop_brief_text(payload)}")
        elif task_ref == TASK_COMBAT_AUTO_LOOP:
            self._append_log(f"自动战斗结果：{render_combat_loop_brief_text(payload)}")
        elif task_ref == TASK_TETROMINOES_AUTO_LOOP:
            self._append_log(f"俄罗斯方块结果：{render_tetrominoes_loop_brief_text(payload)}")
        elif task_ref == TASK_RHYTHM_AUTO_LOOP:
            self._append_log(f"四键音游结果：{render_rhythm_loop_brief_text(payload)}")
        elif task_ref == TASK_PIANO_PLAY_MIDI:
            self._append_log(f"自动弹钢琴结果：{render_piano_play_midi_brief_text(payload)}")

        final_status = str(
            auto_loop_business_status(payload)
            or cafe_loop_business_status(payload)
            or one_cafe_business_status(payload)
            or mahjong_loop_business_status(payload)
            or combat_loop_business_status(payload)
            or tetrominoes_loop_business_status(payload)
            or rhythm_loop_business_status(payload)
            or piano_play_midi_business_status(payload)
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

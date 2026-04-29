from __future__ import annotations

from collections import defaultdict
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from packages.aura_game import SubprocessGameRunner

from .logic import (
    GAME_NAME,
    GuiPreferences,
    LiveUiState,
    TASK_PLAN_READY,
    TASK_RUNTIME_PROBE,
    reduce_live_events,
)


class RunnerBridge(QObject):
    """后台桥接层，避免在 UI 线程中直接调用 SDK。"""

    status_changed = Signal(object)
    tasks_loaded = Signal(object)
    history_loaded = Signal(object)
    event_batch_received = Signal(object)
    live_state_changed = Signal(object)
    task_dispatched = Signal(object)
    run_detail_ready = Signal(str, object)
    plan_info_ready = Signal(object)
    runtime_probe_ready = Signal(object)
    doctor_ready = Signal(object)
    control_message = Signal(object)
    error_occurred = Signal(object)

    def __init__(self, preferences: GuiPreferences) -> None:
        super().__init__()
        self._runner: SubprocessGameRunner | None = None
        self._poll_timer: QTimer | None = None
        self._live_state = LiveUiState()
        self._event_cache: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._pending_detail_cids: set[str] = set()
        self._preferences = preferences

    @Slot()
    def initialize(self) -> None:
        try:
            self.status_changed.emit(
                {
                    "profile": "embedded_full",
                    "scheduler_initialized": False,
                    "scheduler_running": False,
                    "ready": False,
                    "message": "框架启动中",
                }
            )
            self.control_message.emit({"level": "info", "message": "正在后台预热 Aura 框架。"})
            self._runner = SubprocessGameRunner()
            status = self._runner.start()
            self.status_changed.emit(status)
            self.control_message.emit({"level": "info", "message": "Aura 框架预热完成。"})
            self.tasks_loaded.emit(self._runner.list_tasks(GAME_NAME))
            self.history_loaded.emit(self._runner.list_runs(limit=self._preferences.history_limit, game_name=GAME_NAME))

            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(250)
            self._poll_timer.timeout.connect(self.poll_events)
            self._poll_timer.start()

            self._dispatch_internal(TASK_PLAN_READY, {"scenario_tag": "gui_startup"})
            if self._preferences.auto_runtime_probe_on_startup:
                self._dispatch_internal(TASK_RUNTIME_PROBE, {})
        except Exception as exc:  # noqa: BLE001
            self._emit_error("startup", "启动异环运行器失败", str(exc))

    @Slot(object)
    def apply_preferences(self, payload: object) -> None:
        if isinstance(payload, GuiPreferences):
            self._preferences = payload
        elif isinstance(payload, dict):
            self._preferences = GuiPreferences(
                history_limit=int(payload.get("history_limit", self._preferences.history_limit)),
                auto_runtime_probe_on_startup=bool(
                    payload.get(
                        "auto_runtime_probe_on_startup",
                        self._preferences.auto_runtime_probe_on_startup,
                    )
                ),
                expand_developer_tools=bool(
                    payload.get("expand_developer_tools", self._preferences.expand_developer_tools)
                ),
                task_start_delay_sec=int(
                    payload.get("task_start_delay_sec", self._preferences.task_start_delay_sec)
                ),
                quick_stop_hotkey=str(
                    payload.get("quick_stop_hotkey", self._preferences.quick_stop_hotkey)
                ).strip()
                or self._preferences.quick_stop_hotkey,
            )

    @Slot(str, object)
    def run_task(self, task_ref: str, inputs: object) -> None:
        self._dispatch_internal(task_ref, dict(inputs or {}))

    def _dispatch_internal(self, task_ref: str, inputs: dict[str, Any]) -> None:
        if self._runner is None:
            self._emit_error("startup", "运行器不可用", "后台运行器尚未完成初始化。")
            return
        try:
            dispatch = self._runner.run_task(
                game_name=GAME_NAME,
                task_ref=task_ref,
                inputs=inputs,
                wait=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_error("task", f"提交任务失败：{task_ref}", str(exc))
            return

        dispatch = dict(dispatch)
        dispatch["_task_ref"] = task_ref
        cid = str(dispatch.get("cid") or "").strip()
        if cid:
            self._event_cache.setdefault(cid, [])
        self.task_dispatched.emit(dispatch)

    @Slot(str)
    def cancel_task(self, cid: str) -> None:
        normalized = str(cid or "").strip()
        if not normalized:
            self.control_message.emit({"level": "warning", "message": "当前没有可取消的运行任务。"})
            return
        if self._runner is None:
            self._emit_error("runtime", "取消任务失败", "后台运行器尚未完成初始化。")
            return
        try:
            result = self._runner.cancel_task(normalized)
        except Exception as exc:  # noqa: BLE001
            self._emit_error("runtime", f"取消任务失败：{normalized}", str(exc))
            return
        status = str((result or {}).get("status") or "").strip().lower()
        message = str((result or {}).get("message") or result or "已发送取消请求。")
        level = "info" if status in {"success", "ok"} else "warning"
        self.control_message.emit(
            {
                "level": level,
                "cid": normalized,
                "message": f"停止任务请求已发送：{message}",
                "result": result,
            }
        )

    @Slot()
    def poll_events(self) -> None:
        if self._runner is None:
            return
        try:
            events = self._runner.poll_events(limit=50, timeout_sec=0.0)
        except Exception as exc:  # noqa: BLE001
            self._emit_error("runtime", "拉取运行事件失败", str(exc))
            return

        accepted = [event for event in events if self._accept_event(event)]
        if not accepted:
            self._flush_pending_details()
            return

        for event in accepted:
            payload = dict(event.get("payload") or {})
            cid = str(payload.get("cid") or "").strip()
            if cid:
                self._event_cache.setdefault(cid, []).append(event)

        self.event_batch_received.emit(accepted)
        self._live_state, finished_cids = reduce_live_events(self._live_state, accepted)
        self.live_state_changed.emit(
            {
                "active_runs": {cid: dict(item) for cid, item in self._live_state.active_runs.items()},
                "last_event_name": self._live_state.last_event_name,
                "last_error": self._live_state.last_error,
                "latest_metrics": dict(self._live_state.latest_metrics),
                "latest_cid": self._live_state.latest_cid,
                "latest_status": self._live_state.latest_status,
            }
        )

        self._pending_detail_cids.update(cid for cid in finished_cids if cid)
        self._flush_pending_details()

    def _flush_pending_details(self) -> None:
        if self._runner is None or not self._pending_detail_cids:
            return

        refreshed_history = False
        for cid in list(self._pending_detail_cids):
            try:
                detail = self._runner.get_run(cid)
            except Exception:
                continue
            detail_with_events = dict(detail)
            detail_with_events["_live_events"] = list(self._event_cache.get(cid) or [])
            self.run_detail_ready.emit(cid, detail_with_events)

            task_name = str(detail.get("task_name") or "").strip()
            final_result = dict(detail.get("final_result") or {})
            user_data = final_result.get("user_data")
            if task_name == TASK_PLAN_READY and isinstance(user_data, dict):
                self.plan_info_ready.emit(dict(user_data.get("info") or {}))
            elif task_name == TASK_RUNTIME_PROBE and isinstance(user_data, dict):
                self.runtime_probe_ready.emit(dict(user_data))
            self._pending_detail_cids.discard(cid)
            refreshed_history = True

        if refreshed_history:
            self.refresh_history()

    @Slot(str)
    def fetch_run_detail(self, cid: str) -> None:
        if self._runner is None:
            return
        try:
            detail = self._runner.get_run(cid)
            detail_with_events = dict(detail)
            detail_with_events["_live_events"] = list(self._event_cache.get(cid) or [])
            self.run_detail_ready.emit(cid, detail_with_events)
        except Exception as exc:  # noqa: BLE001
            self._emit_error("runtime", f"加载运行详情失败：{cid}", str(exc))

    @Slot()
    def refresh_history(self) -> None:
        if self._runner is None:
            return
        try:
            rows = self._runner.list_runs(limit=self._preferences.history_limit, game_name=GAME_NAME)
            self.history_loaded.emit(rows)
        except Exception as exc:  # noqa: BLE001
            self._emit_error("runtime", "刷新异环历史记录失败", str(exc))

    @Slot()
    def refresh_tasks(self) -> None:
        if self._runner is None:
            return
        try:
            self.tasks_loaded.emit(self._runner.list_tasks(GAME_NAME))
        except Exception as exc:  # noqa: BLE001
            self._emit_error("runtime", "加载异环任务列表失败", str(exc))

    @Slot()
    def refresh_plan_info(self) -> None:
        self._dispatch_internal(TASK_PLAN_READY, {"scenario_tag": "gui_refresh"})

    @Slot()
    def refresh_runtime_probe(self) -> None:
        self._dispatch_internal(TASK_RUNTIME_PROBE, {})

    @Slot()
    def refresh_doctor(self) -> None:
        if self._runner is None:
            return
        try:
            self.doctor_ready.emit(self._runner.doctor(include_shared=True))
        except Exception as exc:  # noqa: BLE001
            self._emit_error("runtime", "加载全局诊断数据失败", str(exc))

    @Slot()
    def shutdown(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer.deleteLater()
            self._poll_timer = None
        if self._runner is not None:
            try:
                self._runner.close()
            except Exception:
                pass
            self._runner = None
        self.status_changed.emit(
            {
                "profile": "embedded_full",
                "scheduler_initialized": False,
                "scheduler_running": False,
                "ready": False,
            }
        )

    def _accept_event(self, event: dict[str, Any]) -> bool:
        name = str(event.get("name") or "").strip()
        payload = dict(event.get("payload") or {})
        game_name = payload.get("game_name")
        if game_name:
            return str(game_name) == GAME_NAME
        return name in {"scheduler.started", "metrics.update"}

    def _emit_error(self, kind: str, title: str, message: str) -> None:
        self.error_occurred.emit(
            {
                "kind": kind,
                "title": title,
                "message": message,
            }
        )

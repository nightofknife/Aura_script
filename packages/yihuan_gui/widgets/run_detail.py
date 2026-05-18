from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QLabel, QPlainTextEdit, QTabWidget, QTextBrowser, QVBoxLayout, QWidget

from ..logic import format_event_stream, format_nodes_timeline, render_history_summary_html, render_json, status_display_name, task_display_name


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

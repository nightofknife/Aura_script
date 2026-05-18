from __future__ import annotations


APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #f5f7fa;
    color: #1f2933;
    font-size: 13px;
}
QMenuBar {
    background: #ffffff;
    border-bottom: 1px solid #d9e0e8;
    padding: 3px 8px;
}
QMenuBar::item {
    padding: 6px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background: #edf4ff;
    color: #1d4ed8;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d9e0e8;
    border-radius: 6px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QLineEdit, QPlainTextEdit, QTextBrowser, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
    background: #ffffff;
    border: 1px solid #cfd8e3;
    border-radius: 5px;
    padding: 4px;
}
QPlainTextEdit#logView {
    background: #111827;
    color: #e5e7eb;
    border: 1px solid #1f2937;
    font-family: Consolas, "Cascadia Mono", monospace;
}
QPushButton {
    background: #eef2f7;
    border: 1px solid #cfd8e3;
    border-radius: 5px;
    padding: 6px 12px;
}
QPushButton:hover {
    background: #e5ecf5;
}
QPushButton:disabled {
    color: #9aa6b2;
    background: #edf1f5;
}
QPushButton#primaryButton {
    background: #2563eb;
    border-color: #1d4ed8;
    color: #ffffff;
    font-weight: 700;
}
QPushButton#primaryButton:hover {
    background: #1d4ed8;
}
QPushButton#dangerButton {
    background: #fff1f2;
    border-color: #fca5a5;
    color: #b91c1c;
    font-weight: 700;
}
QLabel#pageTitle {
    font-size: 24px;
    font-weight: 800;
    color: #111827;
}
QLabel#sectionTitle {
    font-size: 18px;
    font-weight: 800;
    color: #111827;
}
QLabel#mutedText {
    color: #6b7280;
}
QLabel#statusBadge {
    background: #e0f2fe;
    color: #0369a1;
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 700;
}
QLabel#categoryBadge {
    background: #111827;
    color: #ffffff;
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 800;
}
QLabel#kindBadge {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 700;
}
QWidget#topStatusBar, QWidget#taskCard, QWidget#actionBar, QWidget#sideCard {
    background: #ffffff;
    border: 1px solid #d9e0e8;
    border-radius: 6px;
}
QListWidget#taskNav {
    background: transparent;
    border: none;
    padding: 4px;
}
QListWidget#taskNav::item {
    border-radius: 5px;
    padding: 9px 10px;
    margin: 3px 0;
    border-left: 3px solid #d7dde6;
}
QListWidget#taskNav::item:selected {
    background: #1f3f7a;
    color: #ffffff;
    border-left: 3px solid #60a5fa;
}
QListWidget#taskNav::item:disabled {
    color: #334155;
    font-weight: 800;
    background: #e9eef5;
    border-left: 3px solid #94a3b8;
}
QTabWidget::pane {
    border: 1px solid #d9e0e8;
    border-radius: 6px;
    background: #ffffff;
}
QTabBar::tab {
    background: #eef2f7;
    border: 1px solid #d9e0e8;
    padding: 6px 12px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #1d4ed8;
    font-weight: 700;
}
"""

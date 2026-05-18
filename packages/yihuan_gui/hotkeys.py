from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys

from PySide6.QtWidgets import QWidget

from .config_repository import QUICK_STOP_HOTKEY_OPTIONS


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

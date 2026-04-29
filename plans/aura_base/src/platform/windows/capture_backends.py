# -*- coding: utf-8 -*-
from __future__ import annotations

import ctypes
import importlib
import threading
from typing import Any

import cv2
import numpy as np
from PIL import ImageGrab
import win32con
import win32gui
import win32ui

from ..contracts import CaptureResult, TargetRuntimeError
from .window_target import WindowTarget


class BaseWindowsCaptureBackend:
    backend_name = ""

    def __init__(self, target: WindowTarget, config: dict[str, Any] | None = None):
        self.target = target
        self.config = dict(config or {})

    def close(self) -> None:
        return None

    def focus(self) -> bool:
        return self.target.focus()

    def get_client_rect(self) -> tuple[int, int, int, int]:
        return self.target.get_client_rect()

    def get_pixel_color_at(self, x: int, y: int) -> tuple[int, int, int]:
        capture = self.capture((int(x), int(y), 1, 1))
        if not capture.success or capture.image is None:
            raise TargetRuntimeError(
                "windows_pixel_failed",
                f"Failed to read a pixel through backend '{self.backend_name}'.",
                {"backend": self.backend_name},
            )
        red, green, blue = capture.image[0, 0].tolist()
        return int(red), int(green), int(blue)

    def capture(self, rect: tuple[int, int, int, int] | None = None) -> CaptureResult:
        self.target.ensure_valid()
        roi = _normalize_client_roi(self.target, rect)
        client_rect = self.target.get_client_rect()
        image = self._capture_roi(roi)
        return CaptureResult(
            success=True,
            image=image,
            window_rect=client_rect,
            relative_rect=roi,
            backend=self.backend_name,
        )

    def self_check(self) -> dict[str, Any]:
        return {
            "ok": True,
            "backend": self.backend_name,
            "client_rect": list(self.target.get_client_rect()),
        }

    def _capture_roi(self, roi: tuple[int, int, int, int]) -> np.ndarray:
        raise NotImplementedError


class WindowsGdiCaptureBackend(BaseWindowsCaptureBackend):
    backend_name = "gdi"

    def _capture_roi(self, roi: tuple[int, int, int, int]) -> np.ndarray:
        left, top, _, _ = self.target.get_client_rect_screen()
        x, y, width, height = roi
        bbox = (left + x, top + y, left + x + width, top + y + height)
        try:
            image = ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")
        except Exception as exc:
            raise TargetRuntimeError(
                "windows_capture_init_failed",
                f"GDI capture failed for the configured window: {exc}",
                {"backend": self.backend_name, "bbox": list(bbox)},
            ) from exc
        return np.asarray(image)


class WindowsWgcCaptureBackend(BaseWindowsCaptureBackend):
    backend_name = "wgc"

    def __init__(self, target: WindowTarget, config: dict[str, Any] | None = None):
        super().__init__(target, config)
        module_name = str(self.config.get("module_name") or "windows_capture")
        self.capture_cursor = bool(self.config.get("capture_cursor", False))
        self.frame_timeout_ms = max(int(self.config.get("frame_timeout_ms") or 2000), 100)
        self._api_style = ""
        self._windows_capture_cls = None
        try:
            self._module = importlib.import_module(module_name)
        except Exception as exc:
            raise TargetRuntimeError(
                "windows_capture_init_failed",
                f"Unable to import WGC capture module '{module_name}'.",
                {"backend": self.backend_name, "module_name": module_name, "error": str(exc)},
            ) from exc

        self._capturer = self._build_capturer()

    def _build_capturer(self) -> Any:
        # Preferred style: windows_capture.WindowsGraphicsCapture(...)
        capture_cls = getattr(self._module, "WindowsGraphicsCapture", None)
        if callable(capture_cls):
            self._api_style = "windows_graphics_capture"
            try:
                return capture_cls(capture_cursor=self.capture_cursor)
            except TypeError:
                return capture_cls()
            except Exception as exc:
                raise TargetRuntimeError(
                    "windows_capture_init_failed",
                    "Failed to initialize the configured WGC capture class.",
                    {"backend": self.backend_name, "error": str(exc)},
                ) from exc

        # Function style fallback: module.capture_window(hwnd=...)
        capture_fn = getattr(self._module, "capture_window", None)
        if callable(capture_fn):
            self._api_style = "capture_window_function"
            return None

        # windows-capture>=2 exposes WindowsCapture(window_hwnd=...) with frame callbacks.
        capture_cls = getattr(self._module, "WindowsCapture", None)
        if callable(capture_cls):
            self._api_style = "windows_capture_v2"
            self._windows_capture_cls = capture_cls
            return None

        raise TargetRuntimeError(
            "windows_capture_init_failed",
            "The configured WGC capture module does not expose a supported API.",
            {
                "backend": self.backend_name,
                "module_name": getattr(self._module, "__name__", None),
                "expected": ["WindowsGraphicsCapture", "capture_window", "WindowsCapture"],
            },
        )

    def _capture_roi(self, roi: tuple[int, int, int, int]) -> np.ndarray:
        frame = self._capture_full_client_frame()
        x, y, width, height = roi
        return frame[y : y + height, x : x + width].copy()

    def _capture_full_client_frame(self) -> np.ndarray:
        if self._api_style == "windows_capture_v2":
            return self._capture_windows_capture_v2_client_frame()

        hwnd = self.target.hwnd
        frame = None

        if self._capturer is not None and hasattr(self._capturer, "capture_window"):
            try:
                frame = self._capturer.capture_window(hwnd)
            except Exception as exc:
                raise TargetRuntimeError(
                    "windows_capture_failed",
                    f"WGC capture failed for hwnd={hwnd}: {exc}",
                    {"backend": self.backend_name, "hwnd": hwnd},
                ) from exc
        else:
            capture_fn = getattr(self._module, "capture_window", None)
            try:
                frame = capture_fn(hwnd=hwnd, capture_cursor=self.capture_cursor)
            except TypeError:
                frame = capture_fn(hwnd)
            except Exception as exc:
                raise TargetRuntimeError(
                    "windows_capture_failed",
                    f"WGC capture failed for hwnd={hwnd}: {exc}",
                    {"backend": self.backend_name, "hwnd": hwnd},
                ) from exc

        if frame is None:
            raise TargetRuntimeError(
                "windows_capture_failed",
                "WGC capture returned an empty frame.",
                {"backend": self.backend_name, "hwnd": hwnd},
            )
        return _coerce_rgb_frame(frame, backend=self.backend_name)

    def _capture_windows_capture_v2_client_frame(self) -> np.ndarray:
        if self._windows_capture_cls is None:
            raise TargetRuntimeError(
                "windows_capture_init_failed",
                "WGC WindowsCapture API was selected but no capture class is available.",
                {"backend": self.backend_name},
            )

        frame_event = threading.Event()
        state: dict[str, Any] = {"frame": None, "error": None}

        try:
            capturer = self._windows_capture_cls(
                cursor_capture=self.capture_cursor,
                draw_border=False,
                window_hwnd=int(self.target.hwnd),
            )
        except TypeError:
            capturer = self._windows_capture_cls(
                cursor_capture=self.capture_cursor,
                window_hwnd=int(self.target.hwnd),
            )
        except Exception as exc:
            raise TargetRuntimeError(
                "windows_capture_init_failed",
                f"Failed to initialize the configured WGC WindowsCapture class: {exc}",
                {"backend": self.backend_name, "hwnd": int(self.target.hwnd)},
            ) from exc

        @capturer.event
        def on_frame_arrived(frame, control):  # noqa: ANN001
            try:
                frame_buffer = getattr(frame, "frame_buffer", frame)
                state["frame"] = np.asarray(frame_buffer).copy()
                control.stop()
            except Exception as exc:  # noqa: BLE001
                state["error"] = str(exc)
            finally:
                frame_event.set()

        @capturer.event
        def on_closed():
            frame_event.set()

        control = None
        try:
            control = capturer.start_free_threaded()
            if not frame_event.wait(float(self.frame_timeout_ms) / 1000.0):
                raise TargetRuntimeError(
                    "windows_capture_failed",
                    "WGC WindowsCapture did not deliver a frame before timeout.",
                    {
                        "backend": self.backend_name,
                        "hwnd": int(self.target.hwnd),
                        "timeout_ms": int(self.frame_timeout_ms),
                    },
                )
            if state.get("error"):
                raise TargetRuntimeError(
                    "windows_capture_failed",
                    f"WGC WindowsCapture frame callback failed: {state['error']}",
                    {"backend": self.backend_name, "hwnd": int(self.target.hwnd)},
                )
            frame = state.get("frame")
            if frame is None:
                raise TargetRuntimeError(
                    "windows_capture_failed",
                    "WGC WindowsCapture returned no frame.",
                    {"backend": self.backend_name, "hwnd": int(self.target.hwnd)},
                )
            rgb = _coerce_rgb_frame(frame, backend=self.backend_name)
            return self._crop_wgc_frame_to_client(rgb)
        except TargetRuntimeError:
            raise
        except Exception as exc:
            raise TargetRuntimeError(
                "windows_capture_failed",
                f"WGC WindowsCapture failed for hwnd={self.target.hwnd}: {exc}",
                {"backend": self.backend_name, "hwnd": int(self.target.hwnd)},
            ) from exc
        finally:
            if control is not None:
                try:
                    control.stop()
                except Exception:
                    pass
                try:
                    control.wait()
                except Exception:
                    pass

    def _crop_wgc_frame_to_client(self, frame: np.ndarray) -> np.ndarray:
        _, _, client_width, client_height = self.target.get_client_rect()
        frame_height, frame_width = frame.shape[:2]
        if frame_width == client_width and frame_height == client_height:
            return frame.copy()

        client_left, client_top, _, _ = self.target.get_client_rect_screen()
        frame_bounds = _get_dwm_extended_frame_bounds(self.target.hwnd) or self.target.get_window_rect_screen()
        frame_left, frame_top, _, _ = frame_bounds
        offset_x = int(client_left) - int(frame_left)
        offset_y = int(client_top) - int(frame_top)

        if (
            offset_x >= 0
            and offset_y >= 0
            and offset_x + client_width <= frame_width
            and offset_y + client_height <= frame_height
        ):
            return frame[offset_y : offset_y + client_height, offset_x : offset_x + client_width].copy()

        raise TargetRuntimeError(
            "windows_capture_failed",
            "WGC captured frame could not be mapped back to the target client area.",
            {
                "backend": self.backend_name,
                "frame_shape": list(frame.shape),
                "client_rect_screen": [int(client_left), int(client_top), int(client_width), int(client_height)],
                "frame_bounds_screen": list(frame_bounds),
                "client_offset": [int(offset_x), int(offset_y)],
            },
        )

    def self_check(self) -> dict[str, Any]:
        payload = super().self_check()
        payload["module_name"] = getattr(self._module, "__name__", None)
        payload["capture_cursor"] = self.capture_cursor
        return payload


class WindowsDxgiCaptureBackend(BaseWindowsCaptureBackend):
    backend_name = "dxgi"

    def __init__(self, target: WindowTarget, config: dict[str, Any] | None = None):
        super().__init__(target, config)
        module_name = str(self.config.get("module_name") or "dxcam")
        try:
            dxcam = importlib.import_module(module_name)
        except Exception as exc:
            raise TargetRuntimeError(
                "windows_capture_init_failed",
                f"Unable to import DXGI capture module '{module_name}'.",
                {"backend": self.backend_name, "error": str(exc)},
            ) from exc

        device_idx = self.config.get("device_idx")
        output_idx = self.config.get("output_idx")
        create_kwargs = {"output_color": "RGB"}
        if device_idx is not None:
            create_kwargs["device_idx"] = int(device_idx)
        if output_idx is not None:
            create_kwargs["output_idx"] = int(output_idx)

        try:
            self._camera = dxcam.create(**create_kwargs)
        except Exception as exc:
            raise TargetRuntimeError(
                "windows_capture_init_failed",
                "Failed to initialize the configured DXGI capture backend.",
                {"backend": self.backend_name, "error": str(exc), "options": create_kwargs},
            ) from exc

    def close(self) -> None:
        stop_fn = getattr(self._camera, "stop", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                pass

    def _capture_roi(self, roi: tuple[int, int, int, int]) -> np.ndarray:
        left, top, _, _ = self.target.get_client_rect_screen()
        x, y, width, height = roi
        region = (left + x, top + y, left + x + width, top + y + height)
        try:
            frame = self._camera.grab(region=region)
        except Exception as exc:
            raise TargetRuntimeError(
                "windows_capture_failed",
                f"DXGI capture failed for the configured window: {exc}",
                {"backend": self.backend_name, "region": list(region)},
            ) from exc
        if frame is None:
            raise TargetRuntimeError(
                "windows_capture_failed",
                "DXGI capture returned an empty frame.",
                {"backend": self.backend_name, "region": list(region)},
            )
        return np.asarray(frame).copy()


class WindowsPrintWindowCaptureBackend(BaseWindowsCaptureBackend):
    backend_name = "printwindow"

    def _capture_roi(self, roi: tuple[int, int, int, int]) -> np.ndarray:
        _, _, client_width, client_height = self.target.get_client_rect()
        hwnd = self.target.hwnd

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        if not hwnd_dc:
            raise TargetRuntimeError(
                "windows_capture_failed",
                "PrintWindow could not acquire a device context.",
                {"backend": self.backend_name, "hwnd": hwnd},
            )

        src_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        mem_dc = src_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        try:
            bitmap.CreateCompatibleBitmap(src_dc, client_width, client_height)
            mem_dc.SelectObject(bitmap)
            result = ctypes.windll.user32.PrintWindow(hwnd, mem_dc.GetSafeHdc(), 0x00000001)
            if result != 1:
                raise TargetRuntimeError(
                    "windows_capture_failed",
                    "PrintWindow failed to capture the configured client area.",
                    {"backend": self.backend_name, "hwnd": hwnd},
                )
            buffer = bitmap.GetBitmapBits(True)
            info = bitmap.GetInfo()
            image = np.frombuffer(buffer, dtype=np.uint8)
            image = image.reshape((info["bmHeight"], info["bmWidth"], 4))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            x, y, width, height = roi
            return rgb[y : y + height, x : x + width].copy()
        finally:
            mem_dc.DeleteDC()
            src_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            win32gui.DeleteObject(bitmap.GetHandle())


def build_capture_backend(backend: str, target: WindowTarget, config: dict[str, Any]) -> BaseWindowsCaptureBackend:
    normalized = str(backend or "").strip().lower()
    if normalized == "wgc":
        return WindowsWgcCaptureBackend(target, config)
    if normalized == "dxgi":
        return WindowsDxgiCaptureBackend(target, config)
    if normalized == "gdi":
        return WindowsGdiCaptureBackend(target, config)
    if normalized == "printwindow":
        return WindowsPrintWindowCaptureBackend(target, config)
    raise TargetRuntimeError(
        "capture_backend_invalid_for_provider",
        f"Unsupported Windows capture backend '{backend}'.",
        {"backend": backend},
    )


def _normalize_client_roi(target: WindowTarget, rect: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    _, _, client_width, client_height = target.get_client_rect()
    if rect is None:
        return 0, 0, client_width, client_height

    x, y, width, height = [int(value) for value in rect]
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > client_width or y + height > client_height:
        raise TargetRuntimeError(
            "capture_rect_invalid",
            "Capture rect is outside the current window client area.",
            {"rect": [x, y, width, height], "viewport": [0, 0, client_width, client_height]},
        )
    return x, y, width, height


def _coerce_rgb_frame(frame: Any, *, backend: str) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        array = np.asarray(frame)
    elif hasattr(frame, "__array__"):
        array = np.asarray(frame)
    elif hasattr(frame, "to_ndarray"):
        array = np.asarray(frame.to_ndarray())
    elif hasattr(frame, "to_numpy"):
        array = np.asarray(frame.to_numpy())
    elif hasattr(frame, "copy") and callable(frame.copy):
        try:
            array = np.asarray(frame.copy())
        except Exception as exc:
            raise TargetRuntimeError(
                "windows_capture_failed",
                "Capture backend returned an unsupported frame object.",
                {"backend": backend, "error": str(exc), "frame_type": type(frame).__name__},
            ) from exc
    else:
        raise TargetRuntimeError(
            "windows_capture_failed",
            "Capture backend returned an unsupported frame object.",
            {"backend": backend, "frame_type": type(frame).__name__},
        )

    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise TargetRuntimeError(
            "windows_capture_failed",
            "Capture backend returned an array with unsupported shape.",
            {"backend": backend, "shape": list(array.shape)},
        )
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
    return np.asarray(array).copy()


class _RECT(ctypes.Structure):
    _fields_ = (
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    )


def _get_dwm_extended_frame_bounds(hwnd: int) -> tuple[int, int, int, int] | None:
    dwmapi = getattr(ctypes.windll, "dwmapi", None)
    get_attribute = getattr(dwmapi, "DwmGetWindowAttribute", None) if dwmapi else None
    if not callable(get_attribute):
        return None
    rect = _RECT()
    try:
        result = int(get_attribute(int(hwnd), 9, ctypes.byref(rect), ctypes.sizeof(rect)))
    except Exception:
        return None
    if result != 0:
        return None
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        return None
    return int(rect.left), int(rect.top), width, height

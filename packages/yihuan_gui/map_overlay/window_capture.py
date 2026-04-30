from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Any

from PIL import Image

from packages.aura_game import SubprocessGameRunner


class RuntimeCaptureError(RuntimeError):
    """Raised when the Aura runtime cannot provide a target snapshot."""


@dataclass(frozen=True)
class RuntimeTargetInfo:
    title: str
    geometry: tuple[int, int, int, int]
    target: dict[str, Any]

    @property
    def width(self) -> int:
        return max(0, self.geometry[2])

    @property
    def height(self) -> int:
        return max(0, self.geometry[3])


@dataclass(frozen=True)
class RuntimeTargetSnapshot:
    info: RuntimeTargetInfo
    image: Image.Image
    backend: str = ""
    quality_flags: tuple[str, ...] = ()

    @property
    def geometry(self) -> tuple[int, int, int, int]:
        return self.info.geometry


class AuraRuntimeCaptureClient:
    """Small GUI-side facade over the Aura framework target/capture runtime."""

    def __init__(self, *, game_name: str, runner: SubprocessGameRunner | None = None) -> None:
        self.game_name = str(game_name or "").strip()
        self._runner = runner
        self._owns_runner = runner is None

    def target(self) -> RuntimeTargetInfo:
        payload = self._ensure_runner().target_status(game_name=self.game_name)
        if not payload.get("ok", True):
            raise RuntimeCaptureError(str(payload.get("message") or "无法获取目标窗口状态。"))
        return _target_info_from_payload(payload)

    def snapshot(self) -> RuntimeTargetSnapshot:
        payload = self._ensure_runner().target_snapshot(game_name=self.game_name)
        if not payload.get("ok"):
            raise RuntimeCaptureError(str(payload.get("message") or "目标窗口截图失败。"))
        image_bytes = payload.get("image_png")
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise RuntimeCaptureError("框架截图结果缺少图像数据。")
        image = Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB")
        info = _target_info_from_payload(payload, image_size=image.size)
        return RuntimeTargetSnapshot(
            info=info,
            image=image,
            backend=str(payload.get("backend") or ""),
            quality_flags=tuple(str(item) for item in (payload.get("quality_flags") or ())),
        )

    def close(self) -> None:
        if self._owns_runner and self._runner is not None:
            self._runner.close()
        self._runner = None

    def _ensure_runner(self) -> SubprocessGameRunner:
        if self._runner is None:
            self._runner = SubprocessGameRunner()
        return self._runner


def _target_info_from_payload(payload: dict[str, Any], image_size: tuple[int, int] | None = None) -> RuntimeTargetInfo:
    target = dict(payload.get("target") or {})
    geometry = _resolve_geometry(target, image_size=image_size)
    return RuntimeTargetInfo(
        title=str(target.get("title") or target.get("resolved_serial") or payload.get("game_name") or "异环"),
        geometry=geometry,
        target=target,
    )


def _resolve_geometry(target: dict[str, Any], *, image_size: tuple[int, int] | None) -> tuple[int, int, int, int]:
    for key in ("client_rect_screen", "window_rect_screen", "screen_rect", "geometry"):
        rect = _coerce_rect(target.get(key))
        if rect is not None:
            return rect

    client_rect = _coerce_rect(target.get("client_rect"))
    if client_rect is not None and client_rect[2] > 0 and client_rect[3] > 0:
        return client_rect

    if image_size is not None:
        return 0, 0, max(0, int(image_size[0])), max(0, int(image_size[1]))
    return 0, 0, 0, 0


def _coerce_rect(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x, y, width, height = (int(float(value[index])) for index in range(4))
    except (TypeError, ValueError):
        return None
    return x, y, max(0, width), max(0, height)

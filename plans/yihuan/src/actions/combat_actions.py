"""Yihuan automatic combat session action."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
from packages.aura_core.api import action_info, requires_services
from packages.aura_core.observability.logging.core_logger import current_cid, logger
from packages.aura_core.scheduler.cancellation import is_current_task_cancel_requested
from plans.aura_base.src.platform.contracts import TargetRuntimeError

from ..audio_dodge import AudioDodgeRuntime
from ..services.combat_service import YihuanCombatService


class _CombatSessionCancelled(Exception):
    """Internal marker used to stop the sync combat action cooperatively."""


_LOOK_SMOOTH_STEP_PIXELS = 80
_LOOK_SMOOTH_STEP_DELAY_MS = 8


def _coerce_bool(value: bool | str | int | float | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false", "no", "off", "none", "null"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return bool(value)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _combat_cancel_requested() -> bool:
    try:
        return is_current_task_cancel_requested()
    except Exception:
        return False


def _raise_if_cancelled() -> None:
    if _combat_cancel_requested():
        raise _CombatSessionCancelled()


def _sleep_is_mocked() -> bool:
    return hasattr(time.sleep, "mock_calls")


def _sleep_interruptibly(duration_sec: float, *, quantum_sec: float = 0.05) -> None:
    duration = max(float(duration_sec), 0.0)
    _raise_if_cancelled()
    if duration <= 0:
        return
    if _sleep_is_mocked():
        time.sleep(duration)
        _raise_if_cancelled()
        return
    deadline = time.monotonic() + duration
    while True:
        _raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, max(float(quantum_sec), 0.01)))


class _CombatTargetsYoloRuntime:
    def __init__(self, yolo: Any, profile: Mapping[str, Any]) -> None:
        config = dict(profile.get("enemy_health_yolo") or {})
        self.enabled = bool(config.get("enabled"))
        self.yolo = yolo
        self.model_name = str(config.get("model_name") or "yihuan_enemy_hp_bar")
        self.health_labels = {
            str(item).strip().lower()
            for item in (config.get("labels") or ["enemy_hp_bar"])
            if str(item).strip()
        }
        self.direction_labels = {
            str(item).strip().lower()
            for item in (config.get("direction_labels") or ["enemy_direction_marker"])
            if str(item).strip()
        }
        self.reward_enabled = _coerce_bool(config.get("reward_enabled", True))
        self.reward_labels = (
            {
                str(item).strip().lower()
                for item in (config.get("reward_labels") or ["reward_marker"])
                if str(item).strip()
            }
            if self.reward_enabled
            else set()
        )
        self.detection_labels = self.health_labels | self.direction_labels | self.reward_labels
        self.conf = float(config.get("conf") or 0.35)
        self.direction_conf = min(max(_coerce_float(config.get("direction_conf"), self.conf), 0.0), 1.0)
        self.reward_conf = min(max(_coerce_float(config.get("reward_conf"), self.conf), 0.0), 1.0)
        self.iou = float(config.get("iou") or 0.45)
        self.max_det = int(config.get("max_det") or 20)
        self.min_width = int(config.get("min_width") or 14)
        self.min_height = int(config.get("min_height") or 8)
        self.direction_min_width = int(config.get("direction_min_width") or 6)
        self.direction_min_height = int(config.get("direction_min_height") or 6)
        self.reward_min_width = int(config.get("reward_min_width") or 6)
        self.reward_min_height = int(config.get("reward_min_height") or 6)
        self.reward_search_region = _optional_region_tuple(config.get("reward_search_region"))
        self.reward_exclude_regions = [
            region
            for item in (config.get("reward_exclude_regions") or [])
            if (region := _optional_region_tuple(item)) is not None
        ]
        client_size = profile.get("client_size") or (1280, 720)
        self.client_width = max(_coerce_int((client_size or [1280, 720])[0], 1280), 1)
        self.client_height = max(_coerce_int((client_size or [1280, 720])[1], 720), 1)
        self.last_seen_ttl_sec = float(config.get("last_seen_ttl_sec") or 0.0)
        self.fallback_to_hsv_on_error = bool(config.get("fallback_to_hsv_on_error", True))
        self._last_seen_at: float | None = None
        self._last_boxes: list[dict[str, Any]] = []
        self._loaded = False
        self._error_logged = False

    def apply(self, state: Mapping[str, Any], source_image: Any) -> dict[str, Any]:
        resolved_state = dict(state)
        if not self.enabled or self.yolo is None:
            return resolved_state

        now = time.monotonic()
        boxes: list[dict[str, Any]] = []
        health_boxes: list[dict[str, Any]] = []
        direction_boxes: list[dict[str, Any]] = []
        reward_boxes: list[dict[str, Any]] = []
        yolo_error: str | None = None
        provider: str | None = None
        image_height = int(getattr(source_image, "shape", [0, 0])[0] or 0)
        image_width = int(getattr(source_image, "shape", [0, 0])[1] or 0)
        try:
            if not self._loaded:
                self.yolo.preload_model(self.model_name)
                self._loaded = True
            result = self.yolo.detect_image(
                source_image,
                model_name=self.model_name,
                options={
                    "conf": self.conf,
                    "iou": self.iou,
                    "max_det": self.max_det,
                },
            )
            provider = result.get("provider")
            for det in result.get("detections", []):
                label = str(det.get("label") or "").strip().lower()
                if self.detection_labels and label not in self.detection_labels:
                    continue
                bbox = det.get("bbox_global") or det.get("bbox_xywh")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                x, y, width, height = [float(item) for item in bbox]
                score = float(det.get("score") or 0.0)
                box = {
                    "x": int(round(x)),
                    "y": int(round(y)),
                    "width": int(round(width)),
                    "height": int(round(height)),
                    "score": round(score, 3),
                    "label": label or "unknown",
                }
                if label in self.health_labels:
                    if width < self.min_width or height < self.min_height:
                        continue
                    health_boxes.append(box)
                elif label in self.direction_labels:
                    if score < self.direction_conf or width < self.direction_min_width or height < self.direction_min_height:
                        continue
                    direction_boxes.append(box)
                elif self.reward_enabled and label in self.reward_labels:
                    if score < self.reward_conf or width < self.reward_min_width or height < self.reward_min_height:
                        continue
                    if self.reward_search_region is not None and not _box_center_in_scaled_region(
                        box,
                        self.reward_search_region,
                        image_width=image_width,
                        image_height=image_height,
                        client_width=self.client_width,
                        client_height=self.client_height,
                    ):
                        continue
                    if _box_center_in_any_scaled_region(
                        box,
                        self.reward_exclude_regions,
                        image_width=image_width,
                        image_height=image_height,
                        client_width=self.client_width,
                        client_height=self.client_height,
                    ):
                        continue
                    reward_boxes.append(box)
                else:
                    continue
                boxes.append(box)
        except Exception as exc:  # noqa: BLE001 - fallback keeps combat usable if the model is missing/corrupt.
            yolo_error = str(exc)
            if not self._error_logged:
                logger.warning("Combat[yolo_combat_targets] disabled_for_frame error=%s", yolo_error)
                self._error_logged = True
            if self.fallback_to_hsv_on_error:
                debug = dict(resolved_state.get("debug") or {})
                debug["combat_targets_yolo"] = {
                    "enabled": True,
                    "source": "hsv_fallback",
                    "error": yolo_error,
                }
                resolved_state["debug"] = debug
                return resolved_state

        stale = False
        if health_boxes:
            self._last_seen_at = now
            self._last_boxes = [dict(item) for item in health_boxes]
        elif (
            self.last_seen_ttl_sec > 0
            and self._last_seen_at is not None
            and now - self._last_seen_at <= self.last_seen_ttl_sec
            and bool(resolved_state.get("in_combat"))
            and self._last_boxes
        ):
            health_boxes = [dict(item) for item in self._last_boxes]
            stale = True

        direction_boxes = [_with_yolo_direction_region(box, image_width=image_width, image_height=image_height) for box in direction_boxes]
        direction_boxes.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        reward_boxes.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)

        resolved_state["enemy_health_found"] = bool(health_boxes)
        resolved_state["enemy_health_count"] = int(len(health_boxes))
        resolved_state["front_enemy_found"] = bool(health_boxes or resolved_state.get("boss_found"))
        resolved_state["enemy_direction_found"] = bool(direction_boxes)
        resolved_state["enemy_direction_count"] = int(len(direction_boxes))
        resolved_state["enemy_direction_markers"] = [
            {
                "x": int(box["x"]),
                "y": int(box["y"]),
                "width": int(box["width"]),
                "height": int(box["height"]),
                "region": str(box.get("region") or "yolo_unknown"),
                "score": float(box.get("score") or 0.0),
                "label": str(box.get("label") or "enemy_direction_marker"),
            }
            for box in direction_boxes[:10]
        ]
        resolved_state["enemy_direction_primary_side"] = _primary_yolo_direction_side(direction_boxes)
        if self.reward_enabled and not bool(resolved_state.get("remaining_enemy_marker_found")):
            reward_box = reward_boxes[0] if reward_boxes else None
            resolved_state["reward_marker_found"] = reward_box is not None
            resolved_state["reward_marker_confidence"] = round(float((reward_box or {}).get("score") or 0.0), 3)
            if reward_box is not None:
                rx = int(reward_box["x"])
                ry = int(reward_box["y"])
                rw = int(reward_box["width"])
                rh = int(reward_box["height"])
                resolved_state["reward_marker_box"] = [rx, ry, rw, rh]
                resolved_state["reward_marker_center_x"] = int(round(rx + rw / 2.0))
                resolved_state["reward_marker_center_y"] = int(round(ry + rh / 2.0))
            else:
                resolved_state["reward_marker_box"] = []
                resolved_state["reward_marker_center_x"] = None
                resolved_state["reward_marker_center_y"] = None

        debug = dict(resolved_state.get("debug") or {})
        debug["enemy_signal_count"] = int(len(health_boxes))
        debug["front_enemy_signal_count"] = int(len(health_boxes) + int(bool(resolved_state.get("boss_found"))))
        debug["enemy_health_boxes"] = [
            [int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])]
            for box in health_boxes[:10]
        ]
        debug["enemy_direction_markers"] = [dict(box) for box in direction_boxes[:10]]
        debug["enemy_direction_primary_side"] = resolved_state.get("enemy_direction_primary_side")
        debug["reward_marker_box"] = list(resolved_state.get("reward_marker_box") or [])
        debug["combat_targets_yolo"] = {
            "enabled": True,
            "source": "yolo_ttl" if stale else "yolo",
            "model": self.model_name,
            "provider": provider,
            "conf": round(self.conf, 3),
            "direction_conf": round(self.direction_conf, 3),
            "reward_enabled": bool(self.reward_enabled),
            "reward_conf": round(self.reward_conf, 3),
            "count": int(len(boxes)),
            "health_count": int(len(health_boxes)),
            "direction_count": int(len(direction_boxes)),
            "reward_count": int(len(reward_boxes)),
            "boxes": [dict(box) for box in boxes[:20]],
            "health_boxes": [dict(box) for box in health_boxes[:10]],
            "direction_boxes": [dict(box) for box in direction_boxes[:10]],
            "reward_boxes": [dict(box) for box in reward_boxes[:10]],
            "error": yolo_error,
        }
        debug["enemy_health_yolo"] = debug["combat_targets_yolo"]
        resolved_state["debug"] = debug
        return resolved_state


def _optional_region_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    return (
        _coerce_int(value[0], 0),
        _coerce_int(value[1], 0),
        max(_coerce_int(value[2], 0), 1),
        max(_coerce_int(value[3], 0), 1),
    )


def _box_center_in_scaled_region(
    box: Mapping[str, Any],
    region: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    client_width: int,
    client_height: int,
) -> bool:
    if image_width <= 0 or image_height <= 0:
        return True
    scale_x = float(image_width) / float(max(client_width, 1))
    scale_y = float(image_height) / float(max(client_height, 1))
    rx, ry, rw, rh = region
    left = float(rx) * scale_x
    top = float(ry) * scale_y
    right = float(rx + rw) * scale_x
    bottom = float(ry + rh) * scale_y
    center_x = float(_coerce_int(box.get("x"), 0)) + float(_coerce_int(box.get("width"), 0)) / 2.0
    center_y = float(_coerce_int(box.get("y"), 0)) + float(_coerce_int(box.get("height"), 0)) / 2.0
    return left <= center_x <= right and top <= center_y <= bottom


def _box_center_in_any_scaled_region(
    box: Mapping[str, Any],
    regions: list[tuple[int, int, int, int]],
    *,
    image_width: int,
    image_height: int,
    client_width: int,
    client_height: int,
) -> bool:
    return any(
        _box_center_in_scaled_region(
            box,
            region,
            image_width=image_width,
            image_height=image_height,
            client_width=client_width,
            client_height=client_height,
        )
        for region in regions
    )


def _with_yolo_direction_region(box: Mapping[str, Any], *, image_width: int, image_height: int) -> dict[str, Any]:
    resolved = dict(box)
    width = max(int(image_width or 0), 1)
    height = max(int(image_height or 0), 1)
    center_x = (float(_coerce_int(resolved.get("x"), 0)) + float(_coerce_int(resolved.get("width"), 0)) / 2.0) / float(width)
    center_y = (float(_coerce_int(resolved.get("y"), 0)) + float(_coerce_int(resolved.get("height"), 0)) / 2.0) / float(height)
    if center_y >= 0.78:
        side = "bottom"
    elif center_x < 0.5:
        side = "left"
    else:
        side = "right"
    resolved["region"] = f"yolo_{side}"
    return resolved


def _primary_yolo_direction_side(direction_boxes: list[dict[str, Any]]) -> str | None:
    if not direction_boxes:
        return None
    region = str(direction_boxes[0].get("region") or "")
    if region.endswith("_left"):
        return "left"
    if region.endswith("_right"):
        return "right"
    if region.endswith("_bottom"):
        return "bottom"
    return None


def _capture_state(
    app: Any,
    yihuan_combat: YihuanCombatService,
    *,
    profile_name: str,
    enemy_health_yolo: _CombatTargetsYoloRuntime | None = None,
) -> tuple[dict[str, Any], Any]:
    _raise_if_cancelled()
    capture = app.capture()
    if not capture.success or capture.image is None:
        raise RuntimeError("Failed to capture the Yihuan combat screen.")
    state = yihuan_combat.analyze_frame(capture.image, profile_name=profile_name)
    if enemy_health_yolo is not None:
        state = enemy_health_yolo.apply(state, capture.image)
    return state, capture


def _enemy_health_yolo_config(profile: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(profile.get("enemy_health_yolo") or {})
    return {
        "enabled": _coerce_bool(config.get("enabled")),
        "model_name": str(config.get("model_name") or "yihuan_enemy_hp_bar"),
        "labels": {
            str(item).strip().lower()
            for item in (config.get("labels") or ["enemy_hp_bar"])
            if str(item).strip()
        },
        "conf": min(max(_coerce_float(config.get("conf"), 0.35), 0.0), 1.0),
        "iou": min(max(_coerce_float(config.get("iou"), 0.45), 0.0), 1.0),
        "max_det": max(_coerce_int(config.get("max_det"), 20), 1),
        "min_width": max(_coerce_int(config.get("min_width"), 14), 1),
        "min_height": max(_coerce_int(config.get("min_height"), 8), 1),
    }


def _detect_enemy_health_boxes(
    yolo: Any,
    source_image: Any,
    *,
    yolo_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not _coerce_bool(yolo_config.get("enabled")):
        return [], {"ok": False, "error": "enemy_health_yolo disabled", "detections": []}
    if yolo is None:
        return [], {"ok": False, "error": "core/yolo service unavailable", "detections": []}

    result = yolo.detect_image(
        source_image,
        model_name=str(yolo_config.get("model_name") or "yihuan_enemy_hp_bar"),
        options={
            "conf": float(yolo_config.get("conf") or 0.35),
            "iou": float(yolo_config.get("iou") or 0.45),
            "max_det": int(yolo_config.get("max_det") or 20),
        },
    )
    labels = {str(item).strip().lower() for item in (yolo_config.get("labels") or []) if str(item).strip()}
    min_width = max(_coerce_int(yolo_config.get("min_width"), 14), 1)
    min_height = max(_coerce_int(yolo_config.get("min_height"), 8), 1)
    image_height = int(getattr(source_image, "shape", [0, 0])[0] or 0)
    image_width = int(getattr(source_image, "shape", [0, 0])[1] or 0)
    boxes: list[dict[str, Any]] = []
    for det in result.get("detections", []):
        label = str(det.get("label") or "").strip().lower()
        if labels and label not in labels:
            continue
        bbox = det.get("bbox_global") or det.get("bbox_xywh")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x, y, width, height = [float(item) for item in bbox]
        if width < min_width or height < min_height:
            continue
        if image_width > 0 and image_height > 0:
            x = max(0.0, min(x, float(image_width - 1)))
            y = max(0.0, min(y, float(image_height - 1)))
            width = max(1.0, min(width, float(image_width) - x))
            height = max(1.0, min(height, float(image_height) - y))
        boxes.append(
            {
                "class_id": _coerce_int(det.get("class_id"), 0),
                "label": label or "enemy_hp_bar",
                "score": float(det.get("score") or 0.0),
                "x": int(round(x)),
                "y": int(round(y)),
                "width": int(round(width)),
                "height": int(round(height)),
            }
        )
    return boxes, result


def _combat_targets_dataset_root(raw_root: Any = None) -> Path:
    configured = str(raw_root or "").strip()
    if not configured:
        configured = os.environ.get("AURA_YIHUAN_COMBAT_TARGETS_DATASET_ROOT", "").strip()
    if not configured:
        configured = os.environ.get("AURA_YIHUAN_HPBAR_DATASET_ROOT", "").strip()
    if not configured:
        configured = r"D:\aura_yolo_training\yihuan_combat_targets\auto_label"
    return Path(configured)


def _image_digest(source_image: Any) -> str:
    try:
        return hashlib.sha1(source_image.tobytes()).hexdigest()[:16]
    except Exception:
        return hashlib.sha1(repr(source_image).encode("utf-8", errors="ignore")).hexdigest()[:16]


def _rgb_capture_to_bgr(source_image: Any) -> Any:
    if getattr(source_image, "ndim", 0) == 3 and int(source_image.shape[2]) >= 3:
        return cv2.cvtColor(source_image[:, :, :3], cv2.COLOR_RGB2BGR)
    return source_image


def _write_cv_image(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def _write_yolo_label_file(path: Path, boxes: list[dict[str, Any]], *, image_width: int, image_height: int) -> None:
    lines: list[str] = []
    for box in boxes:
        x = float(box["x"])
        y = float(box["y"])
        width = float(box["width"])
        height = float(box["height"])
        class_id = max(_coerce_int(box.get("class_id"), 0), 0)
        cx = (x + width / 2.0) / float(image_width)
        cy = (y + height / 2.0) / float(image_height)
        normalized_width = width / float(image_width)
        normalized_height = height / float(image_height)
        lines.append(
            f"{class_id} {cx:.6f} {cy:.6f} {normalized_width:.6f} {normalized_height:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_annotated_preview(path: Path, source_image: Any, boxes: list[dict[str, Any]]) -> None:
    preview = _rgb_capture_to_bgr(source_image).copy()
    for box in boxes:
        x = int(box["x"])
        y = int(box["y"])
        width = int(box["width"])
        height = int(box["height"])
        score = float(box.get("score") or 0.0)
        label = str(box.get("label") or "enemy_hp_bar")
        cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"{label} {score:.2f}",
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    _write_cv_image(path, preview)


def _parse_label_filter(value: Any, default: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    if isinstance(value, str):
        raw_items = value.replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    elif value is None:
        raw_items = list(default)
    else:
        raw_items = [value]
    labels = {str(item).strip().lower() for item in raw_items if str(item).strip()}
    if not labels:
        labels = {str(item).strip().lower() for item in default if str(item).strip()}
    return labels


def _load_existing_dataset_digests(manifest_path: Path) -> set[str]:
    digests: set[str] = set()
    if not manifest_path.exists():
        return digests
    try:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            digest = str(payload.get("image_digest") or "").strip()
            if digest:
                digests.add(digest)
    except Exception as exc:  # noqa: BLE001 - a stale manifest should not block collection.
        logger.warning("CombatTargetsDataset[manifest] failed_to_read path=%s error=%s", manifest_path, exc)
    return digests


@action_info(
    name="yihuan_combat_collect_hpbar_dataset",
    public=False,
    read_only=False,
    description="Collect auto-labeled Yihuan combat target screenshots with the current YOLO model.",
)
@requires_services(
    app="plans/aura_base/app",
    yihuan_combat="yihuan_combat",
    yolo="core/yolo",
)
def yihuan_combat_collect_hpbar_dataset(
    app: Any,
    yihuan_combat: YihuanCombatService,
    yolo: Any = None,
    max_images: int | str = 0,
    profile_name: str = "default_1280x720_cn",
    model_name: str | None = None,
    labels: str | list[str] | tuple[str, ...] | None = None,
    conf: float | int | str | None = None,
    iou: float | int | str | None = None,
    interval_sec: float | int | str = 1.0,
    dataset_root: str | None = None,
    save_annotated: bool | str = True,
    dedupe: bool | str = False,
    max_seconds: float | int | str = 0,
) -> dict[str, Any]:
    profile = yihuan_combat.load_profile(profile_name)
    resolved_profile = str(profile["profile_name"])
    yolo_config = _enemy_health_yolo_config(profile)
    if str(model_name or "").strip():
        yolo_config["model_name"] = str(model_name).strip()
    yolo_config["labels"] = _parse_label_filter(
        labels,
        ("enemy_hp_bar", "enemy_direction_marker", "reward_marker"),
    )
    if conf is not None and str(conf).strip() != "":
        yolo_config["conf"] = min(max(_coerce_float(conf, 0.25), 0.0), 1.0)
    else:
        yolo_config["conf"] = min(float(yolo_config.get("conf") or 0.35), 0.35)
    if iou is not None and str(iou).strip() != "":
        yolo_config["iou"] = min(max(_coerce_float(iou, 0.45), 0.0), 1.0)
    yolo_config["min_width"] = 1
    yolo_config["min_height"] = 1
    max_images_resolved = max(_coerce_int(max_images, 0), 0)
    interval_sec_resolved = max(_coerce_float(interval_sec, 1.0), 0.1)
    max_seconds_resolved = max(_coerce_float(max_seconds, 0.0), 0.0)
    save_annotated_resolved = _coerce_bool(save_annotated)
    dedupe_resolved = _coerce_bool(dedupe)

    root = _combat_targets_dataset_root(dataset_root)
    image_dir = root / "images"
    label_dir = root / "labels"
    annotated_dir = root / "annotated"
    manifest_path = root / "manifest.jsonl"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if save_annotated_resolved:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not _coerce_bool(yolo_config.get("enabled")):
        raise RuntimeError(f"enemy_health_yolo is disabled in profile {resolved_profile!r}.")
    if yolo is None:
        raise RuntimeError("core/yolo service is unavailable.")

    yolo.preload_model(str(yolo_config["model_name"]))
    existing_digests = _load_existing_dataset_digests(manifest_path) if dedupe_resolved else set()
    start_time = time.monotonic()
    deadline = start_time + max_seconds_resolved if max_seconds_resolved > 0 else None
    scanned_count = 0
    saved_count = 0
    no_detection_count = 0
    duplicate_count = 0
    failed_capture_count = 0
    stopped_reason = "max_images" if max_images_resolved > 0 else "cancelled"

    logger.info(
        "CombatTargetsDataset[start] profile=%s model=%s labels=%s conf=%.3f iou=%.3f max_images=%s interval_sec=%.3f root=%s dedupe=%s save_annotated=%s",
        resolved_profile,
        yolo_config["model_name"],
        sorted(yolo_config["labels"]),
        float(yolo_config["conf"]),
        float(yolo_config["iou"]),
        max_images_resolved,
        interval_sec_resolved,
        root,
        dedupe_resolved,
        save_annotated_resolved,
    )

    while max_images_resolved <= 0 or saved_count < max_images_resolved:
        if _combat_cancel_requested():
            stopped_reason = "cancelled"
            break
        if deadline is not None and time.monotonic() >= deadline:
            stopped_reason = "max_seconds"
            break

        scan_started = time.monotonic()
        capture = app.capture()
        scanned_count += 1
        if not getattr(capture, "success", False) or getattr(capture, "image", None) is None:
            failed_capture_count += 1
            logger.warning(
                "CombatTargetsDataset[capture] failed count=%s error=%s",
                failed_capture_count,
                getattr(capture, "error_message", None) or "unknown",
            )
            _sleep_interruptibly(interval_sec_resolved)
            continue

        source_image = capture.image
        boxes, result = _detect_enemy_health_boxes(yolo, source_image, yolo_config=yolo_config)
        if not boxes:
            no_detection_count += 1
            if no_detection_count == 1 or no_detection_count % 30 == 0:
                logger.info(
                    "CombatTargetsDataset[scan] no_detection scanned=%s no_detection=%s provider=%s",
                    scanned_count,
                    no_detection_count,
                    result.get("provider"),
                )
            elapsed = time.monotonic() - scan_started
            _sleep_interruptibly(max(interval_sec_resolved - elapsed, 0.0))
            continue

        digest = _image_digest(source_image)
        if dedupe_resolved and digest in existing_digests:
            duplicate_count += 1
            logger.info("CombatTargetsDataset[scan] duplicate digest=%s skipped=%s", digest, duplicate_count)
            elapsed = time.monotonic() - scan_started
            _sleep_interruptibly(max(interval_sec_resolved - elapsed, 0.0))
            continue

        image_height = int(source_image.shape[0])
        image_width = int(source_image.shape[1])
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = f"combat_targets_{stamp}_{saved_count + 1:05d}_{digest}"
        image_path = image_dir / f"{stem}.png"
        label_path = label_dir / f"{stem}.txt"
        annotated_path = annotated_dir / f"{stem}.png"

        _write_cv_image(image_path, _rgb_capture_to_bgr(source_image))
        _write_yolo_label_file(label_path, boxes, image_width=image_width, image_height=image_height)
        if save_annotated_resolved:
            _write_annotated_preview(annotated_path, source_image, boxes)

        saved_count += 1
        existing_digests.add(digest)
        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "profile_name": resolved_profile,
            "model_name": yolo_config["model_name"],
            "labels": sorted(yolo_config["labels"]),
            "conf": float(yolo_config["conf"]),
            "iou": float(yolo_config["iou"]),
            "provider": result.get("provider"),
            "image": str(image_path),
            "label": str(label_path),
            "annotated": str(annotated_path) if save_annotated_resolved else None,
            "image_width": image_width,
            "image_height": image_height,
            "image_digest": digest,
            "box_count": len(boxes),
            "boxes": boxes,
        }
        with manifest_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "CombatTargetsDataset[save] saved=%s/%s boxes=%s labels=%s image=%s label=%s",
            saved_count,
            max_images_resolved if max_images_resolved > 0 else "unlimited",
            len(boxes),
            sorted({str(box.get("label") or "") for box in boxes}),
            image_path,
            label_path,
        )

        elapsed = time.monotonic() - scan_started
        _sleep_interruptibly(max(interval_sec_resolved - elapsed, 0.0))

    elapsed_total = time.monotonic() - start_time
    logger.info(
        "CombatTargetsDataset[finish] reason=%s scanned=%s saved=%s no_detection=%s duplicate=%s failed_capture=%s elapsed_sec=%.3f root=%s",
        stopped_reason,
        scanned_count,
        saved_count,
        no_detection_count,
        duplicate_count,
        failed_capture_count,
        elapsed_total,
        root,
    )
    return {
        "status": "success" if stopped_reason in {"max_images", "max_seconds"} else "cancelled",
        "stopped_reason": stopped_reason,
        "profile_name": resolved_profile,
        "model_name": yolo_config["model_name"],
        "labels": sorted(yolo_config["labels"]),
        "dataset_root": str(root),
        "images_dir": str(image_dir),
        "labels_dir": str(label_dir),
        "annotated_dir": str(annotated_dir) if save_annotated_resolved else None,
        "manifest_path": str(manifest_path),
        "scanned_count": scanned_count,
        "saved_count": saved_count,
        "no_detection_count": no_detection_count,
        "duplicate_count": duplicate_count,
        "failed_capture_count": failed_capture_count,
        "elapsed_sec": round(elapsed_total, 3),
    }


def _trim_trace(trace: list[dict[str, Any]], trace_limit: int) -> None:
    if len(trace) > trace_limit:
        del trace[: len(trace) - trace_limit]


def _combat_capture_root() -> Path:
    return Path(__file__).resolve().parents[4] / "logs" / "yihuan_combat_debug"


def _sanitize_capture_label(label: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(label or "").strip().lower())
    normalized = normalized.strip("_")
    return normalized or "unknown"


class _CombatCaptureLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        interval_sec: float,
        max_images: int,
        raw_enabled: bool,
        profile_name: str,
        yihuan_combat: YihuanCombatService,
    ) -> None:
        self.enabled = bool(enabled)
        self.interval_sec = max(float(interval_sec), 0.1)
        self.max_images = max(int(max_images), 1)
        self.raw_enabled = bool(raw_enabled)
        self.profile_name = str(profile_name)
        self.yihuan_combat = yihuan_combat
        self.screenshot_dir: Path | None = None
        self.index_path: Path | None = None
        self.screenshots: list[dict[str, Any]] = []
        self.saved_count = 0
        self.periodic_count = 0
        self.event_count = 0
        self.skipped_max_images_count = 0
        self.capture_failed_count = 0
        self.next_periodic_at: float | None = None
        self._last_save_key: tuple[str, str, str, float] | None = None
        self._cid = str(current_cid() or "-").strip() or "-"

    def start(self, *, start_time: float) -> None:
        if not self.enabled:
            return
        self.next_periodic_at = float(start_time) + float(self.interval_sec)

    def should_capture_periodic(self, *, now: float) -> bool:
        if not self.enabled or self.next_periodic_at is None:
            return False
        return float(now) >= float(self.next_periodic_at)

    def advance_periodic_schedule(self, *, now: float) -> None:
        if not self.enabled or self.next_periodic_at is None:
            return
        next_due = float(self.next_periodic_at)
        while next_due <= float(now):
            next_due += float(self.interval_sec)
        self.next_periodic_at = next_due

    def _ensure_output_dir(self) -> Path:
        if self.screenshot_dir is not None:
            return self.screenshot_dir
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = _combat_capture_root() / f"{timestamp}_cid{_sanitize_capture_label(self._cid)}"
        directory.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir = directory
        self.index_path = directory / "index.json"
        return directory

    def _state_summary(
        self,
        state: Mapping[str, Any] | None,
        *,
        phase: str,
        label: str,
        elapsed_sec: float,
        combat_active: bool | None = None,
    ) -> dict[str, Any]:
        payload = dict(state or {})
        return {
            "t": round(float(elapsed_sec), 3),
            "label": str(label),
            "phase": str(phase),
            "in_combat": bool(payload.get("in_combat")),
            "combat_active": bool(combat_active) if combat_active is not None else bool(payload.get("in_combat")),
            "remaining_enemy_marker_found": bool(payload.get("remaining_enemy_marker_found")),
            "front_enemy_found": bool(payload.get("front_enemy_found")),
            "enemy_health_found": bool(payload.get("enemy_health_found")),
            "enemy_health_count": int(payload.get("enemy_health_count") or 0),
            "enemy_direction_found": bool(payload.get("enemy_direction_found")),
            "enemy_direction_count": int(payload.get("enemy_direction_count") or 0),
            "enemy_direction_primary_side": payload.get("enemy_direction_primary_side"),
            "boss_found": bool(payload.get("boss_found")),
            "target_found": bool(payload.get("target_found")),
            "reward_marker_found": bool(payload.get("reward_marker_found")),
            "reward_marker_center_x": payload.get("reward_marker_center_x"),
            "claim_memento_prompt_found": bool(payload.get("claim_memento_prompt_found")),
            "current_slot": payload.get("current_slot"),
            "skill_state": payload.get("skill_state"),
            "ultimate_state": payload.get("ultimate_state"),
            "confidence": payload.get("confidence"),
        }

    def _write_index(self) -> None:
        if self.index_path is None:
            return
        payload = {
            "screenshot_dir": str(self.screenshot_dir) if self.screenshot_dir is not None else None,
            "screenshots": list(self.screenshots),
            "capture_stats": self.result_payload()["capture_stats"],
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save(
        self,
        *,
        kind: str,
        label: str,
        phase: str,
        source_image: Any,
        state: Mapping[str, Any] | None,
        elapsed_sec: float,
        combat_active: bool | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled or source_image is None:
            return None
        if self.saved_count >= self.max_images:
            self.skipped_max_images_count += 1
            return None
        rounded_elapsed = round(float(elapsed_sec), 2)
        save_key = (str(kind), str(label), str(phase), rounded_elapsed)
        if self._last_save_key == save_key:
            return None
        out_dir = self._ensure_output_dir()
        safe_label = _sanitize_capture_label(label)
        annotated_name = f"{self.saved_count:04d}_{float(elapsed_sec):06.2f}s_{safe_label}_annotated.png"
        raw_name = f"{self.saved_count:04d}_{float(elapsed_sec):06.2f}s_{safe_label}_raw.png"
        overlay = {
            "t": round(float(elapsed_sec), 3),
            "phase": str(phase),
            "note": str(label),
            "combat_active": bool(combat_active) if combat_active is not None else bool((state or {}).get("in_combat")),
        }
        annotated_image, resolved_state = self.yihuan_combat.annotate_frame(
            source_image,
            profile_name=self.profile_name,
            state=state,
            overlay=overlay,
        )
        raw_path: Path | None = None
        if self.raw_enabled:
            raw_path = out_dir / raw_name
            cv2.imwrite(str(raw_path), cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))
        annotated_path = out_dir / annotated_name
        cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        entry = {
            "t": round(float(elapsed_sec), 3),
            "kind": str(kind),
            "label": str(label),
            "phase": str(phase),
            "image_path": str(annotated_path),
            "raw_image_path": str(raw_path) if raw_path is not None else None,
            "state_summary": self._state_summary(
                resolved_state,
                phase=phase,
                label=label,
                elapsed_sec=elapsed_sec,
                combat_active=combat_active,
            ),
        }
        self.screenshots.append(entry)
        self.saved_count += 1
        self.periodic_count += 1 if kind == "periodic" else 0
        self.event_count += 1 if kind == "event" else 0
        self._last_save_key = save_key
        self._write_index()
        return entry

    def capture_event(
        self,
        *,
        label: str,
        phase: str,
        source_image: Any,
        state: Mapping[str, Any] | None,
        elapsed_sec: float,
        combat_active: bool | None = None,
    ) -> dict[str, Any] | None:
        return self._save(
            kind="event",
            label=label,
            phase=phase,
            source_image=source_image,
            state=state,
            elapsed_sec=elapsed_sec,
            combat_active=combat_active,
        )

    def capture_periodic(
        self,
        *,
        phase: str,
        source_image: Any,
        state: Mapping[str, Any] | None,
        elapsed_sec: float,
        combat_active: bool | None = None,
    ) -> dict[str, Any] | None:
        return self._save(
            kind="periodic",
            label=f"periodic_{phase}",
            phase=phase,
            source_image=source_image,
            state=state,
            elapsed_sec=elapsed_sec,
            combat_active=combat_active,
        )

    def record_capture_failure(self) -> None:
        self.capture_failed_count += 1

    def result_payload(self) -> dict[str, Any]:
        return {
            "screenshot_dir": str(self.screenshot_dir) if self.screenshot_dir is not None else None,
            "screenshots": list(self.screenshots),
            "capture_stats": {
                "enabled": bool(self.enabled),
                "saved_count": int(self.saved_count),
                "periodic_count": int(self.periodic_count),
                "event_count": int(self.event_count),
                "skipped_max_images_count": int(self.skipped_max_images_count),
                "capture_failed_count": int(self.capture_failed_count),
                "capture_interval_sec": float(self.interval_sec),
                "capture_max_images": int(self.max_images),
                "capture_raw_enabled": bool(self.raw_enabled),
            },
        }


def _capture_periodic_if_due(
    app: Any,
    capture_debug: _CombatCaptureLogger | None,
    *,
    state: Mapping[str, Any] | None,
    phase: str,
    start_time: float,
    combat_active: bool,
) -> Any | None:
    if capture_debug is None:
        return None
    now = time.monotonic()
    if not capture_debug.should_capture_periodic(now=now):
        return None
    capture = app.capture()
    if not capture.success or capture.image is None:
        capture_debug.record_capture_failure()
        capture_debug.advance_periodic_schedule(now=now)
        return None
    capture_debug.capture_periodic(
        phase=phase,
        source_image=capture.image,
        state=state,
        elapsed_sec=now - start_time,
        combat_active=combat_active,
    )
    capture_debug.advance_periodic_schedule(now=now)
    return capture.image


def _capture_event_image(
    capture_debug: _CombatCaptureLogger | None,
    *,
    label: str,
    phase: str,
    source_image: Any,
    state: Mapping[str, Any] | None,
    start_time: float,
    combat_active: bool,
) -> None:
    if capture_debug is None:
        return
    capture_debug.capture_event(
        label=label,
        phase=phase,
        source_image=source_image,
        state=state,
        elapsed_sec=time.monotonic() - start_time,
        combat_active=combat_active,
    )


def _resolve_terminal_capture_image(app: Any, *, fallback_image: Any, capture_debug: _CombatCaptureLogger | None) -> Any:
    if fallback_image is not None or capture_debug is None or not capture_debug.enabled:
        return fallback_image
    try:
        capture = app.capture()
    except Exception:  # noqa: BLE001
        capture_debug.record_capture_failure()
        return fallback_image
    if not capture.success or capture.image is None:
        capture_debug.record_capture_failure()
        return fallback_image
    return capture.image


def _append_state_trace(
    combat_state_trace: list[dict[str, Any]],
    *,
    start_time: float,
    state: Mapping[str, Any],
    combat_active: bool,
    phase: str,
    note: str,
    trace_limit: int,
) -> None:
    combat_state_trace.append(
        {
            "t": round(time.monotonic() - start_time, 3),
            "phase": str(phase),
            "note": note,
            "combat_active": bool(combat_active),
            "in_supported_scene": bool(state.get("in_supported_scene")),
            "in_combat": bool(state.get("in_combat")),
            "remaining_enemy_marker_found": bool(state.get("remaining_enemy_marker_found")),
            "front_enemy_found": _has_front_enemy(state),
            "target_found": bool(state.get("target_found")),
            "target_confidence": state.get("target_confidence"),
            "enemy_health_found": bool(state.get("enemy_health_found")),
            "enemy_health_count": int(state.get("enemy_health_count") or 0),
            "boss_found": bool(state.get("boss_found")),
            "challenge_success_found": bool(state.get("challenge_success_found")),
            "reward_marker_found": bool(state.get("reward_marker_found")),
            "reward_marker_center_x": state.get("reward_marker_center_x"),
            "claim_memento_prompt_found": bool(state.get("claim_memento_prompt_found")),
            "reward_claim_single_button_found": bool(state.get("reward_claim_single_button_found")),
            "reward_claim_double_button_found": bool(state.get("reward_claim_double_button_found")),
            "reward_result_exit_button_found": bool(state.get("reward_result_exit_button_found")),
            "reward_result_retry_button_found": bool(state.get("reward_result_retry_button_found")),
            "current_slot": state.get("current_slot"),
            "skill_available": bool(state.get("skill_available")),
            "skill_state": state.get("skill_state"),
            "ultimate_available": bool(state.get("ultimate_available")),
            "ultimate_state": state.get("ultimate_state"),
            "confidence": state.get("confidence"),
        }
    )
    _trim_trace(combat_state_trace, trace_limit)


def _final_result(
    *,
    status: str,
    stopped_reason: str,
    failure_reason: str | None,
    profile_name: str,
    strategy_name: str,
    encounters_completed: int,
    current_phase: str,
    last_state: Mapping[str, Any] | None,
    combat_state_trace: list[dict[str, Any]],
    action_trace: list[dict[str, Any]],
    start_time: float,
    dry_run: bool,
    capture_debug: _CombatCaptureLogger | None = None,
    post_combat_reward: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "status": status,
        "stopped_reason": stopped_reason,
        "failure_reason": failure_reason,
        "profile_name": profile_name,
        "strategy_name": strategy_name,
        "encounters_completed": int(encounters_completed),
        "current_phase": str(current_phase),
        "last_state": dict(last_state or {}),
        "combat_state_trace": list(combat_state_trace),
        "action_trace": list(action_trace),
        "elapsed_sec": round(time.monotonic() - start_time, 3),
        "dry_run": bool(dry_run),
        "post_combat_reward": dict(post_combat_reward or {}),
    }
    if capture_debug is not None:
        result.update(capture_debug.result_payload())
    else:
        result.update(
            {
                "screenshot_dir": None,
                "screenshots": [],
                "capture_stats": {
                    "enabled": False,
                    "saved_count": 0,
                    "periodic_count": 0,
                    "event_count": 0,
                    "skipped_max_images_count": 0,
                    "capture_failed_count": 0,
                    "capture_interval_sec": 0.0,
                    "capture_max_images": 0,
                    "capture_raw_enabled": False,
                },
            }
        )
    return result


def _record_action(
    action_trace: list[dict[str, Any]],
    *,
    start_time: float,
    action: str,
    dry_run: bool,
    details: Mapping[str, Any] | None = None,
) -> None:
    entry = {
        "t": round(time.monotonic() - start_time, 3),
        "action": action,
        "dry_run": bool(dry_run),
    }
    if details:
        entry.update(dict(details))
    action_trace.append(entry)
    logger.info(
        "Combat[action] action=%s dry_run=%s details=%s",
        action,
        bool(dry_run),
        dict(details or {}),
    )


def _perform_binding_tap(app: Any, binding: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    normalized = str(binding or "").strip().lower()
    if normalized == "mouse_left":
        app.click(button="left")
    elif normalized == "mouse_right":
        app.click(button="right")
    elif normalized == "mouse_middle":
        app.click(button="middle")
    else:
        app.press_key(str(binding), presses=1)


def _binding_to_input_mapping(binding: str) -> dict[str, Any]:
    normalized = str(binding or "").strip().lower()
    if normalized == "mouse_left":
        return {"type": "mouse_button", "button": "left", "action_name": binding}
    if normalized == "mouse_right":
        return {"type": "mouse_button", "button": "right", "action_name": binding}
    if normalized == "mouse_middle":
        return {"type": "mouse_button", "button": "middle", "action_name": binding}
    return {"type": "key", "key": str(binding), "action_name": binding}


def _execute_input_mapping_phase_with_retry(
    app: Any,
    input_mapping: Any,
    binding: Mapping[str, Any],
    *,
    phase: str,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    if dry_run:
        return
    _call_with_focus_retry(
        app,
        lambda: input_mapping.execute_binding(binding, phase=phase, app=app),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )


def _tap_input_binding_with_retry(
    app: Any,
    input_mapping: Any,
    binding: str,
    *,
    hold_ms: int,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    resolved_binding = _binding_to_input_mapping(binding)
    normalized_hold_ms = max(int(hold_ms), 0)
    if normalized_hold_ms <= 0:
        _execute_input_mapping_phase_with_retry(
            app,
            input_mapping,
            resolved_binding,
            phase="tap",
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            retry_reason=retry_reason,
        )
        return
    held = False
    _execute_input_mapping_phase_with_retry(
        app,
        input_mapping,
        resolved_binding,
        phase="hold",
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )
    held = True
    try:
        _sleep_ms_uninterruptible(normalized_hold_ms, dry_run=dry_run)
    finally:
        if held:
            _execute_input_mapping_phase_with_retry(
                app,
                input_mapping,
                resolved_binding,
                phase="release",
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason=retry_reason,
            )


def _call_with_focus_retry(
    app: Any,
    callback: Any,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    try:
        callback()
        return
    except Exception as exc:  # noqa: BLE001
        if not _is_window_focus_error(exc):
            raise
        if not _try_focus_activation(
            app,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            reason=retry_reason,
        ):
            raise
        callback()


def _perform_move_click_with_retry(
    app: Any,
    x: int,
    y: int,
    *,
    button: str = "left",
    move_duration_sec: float = 0.10,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    if dry_run:
        return
    click_x = int(x)
    click_y = int(y)
    normalized_button = str(button or "left").strip().lower()
    duration = max(float(move_duration_sec), 0.0)

    def _move_and_click() -> None:
        app.move_to(click_x, click_y, duration=duration)
        _sleep_ms_uninterruptible(35, dry_run=False)
        app.click(button=normalized_button)

    _call_with_focus_retry(
        app,
        _move_and_click,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )


def _is_window_focus_error(exc: BaseException) -> bool:
    if isinstance(exc, TargetRuntimeError):
        return exc.code == "window_focus_required"
    return "could not be focused" in str(exc).lower()


def _try_focus_activation(
    app: Any,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    reason: str,
) -> bool:
    if dry_run or not hasattr(app, "focus_with_input"):
        return False
    _record_action(
        action_trace,
        start_time=start_time,
        action="focus_with_input",
        dry_run=dry_run,
        details={"reason": reason},
    )
    try:
        return bool(app.focus_with_input(click_delay=0.15))
    except Exception as exc:  # noqa: BLE001
        action_trace[-1]["success"] = False
        action_trace[-1]["error"] = str(exc)
        return False


def _perform_binding_tap_with_retry(
    app: Any,
    binding: str,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    _call_with_focus_retry(
        app,
        lambda: _perform_binding_tap(app, binding, dry_run=dry_run),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )


def _perform_look_delta_with_retry(
    app: Any,
    dx: int,
    dy: int,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    if dry_run:
        return
    total_dx = int(dx)
    total_dy = int(dy)
    if total_dx == 0 and total_dy == 0:
        return
    step_pixels = max(int(_LOOK_SMOOTH_STEP_PIXELS), 1)
    steps = max(
        int((abs(total_dx) + step_pixels - 1) / step_pixels),
        int((abs(total_dy) + step_pixels - 1) / step_pixels),
        1,
    )
    sent_dx = 0
    sent_dy = 0
    for index in range(steps):
        next_dx_total = int(round(total_dx * float(index + 1) / float(steps)))
        next_dy_total = int(round(total_dy * float(index + 1) / float(steps)))
        step_dx = next_dx_total - sent_dx
        step_dy = next_dy_total - sent_dy
        sent_dx = next_dx_total
        sent_dy = next_dy_total
        if step_dx == 0 and step_dy == 0:
            continue
        _call_with_focus_retry(
            app,
            lambda step_dx=step_dx, step_dy=step_dy: app.look_delta(int(step_dx), int(step_dy)),
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            retry_reason=retry_reason,
        )
        if index + 1 < steps:
            _sleep_ms_uninterruptible(_LOOK_SMOOTH_STEP_DELAY_MS, dry_run=False)


def _perform_mouse_tap_without_move_with_retry(
    app: Any,
    button: str,
    *,
    hold_ms: int,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    if dry_run:
        return
    normalized_button = str(button or "left").strip().lower()
    down_sent = False
    _call_with_focus_retry(
        app,
        lambda: app.mouse_down(normalized_button),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )
    down_sent = True
    try:
        _sleep_ms_uninterruptible(max(int(hold_ms), 0), dry_run=dry_run)
    finally:
        if down_sent:
            _call_with_focus_retry(
                app,
                lambda: app.mouse_up(normalized_button),
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason=retry_reason,
            )


def _tap_binding(
    app: Any,
    binding: str,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    action_name: str,
) -> None:
    _record_action(
        action_trace,
        start_time=start_time,
        action=action_name,
        dry_run=dry_run,
        details={"binding": binding},
    )
    _perform_binding_tap_with_retry(
        app,
        binding,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=action_name,
    )


def _normal_attack(
    app: Any,
    binding: str,
    *,
    mode: str,
    duration_ms: int,
    interval_ms: int,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    interrupt_checker: Any = None,
) -> dict[str, Any]:
    normalized_mode = str(mode or "tap_spam").strip().lower()
    tap_interval_ms = max(int(interval_ms), 1)
    planned_taps = max(1, int((max(int(duration_ms), 0) + tap_interval_ms - 1) / tap_interval_ms))
    tap_hold_ms = min(max(int(round(tap_interval_ms * 0.25)), 12), 24)
    details = {
        "binding": binding,
        "mode": normalized_mode,
        "duration_ms": int(duration_ms),
        "interval_ms": tap_interval_ms,
        "planned_taps": planned_taps,
        "tap_hold_ms": tap_hold_ms,
    }
    _record_action(
        action_trace,
        start_time=start_time,
        action="normal",
        dry_run=dry_run,
        details=details,
    )
    if normalized_mode == "hold":
        normalized = str(binding or "mouse_left").strip().lower()
        duration_sec = max(int(duration_ms), 0) / 1000.0
        if not dry_run:
            if normalized.startswith("mouse_"):
                button = normalized.replace("mouse_", "", 1)
                app.mouse_down(button)
                if duration_sec > 0:
                    time.sleep(duration_sec)
                app.mouse_up(button)
            else:
                app.key_down(str(binding))
                if duration_sec > 0:
                    time.sleep(duration_sec)
                app.key_up(str(binding))
        action_trace[-1]["tap_count"] = 1
        return {"executed": True, "action": "normal", "tap_count": 1}

    tap_count = 0
    interrupt_reason: str | None = None
    interrupt_payload: Mapping[str, Any] | None = None
    normalized_binding = str(binding or "mouse_left").strip().lower()
    _raise_if_cancelled()
    for tap_index in range(planned_taps):
        if interrupt_checker is not None:
            interrupt_payload = interrupt_checker()
            if interrupt_payload is not None:
                interrupt_reason = str(interrupt_payload.get("reason") or "interrupt")
                break
        if normalized_binding == "mouse_left":
            _perform_mouse_tap_without_move_with_retry(
                app,
                "left",
                hold_ms=tap_hold_ms,
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason="normal",
            )
        else:
            _perform_binding_tap_with_retry(
                app,
                binding,
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason="normal",
            )
        tap_count += 1
        if interrupt_checker is not None:
            interrupt_payload = interrupt_checker()
            if interrupt_payload is not None:
                interrupt_reason = str(interrupt_payload.get("reason") or "interrupt")
                break
        if tap_index + 1 < planned_taps:
            _sleep_ms_uninterruptible(tap_interval_ms, dry_run=dry_run)

    action_trace[-1]["tap_count"] = tap_count
    if interrupt_reason is not None:
        action_trace[-1]["interrupt_reason"] = interrupt_reason
    _raise_if_cancelled()
    return {
        "executed": tap_count > 0,
        "action": "normal",
        "tap_count": tap_count,
        "interrupt_reason": interrupt_reason,
        "interrupt_payload": dict(interrupt_payload or {}),
    }


def _release_inputs(app: Any, profile: Mapping[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    if hasattr(app, "release_all"):
        try:
            app.release_all()
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Combat[release] release_all fallback failed: %s", exc)
    try:
        app.mouse_up("left")
        app.mouse_up("right")
        app.mouse_up("middle")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Combat[release] failed to release mouse button: %s", exc)
    for key in profile.get("release_keys") or []:
        try:
            app.key_up(str(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Combat[release] failed to release key %s: %s", key, exc)


def _perform_key_down_with_retry(
    app: Any,
    key: str,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    if dry_run:
        return
    _call_with_focus_retry(
        app,
        lambda: app.key_down(str(key)),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )


def _perform_key_up_with_retry(
    app: Any,
    key: str,
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    retry_reason: str,
) -> None:
    if dry_run:
        return
    _call_with_focus_retry(
        app,
        lambda: app.key_up(str(key)),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=retry_reason,
    )


def _clamp_signed(value: float, *, minimum_abs: int, maximum_abs: int) -> int:
    if value == 0:
        return 0
    sign = 1 if value > 0 else -1
    magnitude = min(max(int(round(abs(value))), int(minimum_abs)), int(maximum_abs))
    return sign * magnitude


def _reward_turn_pixels_for_error(
    error_x: float,
    *,
    align_tolerance_px: int,
    look_pixels_per_px: float,
    min_turn_pixels: int,
    max_turn_pixels: int,
) -> int:
    distance = abs(float(error_x))
    tolerance = max(float(align_tolerance_px), 0.0)
    if distance <= tolerance:
        return 0
    # Reduce correction near the deadzone so repeated scan/turn cycles converge instead of bouncing.
    effective_distance = max(distance - tolerance, 0.0)
    # Match mouse movement to the marker's screen-side: marker left means look left,
    # marker right means look right.
    signed_distance = effective_distance if error_x > 0 else -effective_distance
    return _clamp_signed(
        signed_distance * float(look_pixels_per_px),
        minimum_abs=int(min_turn_pixels),
        maximum_abs=int(max_turn_pixels),
    ) if error_x else 0


def _reward_align_target_x(config: Mapping[str, Any], *, capture_width: int) -> float:
    configured = config.get("align_target_x")
    if configured is None:
        return float(max(int(capture_width), 1)) / 2.0
    target_x = float(_coerce_int(configured, 640))
    return min(max(target_x, 1.0), float(max(int(capture_width), 1)))


def _sleep_ms(milliseconds: int, *, dry_run: bool) -> None:
    _raise_if_cancelled()
    if dry_run:
        return
    seconds = max(int(milliseconds), 0) / 1000.0
    if seconds > 0:
        _sleep_interruptibly(seconds)


def _sleep_ms_uninterruptible(milliseconds: int, *, dry_run: bool) -> None:
    if dry_run:
        return
    seconds = max(int(milliseconds), 0) / 1000.0
    if seconds > 0:
        time.sleep(seconds)


def _is_available(
    state: Mapping[str, Any],
    name: str,
    *,
    ability_lockouts: Mapping[str, float] | None = None,
    now: float | None = None,
) -> bool:
    if not bool(state.get(f"{name}_available")):
        return False
    if ability_lockouts is None:
        return True
    lock_until = float(ability_lockouts.get(name, 0.0))
    current = now if now is not None else time.monotonic()
    return current >= lock_until


def _has_front_enemy(state: Mapping[str, Any]) -> bool:
    if "front_enemy_found" in state:
        return bool(state.get("front_enemy_found"))
    return bool(state.get("enemy_health_found") or state.get("boss_found"))


def _lock_ability(
    ability_lockouts: dict[str, float],
    profile: Mapping[str, Any],
    name: str,
    *,
    now: float | None = None,
) -> None:
    runtime = dict(profile.get("runtime") or {})
    lockout_ms = int(runtime.get(f"{name}_lockout_ms") or 0)
    if lockout_ms <= 0:
        return
    current = now if now is not None else time.monotonic()
    ability_lockouts[name] = current + (lockout_ms / 1000.0)


def _resolve_normal_command(profile: Mapping[str, Any], strategy_name: str) -> dict[str, Any]:
    strategies = dict(profile.get("strategies") or {})
    strategy = dict(strategies.get(strategy_name) or {})
    commands = strategy.get("loop") or []
    if isinstance(commands, list):
        for command in commands:
            if isinstance(command, Mapping) and "normal" in command:
                payload = command["normal"] or {}
                if isinstance(payload, Mapping):
                    return {
                        "mode": str(payload.get("mode") or "tap_spam"),
                        "duration_ms": _coerce_int(payload.get("duration_ms"), 560),
                        "interval_ms": _coerce_int(payload.get("interval_ms"), 70),
                    }
                return {
                    "mode": "tap_spam",
                    "duration_ms": _coerce_int(payload, 560),
                    "interval_ms": 70,
                }
    return {"mode": "tap_spam", "duration_ms": 560, "interval_ms": 70}


def _resolve_runtime_schedule(profile: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(profile.get("runtime") or {})
    ability_schedule = dict(profile.get("ability_schedule") or {})
    switch_policy = dict(profile.get("switch_policy") or {})
    input_timing = dict(profile.get("input_timing") or {})
    combat_exit = dict(profile.get("combat_exit") or {})
    enemy_direction_reacquire = dict(profile.get("enemy_direction_reacquire") or {})
    return {
        "monitor_scan_interval_sec": max(float(runtime.get("monitor_scan_interval_sec") or 2.0), 0.1),
        "combat_scan_interval_sec": max(float(runtime.get("combat_scan_interval_sec") or 3.0), 0.1),
        "action_loop_sleep_sec": max(float(runtime.get("action_loop_sleep_ms") or runtime.get("poll_ms") or 80) / 1000.0, 0.01),
        "skill_interval_sec": max(float(ability_schedule.get("skill_interval_sec") or 3.0), 0.1),
        "ultimate_interval_sec": max(float(ability_schedule.get("ultimate_interval_sec") or 5.0), 0.1),
        "immediate_on_combat_enter": bool(ability_schedule.get("immediate_on_combat_enter", True)),
        "switch_interval_sec": max(float(switch_policy.get("switch_interval_sec") or 10.0), 0.1),
        "post_switch_delay_sec": max(int(switch_policy.get("post_switch_delay_ms") or 0) / 1000.0, 0.0),
        "failed_switch_cooldown_sec": max(int(switch_policy.get("failed_switch_cooldown_ms") or 0) / 1000.0, 0.0),
        "switch_confirm_required_matches": max(int(switch_policy.get("confirm_required_matches") or 2), 1),
        "skill_press_ms": max(int(input_timing.get("skill_press_ms") or 45), 0),
        "ultimate_press_ms": max(int(input_timing.get("ultimate_press_ms") or 55), 0),
        "switch_press_ms": max(int(input_timing.get("switch_press_ms") or 35), 0),
        "exit_confirm_required_scans": max(int(combat_exit.get("confirm_required_scans") or 2), 1),
        "exit_confirm_interval_sec": max(float(combat_exit.get("confirm_interval_sec") or 1.0), 0.1),
        "exit_confirm_missing_sec": max(float(combat_exit.get("confirm_missing_sec") or 3.0), 0.0),
        "exit_post_cooldown_sec": max(int(combat_exit.get("post_exit_cooldown_ms") or 0) / 1000.0, 0.0),
        "exit_challenge_success_immediate": bool(combat_exit.get("challenge_success_immediate", True)),
        "exit_reset_on_reacquire": bool(combat_exit.get("reset_on_reacquire", True)),
        "rear_enemy_turn_interval_sec": max(float(enemy_direction_reacquire.get("turn_interval_sec") or 0.9), 0.1),
        "rear_enemy_turn_pixels_side": max(int(enemy_direction_reacquire.get("turn_pixels_side") or 220), 1),
        "rear_enemy_turn_pixels_bottom": max(int(enemy_direction_reacquire.get("turn_pixels_bottom") or 340), 1),
        "rear_enemy_vertical_delta": int(enemy_direction_reacquire.get("vertical_delta") or 0),
        "rear_enemy_post_turn_delay_sec": max(
            int(enemy_direction_reacquire.get("post_turn_delay_ms") or 0) / 1000.0,
            0.0,
        ),
        "rear_enemy_auto_target_after_turn": bool(enemy_direction_reacquire.get("auto_target_after_turn", True)),
    }


def _scan_interval_sec(*, combat_active: bool, schedule: Mapping[str, Any]) -> float:
    return float(
        schedule["combat_scan_interval_sec"] if combat_active else schedule["monitor_scan_interval_sec"]
    )


def _enqueue_next_scan(*, now: float, combat_active: bool, schedule: Mapping[str, Any]) -> float:
    return now + _scan_interval_sec(combat_active=combat_active, schedule=schedule)


def _trace_scan(
    combat_state_trace: list[dict[str, Any]],
    *,
    start_time: float,
    state: Mapping[str, Any],
    combat_active: bool,
    phase: str,
    note: str,
    trace_limit: int,
) -> None:
    _append_state_trace(
        combat_state_trace,
        start_time=start_time,
        state=state,
        combat_active=combat_active,
        phase=phase,
        note=note,
        trace_limit=trace_limit,
    )
    logger.info(
        "Combat[state] phase=%s note=%s combat_active=%s in_supported_scene=%s in_combat=%s remaining_enemy_marker_found=%s front_enemy_found=%s enemy_health_found=%s enemy_health_count=%s enemy_direction_found=%s enemy_direction_count=%s enemy_direction_side=%s boss_found=%s target_found=%s reward_marker_found=%s reward_marker_center_x=%s claim_memento_prompt_found=%s current_slot=%s skill_state=%s ultimate_state=%s confidence=%s",
        phase,
        note,
        bool(combat_active),
        bool(state.get("in_supported_scene")),
        bool(state.get("in_combat")),
        bool(state.get("remaining_enemy_marker_found")),
        _has_front_enemy(state),
        bool(state.get("enemy_health_found")),
        int(state.get("enemy_health_count") or 0),
        bool(state.get("enemy_direction_found")),
        int(state.get("enemy_direction_count") or 0),
        state.get("enemy_direction_primary_side"),
        bool(state.get("boss_found")),
        bool(state.get("target_found")),
        bool(state.get("reward_marker_found")),
        state.get("reward_marker_center_x"),
        bool(state.get("claim_memento_prompt_found")),
        state.get("current_slot"),
        state.get("skill_state"),
        state.get("ultimate_state"),
        state.get("confidence"),
    )


def _switch_slot(
    app: Any,
    input_mapping: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    target: Any,
    *,
    yihuan_combat: YihuanCombatService,
    profile_name: str,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    press_ms: int = 0,
    fallback_current_slot: int | None = None,
) -> dict[str, Any]:
    team_size = max(_coerce_int(state.get("team_size"), len(profile.get("current_slot_regions") or [])), 1)
    current_slot = state.get("current_slot")
    slot_source = "state"
    if current_slot is None and fallback_current_slot is not None:
        current_slot = fallback_current_slot
        slot_source = "fallback"
    if current_slot is None:
        _record_action(
            action_trace,
            start_time=start_time,
            action="skip_switch",
            dry_run=dry_run,
            details={"reason": "current_slot_unknown", "slot_source": slot_source},
        )
        return {
            "attempted": False,
            "executed": False,
            "reason": "current_slot_unknown",
            "slot": None,
            "state": dict(state),
            "capture_image": None,
        }

    if str(target).lower() == "next":
        slot = (int(current_slot) % team_size) + 1
    else:
        slot = min(max(_coerce_int(target, 1), 1), team_size)

    binding = dict(profile["keys"]).get(f"switch_{slot}", str(slot))
    _record_action(
        action_trace,
        start_time=start_time,
        action="switch",
        dry_run=dry_run,
        details={"binding": binding, "slot": slot, "press_ms": max(int(press_ms), 0), "slot_source": slot_source},
    )
    _tap_input_binding_with_retry(
        app,
        input_mapping,
        binding,
        hold_ms=press_ms,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason="switch",
    )
    action_trace[-1]["slot"] = slot
    if dry_run:
        action_trace[-1]["confirmed"] = True
        return {
            "attempted": True,
            "executed": True,
            "success": True,
            "slot": slot,
            "state": dict(state),
            "capture_image": None,
        }

    confirmed, confirmed_state, confirmed_capture_image = _confirm_switch_slot(
        app,
        yihuan_combat,
        profile_name=profile_name,
        expected_slot=slot,
        confirm_timeout_ms=int(dict(profile.get("switch_policy") or {}).get("confirm_timeout_ms", 0)),
        required_matches=int(dict(profile.get("switch_policy") or {}).get("confirm_required_matches", 1)),
        poll_ms=max(int(dict(profile.get("runtime") or {}).get("poll_ms", 80)), 10),
    )
    action_trace[-1]["confirmed"] = bool(confirmed)
    if not confirmed:
        _record_action(
            action_trace,
            start_time=start_time,
            action="switch_confirm_failed",
            dry_run=dry_run,
            details={"slot": slot},
        )
    return {
        "attempted": True,
        "executed": bool(confirmed),
        "success": bool(confirmed),
        "slot": slot,
        "state": dict(confirmed_state or state),
        "capture_image": confirmed_capture_image,
    }


def _confirm_switch_slot(
    app: Any,
    yihuan_combat: YihuanCombatService,
    *,
    profile_name: str,
    expected_slot: int,
    confirm_timeout_ms: int,
    required_matches: int,
    poll_ms: int,
) -> tuple[bool, dict[str, Any] | None, Any | None]:
    if confirm_timeout_ms <= 0:
        state, capture = _capture_state(app, yihuan_combat, profile_name=profile_name)
        return state.get("current_slot") == expected_slot, state, capture.image

    attempts = max(1, int((max(confirm_timeout_ms, 1) + max(poll_ms, 1) - 1) / max(poll_ms, 1)))
    last_state: dict[str, Any] | None = None
    last_capture_image: Any = None
    consecutive_matches = 0
    required_match_count = max(int(required_matches), 1)
    for attempt in range(attempts):
        state, capture = _capture_state(app, yihuan_combat, profile_name=profile_name)
        last_state = dict(state)
        last_capture_image = capture.image
        if state.get("current_slot") == expected_slot:
            consecutive_matches += 1
            if consecutive_matches >= required_match_count:
                return True, last_state, last_capture_image
        else:
            consecutive_matches = 0
        if attempt + 1 < attempts:
            _sleep_ms(poll_ms, dry_run=False)
    return False, last_state, last_capture_image


def _check_combat_interrupt(
    app: Any,
    yihuan_combat: YihuanCombatService,
    *,
    profile_name: str,
    ability_lockouts: Mapping[str, float],
    audio_dodge_runtime: AudioDodgeRuntime | None,
) -> dict[str, Any] | None:
    if audio_dodge_runtime is not None:
        trigger_event = audio_dodge_runtime.consume_trigger()
        if trigger_event is not None:
            return {"reason": "audio_dodge", "trigger_event": dict(trigger_event)}

    refreshed_state, _capture = _capture_state(app, yihuan_combat, profile_name=profile_name)
    if not bool(refreshed_state.get("in_combat")):
        return {"reason": "combat_ended", "state": dict(refreshed_state)}
    if _is_available(refreshed_state, "ultimate", ability_lockouts=ability_lockouts) or _is_available(
        refreshed_state,
        "skill",
        ability_lockouts=ability_lockouts,
    ):
        return {"reason": "ability_ready", "state": dict(refreshed_state)}
    return None


def _execute_command(
    app: Any,
    input_mapping: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    command: Any,
    *,
    ability_lockouts: dict[str, float],
    yihuan_combat: YihuanCombatService,
    profile_name: str,
    audio_dodge_runtime: AudioDodgeRuntime | None,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    keys = dict(profile["keys"])
    cooldown_ms = int(dict(profile["runtime"])["input_cooldown_ms"])
    input_timing = dict(profile.get("input_timing") or {})

    if isinstance(command, str):
        name = command.strip()
        if name in {"skill", "ultimate"}:
            _record_action(
                action_trace,
                start_time=start_time,
                action=name,
                dry_run=dry_run,
                details={
                    "binding": keys[name],
                    "press_ms": max(int(input_timing.get(f"{name}_press_ms") or 0), 0),
                },
            )
            _tap_input_binding_with_retry(
                app,
                input_mapping,
                keys[name],
                hold_ms=max(int(input_timing.get(f"{name}_press_ms") or 0), 0),
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason=name,
            )
            _lock_ability(ability_lockouts, profile, name)
            _sleep_ms(cooldown_ms, dry_run=dry_run)
            return {"executed": True, "action": name, "used_ability": name}
        if name == "normal":
            result = _normal_attack(
                app,
                keys["normal_attack"],
                mode="tap_spam",
                duration_ms=560,
                interval_ms=70,
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                interrupt_checker=lambda: _check_combat_interrupt(
                    app,
                    yihuan_combat,
                    profile_name=profile_name,
                    ability_lockouts=ability_lockouts,
                    audio_dodge_runtime=audio_dodge_runtime,
                ),
            )
            _sleep_ms(cooldown_ms, dry_run=dry_run)
            return result
        raise ValueError(f"Unsupported combat command: {command!r}")

    if not isinstance(command, Mapping):
        raise ValueError(f"Unsupported combat command type: {type(command).__name__}")

    if "if_available" in command:
        payload = command["if_available"]
        if isinstance(payload, str):
            name = payload
            nested = [name]
        elif isinstance(payload, Mapping):
            name = str(payload.get("name") or "")
            nested_payload = payload.get("then") or [name]
            nested = nested_payload if isinstance(nested_payload, list) else [nested_payload]
        else:
            raise ValueError("if_available command must be a string or mapping.")
        if name not in {"skill", "ultimate"}:
            raise ValueError(f"Unsupported availability target: {name!r}")
        if _is_available(state, name, ability_lockouts=ability_lockouts):
            for nested_command in nested:
                result = _execute_command(
                    app,
                    input_mapping,
                    profile,
                    state,
                    nested_command,
                    ability_lockouts=ability_lockouts,
                    yihuan_combat=yihuan_combat,
                    profile_name=profile_name,
                    audio_dodge_runtime=audio_dodge_runtime,
                    dry_run=dry_run,
                    action_trace=action_trace,
                    start_time=start_time,
                )
                if result.get("executed") or result.get("interrupt_reason"):
                    return result
            return {"executed": False}
        else:
            reason = "not_available"
            if bool(state.get(f"{name}_available")):
                reason = "lockout"
            _record_action(
                action_trace,
                start_time=start_time,
                action=f"skip_{name}",
                dry_run=dry_run,
                details={"reason": reason},
            )
        return {"executed": False, "action": f"skip_{name}", "skip_reason": reason}

    if "normal" in command:
        payload = command["normal"] or {}
        duration_ms = 560
        interval_ms = 70
        mode = "tap_spam"
        if isinstance(payload, Mapping):
            duration_ms = _coerce_int(payload.get("duration_ms"), 560)
            interval_ms = _coerce_int(payload.get("interval_ms"), 70)
            mode = str(payload.get("mode") or "tap_spam")
        elif payload is not None:
            duration_ms = _coerce_int(payload, 560)
        result = _normal_attack(
            app,
            keys["normal_attack"],
            mode=mode,
            duration_ms=duration_ms,
            interval_ms=interval_ms,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            interrupt_checker=lambda: _check_combat_interrupt(
                app,
                yihuan_combat,
                profile_name=profile_name,
                ability_lockouts=ability_lockouts,
                audio_dodge_runtime=audio_dodge_runtime,
            ),
        )
        _sleep_ms(cooldown_ms, dry_run=dry_run)
        return result

    if "keypress" in command:
        payload = command["keypress"]
        binding = payload.get("key") if isinstance(payload, Mapping) else payload
        _tap_binding(
            app,
            str(binding),
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            action_name="keypress",
        )
        _sleep_ms(cooldown_ms, dry_run=dry_run)
        return {"executed": True, "action": "keypress"}

    if "wait" in command:
        payload = command["wait"]
        duration_ms = payload.get("duration_ms") if isinstance(payload, Mapping) else payload
        duration_ms = _coerce_int(duration_ms, 300)
        _record_action(
            action_trace,
            start_time=start_time,
            action="wait",
            dry_run=dry_run,
            details={"duration_ms": duration_ms},
        )
        _sleep_ms(duration_ms, dry_run=dry_run)
        return {"executed": True, "action": "wait"}

    if "switch" in command:
        result = _switch_slot(
            app,
            input_mapping,
            profile,
            state,
            command["switch"],
            yihuan_combat=yihuan_combat,
            profile_name=profile_name,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            press_ms=max(int(input_timing.get("switch_press_ms") or 0), 0),
        )
        _sleep_ms(int(dict(profile["runtime"]).get("input_cooldown_ms", 120)), dry_run=dry_run)
        return {
            "executed": bool(result.get("executed")),
            "action": "switch",
            "state": dict(result.get("state") or state),
            "switch_success": bool(result.get("success")),
            "slot": result.get("slot"),
        }

    raise ValueError(f"Unsupported combat command: {command!r}")


def _execute_strategy(
    app: Any,
    input_mapping: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    strategy_name: str,
    ability_lockouts: dict[str, float],
    yihuan_combat: YihuanCombatService,
    profile_name: str,
    audio_dodge_runtime: AudioDodgeRuntime | None,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    strategies = dict(profile.get("strategies") or {})
    strategy = dict(strategies.get(strategy_name) or {})
    if not strategy:
        raise ValueError(f"Combat strategy not found: {strategy_name}")
    commands = strategy.get("loop") or []
    if not isinstance(commands, list):
        raise ValueError(f"Combat strategy '{strategy_name}' loop must be a list.")
    for command in commands:
        result = _execute_command(
            app,
            input_mapping,
            profile,
            state,
            command,
            ability_lockouts=ability_lockouts,
            yihuan_combat=yihuan_combat,
            profile_name=profile_name,
            audio_dodge_runtime=audio_dodge_runtime,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
        )
        if result.get("interrupt_reason"):
            return result
        if result.get("executed"):
            return result
    return {"executed": False, "action": None}


def _pause_after_encounter(
    app: Any,
    profile: Mapping[str, Any],
    *,
    dry_run: bool,
    cooldown_ms: int,
) -> None:
    _release_inputs(app, profile, dry_run=dry_run)
    _sleep_ms(cooldown_ms, dry_run=dry_run)


def _stage_enter_button_center(state: Mapping[str, Any]) -> tuple[int, int] | None:
    center_x = state.get("stage_enter_button_center_x")
    center_y = state.get("stage_enter_button_center_y")
    if center_x is not None and center_y is not None:
        return int(round(float(center_x))), int(round(float(center_y)))
    box = state.get("stage_enter_button_box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x, y, width, height = [float(item) for item in box]
        return int(round(x + width / 2.0)), int(round(y + height / 2.0))
    return None


def _click_stage_enter_button(
    app: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    capture_debug: _CombatCaptureLogger | None,
    source_image: Any,
    start_time: float,
) -> bool:
    config = dict(profile.get("stage_entry") or {})
    if not bool(config.get("enabled", False)) or not bool(state.get("stage_enter_button_found")):
        return False
    move_duration_sec = max(_coerce_float(config.get("click_move_duration_sec"), 0.10), 0.0)
    center = _stage_enter_button_center(state)
    if center is None:
        _record_action(
            action_trace,
            start_time=start_time,
            action="stage_entry_button_invalid",
            dry_run=dry_run,
            details={"confidence": state.get("stage_enter_button_confidence")},
        )
        return False

    click_x, click_y = center
    _capture_event_image(
        capture_debug,
        label="stage_entry_button_found",
        phase="stage_entry",
        source_image=source_image,
        state=state,
        start_time=start_time,
        combat_active=False,
    )
    _record_action(
        action_trace,
        start_time=start_time,
        action="stage_entry_button_found",
        dry_run=dry_run,
        details={
            "x": int(click_x),
            "y": int(click_y),
            "confidence": state.get("stage_enter_button_confidence"),
            "box": list(state.get("stage_enter_button_box") or []),
        },
    )
    _record_action(
        action_trace,
        start_time=start_time,
        action="stage_entry_click",
        dry_run=dry_run,
        details={"x": int(click_x), "y": int(click_y)},
    )
    if not dry_run:
        _perform_move_click_with_retry(
            app,
            int(click_x),
            int(click_y),
            button="left",
            move_duration_sec=move_duration_sec,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            retry_reason="stage_entry_click",
        )
    _sleep_ms(max(_coerce_int(config.get("wait_after_click_ms"), 800), 0), dry_run=dry_run)
    return True


def _wait_for_stage_started(
    app: Any,
    yihuan_combat: YihuanCombatService,
    profile: Mapping[str, Any],
    *,
    profile_name: str,
    combat_targets_yolo: _CombatTargetsYoloRuntime | None,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    combat_state_trace: list[dict[str, Any]],
    capture_debug: _CombatCaptureLogger | None,
    start_time: float,
    trace_limit: int,
    session_deadline: float | None = None,
) -> dict[str, Any]:
    config = dict(profile.get("stage_entry") or {})
    timeout_sec = max(_coerce_float(config.get("stage_start_timeout_sec"), 30.0), 0.1)
    scan_interval_sec = max(_coerce_float(config.get("scan_interval_sec"), 0.5), 0.05)
    stage_confirm_sec = max(_coerce_float(config.get("stage_confirm_sec"), 3.0), 0.0)
    deadline = time.monotonic() + timeout_sec
    marker_seen_started_at: float | None = None
    _record_action(
        action_trace,
        start_time=start_time,
        action="stage_enter_wait_started",
        dry_run=dry_run,
        details={
            "timeout_sec": round(timeout_sec, 3),
            "scan_interval_sec": round(scan_interval_sec, 3),
            "stage_confirm_sec": round(stage_confirm_sec, 3),
        },
    )

    last_state: dict[str, Any] | None = None
    last_image: Any = None
    while True:
        _raise_if_cancelled()
        now = time.monotonic()
        if session_deadline is not None and now >= session_deadline:
            return {
                "status": "timeout",
                "reason": "session_deadline",
                "state": last_state,
                "capture_image": last_image,
            }
        if now > deadline:
            _record_action(
                action_trace,
                start_time=start_time,
                action="stage_enter_timeout",
                dry_run=dry_run,
                details={"elapsed_sec": round(timeout_sec, 3)},
            )
            return {
                "status": "timeout",
                "reason": "stage_enter_timeout",
                "state": last_state,
                "capture_image": last_image,
            }

        state, capture = _capture_state(
            app,
            yihuan_combat,
            profile_name=profile_name,
            enemy_health_yolo=combat_targets_yolo,
        )
        last_state = dict(state)
        last_image = capture.image
        _trace_scan(
            combat_state_trace,
            start_time=start_time,
            state=state,
            combat_active=False,
            phase="stage_enter_wait",
            note="stage_enter_wait_scan",
            trace_limit=trace_limit,
        )
        _capture_periodic_if_due(
            app,
            capture_debug,
            state=state,
            phase="stage_enter_wait",
            start_time=start_time,
            combat_active=False,
        )
        marker_found = bool(state.get("remaining_enemy_marker_found"))
        if marker_found and marker_seen_started_at is None:
            marker_seen_started_at = time.monotonic()
        if not marker_found:
            marker_seen_started_at = None
        marker_seen_elapsed_sec = (
            max(time.monotonic() - marker_seen_started_at, 0.0)
            if marker_seen_started_at is not None
            else 0.0
        )
        if marker_found and marker_seen_elapsed_sec < stage_confirm_sec:
            _record_action(
                action_trace,
                start_time=start_time,
                action="stage_enter_marker_confirming",
                dry_run=dry_run,
                details={
                    "remaining_enemy_marker_found": True,
                    "marker_seen_elapsed_sec": round(marker_seen_elapsed_sec, 3),
                    "stage_confirm_sec": round(stage_confirm_sec, 3),
                },
            )
        if marker_found and marker_seen_elapsed_sec >= stage_confirm_sec:
            _capture_event_image(
                capture_debug,
                label="stage_enter_confirmed",
                phase="stage_enter_wait",
                source_image=last_image,
                state=state,
                start_time=start_time,
                combat_active=False,
            )
            _record_action(
                action_trace,
                start_time=start_time,
                action="stage_enter_confirmed",
                dry_run=dry_run,
                details={
                    "remaining_enemy_marker_found": bool(state.get("remaining_enemy_marker_found")),
                    "marker_seen_elapsed_sec": round(marker_seen_elapsed_sec, 3),
                    "stage_confirm_sec": round(stage_confirm_sec, 3),
                    "enemy_health_count": int(state.get("enemy_health_count") or 0),
                    "enemy_direction_count": int(state.get("enemy_direction_count") or 0),
                },
            )
            return {
                "status": "success",
                "reason": "remaining_enemy_marker_found",
                "state": dict(state),
                "capture_image": last_image,
            }

        sleep_for = min(scan_interval_sec, max(deadline - time.monotonic(), 0.0))
        if session_deadline is not None:
            sleep_for = min(sleep_for, max(session_deadline - time.monotonic(), 0.0))
        _sleep_interruptibly(max(sleep_for, 0.0))


def _has_approach_enemy_signal(state: Mapping[str, Any]) -> bool:
    return bool(
        state.get("enemy_health_found")
        or state.get("enemy_direction_found")
        or state.get("front_enemy_found")
        or state.get("boss_found")
    )


def _approach_until_enemy_seen(
    app: Any,
    yihuan_combat: YihuanCombatService,
    profile: Mapping[str, Any],
    *,
    profile_name: str,
    combat_targets_yolo: _CombatTargetsYoloRuntime | None,
    initial_state: Mapping[str, Any] | None,
    initial_capture_image: Any,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    combat_state_trace: list[dict[str, Any]],
    capture_debug: _CombatCaptureLogger | None,
    start_time: float,
    trace_limit: int,
    session_deadline: float | None = None,
) -> dict[str, Any]:
    config = dict(profile.get("pre_combat_approach") or {})
    if not bool(config.get("enabled", False)):
        return {
            "status": "skipped",
            "reason": "disabled",
            "state": dict(initial_state or {}),
            "capture_image": initial_capture_image,
        }
    hold_key = str(config.get("hold_key") or "w")
    scan_interval_sec = max(_coerce_float(config.get("scan_interval_sec"), 0.18), 0.05)
    timeout_sec = max(_coerce_float(config.get("max_duration_sec"), 18.0), 0.1)
    min_hold_sec = min(max(_coerce_float(config.get("min_hold_sec"), 0.0), 0.0), timeout_sec)
    release_delay_ms = max(_coerce_int(config.get("release_delay_ms"), 80), 0)
    initial_enemy_seen = initial_state is not None and _has_approach_enemy_signal(initial_state)
    if initial_enemy_seen and min_hold_sec <= 0:
        _record_action(
            action_trace,
            start_time=start_time,
            action="approach_skip_enemy_seen",
            dry_run=dry_run,
            details={
                "enemy_health_count": int(initial_state.get("enemy_health_count") or 0),
                "enemy_direction_count": int(initial_state.get("enemy_direction_count") or 0),
            },
        )
        return {
            "status": "success",
            "reason": "already_seen",
            "state": dict(initial_state),
            "capture_image": initial_capture_image,
        }

    deadline = time.monotonic() + timeout_sec
    last_state = dict(initial_state or {})
    last_image = initial_capture_image
    held = False
    approach_started_at = time.monotonic()
    reason = "timeout"
    enemy_seen_recorded = False

    _record_action(
        action_trace,
        start_time=start_time,
        action="approach_started",
        dry_run=dry_run,
        details={
            "key": hold_key,
            "timeout_sec": round(timeout_sec, 3),
            "min_hold_sec": round(min_hold_sec, 3),
            "scan_interval_sec": round(scan_interval_sec, 3),
        },
    )
    _capture_event_image(
        capture_debug,
        label="approach_started",
        phase="approach_enemy",
        source_image=initial_capture_image,
        state=initial_state,
        start_time=start_time,
        combat_active=False,
    )
    _perform_key_down_with_retry(
        app,
        hold_key,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason="approach_started",
    )
    held = True
    try:
        while True:
            _raise_if_cancelled()
            now = time.monotonic()
            hold_elapsed_sec = max(now - approach_started_at, 0.0)
            if session_deadline is not None and now >= session_deadline:
                reason = "session_deadline"
                break
            if reason == "enemy_seen" and hold_elapsed_sec >= min_hold_sec:
                break
            if now >= deadline:
                reason = "timeout"
                break

            if initial_enemy_seen and not enemy_seen_recorded:
                _capture_event_image(
                    capture_debug,
                    label="approach_enemy_seen",
                    phase="approach_enemy",
                    source_image=last_image,
                    state=last_state,
                    start_time=start_time,
                    combat_active=False,
                )
                _record_action(
                    action_trace,
                    start_time=start_time,
                    action="approach_enemy_seen",
                    dry_run=dry_run,
                    details={
                        "enemy_health_count": int(last_state.get("enemy_health_count") or 0),
                        "enemy_direction_count": int(last_state.get("enemy_direction_count") or 0),
                        "enemy_direction_side": last_state.get("enemy_direction_primary_side"),
                        "hold_elapsed_sec": round(hold_elapsed_sec, 3),
                        "min_hold_sec": round(min_hold_sec, 3),
                    },
                )
                enemy_seen_recorded = True
                reason = "enemy_seen"
                if hold_elapsed_sec >= min_hold_sec:
                    break

            state, capture = _capture_state(
                app,
                yihuan_combat,
                profile_name=profile_name,
                enemy_health_yolo=combat_targets_yolo,
            )
            last_state = dict(state)
            last_image = capture.image
            _trace_scan(
                combat_state_trace,
                start_time=start_time,
                state=state,
                combat_active=False,
                phase="approach_enemy",
                note="approach_scan",
                trace_limit=trace_limit,
            )
            _capture_periodic_if_due(
                app,
                capture_debug,
                state=state,
                phase="approach_enemy",
                start_time=start_time,
                combat_active=False,
            )
            if _has_approach_enemy_signal(state):
                hold_elapsed_sec = max(time.monotonic() - approach_started_at, 0.0)
                if not enemy_seen_recorded:
                    _capture_event_image(
                        capture_debug,
                        label="approach_enemy_seen",
                        phase="approach_enemy",
                        source_image=last_image,
                        state=state,
                        start_time=start_time,
                        combat_active=False,
                    )
                    _record_action(
                        action_trace,
                        start_time=start_time,
                        action="approach_enemy_seen",
                        dry_run=dry_run,
                        details={
                            "enemy_health_count": int(state.get("enemy_health_count") or 0),
                            "enemy_direction_count": int(state.get("enemy_direction_count") or 0),
                            "enemy_direction_side": state.get("enemy_direction_primary_side"),
                            "hold_elapsed_sec": round(hold_elapsed_sec, 3),
                            "min_hold_sec": round(min_hold_sec, 3),
                        },
                    )
                    enemy_seen_recorded = True
                reason = "enemy_seen"
                if hold_elapsed_sec >= min_hold_sec:
                    break

            sleep_for = min(scan_interval_sec, max(deadline - time.monotonic(), 0.0))
            if reason == "enemy_seen" and min_hold_sec > 0:
                sleep_for = min(sleep_for, max(min_hold_sec - (time.monotonic() - approach_started_at), 0.0))
            if session_deadline is not None:
                sleep_for = min(sleep_for, max(session_deadline - time.monotonic(), 0.0))
            _sleep_interruptibly(max(sleep_for, 0.0))
    finally:
        if held:
            _perform_key_up_with_retry(
                app,
                hold_key,
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason="approach_stop",
            )
            if release_delay_ms > 0:
                _sleep_ms(release_delay_ms, dry_run=dry_run)

    _record_action(
        action_trace,
        start_time=start_time,
        action="approach_stop",
        dry_run=dry_run,
        details={
            "reason": reason,
            "key": hold_key,
            "hold_elapsed_sec": round(max(time.monotonic() - approach_started_at, 0.0), 3),
            "min_hold_sec": round(min_hold_sec, 3),
            "enemy_health_count": int((last_state or {}).get("enemy_health_count") or 0),
            "enemy_direction_count": int((last_state or {}).get("enemy_direction_count") or 0),
        },
    )
    return {
        "status": "success" if reason in {"enemy_seen", "timeout", "session_deadline"} else "partial",
        "reason": reason,
        "state": last_state,
        "capture_image": last_image,
    }


def _collect_post_combat_reward(
    app: Any,
    yihuan_combat: YihuanCombatService,
    profile: Mapping[str, Any],
    *,
    profile_name: str,
    combat_targets_yolo: _CombatTargetsYoloRuntime | None = None,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    combat_state_trace: list[dict[str, Any]],
    capture_debug: _CombatCaptureLogger | None,
    start_time: float,
    trace_limit: int,
) -> dict[str, Any]:
    config = dict(profile.get("post_combat_reward") or {})
    click_move_duration_sec = max(_coerce_float(config.get("click_move_duration_sec"), 0.10), 0.0)
    phase = "post_combat_reward"
    if not bool(config.get("enabled", True)):
        result = {"enabled": False, "status": "skipped", "reason": "disabled"}
        _record_action(action_trace, start_time=start_time, action="reward_skipped", dry_run=dry_run, details=result)
        return result
    if dry_run:
        result = {"enabled": True, "status": "skipped", "reason": "dry_run"}
        _record_action(action_trace, start_time=start_time, action="reward_skipped", dry_run=dry_run, details=result)
        return result

    walk_key = str(config.get("walk_key") or "w")
    scan_interval_sec = max(float(config.get("scan_interval_sec") or 0.18), 0.05)
    search_timeout_sec = max(float(config.get("search_timeout_sec") or 14.0), 0.1)
    walk_timeout_sec = max(float(config.get("walk_timeout_sec") or 26.0), 0.1)
    configured_align_target_x = config.get("align_target_x")
    align_tolerance_px = max(int(config.get("align_tolerance_px") or 50), 1)
    look_pixels_per_px = max(float(config.get("look_pixels_per_px") or 0.55), 0.01)
    min_turn_pixels = max(int(config.get("min_turn_pixels") or 8), 1)
    max_turn_pixels = max(int(config.get("max_turn_pixels") or 160), 1)
    post_turn_delay_ms = max(int(config.get("post_turn_delay_ms") or 0), 0)
    walk_step_sec = max(float(config.get("walk_step_sec") or 0.24), 0.05)
    interact_key = str(config.get("interact_key") or "f")
    post_interact_delay_ms = max(int(config.get("post_interact_delay_ms") or 120), 0)
    search_turn_pixels = max(int(config.get("search_turn_pixels") or 520), 1)
    search_turn_interval_sec = max(float(config.get("search_turn_interval_sec") or 0.65), 0.1)
    prompt_required_scans = max(int(config.get("prompt_required_scans") or 1), 1)

    started_at = time.monotonic()
    deadline = started_at + search_timeout_sec + walk_timeout_sec
    last_search_turn_at = 0.0
    search_direction = 1
    walk_started_at: float | None = None
    walking = False
    prompt_scans = 0
    last_state: dict[str, Any] | None = None
    last_capture_image: Any = None

    _record_action(
        action_trace,
        start_time=start_time,
        action="reward_collect_start",
        dry_run=dry_run,
        details={
            "search_timeout_sec": round(search_timeout_sec, 3),
            "walk_timeout_sec": round(walk_timeout_sec, 3),
            "align_target_x": _coerce_int(configured_align_target_x, 640) if configured_align_target_x is not None else None,
            "align_tolerance_px": align_tolerance_px,
        },
    )

    try:
        while time.monotonic() <= deadline:
            state, capture = _capture_state(
                app,
                yihuan_combat,
                profile_name=profile_name,
                enemy_health_yolo=combat_targets_yolo,
            )
            last_state = dict(state)
            last_capture_image = capture.image
            _trace_scan(
                combat_state_trace,
                start_time=start_time,
                state=state,
                combat_active=False,
                phase=phase,
                note="reward_scan",
                trace_limit=trace_limit,
            )
            _capture_periodic_if_due(
                app,
                capture_debug,
                state=state,
                phase=phase,
                start_time=start_time,
                combat_active=False,
            )

            if bool(state.get("claim_memento_prompt_found")):
                prompt_scans += 1
                if prompt_scans >= prompt_required_scans:
                    if walking:
                        _perform_key_up_with_retry(
                            app,
                            walk_key,
                            dry_run=dry_run,
                            action_trace=action_trace,
                            start_time=start_time,
                            retry_reason="reward_stop_at_prompt",
                        )
                        walking = False
                    _capture_event_image(
                        capture_debug,
                        label="reward_claim_prompt_found",
                        phase=phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=False,
                    )
                    result = {
                        "enabled": True,
                        "status": "success",
                        "reason": "claim_prompt_found",
                        "elapsed_sec": round(time.monotonic() - started_at, 3),
                        "reward_marker_found": bool(state.get("reward_marker_found")),
                        "claim_memento_prompt_found": True,
                    }
                    _record_action(
                        action_trace,
                        start_time=start_time,
                        action="reward_claim_prompt_found",
                        dry_run=dry_run,
                        details=result,
                    )
                    _tap_binding(
                        app,
                        interact_key,
                        dry_run=dry_run,
                        action_trace=action_trace,
                        start_time=start_time,
                        action_name="reward_interact",
                    )
                    _sleep_ms(post_interact_delay_ms, dry_run=dry_run)
                    return result | {"state": dict(state)}
                _sleep_interruptibly(scan_interval_sec)
                continue

            prompt_scans = 0
            if walk_started_at is not None and time.monotonic() - walk_started_at >= walk_timeout_sec:
                break

            marker_found = bool(state.get("reward_marker_found"))
            marker_center_x = state.get("reward_marker_center_x")
            capture_width = int((state.get("capture_size") or [1280, 720])[0] or 1280)
            target_x = _reward_align_target_x(config, capture_width=capture_width)
            if marker_found and marker_center_x is not None:
                error_x = float(marker_center_x) - target_x
                if abs(error_x) > align_tolerance_px:
                    if walking:
                        _perform_key_up_with_retry(
                            app,
                            walk_key,
                            dry_run=dry_run,
                            action_trace=action_trace,
                            start_time=start_time,
                            retry_reason="reward_align_marker",
                        )
                        walking = False
                    turn_dx = _reward_turn_pixels_for_error(
                        error_x,
                        align_tolerance_px=align_tolerance_px,
                        look_pixels_per_px=look_pixels_per_px,
                        min_turn_pixels=min_turn_pixels,
                        max_turn_pixels=max_turn_pixels,
                    )
                    _record_action(
                        action_trace,
                        start_time=start_time,
                        action="reward_align_marker",
                        dry_run=dry_run,
                        details={
                            "marker_center_x": int(marker_center_x),
                            "target_x": round(target_x, 1),
                            "error_x": round(error_x, 1),
                            "turn_dx": int(turn_dx),
                        },
                    )
                    _capture_event_image(
                        capture_debug,
                        label="reward_align_marker",
                        phase=phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=False,
                    )
                    _perform_look_delta_with_retry(
                        app,
                        turn_dx,
                        0,
                        dry_run=dry_run,
                        action_trace=action_trace,
                        start_time=start_time,
                        retry_reason="reward_align_marker",
                    )
                    _sleep_ms(post_turn_delay_ms, dry_run=dry_run)
                    continue

                if not walking:
                    _record_action(
                        action_trace,
                        start_time=start_time,
                        action="reward_move_forward_start",
                        dry_run=dry_run,
                        details={
                            "binding": walk_key,
                            "marker_center_x": int(marker_center_x),
                            "target_x": round(target_x, 1),
                            "error_x": round(error_x, 1),
                        },
                    )
                    _capture_event_image(
                        capture_debug,
                        label="reward_move_forward_start",
                        phase=phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=False,
                    )
                    _perform_key_down_with_retry(
                        app,
                        walk_key,
                        dry_run=dry_run,
                        action_trace=action_trace,
                        start_time=start_time,
                        retry_reason="reward_move_forward_start",
                    )
                    walking = True
                    walk_started_at = time.monotonic()
                _sleep_interruptibly(walk_step_sec)
                continue

            if walking:
                _perform_key_up_with_retry(
                    app,
                    walk_key,
                    dry_run=dry_run,
                    action_trace=action_trace,
                    start_time=start_time,
                    retry_reason="reward_marker_lost",
                )
                walking = False
            if time.monotonic() - last_search_turn_at >= search_turn_interval_sec:
                turn_dx = search_direction * search_turn_pixels
                search_direction *= -1
                last_search_turn_at = time.monotonic()
                _record_action(
                    action_trace,
                    start_time=start_time,
                    action="reward_search_turn",
                    dry_run=dry_run,
                    details={"turn_dx": int(turn_dx)},
                )
                _capture_event_image(
                    capture_debug,
                    label="reward_search_turn",
                    phase=phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=False,
                )
                _perform_look_delta_with_retry(
                    app,
                    turn_dx,
                    0,
                    dry_run=dry_run,
                    action_trace=action_trace,
                    start_time=start_time,
                    retry_reason="reward_search_turn",
                )
            _sleep_interruptibly(scan_interval_sec)
    finally:
        if walking:
            _perform_key_up_with_retry(
                app,
                walk_key,
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason="reward_finish_release",
            )

    result = {
        "enabled": True,
        "status": "timeout",
        "reason": "claim_prompt_timeout",
        "elapsed_sec": round(time.monotonic() - started_at, 3),
        "reward_marker_found": bool((last_state or {}).get("reward_marker_found")),
        "claim_memento_prompt_found": bool((last_state or {}).get("claim_memento_prompt_found")),
    }
    _record_action(action_trace, start_time=start_time, action="reward_timeout", dry_run=dry_run, details=result)
    _capture_event_image(
        capture_debug,
        label="reward_timeout",
        phase=phase,
        source_image=last_capture_image,
        state=last_state,
        start_time=start_time,
        combat_active=False,
    )
    return result | {"state": dict(last_state or {})}


def _configured_point(config: Mapping[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
    value = config.get(key)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _coerce_int(value[0], default[0]), _coerce_int(value[1], default[1])
    return int(default[0]), int(default[1])


def _reward_claim_value(claim_mode: str) -> int:
    return 2 if str(claim_mode).strip().lower() == "double" else 1


def _claim_mode_for_remaining_runs(total_runs: int, completed_runs: int) -> str:
    if int(total_runs) <= 0:
        return "single"
    remaining = max(int(total_runs) - int(completed_runs), 0)
    return "double" if remaining >= 2 else "single"


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _click_screen_point(
    app: Any,
    point: tuple[int, int],
    *,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
    action_name: str,
    details: Mapping[str, Any] | None = None,
    move_duration_sec: float = 0.10,
) -> None:
    x, y = int(point[0]), int(point[1])
    payload = {"x": x, "y": y}
    if details:
        payload.update(dict(details))
    _record_action(
        action_trace,
        start_time=start_time,
        action=action_name,
        dry_run=dry_run,
        details=payload,
    )
    if dry_run:
        return
    _perform_move_click_with_retry(
        app,
        x,
        y,
        button="left",
        move_duration_sec=move_duration_sec,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=action_name,
    )


def _wait_for_reward_claim_modal(
    app: Any,
    yihuan_combat: YihuanCombatService,
    profile: Mapping[str, Any],
    *,
    profile_name: str,
    combat_targets_yolo: _CombatTargetsYoloRuntime | None,
    claim_mode: str,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    combat_state_trace: list[dict[str, Any]],
    capture_debug: _CombatCaptureLogger | None,
    start_time: float,
    trace_limit: int,
    session_deadline: float | None = None,
) -> dict[str, Any]:
    config = dict(profile.get("post_combat_reward") or {})
    timeout_sec = max(_coerce_float(config.get("claim_modal_timeout_sec"), 6.0), 0.1)
    scan_interval_sec = max(_coerce_float(config.get("scan_interval_sec"), 0.18), 0.05)
    deadline = time.monotonic() + timeout_sec
    required_key = (
        "reward_claim_double_button_found"
        if str(claim_mode).strip().lower() == "double"
        else "reward_claim_single_button_found"
    )
    last_state: dict[str, Any] | None = None
    last_image: Any = None
    _record_action(
        action_trace,
        start_time=start_time,
        action="reward_claim_modal_wait_started",
        dry_run=dry_run,
        details={"claim_mode": claim_mode, "timeout_sec": round(timeout_sec, 3)},
    )
    while True:
        _raise_if_cancelled()
        now = time.monotonic()
        if session_deadline is not None and now >= session_deadline:
            break
        if now > deadline:
            break
        state, capture = _capture_state(
            app,
            yihuan_combat,
            profile_name=profile_name,
            enemy_health_yolo=combat_targets_yolo,
        )
        last_state = dict(state)
        last_image = capture.image
        _trace_scan(
            combat_state_trace,
            start_time=start_time,
            state=state,
            combat_active=False,
            phase="reward_claim_modal",
            note="reward_claim_modal_scan",
            trace_limit=trace_limit,
        )
        _capture_periodic_if_due(
            app,
            capture_debug,
            state=state,
            phase="reward_claim_modal",
            start_time=start_time,
            combat_active=False,
        )
        modal_found = bool(state.get("reward_claim_single_button_found") or state.get("reward_claim_double_button_found"))
        if modal_found:
            _record_action(
                action_trace,
                start_time=start_time,
                action="reward_confirm_modal_found",
                dry_run=dry_run,
                details={
                    "claim_mode": claim_mode,
                    "single_found": bool(state.get("reward_claim_single_button_found")),
                    "double_found": bool(state.get("reward_claim_double_button_found")),
                    "single_confidence": state.get("reward_claim_single_button_confidence"),
                    "double_confidence": state.get("reward_claim_double_button_confidence"),
                },
            )
            _capture_event_image(
                capture_debug,
                label="reward_confirm_modal_found",
                phase="reward_claim_modal",
                source_image=last_image,
                state=state,
                start_time=start_time,
                combat_active=False,
            )
            if bool(state.get(required_key)):
                return {
                    "status": "success",
                    "reason": "reward_claim_button_found",
                    "state": dict(state),
                    "capture_image": last_image,
                }
            if str(claim_mode).strip().lower() == "double":
                _record_action(
                    action_trace,
                    start_time=start_time,
                    action="reward_claim_failed",
                    dry_run=dry_run,
                    details={"reason": "double_claim_button_not_found"},
                )
                return {
                    "status": "failed",
                    "reason": "double_claim_button_not_found",
                    "state": dict(state),
                    "capture_image": last_image,
                }
        sleep_for = min(scan_interval_sec, max(deadline - time.monotonic(), 0.0))
        if session_deadline is not None:
            sleep_for = min(sleep_for, max(session_deadline - time.monotonic(), 0.0))
        _sleep_interruptibly(max(sleep_for, 0.0))

    reason = "session_deadline" if session_deadline is not None and time.monotonic() >= session_deadline else "reward_claim_modal_timeout"
    _record_action(
        action_trace,
        start_time=start_time,
        action="reward_claim_failed",
        dry_run=dry_run,
        details={"reason": reason, "claim_mode": claim_mode},
    )
    _capture_event_image(
        capture_debug,
        label="reward_claim_modal_timeout",
        phase="reward_claim_modal",
        source_image=last_image,
        state=last_state,
        start_time=start_time,
        combat_active=False,
    )
    return {"status": "failed", "reason": reason, "state": dict(last_state or {}), "capture_image": last_image}


def _wait_for_reward_result_screen(
    app: Any,
    yihuan_combat: YihuanCombatService,
    profile: Mapping[str, Any],
    *,
    profile_name: str,
    combat_targets_yolo: _CombatTargetsYoloRuntime | None,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    combat_state_trace: list[dict[str, Any]],
    capture_debug: _CombatCaptureLogger | None,
    start_time: float,
    trace_limit: int,
    session_deadline: float | None = None,
) -> dict[str, Any]:
    config = dict(profile.get("post_combat_reward") or {})
    timeout_sec = max(_coerce_float(config.get("claim_result_timeout_sec"), 8.0), 0.1)
    scan_interval_sec = max(_coerce_float(config.get("scan_interval_sec"), 0.18), 0.05)
    deadline = time.monotonic() + timeout_sec
    last_state: dict[str, Any] | None = None
    last_image: Any = None
    while True:
        _raise_if_cancelled()
        now = time.monotonic()
        if session_deadline is not None and now >= session_deadline:
            break
        if now > deadline:
            break
        state, capture = _capture_state(
            app,
            yihuan_combat,
            profile_name=profile_name,
            enemy_health_yolo=combat_targets_yolo,
        )
        last_state = dict(state)
        last_image = capture.image
        _trace_scan(
            combat_state_trace,
            start_time=start_time,
            state=state,
            combat_active=False,
            phase="reward_result",
            note="reward_result_scan",
            trace_limit=trace_limit,
        )
        _capture_periodic_if_due(
            app,
            capture_debug,
            state=state,
            phase="reward_result",
            start_time=start_time,
            combat_active=False,
        )
        if bool(state.get("reward_result_retry_button_found") or state.get("reward_result_exit_button_found")):
            _record_action(
                action_trace,
                start_time=start_time,
                action="reward_result_screen_found",
                dry_run=dry_run,
                details={
                    "retry_found": bool(state.get("reward_result_retry_button_found")),
                    "exit_found": bool(state.get("reward_result_exit_button_found")),
                    "retry_confidence": state.get("reward_result_retry_button_confidence"),
                    "exit_confidence": state.get("reward_result_exit_button_confidence"),
                },
            )
            _capture_event_image(
                capture_debug,
                label="reward_result_screen_found",
                phase="reward_result",
                source_image=last_image,
                state=state,
                start_time=start_time,
                combat_active=False,
            )
            return {
                "status": "success",
                "reason": "reward_result_screen_found",
                "state": dict(state),
                "capture_image": last_image,
            }
        sleep_for = min(scan_interval_sec, max(deadline - time.monotonic(), 0.0))
        if session_deadline is not None:
            sleep_for = min(sleep_for, max(session_deadline - time.monotonic(), 0.0))
        _sleep_interruptibly(max(sleep_for, 0.0))

    reason = "session_deadline" if session_deadline is not None and time.monotonic() >= session_deadline else "reward_result_timeout"
    _record_action(
        action_trace,
        start_time=start_time,
        action="reward_claim_failed",
        dry_run=dry_run,
        details={"reason": reason},
    )
    _capture_event_image(
        capture_debug,
        label="reward_result_timeout",
        phase="reward_result",
        source_image=last_image,
        state=last_state,
        start_time=start_time,
        combat_active=False,
    )
    return {"status": "failed", "reason": reason, "state": dict(last_state or {}), "capture_image": last_image}


def _handle_post_reward_claim_flow(
    app: Any,
    yihuan_combat: YihuanCombatService,
    profile: Mapping[str, Any],
    *,
    profile_name: str,
    combat_targets_yolo: _CombatTargetsYoloRuntime | None,
    claim_mode: str,
    completed_runs: int,
    total_runs: int,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    combat_state_trace: list[dict[str, Any]],
    capture_debug: _CombatCaptureLogger | None,
    start_time: float,
    trace_limit: int,
    session_deadline: float | None = None,
) -> dict[str, Any]:
    config = dict(profile.get("post_combat_reward") or {})
    click_move_duration_sec = max(_coerce_float(config.get("click_move_duration_sec"), 0.10), 0.0)
    if dry_run:
        return {
            "status": "success",
            "reason": "dry_run",
            "action": "exit",
            "state": {},
            "capture_image": None,
        }
    modal_result = _wait_for_reward_claim_modal(
        app,
        yihuan_combat,
        profile,
        profile_name=profile_name,
        combat_targets_yolo=combat_targets_yolo,
        claim_mode=claim_mode,
        dry_run=dry_run,
        action_trace=action_trace,
        combat_state_trace=combat_state_trace,
        capture_debug=capture_debug,
        start_time=start_time,
        trace_limit=trace_limit,
        session_deadline=session_deadline,
    )
    if modal_result.get("status") != "success":
        return dict(modal_result) | {"action": "failed"}

    claim_mode_normalized = str(claim_mode).strip().lower()
    if claim_mode_normalized not in {"single", "double"}:
        claim_mode_normalized = "single"
    claim_value = _reward_claim_value(claim_mode_normalized)
    completed_before = max(int(completed_runs), 0)
    total_target = max(int(total_runs), 0)
    completed_after = completed_before + claim_value
    remaining_after = max(total_target - completed_after, 0) if total_target > 0 else 0
    if claim_mode_normalized == "double":
        claim_point = _configured_point(config, "double_claim_button_center", (844, 467))
    else:
        claim_point = _configured_point(config, "single_claim_button_center", (522, 467))
    _click_screen_point(
        app,
        claim_point,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        action_name="reward_claim_click",
        details={
            "mode": claim_mode_normalized,
            "button_center": [int(claim_point[0]), int(claim_point[1])],
            "claim_value": int(claim_value),
            "completed_before": int(completed_before),
            "completed_runs": int(completed_after),
            "total_runs": int(total_target),
            "remaining": int(remaining_after),
        },
        move_duration_sec=click_move_duration_sec,
    )
    _sleep_ms(max(_coerce_int(config.get("post_claim_click_delay_ms"), 350), 0), dry_run=dry_run)

    result_screen = _wait_for_reward_result_screen(
        app,
        yihuan_combat,
        profile,
        profile_name=profile_name,
        combat_targets_yolo=combat_targets_yolo,
        dry_run=dry_run,
        action_trace=action_trace,
        combat_state_trace=combat_state_trace,
        capture_debug=capture_debug,
        start_time=start_time,
        trace_limit=trace_limit,
        session_deadline=session_deadline,
    )
    if result_screen.get("status") != "success":
        return dict(result_screen) | {"action": "failed"}

    should_retry = total_target > 0 and completed_after < total_target
    if should_retry:
        retry_point = _configured_point(config, "retry_challenge_button_center", (796, 622))
        _click_screen_point(
            app,
            retry_point,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            action_name="retry_challenge_click",
            details={
                "completed_runs": int(completed_after),
                "claimed_runs": int(claim_value),
                "total_runs": int(total_target),
                "remaining": int(remaining_after),
            },
            move_duration_sec=click_move_duration_sec,
        )
        _sleep_ms(max(_coerce_int(config.get("post_result_click_delay_ms"), 800), 0), dry_run=dry_run)
        return dict(result_screen) | {
            "action": "retry",
            "claim_mode": claim_mode_normalized,
            "claim_value": int(claim_value),
            "completed_runs": int(completed_after),
            "total_runs": int(total_target),
            "remaining": int(remaining_after),
        }

    exit_point = _configured_point(config, "exit_button_center", (488, 622))
    _click_screen_point(
        app,
        exit_point,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        action_name="exit_after_final_reward_click",
        details={
            "completed_runs": int(completed_after),
            "claimed_runs": int(claim_value),
            "total_runs": int(total_target),
            "remaining": int(remaining_after),
        },
        move_duration_sec=click_move_duration_sec,
    )
    _sleep_ms(max(_coerce_int(config.get("post_result_click_delay_ms"), 800), 0), dry_run=dry_run)
    _record_action(
        action_trace,
        start_time=start_time,
        action="battle_loop_finished",
        dry_run=dry_run,
        details={"completed_runs": int(completed_after), "total_runs": int(total_target), "remaining": int(remaining_after)},
    )
    return dict(result_screen) | {
        "action": "exit",
        "claim_mode": claim_mode_normalized,
        "claim_value": int(claim_value),
        "completed_runs": int(completed_after),
        "total_runs": int(total_target),
        "remaining": int(remaining_after),
    }


def _run_post_battle_reward_sequence(
    app: Any,
    yihuan_combat: YihuanCombatService,
    profile: Mapping[str, Any],
    *,
    profile_name: str,
    combat_targets_yolo: _CombatTargetsYoloRuntime | None,
    claim_mode: str,
    completed_runs: int,
    total_runs: int,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    combat_state_trace: list[dict[str, Any]],
    capture_debug: _CombatCaptureLogger | None,
    start_time: float,
    trace_limit: int,
    session_deadline: float | None = None,
) -> dict[str, Any]:
    reward_result = _collect_post_combat_reward(
        app,
        yihuan_combat,
        profile,
        profile_name=profile_name,
        combat_targets_yolo=combat_targets_yolo,
        dry_run=dry_run,
        action_trace=action_trace,
        combat_state_trace=combat_state_trace,
        capture_debug=capture_debug,
        start_time=start_time,
        trace_limit=trace_limit,
    )
    reward_status = str(reward_result.get("status") or "")
    reward_reason = str(reward_result.get("reason") or "")
    if reward_status != "success":
        if reward_status == "skipped" and reward_reason in {"disabled", "dry_run"}:
            if total_runs > 0 and completed_runs < total_runs and reward_reason != "dry_run":
                return {
                    "status": "failed",
                    "reason": "post_combat_reward_disabled",
                    "action": "failed",
                    "post_combat_reward": dict(reward_result),
                    "state": dict(reward_result.get("state") or {}),
                    "capture_image": reward_result.get("capture_image"),
                }
            return {
                "status": "success",
                "reason": reward_reason,
                "action": "exit",
                "post_combat_reward": dict(reward_result),
                "state": dict(reward_result.get("state") or {}),
                "capture_image": reward_result.get("capture_image"),
            }
        return {
            "status": "failed",
            "reason": reward_reason or "post_combat_reward_failed",
            "action": "failed",
            "post_combat_reward": dict(reward_result),
            "state": dict(reward_result.get("state") or {}),
            "capture_image": reward_result.get("capture_image"),
        }

    claim_result = _handle_post_reward_claim_flow(
        app,
        yihuan_combat,
        profile,
        profile_name=profile_name,
        combat_targets_yolo=combat_targets_yolo,
        claim_mode=claim_mode,
        completed_runs=completed_runs,
        total_runs=total_runs,
        dry_run=dry_run,
        action_trace=action_trace,
        combat_state_trace=combat_state_trace,
        capture_debug=capture_debug,
        start_time=start_time,
        trace_limit=trace_limit,
        session_deadline=session_deadline,
    )
    if claim_result.get("status") != "success":
        return {
            "status": "failed",
            "reason": str(claim_result.get("reason") or "reward_claim_failed"),
            "action": "failed",
            "post_combat_reward": dict(reward_result),
            "claim_result": dict(claim_result),
            "state": dict(claim_result.get("state") or reward_result.get("state") or {}),
            "capture_image": _first_not_none(claim_result.get("capture_image"), reward_result.get("capture_image")),
        }

    if claim_result.get("action") != "retry":
        return {
            "status": "success",
            "reason": str(claim_result.get("reason") or "reward_exit"),
            "action": "exit",
            "claim_mode": str(claim_result.get("claim_mode") or claim_mode),
            "claim_value": int(claim_result.get("claim_value") or _reward_claim_value(claim_mode)),
            "completed_runs": int(claim_result.get("completed_runs") or completed_runs),
            "total_runs": int(claim_result.get("total_runs") or total_runs),
            "remaining": int(claim_result.get("remaining") or 0),
            "post_combat_reward": dict(reward_result),
            "claim_result": dict(claim_result),
            "state": dict(claim_result.get("state") or reward_result.get("state") or {}),
            "capture_image": _first_not_none(claim_result.get("capture_image"), reward_result.get("capture_image")),
        }

    wait_result = _wait_for_stage_started(
        app,
        yihuan_combat,
        profile,
        profile_name=profile_name,
        combat_targets_yolo=combat_targets_yolo,
        dry_run=dry_run,
        action_trace=action_trace,
        combat_state_trace=combat_state_trace,
        capture_debug=capture_debug,
        start_time=start_time,
        trace_limit=trace_limit,
        session_deadline=session_deadline,
    )
    if wait_result.get("status") != "success":
        return {
            "status": "failed",
            "reason": str(wait_result.get("reason") or "stage_enter_timeout"),
            "action": "failed",
            "post_combat_reward": dict(reward_result),
            "claim_result": dict(claim_result),
            "stage_wait_result": dict(wait_result),
            "state": dict(wait_result.get("state") or claim_result.get("state") or {}),
            "capture_image": _first_not_none(wait_result.get("capture_image"), claim_result.get("capture_image")),
        }

    approach_result = _approach_until_enemy_seen(
        app,
        yihuan_combat,
        profile,
        profile_name=profile_name,
        combat_targets_yolo=combat_targets_yolo,
        initial_state=wait_result.get("state"),
        initial_capture_image=wait_result.get("capture_image"),
        dry_run=dry_run,
        action_trace=action_trace,
        combat_state_trace=combat_state_trace,
        capture_debug=capture_debug,
        start_time=start_time,
        trace_limit=trace_limit,
        session_deadline=session_deadline,
    )
    return {
        "status": "success",
        "reason": "retry_stage_started",
        "action": "retry",
        "claim_mode": str(claim_result.get("claim_mode") or claim_mode),
        "claim_value": int(claim_result.get("claim_value") or _reward_claim_value(claim_mode)),
        "completed_runs": int(claim_result.get("completed_runs") or completed_runs),
        "total_runs": int(claim_result.get("total_runs") or total_runs),
        "remaining": int(claim_result.get("remaining") or 0),
        "post_combat_reward": dict(reward_result),
        "claim_result": dict(claim_result),
        "stage_wait_result": dict(wait_result),
        "approach_result": dict(approach_result),
        "state": dict(approach_result.get("state") or wait_result.get("state") or {}),
        "capture_image": _first_not_none(approach_result.get("capture_image"), wait_result.get("capture_image")),
    }


def _consume_audio_dodge_trigger(
    audio_dodge_runtime: AudioDodgeRuntime | None,
    *,
    combat_active: bool,
) -> dict[str, Any] | None:
    if audio_dodge_runtime is None or not combat_active:
        return None
    return audio_dodge_runtime.consume_trigger()


def _schedule_next_due(now: float, interval_sec: float) -> float:
    return float(now) + max(float(interval_sec), 0.0)


def _press_scheduled_ability(
    app: Any,
    input_mapping: Any,
    profile: Mapping[str, Any],
    *,
    ability_name: str,
    action_trace: list[dict[str, Any]],
    start_time: float,
    dry_run: bool,
    press_ms: int = 0,
) -> None:
    binding = dict(profile["keys"])[ability_name]
    _record_action(
        action_trace,
        start_time=start_time,
        action=ability_name,
        dry_run=dry_run,
        details={"binding": binding, "press_ms": max(int(press_ms), 0)},
    )
    _tap_input_binding_with_retry(
        app,
        input_mapping,
        binding,
        hold_ms=press_ms,
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason=ability_name,
    )
    _sleep_ms(int(dict(profile["runtime"]).get("input_cooldown_ms", 120)), dry_run=dry_run)


def _execute_audio_dodge(
    app: Any,
    profile: Mapping[str, Any],
    *,
    trigger_event: Mapping[str, Any],
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> float:
    payload = dict(profile.get("audio_dodge") or {})
    _record_action(
        action_trace,
        start_time=start_time,
        action="audio_dodge",
        dry_run=dry_run,
        details={"score": round(float(trigger_event.get("score") or 0.0), 5)},
    )
    if not dry_run:
        mouse_button = str(payload.get("dodge_mouse_button") or "right")
        dodge_key = str(payload.get("dodge_key") or "shift")
        _call_with_focus_retry(
            app,
            lambda: app.mouse_down(mouse_button),
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            retry_reason="audio_dodge",
        )
        try:
            _sleep_ms_uninterruptible(int(payload.get("right_hold_ms") or 0), dry_run=dry_run)
        finally:
            _call_with_focus_retry(
                app,
                lambda: app.mouse_up(mouse_button),
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason="audio_dodge",
            )
        _sleep_ms_uninterruptible(int(payload.get("post_right_delay_ms") or 0), dry_run=dry_run)
        _call_with_focus_retry(
            app,
            lambda: app.key_down(dodge_key),
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            retry_reason="audio_dodge",
        )
        try:
            _sleep_ms_uninterruptible(int(payload.get("shift_hold_ms") or 0), dry_run=dry_run)
        finally:
            _call_with_focus_retry(
                app,
                lambda: app.key_up(dodge_key),
                dry_run=dry_run,
                action_trace=action_trace,
                start_time=start_time,
                retry_reason="audio_dodge",
            )
    return max(int(payload.get("dodge_pause_ms") or 0), 0) / 1000.0


def _turn_to_rear_enemy(
    app: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    auto_target_enabled: bool,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    markers = list(state.get("enemy_direction_markers") or [])
    if not markers:
        _record_action(
            action_trace,
            start_time=start_time,
            action="skip_rear_enemy_turn",
            dry_run=dry_run,
            details={"reason": "no_markers"},
        )
        return {"executed": False, "auto_targeted": False, "turn_dx": 0, "turn_dy": 0}

    capture_size = list(state.get("capture_size") or [1280, 720])
    frame_width = max(_coerce_int(capture_size[0], 1280), 1)
    primary_side = str(state.get("enemy_direction_primary_side") or "").strip().lower()
    reacquire = dict(profile.get("enemy_direction_reacquire") or {})
    side_turn = max(_coerce_int(reacquire.get("turn_pixels_side"), 220), 1)
    bottom_turn = max(_coerce_int(reacquire.get("turn_pixels_bottom"), 340), 1)
    vertical_delta = _coerce_int(reacquire.get("vertical_delta"), 0)
    auto_target_after_turn = auto_target_enabled and bool(reacquire.get("auto_target_after_turn", True))

    avg_center_x = 0.5
    if markers:
        centers = [
            (float(_coerce_int(marker.get("x"), 0)) + float(_coerce_int(marker.get("width"), 0)) / 2.0)
            / float(frame_width)
            for marker in markers
            if isinstance(marker, Mapping)
        ]
        if centers:
            avg_center_x = sum(centers) / float(len(centers))

    if primary_side == "left":
        turn_dx = -side_turn
    elif primary_side == "right":
        turn_dx = side_turn
    elif primary_side == "bottom":
        turn_dx = -bottom_turn if avg_center_x < 0.45 else bottom_turn
    else:
        turn_dx = -side_turn if avg_center_x < 0.5 else side_turn

    _record_action(
        action_trace,
        start_time=start_time,
        action="turn_to_rear_enemy",
        dry_run=dry_run,
        details={
            "primary_side": primary_side or None,
            "marker_count": len(markers),
            "turn_dx": int(turn_dx),
            "turn_dy": int(vertical_delta),
            "avg_center_x": round(float(avg_center_x), 3),
        },
    )
    _perform_look_delta_with_retry(
        app,
        int(turn_dx),
        int(vertical_delta),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason="turn_to_rear_enemy",
    )

    post_turn_delay_ms = max(_coerce_int(reacquire.get("post_turn_delay_ms"), 0), 0)
    if post_turn_delay_ms > 0:
        _sleep_ms(post_turn_delay_ms, dry_run=dry_run)

    auto_targeted = False
    if auto_target_after_turn:
        _tap_binding(
            app,
            str(dict(profile["keys"]).get("target") or "mouse_middle"),
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            action_name="auto_target_after_turn",
        )
        auto_targeted = True

    return {
        "executed": True,
        "auto_targeted": auto_targeted,
        "turn_dx": int(turn_dx),
        "turn_dy": int(vertical_delta),
    }


def _turn_to_front_enemy(
    app: Any,
    profile: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    auto_target_enabled: bool,
    scan_direction: int,
    dry_run: bool,
    action_trace: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    if list(state.get("enemy_direction_markers") or []):
        result = _turn_to_rear_enemy(
            app,
            profile,
            state,
            auto_target_enabled=auto_target_enabled,
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
        )
        result["used_direction_marker"] = True
        return result

    reacquire = dict(profile.get("enemy_direction_reacquire") or {})
    side_turn = max(_coerce_int(reacquire.get("turn_pixels_side"), 220), 1)
    vertical_delta = _coerce_int(reacquire.get("vertical_delta"), 0)
    turn_dx = side_turn if int(scan_direction or 1) >= 0 else -side_turn
    auto_target_after_turn = auto_target_enabled and bool(reacquire.get("auto_target_after_turn", True))
    _record_action(
        action_trace,
        start_time=start_time,
        action="turn_to_front_enemy",
        dry_run=dry_run,
        details={
            "reason": "front_enemy_missing",
            "used_direction_marker": False,
            "scan_mode": "fixed_direction",
            "turn_dx": int(turn_dx),
            "turn_dy": int(vertical_delta),
        },
    )
    _perform_look_delta_with_retry(
        app,
        int(turn_dx),
        int(vertical_delta),
        dry_run=dry_run,
        action_trace=action_trace,
        start_time=start_time,
        retry_reason="turn_to_front_enemy",
    )
    post_turn_delay_ms = max(_coerce_int(reacquire.get("post_turn_delay_ms"), 0), 0)
    if post_turn_delay_ms > 0:
        _sleep_ms(post_turn_delay_ms, dry_run=dry_run)

    auto_targeted = False
    if auto_target_after_turn:
        _tap_binding(
            app,
            str(dict(profile["keys"]).get("target") or "mouse_middle"),
            dry_run=dry_run,
            action_trace=action_trace,
            start_time=start_time,
            action_name="auto_target_after_turn",
        )
        auto_targeted = True
    return {
        "executed": True,
        "auto_targeted": auto_targeted,
        "turn_dx": int(turn_dx),
        "turn_dy": int(vertical_delta),
        "used_direction_marker": False,
    }


@action_info(
    name="yihuan_combat_run_session",
    public=False,
    read_only=False,
    description="Run the Yihuan-specific automatic combat loop.",
)
@requires_services(
    app="plans/aura_base/app",
    input_mapping="plans/aura_base/input_mapping",
    yihuan_combat="yihuan_combat",
    yolo="core/yolo",
)
def yihuan_combat_run_session(
    app: Any,
    input_mapping: Any,
    yihuan_combat: YihuanCombatService,
    yolo: Any = None,
    profile_name: str = "default_1280x720_cn",
    strategy_name: str = "default",
    max_seconds: float | int | str = 0,
    max_encounters: int | str = 0,
    battle_count: int | str = 0,
    auto_target: bool | str = True,
    auto_dodge: bool | str = True,
    dry_run: bool | str = False,
    debug_enabled: bool | str = False,
    capture_debug_enabled: bool | str = False,
    capture_interval_sec: float | int | str = 2.0,
    capture_max_images: int | str = 120,
    capture_raw_enabled: bool | str = False,
) -> dict[str, Any]:
    start_time = time.monotonic()
    combat_state_trace: list[dict[str, Any]] = []
    action_trace: list[dict[str, Any]] = []
    last_state: dict[str, Any] | None = None
    encounters_completed = 0
    reward_runs_completed = 0
    profile: dict[str, Any] | None = None
    current_phase = "monitor"
    ability_lockouts: dict[str, float] = {"skill": 0.0, "ultimate": 0.0}
    audio_dodge_runtime: AudioDodgeRuntime | None = None
    capture_debug: _CombatCaptureLogger | None = None
    dodge_pause_until = 0.0
    last_capture_image: Any = None
    initial_monitor_captured = False
    last_known_slot: int | None = None
    last_rear_turn_at = 0.0
    front_enemy_scan_direction = 1
    post_combat_reward_result: dict[str, Any] | None = None
    enemy_health_yolo_runtime: _CombatTargetsYoloRuntime | None = None

    try:
        profile = yihuan_combat.load_profile(profile_name)
        resolved_profile = str(profile["profile_name"])
        runtime = dict(profile["runtime"])
        dry_run_enabled = _coerce_bool(dry_run)
        auto_target_enabled = _coerce_bool(auto_target)
        auto_dodge_enabled = _coerce_bool(auto_dodge)
        _ = _coerce_bool(debug_enabled)
        capture_debug_enabled_resolved = _coerce_bool(capture_debug_enabled)
        capture_interval_sec_resolved = max(_coerce_float(capture_interval_sec, 2.0), 0.1)
        capture_max_images_resolved = max(_coerce_int(capture_max_images, 120), 1)
        capture_raw_enabled_resolved = _coerce_bool(capture_raw_enabled)
        normal_command = _resolve_normal_command(profile, str(strategy_name))
        schedule = _resolve_runtime_schedule(profile)
        duration_limit = max(_coerce_float(max_seconds, 0.0), 0.0)
        if dry_run_enabled and duration_limit <= 0:
            duration_limit = float(runtime["dry_run_max_seconds"])
        deadline = time.monotonic() + duration_limit if duration_limit > 0 else None
        legacy_encounter_limit = max(_coerce_int(max_encounters, 0), 0)
        requested_battle_count = max(_coerce_int(battle_count, 0), 0)
        encounter_limit = requested_battle_count if requested_battle_count > 0 else legacy_encounter_limit
        stop_limit_source = "battle_count" if requested_battle_count > 0 else "max_encounters"
        claim_mode = _claim_mode_for_remaining_runs(encounter_limit, reward_runs_completed)
        poll_ms = max(int(runtime["poll_ms"]), 10)
        unsupported_required = int(runtime["unsupported_scene_stable_frames"])
        trace_limit = int(runtime["trace_limit"])
        post_exit_cooldown_ms = int(round(float(schedule["exit_post_cooldown_sec"]) * 1000.0))
        target_retry_interval_sec = max(float(runtime.get("target_retry_interval_ms") or 500) / 1000.0, 0.0)

        logger.info(
            "Combat[session] start profile=%s strategy=%s max_seconds=%.3f max_encounters=%s battle_count=%s resolved_battle_count=%s auto_target=%s auto_dodge=%s dry_run=%s",
            resolved_profile,
            strategy_name,
            duration_limit,
            legacy_encounter_limit,
            requested_battle_count,
            encounter_limit,
            auto_target_enabled,
            auto_dodge_enabled,
            dry_run_enabled,
        )
        if requested_battle_count > 0 and legacy_encounter_limit > 0 and requested_battle_count != legacy_encounter_limit:
            logger.info(
                "Combat[session] battle_count overrides max_encounters battle_count=%s max_encounters=%s",
                requested_battle_count,
                legacy_encounter_limit,
            )
        _record_action(
            action_trace,
            start_time=start_time,
            action="battle_loop_start",
            dry_run=dry_run_enabled,
            details={
                "battle_count": int(encounter_limit),
                "max_encounters": int(legacy_encounter_limit),
                "requested_battle_count": int(requested_battle_count),
                "limit_source": stop_limit_source,
                "claim_mode": claim_mode,
                "claim_strategy": "double_when_remaining_at_least_2",
                "completed_runs": int(reward_runs_completed),
                "remaining": max(int(encounter_limit) - int(reward_runs_completed), 0),
            },
        )
        if encounter_limit > 0:
            _record_action(
                action_trace,
                start_time=start_time,
                action="battle_run_started",
                dry_run=dry_run_enabled,
                details={
                    "index": 1,
                    "total": int(encounter_limit),
                    "claim_mode": claim_mode,
                    "completed_runs": int(reward_runs_completed),
                    "remaining": max(int(encounter_limit) - int(reward_runs_completed), 0),
                },
            )
        logger.info(
            "Combat[config] remaining_enemy_marker_region=%s enemy_health_search_region=%s enemy_direction_regions=%s skill_press_ms=%s ultimate_press_ms=%s switch_press_ms=%s monitor_scan_interval_sec=%.2f combat_scan_interval_sec=%.2f switch_interval_sec=%.2f switch_confirm_required_matches=%s failed_switch_cooldown_sec=%.2f rear_enemy_turn_interval_sec=%.2f exit_confirm_required_scans=%s exit_confirm_interval_sec=%.2f exit_confirm_missing_sec=%.2f exit_post_cooldown_ms=%s",
            dict(profile.get("regions") or {}).get("remaining_enemy_marker"),
            dict(profile.get("regions") or {}).get("enemy_health_search_region"),
            {
                "left": dict(profile.get("regions") or {}).get("enemy_direction_left"),
                "right": dict(profile.get("regions") or {}).get("enemy_direction_right"),
                "bottom": dict(profile.get("regions") or {}).get("enemy_direction_bottom"),
            },
            schedule["skill_press_ms"],
            schedule["ultimate_press_ms"],
            schedule["switch_press_ms"],
            float(schedule["monitor_scan_interval_sec"]),
            float(schedule["combat_scan_interval_sec"]),
            float(schedule["switch_interval_sec"]),
            int(schedule["switch_confirm_required_matches"]),
            float(schedule["failed_switch_cooldown_sec"]),
            float(schedule["rear_enemy_turn_interval_sec"]),
            int(schedule["exit_confirm_required_scans"]),
            float(schedule["exit_confirm_interval_sec"]),
            float(schedule["exit_confirm_missing_sec"]),
            post_exit_cooldown_ms,
        )
        logger.info(
            "Combat[capture] enabled=%s interval_sec=%.2f max_images=%s raw_enabled=%s",
            capture_debug_enabled_resolved,
            capture_interval_sec_resolved,
            capture_max_images_resolved,
            capture_raw_enabled_resolved,
        )
        logger.info(
            "Combat[stage_entry] enabled=%s wait_after_click_ms=%s stage_confirm_sec=%.2f stage_start_timeout_sec=%.2f scan_interval_sec=%.2f",
            bool(dict(profile.get("stage_entry") or {}).get("enabled", False)),
            int(dict(profile.get("stage_entry") or {}).get("wait_after_click_ms") or 0),
            float(dict(profile.get("stage_entry") or {}).get("stage_confirm_sec") or 0.0),
            float(dict(profile.get("stage_entry") or {}).get("stage_start_timeout_sec") or 0.0),
            float(dict(profile.get("stage_entry") or {}).get("scan_interval_sec") or 0.0),
        )
        logger.info(
            "Combat[pre_combat_approach] enabled=%s hold_key=%s max_duration_sec=%.2f min_hold_sec=%.2f scan_interval_sec=%.2f release_delay_ms=%s",
            bool(dict(profile.get("pre_combat_approach") or {}).get("enabled", False)),
            str(dict(profile.get("pre_combat_approach") or {}).get("hold_key") or "w"),
            float(dict(profile.get("pre_combat_approach") or {}).get("max_duration_sec") or 0.0),
            float(dict(profile.get("pre_combat_approach") or {}).get("min_hold_sec") or 0.0),
            float(dict(profile.get("pre_combat_approach") or {}).get("scan_interval_sec") or 0.0),
            int(dict(profile.get("pre_combat_approach") or {}).get("release_delay_ms") or 0),
        )
        enemy_health_yolo_runtime = _CombatTargetsYoloRuntime(yolo, profile)
        enemy_health_yolo_config = dict(profile.get("enemy_health_yolo") or {})
        logger.info(
            "Combat[yolo_combat_targets] enabled=%s model=%s conf=%.3f direction_conf=%.3f reward_conf=%.3f iou=%.3f max_det=%s ttl_sec=%.3f fallback_to_hsv_on_error=%s",
            bool(enemy_health_yolo_runtime.enabled),
            enemy_health_yolo_runtime.model_name,
            enemy_health_yolo_runtime.conf,
            enemy_health_yolo_runtime.direction_conf,
            enemy_health_yolo_runtime.reward_conf,
            enemy_health_yolo_runtime.iou,
            enemy_health_yolo_runtime.max_det,
            enemy_health_yolo_runtime.last_seen_ttl_sec,
            bool(enemy_health_yolo_config.get("fallback_to_hsv_on_error", True)),
        )

        if not dry_run_enabled:
            _try_focus_activation(
                app,
                dry_run=dry_run_enabled,
                action_trace=action_trace,
                start_time=start_time,
                reason="session_start",
            )

        audio_dodge_runtime = AudioDodgeRuntime.from_profile(profile, enabled=auto_dodge_enabled)
        audio_dodge_runtime.start()
        audio_status = audio_dodge_runtime.status() if hasattr(audio_dodge_runtime, "status") else {"status": "unknown"}
        logger.info("Combat[audio_dodge] status=%s", audio_status)
        capture_debug = _CombatCaptureLogger(
            enabled=capture_debug_enabled_resolved,
            interval_sec=capture_interval_sec_resolved,
            max_images=capture_max_images_resolved,
            raw_enabled=capture_raw_enabled_resolved,
            profile_name=resolved_profile,
            yihuan_combat=yihuan_combat,
        )
        capture_debug.start(start_time=start_time)

        combat_active = False
        exit_pending = False
        exit_pending_scans = 0
        exit_pending_started_at: float | None = None
        unsupported_stable = 0
        last_target_attempt = 0.0
        next_scan_at = time.monotonic()
        next_skill_at = float("inf")
        next_ultimate_at = float("inf")
        next_switch_at = float("inf")
        next_exit_confirm_at = float("inf")
        stage_entry_attempted = False
        pre_combat_approach_completed = False

        while True:
            _raise_if_cancelled()
            if deadline is not None and time.monotonic() >= deadline:
                stopped_reason = "dry_run_completed" if dry_run_enabled else "max_seconds"
                return _final_result(
                    status="success" if dry_run_enabled else "partial",
                    stopped_reason=stopped_reason,
                    failure_reason=None,
                    profile_name=resolved_profile,
                    strategy_name=strategy_name,
                    encounters_completed=encounters_completed,
                    current_phase=current_phase,
                    last_state=last_state,
                    combat_state_trace=combat_state_trace,
                    action_trace=action_trace,
                    start_time=start_time,
                    dry_run=dry_run_enabled,
                    capture_debug=capture_debug,
                )

            now = time.monotonic()

            trigger_event = _consume_audio_dodge_trigger(audio_dodge_runtime, combat_active=combat_active)
            if trigger_event is not None:
                current_phase = "audio_dodge"
                dodge_pause_sec = _execute_audio_dodge(
                    app,
                    profile,
                    trigger_event=trigger_event,
                    dry_run=dry_run_enabled,
                    action_trace=action_trace,
                    start_time=start_time,
                )
                dodge_pause_until = max(dodge_pause_until, time.monotonic() + dodge_pause_sec)
                if last_state is not None:
                    _trace_scan(
                        combat_state_trace,
                        start_time=start_time,
                        state=last_state,
                        combat_active=combat_active,
                        phase=current_phase,
                        note="dodge_executed",
                        trace_limit=trace_limit,
                    )
                continue

            if last_state is None or now >= next_scan_at:
                state, _capture = _capture_state(
                    app,
                    yihuan_combat,
                    profile_name=resolved_profile,
                    enemy_health_yolo=enemy_health_yolo_runtime,
                )
                last_state = dict(state)
                if state.get("current_slot") is not None:
                    last_known_slot = _coerce_int(state.get("current_slot"), 0) or last_known_slot
                last_capture_image = _capture.image
                current_phase = "combat" if combat_active else "monitor"
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="combat_scan" if combat_active else "monitor_scan",
                    trace_limit=trace_limit,
                )
                if not initial_monitor_captured:
                    _capture_event_image(
                        capture_debug,
                        label="initial_monitor",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=combat_active,
                    )
                    initial_monitor_captured = True
                _capture_periodic_if_due(
                    app,
                    capture_debug,
                    state=state,
                    phase=current_phase,
                    start_time=start_time,
                    combat_active=combat_active,
                )

                if (
                    not combat_active
                    and not stage_entry_attempted
                    and bool(dict(profile.get("stage_entry") or {}).get("enabled", False))
                    and bool(state.get("stage_enter_button_found"))
                ):
                    current_phase = "stage_entry"
                    stage_entry_attempted = True
                    clicked = _click_stage_enter_button(
                        app,
                        profile,
                        state,
                        dry_run=dry_run_enabled,
                        action_trace=action_trace,
                        capture_debug=capture_debug,
                        source_image=last_capture_image,
                        start_time=start_time,
                    )
                    if clicked:
                        wait_result = _wait_for_stage_started(
                            app,
                            yihuan_combat,
                            profile,
                            profile_name=resolved_profile,
                            combat_targets_yolo=enemy_health_yolo_runtime,
                            dry_run=dry_run_enabled,
                            action_trace=action_trace,
                            combat_state_trace=combat_state_trace,
                            capture_debug=capture_debug,
                            start_time=start_time,
                            trace_limit=trace_limit,
                            session_deadline=deadline,
                        )
                        if wait_result.get("state") is not None:
                            last_state = dict(wait_result.get("state") or {})
                        if wait_result.get("capture_image") is not None:
                            last_capture_image = wait_result.get("capture_image")
                        if wait_result.get("status") != "success":
                            return _final_result(
                                status="failed",
                                stopped_reason=str(wait_result.get("reason") or "stage_enter_timeout"),
                                failure_reason=str(wait_result.get("reason") or "stage_enter_timeout"),
                                profile_name=resolved_profile,
                                strategy_name=strategy_name,
                                encounters_completed=encounters_completed,
                                current_phase="stage_enter_wait",
                                last_state=last_state,
                                combat_state_trace=combat_state_trace,
                                action_trace=action_trace,
                                start_time=start_time,
                                dry_run=dry_run_enabled,
                                capture_debug=capture_debug,
                            )
                        if last_state is not None and bool(last_state.get("in_combat")):
                            approach_result = _approach_until_enemy_seen(
                                app,
                                yihuan_combat,
                                profile,
                                profile_name=resolved_profile,
                                combat_targets_yolo=enemy_health_yolo_runtime,
                                initial_state=last_state,
                                initial_capture_image=last_capture_image,
                                dry_run=dry_run_enabled,
                                action_trace=action_trace,
                                combat_state_trace=combat_state_trace,
                                capture_debug=capture_debug,
                                start_time=start_time,
                                trace_limit=trace_limit,
                                session_deadline=deadline,
                            )
                            pre_combat_approach_completed = True
                            if approach_result.get("state") is not None:
                                last_state = dict(approach_result.get("state") or {})
                            if approach_result.get("capture_image") is not None:
                                last_capture_image = approach_result.get("capture_image")
                        next_scan_at = time.monotonic()
                        continue

                if bool(state.get("in_supported_scene")):
                    unsupported_stable = 0
                else:
                    unsupported_stable += 1
                    if unsupported_stable >= unsupported_required:
                        current_phase = "monitor"
                        return _final_result(
                            status="partial",
                            stopped_reason="not_in_supported_scene",
                            failure_reason=None,
                            profile_name=resolved_profile,
                            strategy_name=strategy_name,
                            encounters_completed=encounters_completed,
                            current_phase=current_phase,
                            last_state=last_state,
                            combat_state_trace=combat_state_trace,
                            action_trace=action_trace,
                            start_time=start_time,
                            dry_run=dry_run_enabled,
                            capture_debug=capture_debug,
                        )

                if combat_active:
                    if bool(state.get("challenge_success_found")) and bool(schedule["exit_challenge_success_immediate"]):
                        current_phase = "post_combat"
                        _capture_event_image(
                            capture_debug,
                            label="challenge_success",
                            phase=current_phase,
                            source_image=last_capture_image,
                            state=state,
                            start_time=start_time,
                            combat_active=False,
                        )
                        _record_action(
                            action_trace,
                            start_time=start_time,
                            action="challenge_success",
                            dry_run=dry_run_enabled,
                        )
                        combat_active = False
                        exit_pending = False
                        exit_pending_scans = 0
                        exit_pending_started_at = None
                        next_exit_confirm_at = float("inf")
                        encounters_completed += 1
                        next_skill_at = float("inf")
                        next_ultimate_at = float("inf")
                        next_switch_at = float("inf")
                        pre_combat_approach_completed = False
                        _trace_scan(
                            combat_state_trace,
                            start_time=start_time,
                            state=state,
                            combat_active=False,
                            phase=current_phase,
                            note="challenge_success",
                            trace_limit=trace_limit,
                        )
                        _pause_after_encounter(
                            app,
                            profile,
                            dry_run=dry_run_enabled,
                            cooldown_ms=post_exit_cooldown_ms,
                        )
                        if encounter_limit > 0:
                            reward_enabled = bool(dict(profile.get("post_combat_reward") or {}).get("enabled", True))
                            if not reward_enabled:
                                if encounters_completed < encounter_limit:
                                    current_phase = "monitor"
                                    next_scan_at = _enqueue_next_scan(
                                        now=time.monotonic(),
                                        combat_active=False,
                                        schedule=schedule,
                                    )
                                    continue
                                return _final_result(
                                    status="success",
                                    stopped_reason=stop_limit_source,
                                    failure_reason=None,
                                    profile_name=resolved_profile,
                                    strategy_name=strategy_name,
                                    encounters_completed=encounters_completed,
                                    current_phase=current_phase,
                                    last_state=last_state,
                                    combat_state_trace=combat_state_trace,
                                    action_trace=action_trace,
                                    start_time=start_time,
                                    dry_run=dry_run_enabled,
                                    capture_debug=capture_debug,
                                )
                            current_phase = "post_combat_reward"
                            current_claim_mode = _claim_mode_for_remaining_runs(encounter_limit, reward_runs_completed)
                            _record_action(
                                action_trace,
                                start_time=start_time,
                                action="battle_run_finished",
                                dry_run=dry_run_enabled,
                                details={
                                    "index": int(encounters_completed),
                                    "total": int(encounter_limit),
                                    "claim_mode": current_claim_mode,
                                    "completed_runs": int(reward_runs_completed),
                                    "remaining": max(int(encounter_limit) - int(reward_runs_completed), 0),
                                },
                            )
                            post_combat_reward_result = _run_post_battle_reward_sequence(
                                app,
                                yihuan_combat,
                                profile,
                                profile_name=resolved_profile,
                                combat_targets_yolo=enemy_health_yolo_runtime,
                                claim_mode=current_claim_mode,
                                completed_runs=reward_runs_completed,
                                total_runs=encounter_limit,
                                dry_run=dry_run_enabled,
                                action_trace=action_trace,
                                combat_state_trace=combat_state_trace,
                                capture_debug=capture_debug,
                                start_time=start_time,
                                trace_limit=trace_limit,
                                session_deadline=deadline,
                            )
                            if post_combat_reward_result.get("state"):
                                last_state = dict(post_combat_reward_result.get("state") or {})
                            if post_combat_reward_result.get("capture_image") is not None:
                                last_capture_image = post_combat_reward_result.get("capture_image")
                            if post_combat_reward_result.get("status") != "success":
                                return _final_result(
                                    status="failed",
                                    stopped_reason=str(post_combat_reward_result.get("reason") or "post_combat_reward_failed"),
                                    failure_reason=str(post_combat_reward_result.get("reason") or "post_combat_reward_failed"),
                                    profile_name=resolved_profile,
                                    strategy_name=strategy_name,
                                    encounters_completed=encounters_completed,
                                    current_phase=current_phase,
                                    last_state=last_state,
                                    combat_state_trace=combat_state_trace,
                                    action_trace=action_trace,
                                    start_time=start_time,
                                    dry_run=dry_run_enabled,
                                    capture_debug=capture_debug,
                                    post_combat_reward=post_combat_reward_result,
                                )
                            reward_runs_completed = max(
                                reward_runs_completed,
                                _coerce_int(post_combat_reward_result.get("completed_runs"), reward_runs_completed),
                            )
                            if post_combat_reward_result.get("action") == "retry":
                                next_claim_mode = _claim_mode_for_remaining_runs(encounter_limit, reward_runs_completed)
                                _record_action(
                                    action_trace,
                                    start_time=start_time,
                                    action="battle_run_started",
                                    dry_run=dry_run_enabled,
                                    details={
                                        "index": int(encounters_completed) + 1,
                                        "total": int(encounter_limit),
                                        "claim_mode": next_claim_mode,
                                        "completed_runs": int(reward_runs_completed),
                                        "remaining": max(int(encounter_limit) - int(reward_runs_completed), 0),
                                    },
                                )
                                current_phase = "monitor"
                                combat_active = False
                                exit_pending = False
                                exit_pending_scans = 0
                                exit_pending_started_at = None
                                next_exit_confirm_at = float("inf")
                                next_skill_at = float("inf")
                                next_ultimate_at = float("inf")
                                next_switch_at = float("inf")
                                pre_combat_approach_completed = True
                                next_scan_at = time.monotonic()
                                continue
                            return _final_result(
                                status="success",
                                stopped_reason=stop_limit_source,
                                failure_reason=None,
                                profile_name=resolved_profile,
                                strategy_name=strategy_name,
                                encounters_completed=encounters_completed,
                                current_phase=current_phase,
                                last_state=last_state,
                                combat_state_trace=combat_state_trace,
                                action_trace=action_trace,
                                start_time=start_time,
                                dry_run=dry_run_enabled,
                                capture_debug=capture_debug,
                                post_combat_reward=post_combat_reward_result,
                            )
                        current_phase = "monitor"
                        next_scan_at = _enqueue_next_scan(
                            now=time.monotonic(),
                            combat_active=False,
                            schedule=schedule,
                        )
                        continue

                    if bool(state.get("in_combat")):
                        if exit_pending and bool(schedule["exit_reset_on_reacquire"]):
                            exit_pending = False
                            exit_pending_scans = 0
                            exit_pending_started_at = None
                            next_exit_confirm_at = float("inf")
                            _record_action(
                                action_trace,
                                start_time=start_time,
                                action="exit_pending_reset_on_reacquire",
                                dry_run=dry_run_enabled,
                                details={
                                    "required_scans": int(schedule["exit_confirm_required_scans"]),
                                    "target_found": bool(state.get("target_found")),
                                },
                            )
                            _trace_scan(
                                combat_state_trace,
                                start_time=start_time,
                                state=state,
                                combat_active=True,
                                phase="combat",
                                note="exit_pending_reset_on_reacquire",
                                trace_limit=trace_limit,
                            )
                    else:
                        current_phase = "exit_pending"
                        if not exit_pending:
                            exit_pending = True
                            exit_pending_scans = 1
                            exit_pending_started_at = time.monotonic()
                            next_exit_confirm_at = time.monotonic() + float(schedule["exit_confirm_interval_sec"])
                            _capture_event_image(
                                capture_debug,
                                label="exit_pending_start",
                                phase=current_phase,
                                source_image=last_capture_image,
                                state=state,
                                start_time=start_time,
                                combat_active=True,
                            )
                            _record_action(
                                action_trace,
                                start_time=start_time,
                                action="exit_pending_start",
                                dry_run=dry_run_enabled,
                                details={
                                    "scan_count": exit_pending_scans,
                                    "required_scans": int(schedule["exit_confirm_required_scans"]),
                                    "confirm_interval_sec": float(schedule["exit_confirm_interval_sec"]),
                                    "confirm_missing_sec": float(schedule["exit_confirm_missing_sec"]),
                                },
                            )
                            _trace_scan(
                                combat_state_trace,
                                start_time=start_time,
                                state=state,
                                combat_active=True,
                                phase=current_phase,
                                note="exit_pending_start",
                                trace_limit=trace_limit,
                            )
                            _release_inputs(app, profile, dry_run=dry_run_enabled)
                            next_scan_at = next_exit_confirm_at
                            continue

                        exit_pending_scans += 1
                        _capture_event_image(
                            capture_debug,
                            label="exit_pending_scan",
                            phase=current_phase,
                            source_image=last_capture_image,
                            state=state,
                            start_time=start_time,
                            combat_active=True,
                        )
                        _record_action(
                            action_trace,
                            start_time=start_time,
                            action="exit_pending_scan",
                            dry_run=dry_run_enabled,
                            details={
                                "scan_count": exit_pending_scans,
                                "required_scans": int(schedule["exit_confirm_required_scans"]),
                                "missing_elapsed_sec": round(
                                    max(time.monotonic() - float(exit_pending_started_at or time.monotonic()), 0.0),
                                    3,
                                ),
                                "confirm_missing_sec": float(schedule["exit_confirm_missing_sec"]),
                            },
                        )
                        _trace_scan(
                            combat_state_trace,
                            start_time=start_time,
                            state=state,
                            combat_active=True,
                            phase=current_phase,
                            note="exit_pending_scan",
                            trace_limit=trace_limit,
                        )
                        missing_elapsed_sec = max(
                            time.monotonic() - float(exit_pending_started_at or time.monotonic()),
                            0.0,
                        )
                        if (
                            exit_pending_scans < int(schedule["exit_confirm_required_scans"])
                            or missing_elapsed_sec < float(schedule["exit_confirm_missing_sec"])
                        ):
                            next_exit_confirm_at = time.monotonic() + float(schedule["exit_confirm_interval_sec"])
                            next_scan_at = next_exit_confirm_at
                            continue

                        current_phase = "post_combat"
                        _capture_event_image(
                            capture_debug,
                            label="exit_combat_confirmed",
                            phase=current_phase,
                            source_image=last_capture_image,
                            state=state,
                            start_time=start_time,
                            combat_active=False,
                        )
                        _record_action(
                            action_trace,
                            start_time=start_time,
                            action="exit_combat_confirmed",
                            dry_run=dry_run_enabled,
                            details={
                                "scan_count": exit_pending_scans,
                                "missing_elapsed_sec": round(missing_elapsed_sec, 3),
                                "confirm_missing_sec": float(schedule["exit_confirm_missing_sec"]),
                            },
                        )
                        combat_active = False
                        exit_pending = False
                        exit_pending_scans = 0
                        exit_pending_started_at = None
                        next_exit_confirm_at = float("inf")
                        encounters_completed += 1
                        next_skill_at = float("inf")
                        next_ultimate_at = float("inf")
                        next_switch_at = float("inf")
                        pre_combat_approach_completed = False
                        _trace_scan(
                            combat_state_trace,
                            start_time=start_time,
                            state=state,
                            combat_active=False,
                            phase=current_phase,
                            note="exit_combat_confirmed",
                            trace_limit=trace_limit,
                        )
                        _pause_after_encounter(
                            app,
                            profile,
                            dry_run=dry_run_enabled,
                            cooldown_ms=post_exit_cooldown_ms,
                        )
                        if encounter_limit > 0:
                            reward_enabled = bool(dict(profile.get("post_combat_reward") or {}).get("enabled", True))
                            if not reward_enabled:
                                if encounters_completed < encounter_limit:
                                    current_phase = "monitor"
                                    next_scan_at = _enqueue_next_scan(
                                        now=time.monotonic(),
                                        combat_active=False,
                                        schedule=schedule,
                                    )
                                    continue
                                return _final_result(
                                    status="success",
                                    stopped_reason=stop_limit_source,
                                    failure_reason=None,
                                    profile_name=resolved_profile,
                                    strategy_name=strategy_name,
                                    encounters_completed=encounters_completed,
                                    current_phase=current_phase,
                                    last_state=last_state,
                                    combat_state_trace=combat_state_trace,
                                    action_trace=action_trace,
                                    start_time=start_time,
                                    dry_run=dry_run_enabled,
                                    capture_debug=capture_debug,
                                )
                            current_phase = "post_combat_reward"
                            current_claim_mode = _claim_mode_for_remaining_runs(encounter_limit, reward_runs_completed)
                            _record_action(
                                action_trace,
                                start_time=start_time,
                                action="battle_run_finished",
                                dry_run=dry_run_enabled,
                                details={
                                    "index": int(encounters_completed),
                                    "total": int(encounter_limit),
                                    "claim_mode": current_claim_mode,
                                    "completed_runs": int(reward_runs_completed),
                                    "remaining": max(int(encounter_limit) - int(reward_runs_completed), 0),
                                },
                            )
                            post_combat_reward_result = _run_post_battle_reward_sequence(
                                app,
                                yihuan_combat,
                                profile,
                                profile_name=resolved_profile,
                                combat_targets_yolo=enemy_health_yolo_runtime,
                                claim_mode=current_claim_mode,
                                completed_runs=reward_runs_completed,
                                total_runs=encounter_limit,
                                dry_run=dry_run_enabled,
                                action_trace=action_trace,
                                combat_state_trace=combat_state_trace,
                                capture_debug=capture_debug,
                                start_time=start_time,
                                trace_limit=trace_limit,
                                session_deadline=deadline,
                            )
                            if post_combat_reward_result.get("state"):
                                last_state = dict(post_combat_reward_result.get("state") or {})
                            if post_combat_reward_result.get("capture_image") is not None:
                                last_capture_image = post_combat_reward_result.get("capture_image")
                            if post_combat_reward_result.get("status") != "success":
                                return _final_result(
                                    status="failed",
                                    stopped_reason=str(post_combat_reward_result.get("reason") or "post_combat_reward_failed"),
                                    failure_reason=str(post_combat_reward_result.get("reason") or "post_combat_reward_failed"),
                                    profile_name=resolved_profile,
                                    strategy_name=strategy_name,
                                    encounters_completed=encounters_completed,
                                    current_phase=current_phase,
                                    last_state=last_state,
                                    combat_state_trace=combat_state_trace,
                                    action_trace=action_trace,
                                    start_time=start_time,
                                    dry_run=dry_run_enabled,
                                    capture_debug=capture_debug,
                                    post_combat_reward=post_combat_reward_result,
                                )
                            reward_runs_completed = max(
                                reward_runs_completed,
                                _coerce_int(post_combat_reward_result.get("completed_runs"), reward_runs_completed),
                            )
                            if post_combat_reward_result.get("action") == "retry":
                                next_claim_mode = _claim_mode_for_remaining_runs(encounter_limit, reward_runs_completed)
                                _record_action(
                                    action_trace,
                                    start_time=start_time,
                                    action="battle_run_started",
                                    dry_run=dry_run_enabled,
                                    details={
                                        "index": int(encounters_completed) + 1,
                                        "total": int(encounter_limit),
                                        "claim_mode": next_claim_mode,
                                        "completed_runs": int(reward_runs_completed),
                                        "remaining": max(int(encounter_limit) - int(reward_runs_completed), 0),
                                    },
                                )
                                current_phase = "monitor"
                                combat_active = False
                                exit_pending = False
                                exit_pending_scans = 0
                                exit_pending_started_at = None
                                next_exit_confirm_at = float("inf")
                                next_skill_at = float("inf")
                                next_ultimate_at = float("inf")
                                next_switch_at = float("inf")
                                pre_combat_approach_completed = True
                                next_scan_at = time.monotonic()
                                continue
                            return _final_result(
                                status="success",
                                stopped_reason=stop_limit_source,
                                failure_reason=None,
                                profile_name=resolved_profile,
                                strategy_name=strategy_name,
                                encounters_completed=encounters_completed,
                                current_phase=current_phase,
                                last_state=last_state,
                                combat_state_trace=combat_state_trace,
                                action_trace=action_trace,
                                start_time=start_time,
                                dry_run=dry_run_enabled,
                                capture_debug=capture_debug,
                                post_combat_reward=post_combat_reward_result,
                            )
                        current_phase = "monitor"
                        next_scan_at = _enqueue_next_scan(
                            now=time.monotonic(),
                            combat_active=False,
                            schedule=schedule,
                        )
                        continue

                elif bool(state.get("in_combat")):
                    if (
                        bool(dict(profile.get("pre_combat_approach") or {}).get("enabled", False))
                        and not pre_combat_approach_completed
                        and not _has_approach_enemy_signal(state)
                    ):
                        current_phase = "approach_enemy"
                        approach_result = _approach_until_enemy_seen(
                            app,
                            yihuan_combat,
                            profile,
                            profile_name=resolved_profile,
                            combat_targets_yolo=enemy_health_yolo_runtime,
                            initial_state=state,
                            initial_capture_image=last_capture_image,
                            dry_run=dry_run_enabled,
                            action_trace=action_trace,
                            combat_state_trace=combat_state_trace,
                            capture_debug=capture_debug,
                            start_time=start_time,
                            trace_limit=trace_limit,
                            session_deadline=deadline,
                        )
                        pre_combat_approach_completed = True
                        if approach_result.get("state") is not None:
                            last_state = dict(approach_result.get("state") or {})
                        if approach_result.get("capture_image") is not None:
                            last_capture_image = approach_result.get("capture_image")
                        next_scan_at = time.monotonic()
                        continue

                    combat_active = True
                    exit_pending = False
                    exit_pending_scans = 0
                    exit_pending_started_at = None
                    next_exit_confirm_at = float("inf")
                    current_phase = "enemy_detected"
                    _trace_scan(
                        combat_state_trace,
                        start_time=start_time,
                        state=state,
                        combat_active=True,
                        phase=current_phase,
                        note="enter_combat",
                        trace_limit=trace_limit,
                    )
                    _capture_event_image(
                        capture_debug,
                        label="enter_combat",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=True,
                    )
                    if auto_target_enabled and _has_front_enemy(state) and not bool(state.get("target_found")):
                        current_phase = "acquire_target"
                        _capture_event_image(
                            capture_debug,
                            label="auto_target_attempt",
                            phase=current_phase,
                            source_image=last_capture_image,
                            state=state,
                            start_time=start_time,
                            combat_active=True,
                        )
                        _tap_binding(
                            app,
                            dict(profile["keys"])["target"],
                            dry_run=dry_run_enabled,
                            action_trace=action_trace,
                            start_time=start_time,
                            action_name="auto_target",
                        )
                        last_target_attempt = time.monotonic()
                    if bool(schedule["immediate_on_combat_enter"]):
                        next_skill_at = time.monotonic()
                        next_ultimate_at = time.monotonic()
                    else:
                        next_skill_at = _schedule_next_due(time.monotonic(), float(schedule["skill_interval_sec"]))
                        next_ultimate_at = _schedule_next_due(time.monotonic(), float(schedule["ultimate_interval_sec"]))
                    next_switch_at = _schedule_next_due(time.monotonic(), float(schedule["switch_interval_sec"]))
                    next_scan_at = _enqueue_next_scan(
                        now=time.monotonic(),
                        combat_active=True,
                        schedule=schedule,
                    )
                    continue

                next_scan_at = _enqueue_next_scan(
                    now=time.monotonic(),
                    combat_active=combat_active,
                    schedule=schedule,
                )

            if not combat_active or last_state is None:
                current_phase = "monitor"
                _capture_periodic_if_due(
                    app,
                    capture_debug,
                    state=last_state,
                    phase=current_phase,
                    start_time=start_time,
                    combat_active=False,
                )
                remaining = max(next_scan_at - time.monotonic(), 0.0)
                _sleep_interruptibly(min(float(schedule["action_loop_sleep_sec"]), remaining if remaining > 0 else float(schedule["action_loop_sleep_sec"])))
                continue

            if time.monotonic() < dodge_pause_until:
                current_phase = "combat"
                _capture_periodic_if_due(
                    app,
                    capture_debug,
                    state=last_state,
                    phase=current_phase,
                    start_time=start_time,
                    combat_active=True,
                )
                _sleep_interruptibly(min(float(schedule["action_loop_sleep_sec"]), max(dodge_pause_until - time.monotonic(), 0.0)))
                continue

            if exit_pending:
                current_phase = "exit_pending"
                _capture_periodic_if_due(
                    app,
                    capture_debug,
                    state=last_state,
                    phase=current_phase,
                    start_time=start_time,
                    combat_active=True,
                )
                remaining = max(next_scan_at - time.monotonic(), 0.0)
                _sleep_interruptibly(
                    min(
                        float(schedule["action_loop_sleep_sec"]),
                        remaining if remaining > 0 else float(schedule["action_loop_sleep_sec"]),
                    )
                )
                continue

            state = dict(last_state)
            current_phase = "combat"

            if not _has_front_enemy(state):
                current_phase = "reacquire_target"
                if time.monotonic() - last_rear_turn_at >= float(schedule["rear_enemy_turn_interval_sec"]):
                    _capture_event_image(
                        capture_debug,
                        label="auto_target_attempt_after_turn",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=True,
                    )
                    rear_turn_result = _turn_to_front_enemy(
                        app,
                        profile,
                        state,
                        auto_target_enabled=auto_target_enabled,
                        scan_direction=front_enemy_scan_direction,
                        dry_run=dry_run_enabled,
                        action_trace=action_trace,
                        start_time=start_time,
                    )
                    last_rear_turn_at = time.monotonic()
                    if bool(rear_turn_result.get("auto_targeted")):
                        last_target_attempt = time.monotonic()
                    _trace_scan(
                        combat_state_trace,
                        start_time=start_time,
                        state=state,
                        combat_active=True,
                        phase=current_phase,
                        note="front_enemy_reacquire",
                        trace_limit=trace_limit,
                    )
                _capture_periodic_if_due(
                    app,
                    capture_debug,
                    state=state,
                    phase=current_phase,
                    start_time=start_time,
                    combat_active=True,
                )
                remaining = max(next_scan_at - time.monotonic(), 0.0)
                _sleep_interruptibly(
                    min(
                        float(schedule["action_loop_sleep_sec"]),
                        remaining if remaining > 0 else float(schedule["action_loop_sleep_sec"]),
                    )
                )
                continue

            if (
                auto_target_enabled
                and not bool(state.get("target_found"))
                and time.monotonic() - last_target_attempt >= target_retry_interval_sec
            ):
                current_phase = "acquire_target"
                _capture_event_image(
                    capture_debug,
                    label="auto_target_attempt",
                    phase=current_phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=True,
                )
                _tap_binding(
                    app,
                    dict(profile["keys"])["target"],
                    dry_run=dry_run_enabled,
                    action_trace=action_trace,
                    start_time=start_time,
                    action_name="auto_target",
                )
                last_target_attempt = time.monotonic()
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="auto_target_attempt",
                    trace_limit=trace_limit,
                )

            if time.monotonic() >= next_ultimate_at:
                _capture_event_image(
                    capture_debug,
                    label="ultimate_due",
                    phase=current_phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=True,
                )
                _press_scheduled_ability(
                    app,
                    input_mapping,
                    profile,
                    ability_name="ultimate",
                    action_trace=action_trace,
                    start_time=start_time,
                    dry_run=dry_run_enabled,
                    press_ms=int(schedule["ultimate_press_ms"]),
                )
                next_ultimate_at = _schedule_next_due(time.monotonic(), float(schedule["ultimate_interval_sec"]))
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="ultimate_due",
                    trace_limit=trace_limit,
                )
                continue

            if time.monotonic() >= next_skill_at:
                _capture_event_image(
                    capture_debug,
                    label="skill_due",
                    phase=current_phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=True,
                )
                _press_scheduled_ability(
                    app,
                    input_mapping,
                    profile,
                    ability_name="skill",
                    action_trace=action_trace,
                    start_time=start_time,
                    dry_run=dry_run_enabled,
                    press_ms=int(schedule["skill_press_ms"]),
                )
                next_skill_at = _schedule_next_due(time.monotonic(), float(schedule["skill_interval_sec"]))
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="skill_due",
                    trace_limit=trace_limit,
                )
                continue

            if time.monotonic() >= next_switch_at:
                _capture_event_image(
                    capture_debug,
                    label="switch_attempt",
                    phase=current_phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=True,
                )
                switch_result = _switch_slot(
                    app,
                    input_mapping,
                    profile,
                    state,
                    "next",
                    yihuan_combat=yihuan_combat,
                    profile_name=resolved_profile,
                    dry_run=dry_run_enabled,
                    action_trace=action_trace,
                    start_time=start_time,
                    press_ms=int(schedule["switch_press_ms"]),
                    fallback_current_slot=last_known_slot,
                )
                if switch_result.get("state") is not None:
                    last_state = dict(switch_result.get("state") or {})
                    state = dict(last_state)
                    if state.get("current_slot") is not None:
                        last_known_slot = _coerce_int(state.get("current_slot"), 0) or last_known_slot
                if switch_result.get("capture_image") is not None:
                    last_capture_image = switch_result.get("capture_image")
                if bool(switch_result.get("success")):
                    if switch_result.get("slot") is not None:
                        last_known_slot = _coerce_int(switch_result.get("slot"), 0) or last_known_slot
                    _capture_event_image(
                        capture_debug,
                        label="switch_success",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=True,
                    )
                    next_switch_at = _schedule_next_due(time.monotonic(), float(schedule["switch_interval_sec"]))
                else:
                    _capture_event_image(
                        capture_debug,
                        label="switch_confirm_failed",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=True,
                    )
                    retry_delay_sec = float(schedule["failed_switch_cooldown_sec"]) or float(schedule["switch_interval_sec"])
                    next_switch_at = _schedule_next_due(time.monotonic(), retry_delay_sec)
                    _record_action(
                        action_trace,
                        start_time=start_time,
                        action="switch_retry_scheduled",
                        dry_run=dry_run_enabled,
                        details={"retry_delay_sec": round(float(retry_delay_sec), 3)},
                    )
                    _capture_event_image(
                        capture_debug,
                        label="switch_retry_scheduled",
                        phase=current_phase,
                        source_image=last_capture_image,
                        state=state,
                        start_time=start_time,
                        combat_active=True,
                    )
                if bool(switch_result.get("success")) and float(schedule["post_switch_delay_sec"]) > 0:
                    _sleep_interruptibly(float(schedule["post_switch_delay_sec"]))
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="switch_success" if bool(switch_result.get("success")) else "switch_failed",
                    trace_limit=trace_limit,
                )
                continue

            normal_result = _normal_attack(
                app,
                dict(profile["keys"])["normal_attack"],
                mode=str(normal_command["mode"]),
                duration_ms=_coerce_int(normal_command["duration_ms"], 560),
                interval_ms=_coerce_int(normal_command["interval_ms"], 70),
                dry_run=dry_run_enabled,
                action_trace=action_trace,
                start_time=start_time,
                interrupt_checker=lambda: (
                    {
                        "reason": "audio_dodge",
                        "trigger_event": trigger,
                    }
                    if (trigger := _consume_audio_dodge_trigger(audio_dodge_runtime, combat_active=True)) is not None
                    else None
                ),
            )
            interrupt_reason = normal_result.get("interrupt_reason")
            interrupt_payload = dict(normal_result.get("interrupt_payload") or {})
            if interrupt_reason == "audio_dodge":
                current_phase = "audio_dodge"
                _capture_event_image(
                    capture_debug,
                    label="audio_dodge",
                    phase=current_phase,
                    source_image=last_capture_image,
                    state=state,
                    start_time=start_time,
                    combat_active=True,
                )
                dodge_pause_sec = _execute_audio_dodge(
                    app,
                    profile,
                    trigger_event=dict(interrupt_payload.get("trigger_event") or {}),
                    dry_run=dry_run_enabled,
                    action_trace=action_trace,
                    start_time=start_time,
                )
                dodge_pause_until = max(dodge_pause_until, time.monotonic() + dodge_pause_sec)
                _trace_scan(
                    combat_state_trace,
                    start_time=start_time,
                    state=state,
                    combat_active=combat_active,
                    phase=current_phase,
                    note="dodge_preempted_action",
                    trace_limit=trace_limit,
                )
                continue

            _trace_scan(
                combat_state_trace,
                start_time=start_time,
                state=state,
                combat_active=combat_active,
                phase=current_phase,
                note=f"combat_loop:{normal_result.get('action') or 'idle'}",
                trace_limit=trace_limit,
            )
            _capture_periodic_if_due(
                app,
                capture_debug,
                state=state,
                phase=current_phase,
                start_time=start_time,
                combat_active=True,
            )
            remaining_to_scan = max(next_scan_at - time.monotonic(), 0.0)
            if remaining_to_scan > 0:
                _sleep_interruptibly(min(float(schedule["action_loop_sleep_sec"]), remaining_to_scan))

    except _CombatSessionCancelled:
        logger.info("Combat[session] cancelled phase=%s", current_phase)
        last_capture_image = _resolve_terminal_capture_image(app, fallback_image=last_capture_image, capture_debug=capture_debug)
        _capture_event_image(
            capture_debug,
            label="cancelled",
            phase=current_phase,
            source_image=last_capture_image,
            state=last_state,
            start_time=start_time,
            combat_active=bool((last_state or {}).get("in_combat")),
        )
        _release_inputs(app, profile or {}, dry_run=_coerce_bool(dry_run))
        return _final_result(
            status="cancelled",
            stopped_reason="cancelled",
            failure_reason="cancelled",
            profile_name=str((profile or {}).get("profile_name") or profile_name),
            strategy_name=str(strategy_name),
            encounters_completed=encounters_completed,
            current_phase=current_phase,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
            capture_debug=capture_debug,
        )

    except FileNotFoundError as exc:
        logger.warning("Combat[session] profile failed: %s", exc)
        last_capture_image = _resolve_terminal_capture_image(app, fallback_image=last_capture_image, capture_debug=capture_debug)
        _capture_event_image(
            capture_debug,
            label="failure",
            phase=current_phase,
            source_image=last_capture_image,
            state=last_state,
            start_time=start_time,
            combat_active=bool((last_state or {}).get("in_combat")),
        )
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason="profile_not_found",
            profile_name=str(profile_name),
            strategy_name=str(strategy_name),
            encounters_completed=encounters_completed,
            current_phase=current_phase,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
            capture_debug=capture_debug,
        )
    except TargetRuntimeError as exc:
        failure_reason = "window_focus_failed" if exc.code == "window_focus_required" else str(exc.code or "input_failed")
        logger.warning("Combat[session] runtime failure [%s]: %s", exc.code, exc)
        last_capture_image = _resolve_terminal_capture_image(app, fallback_image=last_capture_image, capture_debug=capture_debug)
        _capture_event_image(
            capture_debug,
            label="failure",
            phase=current_phase,
            source_image=last_capture_image,
            state=last_state,
            start_time=start_time,
            combat_active=bool((last_state or {}).get("in_combat")),
        )
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason=failure_reason,
            profile_name=str(profile_name),
            strategy_name=str(strategy_name),
            encounters_completed=encounters_completed,
            current_phase=current_phase,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
            capture_debug=capture_debug,
        )
    except RuntimeError as exc:
        logger.warning("Combat[session] capture failed: %s", exc)
        last_capture_image = _resolve_terminal_capture_image(app, fallback_image=last_capture_image, capture_debug=capture_debug)
        _capture_event_image(
            capture_debug,
            label="failure",
            phase=current_phase,
            source_image=last_capture_image,
            state=last_state,
            start_time=start_time,
            combat_active=bool((last_state or {}).get("in_combat")),
        )
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason="capture_failed",
            profile_name=str(profile_name),
            strategy_name=str(strategy_name),
            encounters_completed=encounters_completed,
            current_phase=current_phase,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
            capture_debug=capture_debug,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Combat[session] unexpected failure: %s", exc)
        last_capture_image = _resolve_terminal_capture_image(app, fallback_image=last_capture_image, capture_debug=capture_debug)
        _capture_event_image(
            capture_debug,
            label="failure",
            phase=current_phase,
            source_image=last_capture_image,
            state=last_state,
            start_time=start_time,
            combat_active=bool((last_state or {}).get("in_combat")),
        )
        return _final_result(
            status="failed",
            stopped_reason="failed",
            failure_reason="exception",
            profile_name=str(profile_name),
            strategy_name=str(strategy_name),
            encounters_completed=encounters_completed,
            current_phase=current_phase,
            last_state=last_state,
            combat_state_trace=combat_state_trace,
            action_trace=action_trace,
            start_time=start_time,
            dry_run=_coerce_bool(dry_run),
            capture_debug=capture_debug,
        )
    finally:
        if audio_dodge_runtime is not None:
            audio_dodge_runtime.stop()
        if profile is not None:
            _release_inputs(app, profile, dry_run=_coerce_bool(dry_run))

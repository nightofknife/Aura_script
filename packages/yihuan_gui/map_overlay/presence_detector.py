from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


_URBAN_FUN_TEMPLATE_PNG_BASE64 = (
    "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"
    "kjPu+37HsY5QV5czO6+YPSxatEgpdezYsam/6O577mk7dKitrS2VSsXmrrXG76aiprXu7OxIV6dn1ddTQt2k63ne"
    "4cP4SYPRw+XBXQXkdxznkW9+M5VO/3DDhpMnT65bt67ik79+7bWtW7fGf963du1LL77Y1ta2afPmmHUppR76yle4"
    "IcDFIMBDd4CzPT2N8+fbDT7fe/7UKYPlJDylv1SLapMA/B+eeML2fN9//LHH8AOAIPjHp576zrp1j3772z95+mnP"
    "G1vA5o7T2NiYrbEn4nDHHXdcvXAhIeTGm27inL+1destS5c+u3Hj5+6/v6uz8/HHH59mPmDt6up65OGH7Rrsg/19"
    "/W0H26xAeoIj0jFyG5+KxLAfxtqPAmwnmUyeP3/+hz/4wZq77/7wgw/e3bFj1erVWus333jD3lBfX//TZ56J39e4"
    "YMH1zc3IBOfM8Txvx/btzTfcsGvnzjvXrDl9+vS2bdsSicTK2293zPefJpQglhb94rvb3jVFL4Yx6NI6j8wwEp9+"
    "6YtftNfXv/pVe8uau+569uc/7+/vf2z9epu/rF69+o9XrZrolT/buPGvPv95pdR/vvLK+u9+F3Of2tr/eOml+fPn"
    "p9Ppz953X0dHh1LqzJkz9v5UKi2lPHbkWCTuFFy99NGtMTTzx/8DobHjhVZxF9IAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class BigMapPresenceResult:
    is_candidate: bool
    score: float
    reasons: list[str]
    debug: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_candidate": self.is_candidate,
            "score": self.score,
            "reasons": list(self.reasons),
            "debug": dict(self.debug),
        }


class BigMapPresenceDetector:
    """Detect the Yihuan big-map UI by matching the bottom-left urban-fun icon."""

    candidate_threshold = 0.72
    _template_scales = (0.55, 0.65, 0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.25, 1.35, 1.50, 1.70)

    def detect(self, image: Image.Image) -> BigMapPresenceResult:
        array = _client_like_array(image)
        height, width = array.shape[:2]
        if height < 240 or width < 320:
            return BigMapPresenceResult(
                is_candidate=False,
                score=0.0,
                reasons=[],
                debug={"reason": "image_too_small", "size": (width, height)},
            )

        match = self._match_urban_fun_template(array)
        is_candidate = bool(match.get("matched"))
        return BigMapPresenceResult(
            is_candidate=is_candidate,
            score=round(float(match.get("score", 0.0)), 4),
            reasons=["urban_fun_template"] if is_candidate else [],
            debug={
                "client_size": (width, height),
                "urban_fun_template": match,
            },
        )

    def _match_urban_fun_template(self, array: np.ndarray) -> dict[str, Any]:
        cv2, cv2_error = _cv2_module()
        if cv2 is None:
            return {"available": False, "matched": False, "score": 0.0, "error": cv2_error}

        gray = array.mean(axis=2).astype(np.uint8) if array.ndim == 3 else array.astype(np.uint8)
        height, width = gray.shape[:2]
        left, top, right, bottom = _roi_bounds(width, height, 0.0, 0.58, 0.28, 1.0)
        search = np.ascontiguousarray(gray[top:bottom, left:right])
        template = _urban_fun_template_gray()

        best_score = -1.0
        best_location = (0, 0)
        best_size = (0, 0)
        best_scale = 0.0
        for scale in self._template_scales:
            template_width = max(8, int(round(template.shape[1] * scale)))
            template_height = max(8, int(round(template.shape[0] * scale)))
            if template_height >= search.shape[0] or template_width >= search.shape[1]:
                continue
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(template, (template_width, template_height), interpolation=interpolation)
            scores = cv2.matchTemplate(search, resized, cv2.TM_CCOEFF_NORMED)
            _, max_score, _, max_location = cv2.minMaxLoc(scores)
            if np.isfinite(max_score) and float(max_score) > best_score:
                best_score = float(max_score)
                best_location = (int(max_location[0] + left), int(max_location[1] + top))
                best_size = (template_width, template_height)
                best_scale = float(scale)

        score = max(0.0, best_score)
        return {
            "available": True,
            "matched": score >= self.candidate_threshold,
            "score": round(float(score), 4),
            "location": best_location,
            "scale": round(best_scale, 3),
            "template_size": best_size,
            "threshold": self.candidate_threshold,
            "search_roi": (left, top, right, bottom),
        }


def _client_like_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"))
    return _strip_captured_window_frame(array)


def _strip_captured_window_frame(array: np.ndarray) -> np.ndarray:
    gray = array.mean(axis=2)
    height, width = gray.shape
    max_search = max(1, int(height * 0.14))
    bright_rows = gray.mean(axis=1) > 180.0

    title_start: int | None = None
    title_end: int | None = None
    y = 0
    while y < max_search:
        if not bright_rows[y]:
            y += 1
            continue
        start = y
        while y < max_search and bright_rows[y]:
            y += 1
        end = y
        if end - start >= 10:
            title_start = start
            title_end = end
            break

    if title_start is None or title_end is None:
        return array

    title_band = gray[title_start:title_end]
    column_mean = title_band.mean(axis=0)
    columns = np.where(column_mean > 160.0)[0]
    if len(columns) <= 0:
        return array[title_end:, :]
    left = max(0, int(columns[0]))
    right = min(width, int(columns[-1]) + 1)
    if right - left < width * 0.5:
        return array[title_end:, :]
    return array[title_end:, left:right]


def _roi_bounds(width: int, height: int, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(width * x1)))
    top = max(0, min(height - 1, int(height * y1)))
    right = max(left + 1, min(width, int(width * x2)))
    bottom = max(top + 1, min(height, int(height * y2)))
    return left, top, right, bottom


@lru_cache(maxsize=1)
def _urban_fun_template_gray() -> np.ndarray:
    raw = base64.b64decode(_URBAN_FUN_TEMPLATE_PNG_BASE64)
    image = Image.open(BytesIO(raw)).convert("L")
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


@lru_cache(maxsize=1)
def _cv2_module() -> tuple[Any | None, str | None]:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    return cv2, None

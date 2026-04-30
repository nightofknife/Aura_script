from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .models import MapMatchResult


class BigMapLocator:
    def __init__(self) -> None:
        self._reference_path: tuple[Path, tuple[int, int] | None] | None = None
        self._reference_cache: dict[str, Any] | None = None

    def locate(
        self,
        screenshot: Image.Image,
        reference_gray_path: Path | str,
        *,
        reference_map_size: tuple[int, int] | None = None,
    ) -> MapMatchResult:
        try:
            import cv2
            import numpy as np
        except ModuleNotFoundError as exc:
            return MapMatchResult(False, 0.0, message=f"缺少图像匹配依赖：{exc.name}")

        reference_path = Path(reference_gray_path)
        if not reference_path.is_file():
            return MapMatchResult(False, 0.0, message="ZeroLuck 底图尚未缓存。")

        try:
            reference = self._load_reference(reference_path, cv2, reference_map_size=reference_map_size)
            crop, offset = _crop_screenshot_for_map(screenshot, cv2, np)
            crop_gray = _preprocess(crop, cv2)
            detector_name, detector, norm_type = _create_detector(cv2)
            keypoints_query, descriptors_query = detector.detectAndCompute(crop_gray, None)
            keypoints_ref = reference["keypoints"]
            descriptors_ref = reference["descriptors"]
            if descriptors_query is None or descriptors_ref is None:
                return MapMatchResult(False, 0.0, message="未检测到足够的大地图特征。")
            if len(keypoints_query) < 20 or len(keypoints_ref) < 20:
                return MapMatchResult(False, 0.0, message="大地图特征点过少。")

            matches = _match_descriptors(cv2, descriptors_query, descriptors_ref, norm_type)
            if len(matches) < 24:
                return MapMatchResult(
                    False,
                    0.0,
                    message="请打开异环大地图后刷新，或调整缩放/移动地图后重试。",
                    debug_info={"good_matches": len(matches), "detector": detector_name},
                )

            src_ref = np.float32([keypoints_ref[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
            dst_query = np.float32([keypoints_query[item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
            dst_query[:, 0, 0] += float(offset[0])
            dst_query[:, 0, 1] += float(offset[1])
            homography, inlier_mask = cv2.findHomography(src_ref, dst_query, cv2.RANSAC, 4.0)
            if homography is None or inlier_mask is None:
                return MapMatchResult(False, 0.0, message="大地图配准失败。")

            inliers = int(inlier_mask.ravel().sum())
            inlier_ratio = inliers / max(1, len(matches))
            reprojection_error = _mean_reprojection_error(cv2, np, src_ref, dst_query, homography, inlier_mask)
            confidence = _confidence(inliers, inlier_ratio, reprojection_error)
            scale_x = float(reference["source_scale_x"])
            scale_y = float(reference["source_scale_y"])
            full_transform = homography @ np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=float)
            transform_ok = _validate_transform(full_transform, screenshot.size)
            success = inliers >= 30 and inlier_ratio >= 0.18 and reprojection_error <= 10.0 and transform_ok
            message = (
                f"已匹配 {inliers} 个特征，置信度 {confidence:.0%}"
                if success
                else "请打开异环大地图后刷新，或调整缩放/移动地图后重试。"
            )
            return MapMatchResult(
                success=success,
                confidence=confidence if success else min(confidence, 0.49),
                transform=full_transform,
                visible_polygon=_project_reference_corners(np, full_transform, reference["full_size"]),
                screen_rect=(0, 0, int(screenshot.width), int(screenshot.height)),
                message=message,
                debug_info={
                    "detector": detector_name,
                    "good_matches": len(matches),
                    "inliers": inliers,
                    "inlier_ratio": inlier_ratio,
                    "reprojection_error": reprojection_error,
                    "reference_scale_x": scale_x,
                    "reference_scale_y": scale_y,
                    "transform_ok": transform_ok,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return MapMatchResult(False, 0.0, message=f"大地图定位异常：{exc}")

    def _load_reference(
        self,
        reference_path: Path,
        cv2: Any,
        *,
        reference_map_size: tuple[int, int] | None,
    ) -> dict[str, Any]:
        cache_key = (reference_path, reference_map_size)
        if self._reference_path == cache_key and self._reference_cache is not None:
            return self._reference_cache

        reference_gray = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if reference_gray is None:
            raise ValueError("无法读取 ZeroLuck 底图缓存。")
        image_height, image_width = reference_gray.shape[:2]
        map_width, map_height = reference_map_size or (image_width, image_height)
        resize_scale = min(1.0, 3200.0 / max(image_width, image_height))
        if resize_scale < 1.0:
            reference_gray = cv2.resize(reference_gray, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
        reference_gray = _preprocess(reference_gray, cv2)
        detector_name, detector, _norm_type = _create_detector(cv2)
        keypoints, descriptors = detector.detectAndCompute(reference_gray, None)
        resized_height, resized_width = reference_gray.shape[:2]
        self._reference_path = cache_key
        self._reference_cache = {
            "detector": detector_name,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "source_scale_x": resized_width / max(1, map_width),
            "source_scale_y": resized_height / max(1, map_height),
            "full_size": (map_width, map_height),
        }
        return self._reference_cache


def _crop_screenshot_for_map(screenshot: Image.Image, cv2: Any, np: Any) -> tuple[Any, tuple[int, int]]:
    image = screenshot.convert("RGB")
    array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = array.shape[:2]
    left = int(width * 0.10)
    top = int(height * 0.07)
    right = int(width * 0.97)
    bottom = int(height * 0.93)
    return array[top:bottom, left:right], (left, top)


def _preprocess(image: Any, cv2: Any) -> Any:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _create_detector(cv2: Any) -> tuple[str, Any, int]:
    if hasattr(cv2, "SIFT_create"):
        return "SIFT", cv2.SIFT_create(nfeatures=16000, contrastThreshold=0.01), cv2.NORM_L2
    if hasattr(cv2, "AKAZE_create"):
        return "AKAZE", cv2.AKAZE_create(), cv2.NORM_HAMMING
    return "ORB", cv2.ORB_create(nfeatures=12000), cv2.NORM_HAMMING


def _match_descriptors(cv2: Any, query_descriptors: Any, reference_descriptors: Any, norm_type: int) -> list[Any]:
    matcher = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(query_descriptors, reference_descriptors, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.76 * second.distance:
            good_matches.append(first)
    return good_matches


def _mean_reprojection_error(cv2: Any, np: Any, src: Any, dst: Any, homography: Any, inlier_mask: Any) -> float:
    projected = cv2.perspectiveTransform(src, homography)
    errors = np.linalg.norm(projected.reshape(-1, 2) - dst.reshape(-1, 2), axis=1)
    inliers = inlier_mask.ravel().astype(bool)
    if not inliers.any():
        return float("inf")
    return float(errors[inliers].mean())


def _confidence(inliers: int, inlier_ratio: float, reprojection_error: float) -> float:
    inlier_score = min(1.0, inliers / 180.0)
    ratio_score = min(1.0, inlier_ratio / 0.65)
    error_score = max(0.0, min(1.0, (12.0 - reprojection_error) / 12.0))
    return max(0.0, min(1.0, 0.45 * inlier_score + 0.35 * ratio_score + 0.20 * error_score))


def _validate_transform(transform: Any, screen_size: tuple[int, int]) -> bool:
    try:
        import numpy as np

        if not np.isfinite(transform).all():
            return False
        affine = transform[:2, :2]
        _u, singular_values, _vh = np.linalg.svd(affine)
        if min(singular_values) <= 0:
            return False
        anisotropy = max(singular_values) / min(singular_values)
        screen_max = max(screen_size)
        scale_ok = 0.02 <= max(singular_values) <= max(8.0, screen_max / 250.0)
        return anisotropy <= 2.2 and scale_ok
    except Exception:
        return False


def _project_reference_corners(np: Any, transform: Any, full_size: tuple[int, int]) -> list[tuple[float, float]]:
    width, height = full_size
    points = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=float).T
    projected = transform @ points
    projected /= projected[2:3, :]
    return [(float(projected[0, index]), float(projected[1, index])) for index in range(projected.shape[1])]

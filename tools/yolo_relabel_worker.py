from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.yolo_project_lib import (
    WORKER_RESULT_PREFIX,
    YoloProjectError,
    build_run_id,
    current_model_path,
    load_project_config,
    load_samples,
    write_samples,
    SampleRecord,
    utc_now,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate YOLO draft labels for unlabeled samples.")
    parser.add_argument("--project-root", required=True, help="Absolute or repo-relative project root.")
    parser.add_argument("--run-id", help="Optional explicit relabel run id.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = run_relabel(project_root=Path(args.project_root).expanduser().resolve(), run_id=args.run_id)
        print(WORKER_RESULT_PREFIX + json.dumps(payload, ensure_ascii=False))
    except Exception as exc:
        print(WORKER_RESULT_PREFIX + json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1) from exc


def run_relabel(*, project_root: Path, run_id: str | None) -> dict[str, object]:
    if not project_root.is_dir():
        raise YoloProjectError(f"Project directory not found: {project_root}")

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as exc:
        raise YoloProjectError("ultralytics and torch are required for relabeling.") from exc

    project = load_project_config(project_root)
    model_path = current_model_path(project_root)
    if not model_path.is_file():
        raise YoloProjectError(f"Active model was not found: {model_path}")

    relabel_cfg = project.get("relabel", {}) if isinstance(project.get("relabel"), dict) else {}
    conf = float(relabel_cfg.get("conf", 0.25))
    iou = float(relabel_cfg.get("iou", 0.45))
    max_det = int(relabel_cfg.get("max_det", 100))
    class_count = len(project.get("class_names") or [])
    resolved_run_id = run_id or build_run_id("relabel")

    model = YOLO(str(model_path))
    samples = load_samples(project_root)
    updated: list[SampleRecord] = []
    processed_count = 0
    draft_count = 0
    predictions_summary: list[dict[str, object]] = []
    draft_dir = project_root / "labels" / "draft"
    draft_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        if sample.status != "unlabeled":
            updated.append(sample)
            continue

        image_path = project_root / sample.image_relpath
        if not image_path.is_file():
            updated.append(sample)
            continue

        print(f"[relabel] sample={sample.sample_id} image={image_path.name}")
        predictions = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )
        label_lines = _build_yolo_label_lines(predictions, class_count=class_count)
        draft_path = draft_dir / f"{sample.sample_id}.txt"
        draft_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        processed_count += 1
        draft_count += 1
        predictions_summary.append(
            {
                "sample_id": sample.sample_id,
                "draft_label": draft_path.relative_to(project_root).as_posix(),
                "detection_count": len(label_lines),
            }
        )
        updated.append(
            SampleRecord(
                sample_id=sample.sample_id,
                image_relpath=sample.image_relpath,
                approved_label_relpath=sample.approved_label_relpath,
                draft_label_relpath=draft_path.relative_to(project_root).as_posix(),
                status="draft_generated",
                source="model",
                last_session_id=sample.last_session_id,
                last_model_run_id=resolved_run_id,
                updated_at=utc_now(),
            )
        )

    write_samples(project_root, updated)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "ok": True,
        "run_id": resolved_run_id,
        "processed_count": processed_count,
        "draft_count": draft_count,
        "predictions": predictions_summary,
    }


def _build_yolo_label_lines(predictions, *, class_count: int) -> list[str]:
    lines: list[str] = []
    for prediction in predictions or []:
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            continue
        xywhn_values = _to_native_sequence(getattr(boxes, "xywhn", None))
        cls_values = _to_native_sequence(getattr(boxes, "cls", None))
        for index, xywhn in enumerate(xywhn_values):
            class_id = int(cls_values[index]) if index < len(cls_values) else -1
            if class_id < 0 or (class_count and class_id >= class_count):
                continue
            x_center, y_center, width, height = [float(value) for value in xywhn]
            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
    return lines


def _to_native_sequence(value) -> list:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


if __name__ == "__main__":
    main()

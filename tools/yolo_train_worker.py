from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.yolo_project_lib import WORKER_RESULT_PREFIX, YoloProjectError, build_run_id, export_training_dataset, load_project_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO model for one project.")
    parser.add_argument("--project-root", required=True, help="Absolute or repo-relative project root.")
    parser.add_argument("--export-dir", help="Optional pre-exported dataset directory.")
    parser.add_argument("--run-id", help="Optional explicit train run id.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = run_training(
            project_root=Path(args.project_root).expanduser().resolve(),
            export_dir=Path(args.export_dir).expanduser().resolve() if args.export_dir else None,
            run_id=args.run_id,
        )
        print(WORKER_RESULT_PREFIX + json.dumps(payload, ensure_ascii=False))
    except Exception as exc:
        print(WORKER_RESULT_PREFIX + json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1) from exc


def run_training(*, project_root: Path, export_dir: Path | None, run_id: str | None) -> dict[str, object]:
    if not project_root.is_dir():
        raise YoloProjectError(f"Project directory not found: {project_root}")

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as exc:
        raise YoloProjectError("ultralytics and torch are required for training.") from exc

    project = load_project_config(project_root)
    dataset_info = export_training_dataset(project_root) if export_dir is None else {
        "export_id": Path(export_dir).name,
        "export_dir": str(export_dir),
        "dataset_yaml": str(export_dir / "dataset.yaml"),
    }
    dataset_dir = Path(dataset_info["export_dir"]).resolve()
    dataset_yaml = dataset_dir / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise YoloProjectError(f"Dataset yaml not found: {dataset_yaml}")

    resolved_run_id = run_id or build_run_id("train")
    base_model = str(project.get("base_model") or "yolo11n.pt")
    run_root = project_root / "runs" / "train"
    print(f"[train] dataset={dataset_yaml}")
    print(f"[train] base_model={base_model} run_id={resolved_run_id}")

    model = YOLO(base_model)
    results = model.train(
        data=str(dataset_yaml),
        imgsz=int(project.get("imgsz", 640)),
        epochs=int(project.get("epochs", 100)),
        batch=int(project.get("batch", 16)),
        device=str(project.get("device", "auto")),
        project=str(run_root),
        name=resolved_run_id,
        exist_ok=False,
        verbose=True,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        save_dir = run_root / resolved_run_id
    save_dir = Path(save_dir).resolve()
    best_model_path = save_dir / "weights" / "best.pt"
    if not best_model_path.is_file():
        raise YoloProjectError(f"Training completed but best.pt was not found: {best_model_path}")

    metrics = _extract_metrics(results)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "ok": True,
        "run_id": resolved_run_id,
        "run_dir": str(save_dir),
        "best_model_path": str(best_model_path),
        "dataset_export_id": dataset_info["export_id"],
        "metrics": metrics,
    }


def _extract_metrics(results) -> dict[str, object]:
    metrics: dict[str, object] = {}
    summary = getattr(results, "results_dict", None)
    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, (int, float, str)):
                metrics[str(key)] = value
    best_epoch = getattr(results, "best_epoch", None)
    if isinstance(best_epoch, int):
        metrics["best_epoch"] = best_epoch
    return metrics


if __name__ == "__main__":
    main()

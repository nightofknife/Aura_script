from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Yihuan 3-class combat-target detector.")
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=15)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    import torch
    from ultralytics import YOLO

    thread_limit = int(os.environ.get("AURA_TORCH_NUM_THREADS") or "0")
    if thread_limit > 0:
        torch.set_num_threads(thread_limit)
        torch.set_num_interop_threads(max(1, min(thread_limit, 2)))

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch_num_threads": torch.get_num_threads(),
                "dataset_yaml": str(args.dataset_yaml),
                "base_model": str(args.base_model),
                "run": str(args.project / args.name),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    model = YOLO(str(args.base_model.resolve()))
    results = model.train(
        data=str(args.dataset_yaml.resolve()),
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch),
        device=str(args.device),
        project=str(args.project.resolve()),
        name=str(args.name),
        exist_ok=False,
        patience=int(args.patience),
        workers=int(args.workers),
        cache=False,
        verbose=True,
    )
    print(
        json.dumps(
            {
                "save_dir": str(results.save_dir),
                "results": getattr(results, "results_dict", None),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plans.yihuan.src.services.combat_service import YihuanCombatService  # noqa: E402


CLASS_NAMES = ["enemy_hp_bar", "enemy_direction_marker", "reward_marker"]
HP_CLASS_ID = 0
DIRECTION_CLASS_ID = 1
REWARD_CLASS_ID = 2


@dataclass
class Box:
    class_id: int
    x: float
    y: float
    w: float
    h: float
    source: str
    confidence: float | None = None


@dataclass
class Sample:
    image_path: Path
    image_hash: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a 3-class Yihuan combat-target YOLO dataset.")
    parser.add_argument("--logs-root", type=Path, default=REPO_ROOT / "logs" / "yihuan_combat_debug")
    parser.add_argument("--hp-dataset", type=Path, required=True)
    parser.add_argument("--hp-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-conf", type=float, default=0.82)
    parser.add_argument("--hp-conf", type=float, default=0.50)
    parser.add_argument("--negative-limit", type=int, default=420)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=768)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_dataset(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / rel).mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    samples: dict[str, Sample] = {}
    combat = YihuanCombatService()

    hp_dataset_stats = import_hp_dataset(
        hp_dataset=args.hp_dataset.resolve(),
        samples=samples,
        combat=combat,
    )

    raw_records = collect_raw_records(args.logs_root.resolve())
    detector_stats = add_combat_detector_labels(
        records=raw_records,
        samples=samples,
        combat=combat,
        reward_conf=float(args.reward_conf),
    )

    hp_model = YOLO(str(args.hp_model.resolve()))
    hp_predict_stats = add_hp_predictions_for_raw_records(
        records=raw_records,
        samples=samples,
        model=hp_model,
        conf=float(args.hp_conf),
        imgsz=int(args.imgsz),
    )

    dedupe_all_boxes(samples)
    selected = select_samples(samples, negative_limit=int(args.negative_limit), rng=rng)
    train_samples, val_samples = split_samples(selected, val_ratio=float(args.val_ratio), rng=rng)
    export_samples(output_dir, train_samples, "train")
    export_samples(output_dir, val_samples, "val")
    dataset_yaml = write_dataset_yaml(output_dir)
    manifest = {
        "classes": CLASS_NAMES,
        "output_dir": str(output_dir),
        "dataset_yaml": str(dataset_yaml),
        "stats": {
            "hp_dataset": hp_dataset_stats,
            "raw_records": len(raw_records),
            "combat_detector": detector_stats,
            "hp_predictions": hp_predict_stats,
            "unique_samples_before_selection": len(samples),
            "selected_samples": len(selected),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "class_box_counts": count_boxes_by_class(selected),
            "negative_samples": sum(1 for item in selected if not item.boxes),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def import_hp_dataset(*, hp_dataset: Path, samples: dict[str, Sample], combat: YihuanCombatService) -> dict[str, Any]:
    stats = {"images": 0, "labels": 0, "boxes": 0, "direction_boxes_added": 0}
    for split in ("train", "val"):
        image_dir = hp_dataset / "images" / split
        label_dir = hp_dataset / "labels" / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(iter_images(image_dir)):
            image = read_rgb(image_path)
            if image is None:
                continue
            sample = upsert_sample(samples, image_path, image, source=f"hp_dataset:{split}")
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(float(parts[0]))
                    if class_id != 0:
                        continue
                    cx, cy, w, h = [float(item) for item in parts[1:]]
                    sample.boxes.append(
                        yolo_norm_to_box(
                            class_id=HP_CLASS_ID,
                            cx=cx,
                            cy=cy,
                            w=w,
                            h=h,
                            width=sample.width,
                            height=sample.height,
                            source=f"hp_dataset:{split}",
                        )
                    )
                    stats["boxes"] += 1
                stats["labels"] += 1
            # Add direction labels when the screenshot also contains the behind/side enemy marker.
            state = combat.analyze_frame(image)
            for marker in state.get("enemy_direction_markers") or []:
                box = expanded_box(
                    class_id=DIRECTION_CLASS_ID,
                    x=float(marker["x"]),
                    y=float(marker["y"]),
                    w=float(marker["width"]),
                    h=float(marker["height"]),
                    image_w=sample.width,
                    image_h=sample.height,
                    pad=6,
                    source="combat_hsv_direction_on_hp_dataset",
                )
                sample.boxes.append(box)
                stats["direction_boxes_added"] += 1
            stats["images"] += 1
    return stats


def collect_raw_records(logs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index_path in sorted(logs_root.glob("*/index.json")):
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for shot in payload.get("screenshots") or []:
            raw = shot.get("raw_image_path")
            if not raw:
                continue
            path = Path(raw)
            if path.is_file():
                records.append(
                    {
                        "path": path,
                        "phase": shot.get("phase"),
                        "label": shot.get("label"),
                        "index": str(index_path),
                    }
                )
    return records


def add_combat_detector_labels(
    *,
    records: list[dict[str, Any]],
    samples: dict[str, Sample],
    combat: YihuanCombatService,
    reward_conf: float,
) -> dict[str, int]:
    stats = {
        "records_read": 0,
        "direction_frames": 0,
        "direction_boxes": 0,
        "reward_frames": 0,
        "reward_boxes": 0,
    }
    for record in records:
        image_path = Path(record["path"])
        image = read_rgb(image_path)
        if image is None:
            continue
        sample = upsert_sample(samples, image_path, image, source="combat_raw")
        state = combat.analyze_frame(image)
        stats["records_read"] += 1
        direction_count = 0
        for marker in state.get("enemy_direction_markers") or []:
            sample.boxes.append(
                expanded_box(
                    class_id=DIRECTION_CLASS_ID,
                    x=float(marker["x"]),
                    y=float(marker["y"]),
                    w=float(marker["width"]),
                    h=float(marker["height"]),
                    image_w=sample.width,
                    image_h=sample.height,
                    pad=6,
                    source="combat_hsv_direction_raw",
                )
            )
            direction_count += 1
        if direction_count:
            stats["direction_frames"] += 1
            stats["direction_boxes"] += direction_count

        reward_score = float(state.get("reward_marker_confidence") or 0.0)
        reward_box = state.get("reward_marker_box") or []
        if (
            record.get("phase") == "post_combat_reward"
            and len(reward_box) == 4
            and reward_score >= reward_conf
        ):
            sample.boxes.append(
                expanded_box(
                    class_id=REWARD_CLASS_ID,
                    x=float(reward_box[0]),
                    y=float(reward_box[1]),
                    w=float(reward_box[2]),
                    h=float(reward_box[3]),
                    image_w=sample.width,
                    image_h=sample.height,
                    pad=2,
                    source=f"combat_template_reward_raw:{reward_score:.3f}",
                    confidence=reward_score,
                )
            )
            stats["reward_frames"] += 1
            stats["reward_boxes"] += 1
    return stats


def add_hp_predictions_for_raw_records(
    *,
    records: list[dict[str, Any]],
    samples: dict[str, Sample],
    model: Any,
    conf: float,
    imgsz: int,
) -> dict[str, int]:
    stats = {"records_seen": 0, "frames_with_hp": 0, "boxes": 0}
    paths = [Path(record["path"]) for record in records]
    for image_path in paths:
        image = read_rgb(image_path)
        if image is None:
            continue
        sample = upsert_sample(samples, image_path, image, source="combat_raw_hp_predict")
        result = model.predict(
            source=str(image_path),
            imgsz=int(imgsz),
            conf=float(conf),
            iou=0.45,
            max_det=20,
            verbose=False,
            device=0,
        )[0]
        added = 0
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),))
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((len(xyxy),))
            for coords, score, cls in zip(xyxy, confs, classes):
                if int(cls) != 0:
                    continue
                x1, y1, x2, y2 = [float(item) for item in coords]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width < 8 or height < 2:
                    continue
                sample.boxes.append(
                    Box(
                        class_id=HP_CLASS_ID,
                        x=x1,
                        y=y1,
                        w=width,
                        h=height,
                        source="hp_model_raw",
                        confidence=float(score),
                    )
                )
                added += 1
        stats["records_seen"] += 1
        if added:
            stats["frames_with_hp"] += 1
            stats["boxes"] += added
    return stats


def select_samples(samples: dict[str, Sample], *, negative_limit: int, rng: random.Random) -> list[Sample]:
    positives = [sample for sample in samples.values() if sample.boxes]
    negatives = [sample for sample in samples.values() if not sample.boxes]
    rng.shuffle(negatives)
    selected = positives + negatives[: max(0, int(negative_limit))]
    rng.shuffle(selected)
    return selected


def split_samples(samples: list[Sample], *, val_ratio: float, rng: random.Random) -> tuple[list[Sample], list[Sample]]:
    positives_by_class: dict[int, list[Sample]] = {0: [], 1: [], 2: []}
    negatives: list[Sample] = []
    assigned: set[str] = set()
    for sample in samples:
        if not sample.boxes:
            negatives.append(sample)
            continue
        classes = sorted({box.class_id for box in sample.boxes})
        primary = classes[0]
        positives_by_class[primary].append(sample)

    train: list[Sample] = []
    val: list[Sample] = []
    for group in list(positives_by_class.values()) + [negatives]:
        rng.shuffle(group)
        val_count = max(1, int(round(len(group) * val_ratio))) if len(group) >= 10 else max(0, int(round(len(group) * val_ratio)))
        for index, sample in enumerate(group):
            if sample.image_hash in assigned:
                continue
            assigned.add(sample.image_hash)
            (val if index < val_count else train).append(sample)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def export_samples(output_dir: Path, samples: list[Sample], split: str) -> None:
    for index, sample in enumerate(samples):
        suffix = sample.image_path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            suffix = ".png"
        stem = f"{index:06d}_{sample.image_hash[:12]}"
        out_image = output_dir / "images" / split / f"{stem}{suffix}"
        out_label = output_dir / "labels" / split / f"{stem}.txt"
        shutil.copy2(sample.image_path, out_image)
        lines = [box_to_yolo_line(box, sample.width, sample.height) for box in sample.boxes]
        out_label.write_text("".join(lines), encoding="utf-8")


def write_dataset_yaml(output_dir: Path) -> Path:
    payload = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dataset_yaml


def upsert_sample(samples: dict[str, Sample], image_path: Path, image_rgb: np.ndarray, *, source: str) -> Sample:
    digest = file_sha1(image_path)
    height, width = image_rgb.shape[:2]
    sample = samples.get(digest)
    if sample is None:
        sample = Sample(
            image_path=image_path,
            image_hash=digest,
            width=int(width),
            height=int(height),
        )
        samples[digest] = sample
    sample.sources.add(source)
    return sample


def dedupe_all_boxes(samples: dict[str, Sample]) -> None:
    for sample in samples.values():
        kept: list[Box] = []
        for box in sorted(sample.boxes, key=lambda item: (item.class_id, -(item.confidence or 0.0), item.source)):
            if any(box.class_id == other.class_id and box_iou(box, other) >= 0.72 for other in kept):
                continue
            kept.append(box)
        sample.boxes = kept


def count_boxes_by_class(samples: Iterable[Sample]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for sample in samples:
        for box in sample.boxes:
            counts[CLASS_NAMES[int(box.class_id)]] += 1
    return counts


def read_rgb(path: Path) -> np.ndarray | None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            yield path


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yolo_norm_to_box(
    *,
    class_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    width: int,
    height: int,
    source: str,
) -> Box:
    abs_w = float(w) * width
    abs_h = float(h) * height
    x = float(cx) * width - abs_w / 2.0
    y = float(cy) * height - abs_h / 2.0
    return Box(class_id=class_id, x=x, y=y, w=abs_w, h=abs_h, source=source)


def expanded_box(
    *,
    class_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    image_w: int,
    image_h: int,
    pad: float,
    source: str,
    confidence: float | None = None,
) -> Box:
    x1 = max(0.0, x - pad)
    y1 = max(0.0, y - pad)
    x2 = min(float(image_w), x + w + pad)
    y2 = min(float(image_h), y + h + pad)
    return Box(
        class_id=class_id,
        x=x1,
        y=y1,
        w=max(1.0, x2 - x1),
        h=max(1.0, y2 - y1),
        source=source,
        confidence=confidence,
    )


def box_to_yolo_line(box: Box, image_w: int, image_h: int) -> str:
    x1 = max(0.0, min(float(image_w), box.x))
    y1 = max(0.0, min(float(image_h), box.y))
    x2 = max(0.0, min(float(image_w), box.x + box.w))
    y2 = max(0.0, min(float(image_h), box.y + box.h))
    cx = ((x1 + x2) / 2.0) / float(image_w)
    cy = ((y1 + y2) / 2.0) / float(image_h)
    w = max(0.0, x2 - x1) / float(image_w)
    h = max(0.0, y2 - y1) / float(image_h)
    return f"{int(box.class_id)} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n"


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.w) * max(0.0, a.h)
    area_b = max(0.0, b.w) * max(0.0, b.h)
    return inter / max(area_a + area_b - inter, 1e-9)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.aura_core.services.yolo_contract import (  # noqa: E402
    EXPORT_OUTPUT_FORMAT,
    YoloModelMetadata,
    YoloPreprocessMetadata,
    infer_variant_from_name,
    metadata_path_for_model,
    normalize_family_token,
    write_model_metadata,
)


def _load_ultralytics_yolo_class():
    # Ultralytics export performs its own requirement auto-install unless disabled.
    # We keep the environment deterministic and rely on our optional requirements files instead.
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for export. Install requirements/optional-yolo-export.txt."
        ) from exc
    return YOLO


def _load_onnx_module():
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "onnx is required for export validation. Install requirements/optional-yolo-export.txt."
        ) from exc
    return onnx


def _load_onnxruntime_module():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for export validation. Install requirements/optional-vision-onnx-cpu.txt "
            "or requirements/optional-vision-onnx-cuda.txt."
        ) from exc
    return ort


def _extract_class_names(model: Any) -> list[str]:
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        normalized = [str(value).strip() for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    elif isinstance(names, list):
        normalized = [str(value).strip() for value in names]
    else:
        normalized = []
    normalized = [item for item in normalized if item]
    if not normalized:
        raise RuntimeError("Unable to derive class_names from the .pt model.")
    return normalized


def _detect_task(model: Any) -> str:
    task = getattr(model, "task", None)
    if isinstance(task, str) and task.strip():
        return task.strip().lower()
    overrides = getattr(model, "overrides", None)
    if isinstance(overrides, dict) and isinstance(overrides.get("task"), str):
        return str(overrides["task"]).strip().lower()
    inner_model = getattr(model, "model", None)
    inner_task = getattr(inner_model, "task", None)
    if isinstance(inner_task, str) and inner_task.strip():
        return inner_task.strip().lower()
    return ""


def _load_onnx_graph(path: Path) -> Any:
    onnx = _load_onnx_module()
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return model


def _resolve_output_layout(*, onnx_model: Any, class_count: int) -> str:
    outputs = list(getattr(onnx_model.graph, "output", []))
    if not outputs:
        raise RuntimeError("Exported ONNX graph has no outputs.")
    output = outputs[0]
    tensor_type = getattr(output, "type", None)
    shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
    dims = list(getattr(shape, "dim", []))
    dim_values: list[Optional[int]] = []
    for dim in dims:
        value = getattr(dim, "dim_value", 0)
        dim_values.append(int(value) if int(value) > 0 else None)
    if len(dim_values) != 3:
        raise RuntimeError(f"Unsupported ONNX output rank for detect export: {dim_values}")

    expected_channels = 4 + int(class_count)
    if dim_values[1] == expected_channels and dim_values[2] != expected_channels:
        return "bcn"
    if dim_values[2] == expected_channels and dim_values[1] != expected_channels:
        return "bnc"
    raise RuntimeError(
        f"Unable to determine output layout from ONNX output shape {dim_values}; expected channel size {expected_channels}."
    )


def _validate_onnxruntime_session(path: Path) -> str:
    ort = _load_onnxruntime_module()
    available = list(ort.get_available_providers())
    providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available[:1]
    if not providers:
        raise RuntimeError("No ONNX Runtime execution provider is available for export validation.")
    session = ort.InferenceSession(str(path), providers=providers)
    active = list(session.get_providers())
    return active[0] if active else providers[0]


def export_yolo_onnx(
    *,
    model_path: Path,
    family: str,
    imgsz: int,
    out_dir: Path,
    name: Optional[str] = None,
    variant: Optional[str] = None,
    opset: Optional[int] = None,
) -> dict[str, Any]:
    normalized_family = normalize_family_token(family)
    resolved_name = str(name or model_path.stem or "model").strip()
    if not resolved_name:
        raise RuntimeError("Unable to determine output model name.")

    YOLO = _load_ultralytics_yolo_class()
    model = YOLO(str(model_path))
    task = _detect_task(model)
    if task != "detect":
        raise RuntimeError(f"Only detect models are supported for export. Received task={task or None}.")

    class_names = _extract_class_names(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_kwargs: dict[str, Any] = {
        "format": "onnx",
        "imgsz": int(imgsz),
        "dynamic": False,
        "half": False,
        "int8": False,
        "end2end": False,
    }
    if opset is not None:
        export_kwargs["opset"] = int(opset)

    exported = model.export(**export_kwargs)
    exported_path = Path(str(exported)).resolve()
    if not exported_path.is_file():
        raise RuntimeError(f"Ultralytics export did not produce an ONNX file: {exported_path}")

    target_path = (out_dir / f"{resolved_name}.onnx").resolve()
    if exported_path != target_path:
        shutil.copy2(exported_path, target_path)

    onnx_model = _load_onnx_graph(target_path)
    output_layout = _resolve_output_layout(onnx_model=onnx_model, class_count=len(class_names))
    metadata = YoloModelMetadata(
        schema_version=1,
        task="detect",
        family=normalized_family,
        variant=(variant or infer_variant_from_name(model_path.stem)),
        input_size=(int(imgsz), int(imgsz)),
        input_format="rgb",
        input_layout="nchw",
        preprocess=YoloPreprocessMetadata(letterbox=True, pad_value=114, normalize="divide_255"),
        output_format=EXPORT_OUTPUT_FORMAT,
        output_layout=output_layout,
        class_names=class_names,
        default_conf=0.25,
        default_iou=0.45,
    )
    metadata_path = metadata_path_for_model(target_path)
    write_model_metadata(metadata_path, metadata)
    provider = _validate_onnxruntime_session(target_path)
    return {
        "ok": True,
        "model": str(target_path),
        "metadata": str(metadata_path),
        "family": normalized_family,
        "variant": metadata.variant,
        "imgsz": int(imgsz),
        "output_layout": output_layout,
        "provider": provider,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a YOLO .pt model into Aura runtime ONNX artifacts.")
    parser.add_argument("--model", required=True, type=Path, help="Path to the source .pt model.")
    parser.add_argument("--family", required=True, choices=["yolo8", "yolo11", "yolo26"], help="YOLO family.")
    parser.add_argument("--imgsz", required=True, type=int, help="Fixed square export size.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for exported artifacts.")
    parser.add_argument("--name", default=None, help="Optional output artifact stem.")
    parser.add_argument("--variant", default=None, help="Optional metadata-only variant value.")
    parser.add_argument("--opset", default=None, type=int, help="Optional ONNX opset to pass through.")
    return parser


def run_cli(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = export_yolo_onnx(
        model_path=args.model,
        family=args.family,
        imgsz=args.imgsz,
        out_dir=args.out_dir,
        name=args.name,
        variant=args.variant,
        opset=args.opset,
    )
    print(result)
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()

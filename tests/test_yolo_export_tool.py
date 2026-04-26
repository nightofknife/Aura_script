from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from packages.aura_core.services.yolo_contract import load_model_metadata
from tools import export_yolo_onnx as export_tool


class _FakeExportedModel:
    def __init__(self, model_path: str, *, task: str = "detect", names=None, export_target: Path | None = None):
        self.model_path = model_path
        self.task = task
        self.names = names or {0: "enemy", 1: "ally"}
        self.export_target = export_target
        self.export_calls = []

    def export(self, **kwargs):
        self.export_calls.append(kwargs)
        if self.export_target is None:
            raise RuntimeError("missing export target")
        self.export_target.write_bytes(b"fake-onnx")
        return str(self.export_target)


class _FakeYoloFactory:
    def __init__(self, model: _FakeExportedModel):
        self.model = model
        self.calls = []

    def __call__(self, path: str):
        self.calls.append(path)
        return self.model


class _FakeOnnxModule:
    def __init__(self, dims):
        self._dims = dims
        self.checker = SimpleNamespace(check_model=self._check_model)
        self.checked = []
        self.loaded = []

    def load(self, path: str):
        self.loaded.append(path)
        dim_objects = [SimpleNamespace(dim_value=value) for value in self._dims]
        return SimpleNamespace(
            graph=SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type=SimpleNamespace(
                            tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dim_objects))
                        )
                    )
                ]
            )
        )

    def _check_model(self, model):
        self.checked.append(model)


class _FakeOrtModule:
    def __init__(self, providers=None):
        self._providers = providers or ["CPUExecutionProvider"]
        self.sessions = []

    def get_available_providers(self):
        return list(self._providers)

    def InferenceSession(self, path, providers=None):
        session = SimpleNamespace(path=path, providers=list(providers or self._providers), get_providers=lambda: list(providers or self._providers))
        self.sessions.append(session)
        return session


class TestYoloExportTool(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_export_generates_model_and_metadata(self):
        source_model = self.root / "best.pt"
        source_model.write_bytes(b"fake-pt")
        exported_source = self.root / "ultralytics_tmp.onnx"
        fake_model = _FakeExportedModel(str(source_model), export_target=exported_source)
        fake_factory = _FakeYoloFactory(fake_model)
        fake_onnx = _FakeOnnxModule([1, 6, 8400])
        fake_ort = _FakeOrtModule()
        out_dir = self.root / "artifacts"

        with patch.object(export_tool, "_load_ultralytics_yolo_class", return_value=fake_factory), patch.object(export_tool, "_load_onnx_module", return_value=fake_onnx), patch.object(export_tool, "_load_onnxruntime_module", return_value=fake_ort):
            result = export_tool.export_yolo_onnx(
                model_path=source_model,
                family="yolo11",
                imgsz=640,
                out_dir=out_dir,
                name="enemy_detector",
                variant="n",
            )

        model_path = out_dir / "enemy_detector.onnx"
        metadata_path = out_dir / "enemy_detector.meta.json"
        self.assertTrue(model_path.is_file())
        self.assertTrue(metadata_path.is_file())
        metadata = load_model_metadata(metadata_path)
        self.assertEqual(metadata.family, "yolo11")
        self.assertEqual(metadata.variant, "n")
        self.assertEqual(metadata.input_size, (640, 640))
        self.assertEqual(metadata.output_layout, "bcn")
        self.assertEqual(metadata.class_names, ["enemy", "ally"])
        self.assertEqual(result["provider"], "CPUExecutionProvider")
        self.assertEqual(result["output_layout"], "bcn")
        self.assertEqual(fake_factory.calls, [str(source_model)])
        self.assertEqual(
            fake_model.export_calls[0],
            {
                "format": "onnx",
                "imgsz": 640,
                "dynamic": False,
                "half": False,
                "int8": False,
                "end2end": False,
            },
        )

    def test_export_rejects_non_detect_models(self):
        source_model = self.root / "best.pt"
        source_model.write_bytes(b"fake-pt")
        fake_factory = _FakeYoloFactory(_FakeExportedModel(str(source_model), task="segment"))

        with patch.object(export_tool, "_load_ultralytics_yolo_class", return_value=fake_factory):
            with self.assertRaisesRegex(RuntimeError, "Only detect models are supported"):
                export_tool.export_yolo_onnx(
                    model_path=source_model,
                    family="yolo11",
                    imgsz=640,
                    out_dir=self.root / "artifacts",
                )

    def test_export_rejects_unsupported_layout(self):
        source_model = self.root / "best.pt"
        source_model.write_bytes(b"fake-pt")
        exported_source = self.root / "ultralytics_tmp.onnx"
        fake_model = _FakeExportedModel(str(source_model), export_target=exported_source)
        fake_factory = _FakeYoloFactory(fake_model)
        fake_onnx = _FakeOnnxModule([1, 7, 8400])
        fake_ort = _FakeOrtModule()

        with patch.object(export_tool, "_load_ultralytics_yolo_class", return_value=fake_factory), patch.object(export_tool, "_load_onnx_module", return_value=fake_onnx), patch.object(export_tool, "_load_onnxruntime_module", return_value=fake_ort):
            with self.assertRaisesRegex(RuntimeError, "Unable to determine output layout"):
                export_tool.export_yolo_onnx(
                    model_path=source_model,
                    family="yolo11",
                    imgsz=640,
                    out_dir=self.root / "artifacts",
                )

    def test_export_rejects_unsupported_family(self):
        source_model = self.root / "best.pt"
        source_model.write_bytes(b"fake-pt")
        with self.assertRaisesRegex(ValueError, "Unsupported YOLO family token"):
            export_tool.export_yolo_onnx(
                model_path=source_model,
                family="yolo10",
                imgsz=640,
                out_dir=self.root / "artifacts",
            )


if __name__ == "__main__":
    unittest.main()

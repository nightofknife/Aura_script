# OCR ONNX Runtime 部署、导出与使用指南

本文说明 Aura 当前的 OCR 运行时方案。OCR 已从 `PaddleOCR + paddlepaddle-gpu` 运行时切换到 `ONNX Runtime`，并与 YOLO 共用同一套 CPU/CUDA provider 选择逻辑。

## 1. 总体流程

标准部署流程固定为：

1. 准备 PaddleOCR inference 模型目录。
2. 在导出机安装 OCR export 依赖。
3. 使用 `tools/export_paddleocr_onnx.py` 生成 ONNX 部署包。
4. 部署机只安装 ONNX Runtime CPU 或 CUDA 包。
5. 运行现有 `preload_ocr`、`find_text`、`recognize_all_text` 等动作。

运行时不再支持直接加载 `.pdmodel/.pdiparams`，也不再保留 PaddleOCR fallback。部署机不需要 `paddleocr`、`paddlex`、`paddlepaddle` 或 `paddlepaddle-gpu`。

## 2. 安装依赖

### CPU 部署

```powershell
.\scripts\setup_python_runtime.ps1 -VisionProvider cpu
```

等价的手动安装方式：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements\optional-vision-onnx-cpu.txt
```

### CUDA12 部署

```powershell
.\scripts\setup_python_runtime.ps1 -VisionProvider cuda
```

等价的手动安装方式：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements\optional-vision-onnx-cuda.txt
```

CPU 和 CUDA 部署包二选一：CPU 环境安装 `onnxruntime`，CUDA 环境安装 `onnxruntime-gpu`。不要同时安装这两个包，否则 Windows 下可能出现 `CUDAExecutionProvider` 不可见或回退到 CPU 的情况。

### OCR 导出机

导出机需要额外安装 Paddle/Paddle2ONNX 工具链：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements\optional-ocr-export.txt
```

当前 PP-OCRv5 导出已在 WSL/Linux 下验证通过，推荐使用 Linux/WSL 作为导出环境。Windows 原生 `paddle2onnx 2.1.0` wheel 在部分 Python 3.12 + Paddle 组合下可能出现 native DLL ABI 加载失败；旧版 `paddle2onnx 1.3.1` 又不能正确转换当前 PP-OCRv5 inference 模型。

如果导出机也要做 Paddle GPU 导出或验证，请按 Paddle 官方 CUDA wheel 源额外安装 GPU 版 Paddle，例如：

```powershell
.venv\Scripts\python.exe -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

这些依赖只属于导出机，不属于部署机。

## 3. 导出 OCR ONNX 模型

当前仓库内置的 PaddleOCR inference 模型位于：

```text
plans/aura_base/src/services/ocr_model
```

导出命令：

```powershell
.venv\Scripts\python.exe tools\export_paddleocr_onnx.py `
  --model-root plans\aura_base\src\services\ocr_model `
  --out-dir models\ocr `
  --name ppocrv5_server
```

在本机 WSL 中验证过的导出命令为：

```powershell
wsl.exe bash -lc 'cd /mnt/d/python_project/Aura_script && export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" && python3 tools/export_paddleocr_onnx.py --model-root plans/aura_base/src/services/ocr_model --out-dir models/ocr --name ppocrv5_server'
```

导出工具会优先使用每个 Paddle inference 子目录中的 `inference.json`，再回退到 `inference.pdmodel`。这是为了兼容 Paddle 3.x 对旧 `.pdmodel` 的弃用和 `load_combine` 参数读取限制。

可选指定 ONNX opset：

```powershell
.venv\Scripts\python.exe tools\export_paddleocr_onnx.py `
  --model-root plans\aura_base\src\services\ocr_model `
  --out-dir models\ocr `
  --name ppocrv5_server `
  --opset 13
```

导出后标准目录结构为：

```text
models/
`-- ocr/
    `-- ppocrv5_server/
        |-- ocr.meta.json
        |-- det.onnx
        |-- rec.onnx
        `-- textline_orientation.onnx
```

`doc_orientation.onnx` 是可选文件。当前运行时默认关闭文档方向分类，以保持与旧实现 `use_doc_orientation_classify=False` 的行为一致。

导出工具会做以下校验：

- 检查必需 Paddle inference 文件是否存在。
- 用 `paddle2onnx` 导出 ONNX。
- 用 `onnx.checker` 校验模型。
- 用 ONNX Runtime 构造 session。
- 写出 `ocr.meta.json`。

## 4. Runtime 配置

默认配置如下：

```yaml
ocr:
  default_model: ppocrv5_server
  models_root: models/ocr
  execution_provider: auto
  session:
    intra_op_num_threads: 0
    inter_op_num_threads: 0
    graph_optimization_level: all
```

字段说明：

- `ocr.default_model`: 默认 OCR 部署包目录名。
- `ocr.models_root`: OCR 部署包根目录。
- `ocr.execution_provider`: `auto`、`cpu` 或 `cuda`。
- `auto`: 优先 `CUDAExecutionProvider`，不可用时回退 CPU。
- `cpu`: 只允许 `CPUExecutionProvider`。
- `cuda`: 必须使用 `CUDAExecutionProvider`，不可用时直接失败。
- `ocr.session.*`: ONNX Runtime session 线程和图优化配置，与 YOLO 的 `yolo.session.*` 语义一致。

## 5. 任务 YAML 使用方式

OCR action 名称和参数保持兼容。已有任务 YAML 不需要因为 ONNX 迁移而改写。

预加载：

```yaml
steps:
  preload:
    action: preload_ocr
    params:
      warmup: true
```

查找文本：

```yaml
steps:
  find_start:
    action: find_text
    params:
      text_to_find: 开始
      region: [100, 100, 600, 300]
      match_mode: contains
```

读取区域内全部文本：

```yaml
steps:
  read_panel:
    action: get_text_in_region
    params:
      region: [100, 100, 600, 300]
      join_with: " "
```

`preload_ocr` 现在会在原有返回外追加：

```json
{
  "backend": "onnxruntime",
  "provider": "CUDAExecutionProvider",
  "model": "ppocrv5_server"
}
```

`OcrResult` 和 `MultiOcrResult` 的公共字段保持不变：

- `found`
- `text`
- `center_point`
- `rect`
- `confidence`
- `debug_info`

`debug_info` 中会附带 `backend`、`provider` 和原始 polygon，便于诊断。

## 6. 环境诊断

检查 ONNX Runtime、CUDA provider、YOLO 和 OCR：

```powershell
.venv\Scripts\python.exe tools\gpu_runtime_diagnostics.py --probe-ocr --json
```

同时探测 YOLO 模型：

```powershell
.venv\Scripts\python.exe tools\gpu_runtime_diagnostics.py `
  --probe-ocr `
  --onnx-model models\yolo\yolo11n.onnx `
  --json
```

关键字段：

- `onnxruntime.available_providers` 应包含 `CUDAExecutionProvider`，表示 GPU provider 可见。
- `ocr_service.backend` 应为 `onnxruntime`。
- `ocr_service.provider` 应为 `CUDAExecutionProvider` 或 `CPUExecutionProvider`。
- `ocr_service.model` 应为当前加载的 OCR bundle 名称。

诊断输出里仍保留 Paddle 信息，但它只用于导出机排查，不再是部署机运行成功的条件。

## 7. 常见问题

### 缺少 `ocr.meta.json`

现象：`preload_ocr` 报 `Missing OCR metadata sidecar`。

处理：先运行 `tools/export_paddleocr_onnx.py`，并确认目录为：

```text
models/ocr/ppocrv5_server/ocr.meta.json
```

### 传入 Paddle 模型目录无法运行

运行时只接受导出后的 ONNX 部署包。`.pdmodel/.pdiparams` 只属于导出阶段。

### CUDA provider 不可用

先检查是否同时安装了 `onnxruntime` 和 `onnxruntime-gpu`：

```powershell
.venv\Scripts\python.exe -m pip show onnxruntime onnxruntime-gpu
```

CUDA 部署环境应只保留 `onnxruntime-gpu`。

### OCR 和 YOLO provider 不一致

两者共用 `packages/aura_core/services/onnx_runtime_backend.py` 的 provider 逻辑，但分别读取自己的配置前缀：

- OCR 读取 `ocr.execution_provider`
- YOLO 读取 `yolo.execution_provider`

如果希望两者都强制走 GPU，请同时配置：

```yaml
ocr:
  execution_provider: cuda
yolo:
  execution_provider: cuda
```

## 8. 最小落地流程

CUDA12 部署机推荐流程：

```powershell
.\scripts\setup_python_runtime.ps1 -VisionProvider cuda
.venv\Scripts\python.exe tools\export_paddleocr_onnx.py `
  --model-root plans\aura_base\src\services\ocr_model `
  --out-dir models\ocr `
  --name ppocrv5_server
.venv\Scripts\python.exe tools\gpu_runtime_diagnostics.py --probe-ocr --json
```

部署交付物只需要：

```text
models/ocr/ppocrv5_server/ocr.meta.json
models/ocr/ppocrv5_server/det.onnx
models/ocr/ppocrv5_server/rec.onnx
models/ocr/ppocrv5_server/textline_orientation.onnx
```

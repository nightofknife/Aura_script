# Python 运行环境与入口

Aura Game Framework 当前默认使用 Python 3.12 虚拟环境。

## 1. 初始化运行环境

在项目根目录执行：

```powershell
.\scripts\setup_python_runtime.ps1
```

关键行为：

- 自动解析本机 Python `3.12.x`
- 创建或复用 `.venv`
- 安装 `requirements/runtime.txt`
- 强制从 Paddle 官方 `cu129` 源安装 `paddlepaddle-gpu==3.3.1`
- 如果存在 `requirements/runtime.lock` 则优先使用 lock
- 运行 `pip check`

## 2. 启动前校验

推荐在迁移、切换依赖或移动目录后执行：

```powershell
.\scripts\build_preflight.ps1
```

当前会检查：

- `.venv` 必须是 Python 3.12
- `include-system-site-packages = false`
- `PYTHONNOUSERSITE=1` 生效
- 如果存在 `requirements/runtime.lock`，已安装包必须与 lock 一致
- `pip check` 通过
- `cli.py --help` 可运行
- `packages.aura_game.EmbeddedGameRunner` 可以加载并识别 `aura_benchmark`

## 3. CLI 入口

### 查看游戏模块

```powershell
.venv\Scripts\python.exe cli.py games --all
```

### 查看任务

```powershell
.venv\Scripts\python.exe cli.py tasks aura_benchmark
```

### 运行任务

```powershell
.venv\Scripts\python.exe cli.py run aura_benchmark tasks:single_sleep.yaml --inputs "{\"duration_ms\": 50, \"scenario\": \"demo\"}"
```

在 PowerShell 下，推荐把输入写入 JSON 文件后配合 `--inputs-file` 使用，避免命令行引号转义问题。

### TUI

```powershell
.venv\Scripts\python.exe cli.py tui
```

说明：

- TUI 入口使用 `tui_manual` profile
- 适合人工执行 entry task 或调试调度项
- 不依赖 HTTP 服务

## 4. Python SDK

### EmbeddedRunner

```python
from packages.aura_game import EmbeddedGameRunner

runner = EmbeddedGameRunner()
runner.start()
print(runner.list_games())
runner.close()
```

### SubprocessRunner

```python
from packages.aura_game import SubprocessGameRunner

if __name__ == "__main__":
    with SubprocessGameRunner() as runner:
        print(runner.list_games())
```

说明：

- `EmbeddedGameRunner`
  适合脚本、测试、内部工具直接调用。
- `SubprocessGameRunner`
  适合宿主 GUI 或桌面程序把执行逻辑隔离到独立子进程。

## 5. 依赖文件

- `requirements/runtime.txt`
  运行时依赖。
- `requirements/dev.txt`
  开发与测试依赖。
- `requirements/optional-yolo-cpu.txt`
  ONNX Runtime CPU 推理依赖。
- `requirements/optional-yolo-cuda.txt`
  ONNX Runtime CUDA 12 推理依赖。
- `requirements/optional-yolo-export.txt`
  YOLO `.pt -> .onnx + .meta.json` 导出依赖。

## 6. YOLO 部署流

运行时不再直接加载 `.pt`，标准部署流程是：

1. 训练得到 `.pt` 模型。
2. 使用导出工具生成标准产物：

```powershell
.venv\Scripts\python.exe tools\export_yolo_onnx.py --model path\to\best.pt --family yolo11 --imgsz 640 --out-dir path\to\artifacts
```

3. 交付并加载：
   - `model.onnx`
   - `model.meta.json`

CPU 环境安装 `requirements/optional-yolo-cpu.txt`，CUDA 环境安装 `requirements/optional-yolo-cuda.txt`。

完整说明见：

- [YOLO ONNX Runtime 部署、导出与使用指南](../project-reference/yolo-onnx-runtime.md)

## 7. GPU 栈

当前仓库默认统一到 CUDA 12 路线：

- OCR 使用 Paddle 官方 `cu129` wheel
- YOLO 运行时使用 PyPI 默认 `onnxruntime-gpu`（CUDA 12）

如果需要 GPU 能力，建议避免在同一个环境里混用 CUDA 11 专用 ORT 包和旧版 Paddle `cu118` wheel。

可以使用以下命令快速检查当前环境里的 GPU 栈、YOLO 运行时和 OCR 运行时：

```powershell
.venv\Scripts\python.exe tools\gpu_runtime_diagnostics.py --probe-ocr --onnx-model .runtime\smoke_yolo\smoke_yolo11n.onnx
```

## CUDA12 Notes

- For YOLO, install exactly one ONNX Runtime package:
  `requirements/optional-yolo-cpu.txt` or `requirements/optional-yolo-cuda.txt`
- `requirements/optional-yolo-export.txt` is only for export tooling and must be combined with one runtime package above
- Do not install `onnxruntime` and `onnxruntime-gpu` into the same environment, or YOLO may fall back to CPU-only providers
- If GPU probing looks wrong, rerun:

```powershell
.venv\Scripts\python.exe tools\gpu_runtime_diagnostics.py --probe-ocr --onnx-model .runtime\smoke_yolo\smoke_yolo11n.onnx --json
```

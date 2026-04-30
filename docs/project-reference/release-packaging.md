# CPU / GPU+CPU ONNX Release Packaging

本文档记录 Aura Windows x64 运行包的本地打包和 GitHub Actions 自动发布流程。CPU 包保持 CLI-only；GPU+CPU 主包额外包含 `AuraYihuan.exe`，用户解压后可以双击启动异环 GUI。

## 产物

正式 Release 上传这些文件：

```text
aura-release-cpu-onnx.zip
aura-release-gpu-onnx.zip
aura-release-gpu-onnx-nvidia-overlay.zip
SHA256SUMS.txt
```

用户视角分为三个下载项：

```text
CPU-only: 下载 aura-release-cpu-onnx.zip，保持 CLI-only，用 run.ps1 启动
GPU+CPU: 下载 aura-release-gpu-onnx.zip，解压后双击 AuraYihuan.exe 启动 GUI
NVIDIA overlay: 需要离线 CUDA/cuDNN runtime 时，额外下载 aura-release-gpu-onnx-nvidia-overlay.zip
```

CPU 包使用 `onnxruntime`，release 内默认配置为 `ocr.execution_provider: cpu`。GPU+CPU 包使用 `onnxruntime-gpu`，release 内默认配置为 `ocr.execution_provider: auto`，优先 CUDA，CUDA 不可用时回落 CPU。GPU 主包不默认内置完整 NVIDIA CUDA/cuDNN DLL；overlay 包解压后会提供 `runtime/_internal/nvidia`。

NVIDIA overlay 的 zip 根目录与 GPU 主包保持同名顶层目录。最简单的用户流程是：把 `aura-release-gpu-onnx.zip` 和 `aura-release-gpu-onnx-nvidia-overlay.zip` 放在同一个文件夹，两个都解压到当前文件夹；如果系统提示合并或覆盖，选择允许。最终仍然只进入 `aura-release-gpu-onnx` 文件夹并双击 `AuraYihuan.exe`。

## 本地构建

CPU 和 GPU 必须使用两个干净 venv。不要在同一个 venv 中同时安装 `onnxruntime` 和 `onnxruntime-gpu`。

CPU-only：

```powershell
py -3.12 -m venv .venv-release-cpu-onnx
.\.venv-release-cpu-onnx\Scripts\python.exe -m pip install -U pip
.\.venv-release-cpu-onnx\Scripts\python.exe -m pip install `
  -r requirements\runtime.txt `
  -r requirements\optional-vision-onnx-cpu.txt `
  pyinstaller==6.14.2

powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\build_release.ps1 `
  -VenvPython .venv-release-cpu-onnx\Scripts\python.exe `
  -ReleaseName aura-release-cpu-onnx `
  -RuntimeRoot .runtime-cpu `
  -OnnxRuntimeProfile cpu `
  -CreateZip
```

GPU+CPU 主包：

```powershell
py -3.12 -m venv .venv-release-gpu-onnx
.\.venv-release-gpu-onnx\Scripts\python.exe -m pip install -U pip
.\.venv-release-gpu-onnx\Scripts\python.exe -m pip install `
  -r requirements\gui.txt `
  -r requirements\optional-vision-onnx-cuda.txt `
  pyinstaller==6.14.2

powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\build_release.ps1 `
  -VenvPython .venv-release-gpu-onnx\Scripts\python.exe `
  -ReleaseName aura-release-gpu-onnx `
  -RuntimeRoot .runtime-gpu `
  -OnnxRuntimeProfile gpu `
  -IncludeGui `
  -CreateZip
```

NVIDIA runtime overlay：

```powershell
.\.venv-release-gpu-onnx\Scripts\python.exe -m pip install `
  -r requirements\optional-nvidia-runtime-cu12.txt

powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\build_release.ps1 `
  -VenvPython .venv-release-gpu-onnx\Scripts\python.exe `
  -ReleaseName aura-release-gpu-onnx `
  -RuntimeRoot .runtime-gpu-overlay `
  -OnnxRuntimeProfile gpu `
  -SkipBuild `
  -SkipAssemble `
  -CreateNvidiaOverlay
```

每次生成 zip 或 overlay 后，脚本会刷新对应 `RuntimeRoot\release\SHA256SUMS.txt`。GitHub Actions 会在下载 CPU/GPU/overlay artifacts 后重新生成一份合并的 `SHA256SUMS.txt` 并上传到正式 Release。

不要把下面内容提交到 Git：

```text
.runtime/
.runtime-*/
.venv-release-*/
models/
*.onnx
*.meta.json
release zip 产物
```

## OCR 模型资产

仓库 `.gitignore` 会忽略 `models/` 和 ONNX 模型文件。GitHub Actions 打包前需要从 GitHub Release 下载模型资产。

建议创建一个模型专用 Release：

```text
tag: model-ppocrv5-server-v1
asset: models-ocr-ppocrv5_server-v1.zip
asset: models-ocr-ppocrv5_server-v1.sha256
```

模型 zip 推荐根目录结构：

```text
ppocrv5_server/
  ocr.meta.json
  det.onnx
  rec.onnx
  doc_orientation.onnx
  textline_orientation.onnx
```

本地生成模型资产：

```powershell
New-Item -ItemType Directory -Force -Path .runtime\model-assets | Out-Null
Compress-Archive `
  -Path models\ocr\ppocrv5_server `
  -DestinationPath .runtime\model-assets\models-ocr-ppocrv5_server-v1.zip `
  -Force
(Get-FileHash .runtime\model-assets\models-ocr-ppocrv5_server-v1.zip -Algorithm SHA256).Hash.ToLowerInvariant() |
  Set-Content .runtime\model-assets\models-ocr-ppocrv5_server-v1.sha256 -Encoding ascii
```

## GitHub 设置

在 `Settings -> Actions -> General` 中设置：

```text
Actions permissions: Allow GitHub-created actions and reusable workflows
Workflow permissions: Read and write permissions
Allow GitHub Actions to create and approve pull requests: 不需要开启
Artifact and log retention: 建议 3-7 天
```

在 `Settings -> Secrets and variables -> Actions -> Variables` 中设置：

```text
AURA_PYTHON_VERSION=3.12
AURA_OCR_MODEL_RELEASE_TAG=model-ppocrv5-server-v1
AURA_OCR_MODEL_ASSET=models-ocr-ppocrv5_server-v1.zip
AURA_OCR_MODEL_SHA256_ASSET=models-ocr-ppocrv5_server-v1.sha256
```

workflow 使用默认 `GITHUB_TOKEN`，不需要 Personal Access Token。若仓库限制 Actions 来源，需要允许这些官方 actions：

```text
actions/checkout@v4
actions/setup-python@v5
actions/upload-artifact@v4
actions/download-artifact@v4
```

如果需要真实 CUDA 验证，在 `Settings -> Actions -> Runners` 添加一台 Windows GPU self-hosted runner，并设置标签：

```text
self-hosted
windows
gpu
```

默认 workflow 不跑 self-hosted GPU 烟测；只有手动触发并设置 `run_gpu_smoke=true` 且 `create_overlay=true` 时才运行。

## GitHub Actions 使用

手动触发 `.github/workflows/release.yml`：

```text
release_tag: v0.1.0
publish_release: true
create_overlay: true
run_gpu_smoke: false
```

推送 `v*` tag 时也会触发构建并发布 Release assets。

workflow 分为五类 job：

```text
build-cpu: 构建 aura-release-cpu-onnx.zip
build-gpu: 构建带 AuraYihuan.exe 的 aura-release-gpu-onnx.zip
build-nvidia-overlay: 构建 aura-release-gpu-onnx-nvidia-overlay.zip
publish-assets: 生成 SHA256SUMS.txt，并在需要时上传到 GitHub Release
gpu-smoke: 可选 self-hosted CUDA 强制烟测
```

## 验收

CPU 包必须满足：

```text
存在 models/ocr/ppocrv5_server/ocr.meta.json
存在 models/ocr/ppocrv5_server/det.onnx
存在 models/ocr/ppocrv5_server/rec.onnx
不存在 runtime/_internal/onnxruntime/capi/onnxruntime_providers_cuda.dll
不存在 runtime/_internal/nvidia
不存在 runtime/_internal/paddle
不存在 runtime/_internal/paddleocr
不存在 runtime/_internal/paddlex
不存在 plans/aura_base/src/services/ocr_model
config.yaml 中 ocr.execution_provider=cpu
BUILD-INFO.txt 中 release_profile=cpu
```

GPU+CPU 主包必须满足：

```text
存在 AuraYihuan.exe
存在 runtime/AuraYihuanRuntime.exe
存在 runtime/_internal/PySide6
存在 models/ocr/ppocrv5_server/ocr.meta.json
存在 models/ocr/ppocrv5_server/det.onnx
存在 models/ocr/ppocrv5_server/rec.onnx
存在 runtime/_internal/onnxruntime/capi/onnxruntime_providers_cuda.dll
不存在 runtime/_internal/nvidia，除非显式使用 -IncludeNvidia
不存在 runtime/_internal/paddle
不存在 runtime/_internal/paddleocr
不存在 runtime/_internal/paddlex
不存在 plans/aura_base/src/services/ocr_model
config.yaml 中 ocr.execution_provider=auto
BUILD-INFO.txt 中 release_profile=gpu
BUILD-INFO.txt 中 gui=true
```

Overlay 必须满足：

```text
zip 顶层目录必须是 aura-release-gpu-onnx/
存在 runtime/_internal/nvidia/cuda_runtime/bin/cudart64_12.dll
存在 runtime/_internal/nvidia/cublas/bin/cublas64_12.dll
存在 runtime/_internal/nvidia/cublas/bin/cublasLt64_12.dll
存在 runtime/_internal/nvidia/cudnn/bin/cudnn64_9.dll
存在 runtime/_internal/nvidia/cufft/bin/cufft64_11.dll
```

CPU 和 GPU 主包都要跑 CLI smoke：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File <release>\run.ps1 games --all
powershell -NoProfile -ExecutionPolicy Bypass -File <release>\run.ps1 tasks yihuan
powershell -NoProfile -ExecutionPolicy Bypass -File <release>\run.ps1 run yihuan tasks:checks:plan_loaded.yaml --timeout-sec 120
```

GPU 主包还要跑 GUI runtime self-check：

```powershell
$env:AURA_BASE_PATH = "<release>"
$env:PYTHONNOUSERSITE = "1"
& <release>\runtime\AuraYihuanRuntime.exe --self-check
```

体积目标：

```text
CPU zip < 700 MiB
GPU main zip < 900 MiB
NVIDIA overlay zip < 1.9 GiB
单个 GitHub Release asset < 2 GiB
```

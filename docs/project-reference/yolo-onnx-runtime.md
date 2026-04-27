# YOLO ONNX Runtime 部署、导出与使用指南

本文档说明当前仓库中 `aura_base` / `aura_core` 的 YOLO 运行方案，覆盖以下内容：

- 需要安装什么
- 如何初始化和验证运行环境
- 如何把训练好的 `.pt` 模型导出为可部署的 `.onnx + .meta.json`
- 模型文件应该如何放置
- `meta.json` 的结构和含义
- 在任务 YAML 中如何调用 `yolo_*` 动作
- 常见错误与排查方式

本文档基于当前已落地的 CUDA12 方案编写，目标是让训练机、导出机和部署机都能按同一套约定工作。

## 1. 方案总览

当前仓库的 YOLO 方案已经从运行时 `ultralytics` / PyTorch 推理切换为 `ONNX Runtime`。

标准流程如下：

1. 训练阶段得到 `.pt` 模型
2. 使用仓库内导出工具 `tools/export_yolo_onnx.py`
3. 产出标准部署物：
   - `model.onnx`
   - `model.meta.json`
4. 把这两个文件放到运行时约定的位置
5. 运行时由 `packages/aura_core/services/yolo_service.py` 使用 `onnxruntime` 或 `onnxruntime-gpu` 推理

这套方案的核心目标是：

- 运行时不再依赖 `ultralytics`
- 部署机不需要 PyTorch
- 同时支持 CPU 和 CUDA12
- 保留 `aura_base` 原有 `yolo_*` 动作接口

## 2. 当前支持范围

### 2.1 支持的模型家族

当前运行时只支持以下 YOLO 家族：

- `yolo8`
- `yolo11`
- `yolo26`

支持的别名：

- `yolo8` / `yolov8` / `v8`
- `yolo11` / `yolov11` / `v11`
- `yolo26` / `yolov26` / `v26`

### 2.2 当前限制

当前版本只支持：

- `detect` 任务
- `FP32`
- 固定输入尺寸
- 标准导出格式：`end2end=False`
- 由本仓库导出工具生成的标准产物

当前版本不支持：

- 运行时直接加载 `.pt`
- `segment` / `pose` / `classify`
- 动态输入尺寸
- `FP16` / `INT8`
- 外部第三方 ONNX 模型直接接入
- 缺少 `.meta.json` 的裸 `.onnx`

如果把 `.pt` 路径直接传给运行时，会明确报错，提示先执行导出工具。

## 3. 安装说明

### 3.1 基础运行时

在仓库根目录执行：

```powershell
.\scripts\setup_python_runtime.ps1
```

这个脚本会完成以下事情：

- 自动查找 Python `3.12.x`
- 创建或复用 `.venv`
- 安装基础运行依赖 `requirements/runtime.txt`
- 从 Paddle 官方 `cu129` 源安装 `paddlepaddle-gpu==3.3.1`
- 拉取 `aura_base` 需要的 MuMu 运行资源
- 执行 `pip check`

注意：

- 这个基础脚本已经覆盖 OCR 的 CUDA12 运行时
- 这个脚本不会自动安装 YOLO 的可选运行依赖

### 3.2 YOLO 运行时依赖

#### 只跑 CPU

```powershell
.venv\Scripts\python.exe -m pip install -r requirements\optional-yolo-cpu.txt
```

#### 跑 CUDA12

```powershell
.venv\Scripts\python.exe -m pip install -r requirements\optional-yolo-cuda.txt
```

`requirements/optional-yolo-cuda.txt` 当前包含：

- `numpy<2.4`
- `onnxruntime-gpu`

这层约束是必要的，因为 OCR 依赖链里的 `paddlex` 当前要求 `numpy<2.4`。

### 3.3 导出工具依赖

如果这台机器除了部署还要负责把 `.pt` 导出为 ONNX，还需要额外安装：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements\optional-yolo-export.txt
```

它当前包含：

- `ultralytics`
- `onnx`
- `onnxslim>=0.1.71`

导出工具不会再允许 Ultralytics 自动偷偷安装额外依赖，因此建议始终通过仓库里的 requirements 文件安装，不要临时手装别的 ORT 包。

### 3.4 推荐安装组合

#### 部署机，CPU 推理

```powershell
.\scripts\setup_python_runtime.ps1
.venv\Scripts\python.exe -m pip install -r requirements\optional-yolo-cpu.txt
```

#### 部署机，CUDA12 推理

```powershell
.\scripts\setup_python_runtime.ps1
.venv\Scripts\python.exe -m pip install -r requirements\optional-yolo-cuda.txt
```

#### 导出机，同时也要做 CUDA12 运行验证

```powershell
.\scripts\setup_python_runtime.ps1
.venv\Scripts\python.exe -m pip install -r requirements\optional-yolo-cuda.txt -r requirements\optional-yolo-export.txt
```

### 3.5 重要安装约束

#### 不要同时安装 `onnxruntime` 和 `onnxruntime-gpu`

二者同时存在时，Windows 环境里很容易出现：

- `onnxruntime-gpu` 已安装
- 但 `CUDAExecutionProvider` 不可见
- 最终 YOLO 退回 CPU

当前仓库已经在诊断脚本和 `YoloService` 中加了这类冲突检测，但最好的做法仍然是：

- CPU 环境只装 `onnxruntime`
- CUDA 环境只装 `onnxruntime-gpu`

#### 运行时不需要 `ultralytics`

运行时只需要：

- `onnxruntime` 或 `onnxruntime-gpu`
- `.onnx + .meta.json`

部署机不需要 `ultralytics`、`torch`、`torchvision`。

#### 导出环境里出现 `torch==...+cpu` 是正常现象

导出工具依赖 `ultralytics`，而 `ultralytics` 会带上 PyTorch。当前环境中看到 `torch` 是正常的，但这不代表运行时推理会用 PyTorch。真正运行 YOLO 推理的是 `onnxruntime`。

## 4. 环境验证与常用命令

### 4.1 预检

基础环境装完后建议执行：

```powershell
.\scripts\build_preflight.ps1
```

当前会检查：

- `.venv` 是否是 Python 3.12
- 虚拟环境是否隔离
- `requirements/runtime.lock` 与当前基础环境是否一致
- MuMu 运行资源是否存在
- `pip check` 是否通过
- CLI 和基础计划是否能正常启动

### 4.2 GPU 诊断

推荐使用：

```powershell
.venv\Scripts\python.exe tools\gpu_runtime_diagnostics.py --probe-ocr --onnx-model .runtime\smoke_yolo\smoke_yolo11n.onnx --json
```

这个脚本会输出：

- Python 版本和路径
- `nvidia-smi`
- Paddle 状态
- ONNX Runtime provider 列表
- YoloService 实际推理 provider
- OcrService 实际预加载设备

典型正确结果应该类似：

- OCR: `preload_device = gpu`
- Paddle: `compiled_with_cuda = true`
- ONNX Runtime: `available_providers` 中包含 `CUDAExecutionProvider`
- YOLO probe: `provider = CUDAExecutionProvider`

### 4.3 CLI 基础命令

```powershell
.venv\Scripts\python.exe cli.py games --all
.venv\Scripts\python.exe cli.py tasks aura_benchmark
.venv\Scripts\python.exe cli.py tui
```

如果只是验证框架入口，这几条命令就够了。YOLO 本身主要通过 `aura_base` 的动作在任务里使用，而不是单独提供 CLI 子命令。

## 5. 模型文件组织方式

### 5.1 标准部署物

每个模型必须有两个文件，且同名同 stem：

```text
xxx.onnx
xxx.meta.json
```

例如：

```text
boss_detector.onnx
boss_detector.meta.json
```

缺少任意一个都不能被运行时接受。

### 5.2 默认模型目录

运行时的默认模型根目录来自配置键：

- `yolo.models_root`

默认值为：

```text
models/yolo
```

所以默认推荐目录结构是：

```text
models/
`-- yolo/
    |-- yolo11n.onnx
    |-- yolo11n.meta.json
    |-- boss_detector.onnx
    `-- boss_detector.meta.json
```

### 5.3 命名模型解析规则

如果你在动作或服务里写：

```yaml
model_name: yolo11
```

运行时会按以下逻辑解析：

- 家族：`yolo11`
- 变体：来自 `yolo.default_variant`
- 默认变体未配置时为 `n`

因此：

- `yolo11` 默认解析为 `yolo11n.onnx`
- `yolo8` 默认解析为 `yolov8n.onnx`
- `yolo26` 默认解析为 `yolo26n.onnx`

### 5.4 直接按文件名或路径加载

也可以直接写：

```yaml
model_name: boss_detector.onnx
```

或者：

```yaml
model_name: D:\models\boss_detector.onnx
```

要求：

- 后缀必须是 `.onnx`
- 同目录下必须有 `boss_detector.meta.json`

## 6. 如何从 `.pt` 导出为标准 ONNX

### 6.1 命令格式

```powershell
.venv\Scripts\python.exe tools\export_yolo_onnx.py `
  --model path\to\best.pt `
  --family yolo11 `
  --imgsz 640 `
  --out-dir path\to\artifacts `
  --name boss_detector `
  --variant n
```

### 6.2 参数说明

#### 必填参数

- `--model`
  训练好的 `.pt` 模型路径
- `--family`
  只能是 `yolo8` / `yolo11` / `yolo26`
- `--imgsz`
  固定导出尺寸，整数，例如 `640`
- `--out-dir`
  导出目录

#### 可选参数

- `--name`
  输出文件名 stem，默认使用输入 `.pt` 的 stem
- `--variant`
  仅写入 metadata，不影响运行时逻辑
- `--opset`
  可选传给 Ultralytics 的 ONNX opset

### 6.3 导出工具内部做了什么

导出脚本会固定使用以下导出约束：

- `format="onnx"`
- `dynamic=False`
- `half=False`
- `int8=False`
- `end2end=False`

并自动完成：

- 校验模型任务必须是 `detect`
- 读取 `.pt` 中的类名
- 用 `onnx.checker` 校验导出的 ONNX
- 通过 ONNX 输出张量形状判断 `output_layout`
- 写出 `meta.json`
- 构造一次 ONNX Runtime session 作为导出后校验

### 6.4 导出结果

例如：

```powershell
.venv\Scripts\python.exe tools\export_yolo_onnx.py --model yolo11n.pt --family yolo11 --imgsz 640 --out-dir .runtime\exports --name smoke_yolo11n
```

将生成：

```text
.runtime/exports/smoke_yolo11n.onnx
.runtime/exports/smoke_yolo11n.meta.json
```

### 6.5 关于导出结果里的 `provider`

导出工具最终打印的返回结果里包含 `provider` 字段，例如：

```python
{
  "provider": "CPUExecutionProvider"
}
```

这只是“导出后校验会话”实际使用的 provider，不等于部署时只能跑 CPU。部署时真正使用什么 provider，由运行时机器安装的是 `onnxruntime` 还是 `onnxruntime-gpu` 决定。

## 7. `meta.json` 生成规则与编写要求

### 7.1 正常情况下不要手写

推荐做法是：

- 始终用 `tools/export_yolo_onnx.py` 自动生成 `meta.json`
- 不要手工编辑

因为以下字段非常容易写错：

- `output_layout`
- `class_names`
- `input_size`
- `default_conf`
- `default_iou`

### 7.2 只有调试时才建议手工检查

如果你需要确认文件内容，可以参照下面这份结构。

### 7.3 `meta.json` 示例

```json
{
  "schema_version": 1,
  "task": "detect",
  "family": "yolo11",
  "variant": "n",
  "input_size": [640, 640],
  "input_format": "rgb",
  "input_layout": "nchw",
  "preprocess": {
    "letterbox": true,
    "pad_value": 114,
    "normalize": "divide_255"
  },
  "output_format": "ultralytics_detect_raw_v1",
  "output_layout": "bcn",
  "class_names": ["enemy", "ally", "boss"],
  "default_conf": 0.25,
  "default_iou": 0.45
}
```

### 7.4 字段说明

- `schema_version`
  当前必须是 `1`
- `task`
  当前必须是 `"detect"`
- `family`
  必须是 `"yolo8"` / `"yolo11"` / `"yolo26"`
- `variant`
  通常为 `"n"` / `"s"` / `"m"` / `"l"` / `"x"`，也可为 `null`
- `input_size`
  `[width, height]`
- `input_format`
  当前必须是 `"rgb"`
- `input_layout`
  当前必须是 `"nchw"`
- `preprocess.letterbox`
  当前必须是 `true`
- `preprocess.pad_value`
  当前默认 `114`
- `preprocess.normalize`
  当前必须是 `"divide_255"`
- `output_format`
  当前必须是 `"ultralytics_detect_raw_v1"`
- `output_layout`
  当前只能是 `"bcn"` 或 `"bnc"`
- `class_names`
  类名数组，顺序必须与模型输出类别通道顺序一致
- `default_conf`
  默认置信度阈值，范围 `0.0 ~ 1.0`
- `default_iou`
  默认 NMS IoU 阈值，范围 `0.0 ~ 1.0`

### 7.5 `output_layout` 的含义

- `bcn`
  输出张量逻辑为 `B x C x N`
- `bnc`
  输出张量逻辑为 `B x N x C`

其中：

- `B` 是 batch
- `C` 是通道数，等于 `4 + class_count`
- `N` 是候选框数量

运行时不会猜测布局，而是完全依赖 `meta.json`。

## 8. 运行时配置如何编写

运行时读取的主要键如下：

- `yolo.default_model`
- `yolo.default_variant`
- `yolo.models_root`
- `yolo.execution_provider`
- `yolo.session.intra_op_num_threads`
- `yolo.session.inter_op_num_threads`
- `yolo.session.graph_optimization_level`

示例配置可以写成层级 YAML 形式：

```yaml
yolo:
  default_model: yolo11
  default_variant: n
  models_root: models/yolo
  execution_provider: cuda
  session:
    intra_op_num_threads: 0
    inter_op_num_threads: 0
    graph_optimization_level: all
```

含义如下：

- `default_model`
  未显式传 `model_name` 时使用的模型，默认是 `yolo11`
- `default_variant`
  当 `model_name` 是家族别名时使用的默认变体，默认是 `n`
- `models_root`
  命名模型的根目录，默认是 `models/yolo`
- `execution_provider`
  只能是 `auto` / `cpu` / `cuda`
- `intra_op_num_threads`
  ONNX Runtime 的 intra-op 线程数，`0` 表示默认
- `inter_op_num_threads`
  ONNX Runtime 的 inter-op 线程数，`0` 表示默认
- `graph_optimization_level`
  只能是 `disabled` / `basic` / `extended` / `all`

关于 `execution_provider`：

- `auto`
  优先尝试 `CUDAExecutionProvider`，不行再退回 CPU
- `cpu`
  只允许 CPU
- `cuda`
  必须用 CUDA，不允许自动退回 CPU

## 9. 任务 YAML 如何编写

### 9.1 最小预加载示例

```yaml
meta:
  title: YOLO 预加载示例

steps:
  preload_yolo:
    action: yolo_preload_model
    params:
      model_name: yolo11
```

### 9.2 屏幕检测示例

```yaml
meta:
  title: YOLO 屏幕检测示例

steps:
  detect_enemy:
    action: yolo_detect_on_screen
    params:
      model_name: yolo11
      roi: [100, 100, 1200, 700]
      options:
        conf: 0.35
        iou: 0.45
        max_det: 20
        agnostic_nms: false
```

参数说明：

- `roi`
  格式固定为 `[x, y, w, h]`
- `options.conf`
  置信度阈值
- `options.iou`
  NMS IoU 阈值
- `options.max_det`
  最大保留框数
- `options.agnostic_nms`
  是否做类别无关 NMS

### 9.3 按标签查找并点击

```yaml
meta:
  title: YOLO 点击目标示例

steps:
  click_target:
    action: yolo_find_and_click_target
    params:
      model_name: boss_detector.onnx
      target_labels: ["boss"]
      roi: [200, 120, 1000, 600]
      min_confidence: 0.50
      sort_mode: highest_confidence
      click_offset: [0, 0]
      button: left
      clicks: 1
      interval: 0.1
      post_delay_sec: 0.2
      options:
        conf: 0.30
        iou: 0.45
        max_det: 10
        agnostic_nms: false
```

### 9.4 等待目标出现

```yaml
meta:
  title: YOLO 等待目标出现

steps:
  wait_enemy:
    action: yolo_wait_for_target
    params:
      model_name: yolo11
      target_labels: ["enemy"]
      timeout_sec: 15.0
      poll_interval_sec: 0.3
      roi: [50, 50, 1000, 700]
      min_confidence: 0.45
      sort_mode: nearest_to_point
      anchor_point: [640, 360]
```

### 9.5 常见动作列表

当前 `aura_base` 中常用的 YOLO 动作包括：

- `yolo_preload_model`
- `yolo_set_active_model`
- `yolo_unload_model`
- `yolo_list_loaded_models`
- `yolo_get_active_model`
- `yolo_get_class_names`
- `yolo_resolve_class_ids`
- `yolo_detect_on_screen`
- `yolo_detect_image`
- `yolo_count_targets`
- `yolo_find_target`
- `yolo_wait_for_target`
- `yolo_wait_for_target_disappear`
- `yolo_find_and_click_target`
- `yolo_click_all_targets`
- `yolo_find_target_and_press_key`

### 9.6 `target_labels` 和 `options.classes` 的区别

- `target_labels`
  动作层做字符串标签过滤，例如 `["boss", "enemy"]`
- `options.classes`
  运行时按 `class_id` 过滤，例如 `[0, 2]`

如果你已经知道模型类别索引，`options.classes` 更靠近底层过滤；如果是任务 YAML 直接写业务逻辑，通常 `target_labels` 更直观。

### 9.7 兼容参数说明

以下参数当前只是兼容老接口，不再真正生效：

- `options.device`
- `options.half`

当前行为是：

- 记录 warning
- 不报错
- 忽略这些参数

`options.imgsz` 如果与 metadata 中的 `input_size` 不一致，也会被忽略并记录 warning。

## 10. 运行结果数据结构

### 10.1 顶层结果

`detect` / `detect_image` / `detect_on_screen` 的结果会包含：

- `ok`
- `model`
- `family`
- `image_size`
- `backend`
- `provider`
- `detections`

其中：

- `backend` 当前固定为 `"onnxruntime"`
- `provider` 一般是 `"CPUExecutionProvider"` 或 `"CUDAExecutionProvider"`

### 10.2 单个检测框

每个 detection 目前包含：

- `class_id`
- `label`
- `score`
- `bbox_xyxy`
- `bbox_xywh`
- `bbox_global`
- `image_index`

说明：

- `bbox_xyxy`
  `[x1, y1, x2, y2]`
- `bbox_xywh`
  `[x, y, w, h]`
- `bbox_global`
  在 `detect_on_screen` 中表示屏幕全局坐标
- `bbox_global`
  在 `detect_image` 中会兼容性地写成图像局部坐标版的 `xywh`

### 10.3 结果示例

```json
{
  "ok": true,
  "model": "yolo11n",
  "family": "yolo11",
  "image_size": [1280, 720],
  "backend": "onnxruntime",
  "provider": "CUDAExecutionProvider",
  "detections": [
    {
      "class_id": 0,
      "label": "enemy",
      "score": 0.91,
      "bbox_xyxy": [100.5, 120.0, 260.5, 300.0],
      "bbox_xywh": [100.5, 120.0, 160.0, 180.0],
      "bbox_global": [220, 260, 160, 180],
      "image_index": 0
    }
  ]
}
```

## 11. 常见问题与排查

### 11.1 传入 `.pt` 报错

现象：

- 运行时提示只支持 `.onnx`

原因：

- 当前运行时明确不再支持 `.pt`

处理方式：

- 先执行 `tools/export_yolo_onnx.py`
- 运行时只加载 `.onnx + .meta.json`

### 11.2 缺少 `.meta.json`

现象：

- preload 时提示 `Missing YOLO metadata sidecar`

原因：

- `.onnx` 同 stem 的 `meta.json` 不存在

处理方式：

- 确保 `foo.onnx` 和 `foo.meta.json` 同目录同 stem

### 11.3 `CUDAExecutionProvider` 不可用

先执行：

```powershell
.venv\Scripts\python.exe tools\gpu_runtime_diagnostics.py --onnx-model path\to\your_model.onnx --json
```

重点检查：

- `onnxruntime.distributions`
- `onnxruntime.available_providers`
- `yolo_service.provider`

常见原因：

- 装成了 `optional-yolo-cpu.txt`
- 同时安装了 `onnxruntime` 和 `onnxruntime-gpu`
- CUDA 运行库可见性有问题

### 11.4 看到 `torch` 是 CPU 版，会不会影响部署

不会。

原因：

- `torch` 只服务于导出阶段
- 真正运行 YOLO 推理的是 `onnxruntime`

只要诊断结果里：

- `available_providers` 包含 `CUDAExecutionProvider`
- `yolo_service.provider = CUDAExecutionProvider`

就说明运行时已经在走 GPU。

### 11.5 导出时返回 `provider = CPUExecutionProvider`

这不代表部署只能跑 CPU。

原因：

- 导出工具最后的 session 校验优先选 CPU provider 做兼容校验

部署是否用 GPU，要看部署机：

- 装的是 `onnxruntime-gpu`
- `YoloService` 实际拿到的 provider 是什么

### 11.6 `pip check` 报 `numpy` 冲突

当前仓库已经在以下文件中固定了兼容约束：

- `requirements/optional-yolo-cpu.txt`
- `requirements/optional-yolo-cuda.txt`
- `scripts/setup_python_runtime.ps1`

如果你是手工安装依赖，请务必沿用仓库命令，不要自己单独升级 `numpy`。

## 12. 一份推荐的最小落地流程

### 12.1 部署机

```powershell
.\scripts\setup_python_runtime.ps1
.venv\Scripts\python.exe -m pip install -r requirements\optional-yolo-cuda.txt
.\scripts\build_preflight.ps1
```

把模型放到：

```text
models/yolo/yolo11n.onnx
models/yolo/yolo11n.meta.json
```

然后执行诊断：

```powershell
.venv\Scripts\python.exe tools\gpu_runtime_diagnostics.py --probe-ocr --onnx-model models\yolo\yolo11n.onnx --json
```

### 12.2 导出机

```powershell
.\scripts\setup_python_runtime.ps1
.venv\Scripts\python.exe -m pip install -r requirements\optional-yolo-cuda.txt -r requirements\optional-yolo-export.txt
.venv\Scripts\python.exe tools\export_yolo_onnx.py --model path\to\best.pt --family yolo11 --imgsz 640 --out-dir path\to\artifacts --name boss_detector
```

交付给部署机的只有：

- `boss_detector.onnx`
- `boss_detector.meta.json`

## 13. 结论

当前仓库中的标准 YOLO 使用方式可以概括为一句话：

> 训练阶段使用 `.pt`，部署阶段只认 `.onnx + .meta.json`，推理由 ONNX Runtime 执行，CPU 和 CUDA12 只通过安装不同的 ORT 包来切换。

如果你后续继续维护模型，建议始终遵守这条边界：

- 训练和导出逻辑留在导出机
- 部署机只保留运行时必须依赖
- 所有部署模型都用本仓库导出工具生成

## Shared Vision ONNX Runtime note

YOLO and OCR now use the same ONNX Runtime provider/session helper in `packages/aura_core/services/onnx_runtime_backend.py`.

For new deployments prefer the shared dependency files:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements\optional-vision-onnx-cpu.txt
.venv\Scripts\python.exe -m pip install -r requirements\optional-vision-onnx-cuda.txt
```

The legacy YOLO optional files remain as compatibility entrypoints and forward to the shared vision files:

```powershell
requirements\optional-yolo-cpu.txt
requirements\optional-yolo-cuda.txt
```

The rule is unchanged: install exactly one ONNX Runtime package per environment. CPU deployments use `onnxruntime`; CUDA12 deployments use `onnxruntime-gpu`.

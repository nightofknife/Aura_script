# Aura Game Framework

从 `Aura` 主仓库拆分出来的独立项目，定位为一个专攻游戏脚本的本地自动化框架。

它当前聚焦：

- `Windows` 主机环境
- `CLI + YAML` 作为主入口
- `Python` 本地 SDK 作为一等集成面
- `aura_base` 共享运行时能力层
- `aura_benchmark` 作为回归与 smoke 基线

它不再把 `HTTP / FastAPI / Web GUI` 视为框架核心边界。

## 目录概览

- `packages/aura_core`
  任务编排、调度、状态、观测、DSL 执行内核。
- `packages/aura_game`
  面向宿主脚本和桌面程序的本地 SDK，提供嵌入式和子进程两种调用模式。
- `plans/aura_base`
  共享设备能力层，当前以 MuMu/ADB/OCR/视觉动作为主，并开始收敛到统一 runtime adapter 语义。
- `plans/aura_benchmark`
  基准与回归任务，优先保证无具体设备依赖。
- `cli.py`
  本地 CLI，聚焦游戏模块、任务执行和运行查询。

当前仓库仍保留 `plans/`、`plan_name`、`task_ref` 等兼容术语，以降低迁移成本；对外可以直接把 `plan` 理解为“游戏模块”。

## 快速开始

1. 准备 Python 3.12 虚拟环境：

```powershell
.\scripts\setup_python_runtime.ps1
```

2. 运行预检：

```powershell
.\scripts\build_preflight.ps1
```

3. 查看游戏模块和任务：

```powershell
.\scripts\run_cli.ps1 games --all
.\scripts\run_cli.ps1 tasks aura_benchmark
```

4. 跑一个基准任务：

```powershell
.\scripts\run_cli.ps1 run aura_benchmark tasks:single_sleep.yaml --inputs "{\"duration_ms\": 100, \"scenario\": \"demo\"}"
```

如果你在 PowerShell 里不想处理 JSON 引号转义，优先用 `--inputs-file`。

## Yihuan GUI

仓库现在提供了面向 `plans/yihuan` 的 Windows 桌面 GUI。

安装可选 GUI 依赖：

```powershell
python -m pip install -r requirements\gui.txt
```

启动 GUI：

```powershell
python cli.py gui yihuan
```

如果需要打包成独立 GUI 程序，可使用 `packaging/pyinstaller/yihuan_gui.spec`。

## 模型与本地资产

GitHub 仓库只保留代码、文档、配置和测试夹具，不提交本地模型权重或运行产物。

默认忽略的资产包括：

- `models/`
- `*.onnx`、`*.pt`、`*.pth`、`*.safetensors`
- PaddleOCR 本地模型目录 `plans/aura_base/src/services/ocr_model/`
- MuMu 运行资产目录 `plans/aura_base/assets/mumu/`
- `.runtime/`、`logs/`、`tmp/`

MuMu 运行资产不随仓库提交。运行 `.\scripts\setup_python_runtime.ps1` 时会默认下载；也可以手动执行：

```powershell
.venv\Scripts\python.exe scripts\fetch_mumu_runtime_assets.py
```

YOLO 部署模型请由使用者自行准备，并按文档放到本机的 `models/yolo/`：

```text
models/yolo/yolo11n.onnx
models/yolo/yolo11n.meta.json
```

如果需要从 `.pt` 导出 ONNX，请先安装可选导出依赖：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements\optional-yolo-export.txt
```

OCR 不再要求仓库内置 PaddleOCR 模型目录。若本地存在 `plans/aura_base/src/services/ocr_model/`，运行时会优先使用；若不存在，则交给 PaddleOCR 按它自己的缓存/下载机制解析模型。

## Python SDK

```python
from packages.aura_game import EmbeddedGameRunner

runner = EmbeddedGameRunner()
print(runner.list_games())
print(runner.list_tasks("aura_benchmark"))
print(
    runner.run_task(
        game_name="aura_benchmark",
        task_ref="tasks:single_sleep.yaml",
        inputs={"duration_ms": 10, "scenario": "sdk_demo"},
        wait=True,
    )
)
runner.close()
```

如果你想把执行和宿主程序隔离开：

```python
from packages.aura_game import SubprocessGameRunner

if __name__ == "__main__":
    with SubprocessGameRunner() as runner:
        print(runner.list_games())
```

## 文档入口

- [文档总览](docs/README.md)
- [环境与入口](docs/getting-started/01-python-runtime.md)
- [任务 YAML 指南](docs/getting-started/03-task-yaml-guide.md)
- [aura_base 运行时说明](docs/project-reference/aura-base-runtime.md)
- [基准任务说明](docs/project-reference/benchmark-plan.md)

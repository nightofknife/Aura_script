# Aura Game Framework 重构计划

## 1. 目标

将当前拆分出的 `Aura Game Framework` 继续收敛为一个专攻游戏脚本的自动化框架。

目标形态：

- 面向 `Windows` 主机环境
- 同时支持 `PC 原生窗口游戏` 与 `Android 模拟器游戏`
- 以 `CLI + YAML` 为主入口
- 以 `Python 本地 SDK` 作为一等集成面
- 不再把 `HTTP / FastAPI / Web GUI` 作为核心框架边界

本计划文件用于迁移到新目录后的后续改造，不要求一次性完成，按阶段推进即可。

---

## 2. 当前基线状态

当前迁移目录已经完成的工作：

- 保留了 `packages/aura_core`
- 保留了 `plans/aura_base`
- 保留了 `plans/aura_benchmark`
- 移除了 `backend/`
- 移除了 `aura_gui/`
- 移除了 `plans/resonance/`
- 新增了本地 SDK：
  - `packages/aura_game/runner.py`
  - `EmbeddedGameRunner`
  - `SubprocessGameRunner`
- 新增了精简 CLI：
  - `cli.py`
- 新增了项目说明和精简文档：
  - `README.md`
  - `docs/`

当前项目已经可以作为“拆分后的独立骨架”运行，但内部仍然保留不少旧框架心智和兼容术语，例如：

- `plans/`
- `plan_name`
- `task_ref`
- `manifest.yaml`
- `target.provider / target.mumu`

这些内容暂时保留是为了降低迁移成本，不代表最终形态。

---

## 3. 重构总原则

### 3.1 保留的核心能力

以下内容应继续保留并作为新框架核心：

- 调度器与任务编排内核
- YAML 任务 DSL
- `depends_on / when / loop / retry / returns` 等运行语义
- action / service 注入机制
- 运行记录、观测、错误语义
- `aura_base` 中的设备控制、截图、OCR、视觉能力
- 本地 SDK 与 CLI

### 3.2 要收敛或弱化的内容

以下内容不再作为核心能力继续扩张：

- 以 HTTP 为中心的服务平台形态
- 面向通用自动化平台的 package 心智
- 与游戏脚本无关的通用平台产品面
- 以具体业务游戏为中心的内置模块

### 3.3 改造策略

采用 `温和破坏升级`：

- 不推翻执行引擎
- 不重写任务 DSL
- 允许做术语收敛、目录收敛、配置收敛
- 对旧写法保留兼容层
- 新项目文档、CLI、SDK 优先使用新术语

---

## 4. 目标架构

建议收敛成四层结构：

### 4.1 Framework Core

目录基础：

- `packages/aura_core`

职责：

- runtime / scheduler
- packaging / task loading
- execution engine / node executor
- observability
- config / context / registries

要求：

- 保持执行语义稳定
- 不再依赖 `backend/`
- 不再默认面向 API server

### 4.2 Local SDK

目录基础：

- `packages/aura_game`

职责：

- 暴露统一本地调用接口
- 提供嵌入式执行模式
- 提供子进程隔离执行模式
- 为未来宿主 GUI 提供非 HTTP 集成面

目标公开接口：

- `EmbeddedGameRunner`
- `SubprocessGameRunner`

建议稳定方法：

- `start()`
- `stop()`
- `status()`
- `list_games()`
- `list_tasks(game_name)`
- `run_task(game_name, task_ref, inputs, wait=False, timeout_sec=...)`
- `get_run(cid)`
- `list_runs(...)`
- `poll_events(...)`
- `doctor()`

### 4.3 Shared Runtime Layer

目录基础：

- `plans/aura_base`

职责：

- 共享设备能力层
- 输入控制
- 截图
- OCR
- 图像识别
- 高层 `app` 交互 facade

当前状态：

- 真实能力边界仍是 `MuMu-only V1`

长期目标：

- 从“MuMu 实现”提升为“平台适配层 + 公共交互层”

### 4.4 Sample / Benchmark Layer

目录基础：

- `plans/aura_benchmark`

职责：

- 框架回归基线
- 编排语义验证
- CLI / SDK smoke test 基线

要求：

- 始终保持不依赖具体设备
- 始终作为第一优先级回归测试模块

---

## 5. 阶段化改造计划

## 阶段 A：稳定拆分项目基线

目标：

- 让当前迁移出的目录成为稳定的“新仓库起点”

需要做的事：

- 清理仍带有旧平台心智的文档描述
- 检查并修正 `README.md`、`docs/` 中所有对 `backend / FastAPI / GUI` 的核心化描述
- 保持 CLI、SDK、benchmark 的 smoke 验证始终可用
- 固定 manifest 生成结果，避免每次加载都产生不必要 diff

完成标准：

- 新目录可单独运行
- `cli.py games`
- `cli.py tasks aura_benchmark`
- `cli.py run aura_benchmark ...`
- `EmbeddedGameRunner`
- `SubprocessGameRunner`
  全部可用

---

## 阶段 B：收敛术语与用户心智

目标：

- 对外从“自动化框架 / plan 平台”收敛为“游戏脚本框架”

需要做的事：

- 新文档中优先使用：
  - `game`
  - `game module`
  - `game task`
- 保留内部兼容字段：
  - `plan_name`
  - `plans/`
- 在 SDK 和 CLI 输出中新增：
  - `game_name`
- 后续逐步减少用户直接接触 `plan` 的频率

建议策略：

- 内部实现先不大改
- 先改用户面：文档、CLI 输出、SDK 返回结构

完成标准：

- 新用户只看新项目文档时，不会认为这是一个 HTTP 自动化平台
- 新用户能把 `plan` 自然理解为“游戏模块”

---

## 阶段 C：平台适配层抽象化

目标：

- 将 `aura_base` 从 MuMu-only 能力组织，提升为可扩展的游戏平台适配层

建议新增抽象：

- `RuntimeAdapter`
- `CaptureBackend`
- `InputBackend`

建议公共契约：

- `capture()`
- `focus()`
- `get_client_rect()`
- `get_pixel_color_at()`
- `click()`
- `move_to()`
- `drag_to()`
- `scroll()`
- `press_key()`
- `key_down()`
- `key_up()`
- `type_text()`
- `release_all()`
- `self_check()`

第一阶段平台实现建议：

- `MumuAndroidAdapter`
- `WindowsDesktopAdapter`

重构要求：

- `screen / controller / app` 只依赖统一 adapter，不再直接依赖 MuMu session 细节
- OCR / 视觉 / 流程动作层不能写 MuMu 专属逻辑分支
- MuMu 逻辑收敛到 adapter 或 provider 目录

完成标准：

- `aura_base` 的公共服务层不再写死 MuMu-only 术语
- 增加一个最小的 Windows 桌面适配实现骨架，即使初期能力不完整也可

---

## 阶段 D：配置收敛

目标：

- 把当前配置从“旧 runtime + 旧 target 命名”收敛为面向游戏框架的结构

建议目标结构：

```yaml
runtime:
  family: windows_desktop | android_emulator
  provider: windows | mumu
  startup_timeout_sec: 10
  providers:
    mumu: ...
    windows: ...
```

兼容要求：

- 继续兼容旧配置：
  - `target.provider`
  - `target.mumu`
- 新配置优先，旧配置自动映射
- 在检测到旧配置时给出 warning，但不直接中断

完成标准：

- 新项目示例文档只写新配置结构
- 老项目脚本复制进来后还能通过兼容层运行

---

## 阶段 E：目录与声明文件收敛

目标：

- 让目录和声明文件更贴近“游戏模块”语义，而不是通用 package 平台

长期建议：

- `plans/` 逐步演化到 `games/`
- `manifest.yaml` 逐步演化到更瘦身的 `game.yaml`

但当前不建议立刻硬切：

- 先保留 `plans/` 兼容加载
- 先保留 `manifest.yaml`
- 先从用户文档与脚手架层引导新结构

建议的推进方式：

1. 保留旧 loader
2. 增加对新目录和新声明文件的支持
3. CLI 与文档优先输出新结构
4. 最后再考虑是否弃用旧结构

完成标准：

- 新项目可以同时加载旧结构与新结构
- 新脚手架默认生成新结构

---

## 阶段 F：测试与回归体系固化

目标：

- 让这个项目在脱离原仓库后仍然可以独立迭代

至少要保留的测试层：

- `aura_benchmark` 运行验证
- `aura_base` MuMu runtime 测试
- graph / scheduler regression 测试
- SDK smoke test

建议新增测试：

- `EmbeddedGameRunner` 生命周期测试
- `SubprocessGameRunner` 生命周期测试
- adapter contract tests
- CLI 基础命令 smoke test
- 配置兼容测试
- 旧 `plan_name` / 新 `game_name` 输出兼容测试

完成标准：

- 在没有 `backend` / `aura_gui` / `resonance` 的情况下，测试体系仍然自洽

---

## 6. 当前不做的事

以下内容不在本阶段范围内：

- 恢复 HTTP API
- 恢复 GUI
- 保留任何 `resonance` 业务代码在主仓库中
- 为 PC 游戏提供内存注入、读写进程状态、反作弊敏感能力
- 重写任务 DSL

---

## 7. 优先级建议

如果迁移到新路径后要继续开发，建议按下面顺序推进：

1. 阶段 A：稳定拆分项目基线
2. 阶段 B：收敛术语与用户心智
3. 阶段 C：平台适配层抽象化
4. 阶段 D：配置收敛
5. 阶段 F：测试与回归体系固化
6. 阶段 E：目录与声明文件收敛

原因：

- A/B 能最快提升“这已经是个新项目”的清晰度
- C/D 是真正决定未来扩展能力的技术骨架
- E 虽然重要，但不是第一优先级，否则迁移成本会陡增

---

## 8. 迁移后验收清单

迁移到新目录并继续改造后，建议以以下结果作为阶段验收：

- `README.md` 与 `docs/` 不再把项目描述为 HTTP 自动化平台
- `cli.py` 能完成游戏模块枚举、任务枚举、任务运行、运行详情查询
- `EmbeddedGameRunner` 可直接嵌入脚本或工具使用
- `SubprocessGameRunner` 可被宿主 GUI 调用
- `aura_benchmark` 始终可跑通
- `aura_base` 保持稳定，且开始出现统一 adapter 抽象
- 没有 `backend` / `aura_gui` / `resonance` 依赖残留在核心使用路径上

---

## 9. 备注

当前拆分出的版本已经是一个“可运行的新项目起点”，不是纯文档草稿。

因此后续修改建议遵循：

- 先稳定对外使用路径
- 再逐步收敛内部结构
- 避免为了术语或目录一次性重写整个内核

这能最大限度保留现有可用能力，同时让它逐步变成一个真正专攻游戏脚本的框架。

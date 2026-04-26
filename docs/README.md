# 文档总览

这份文档集对应拆分后的 `Aura Game Framework`，重点只保留游戏脚本框架本身相关的内容。

## 推荐阅读顺序

1. [环境与入口](./getting-started/01-python-runtime.md)
2. [架构总览](./getting-started/02-architecture-overview.md)
3. [任务 YAML 指南](./getting-started/03-task-yaml-guide.md)
4. [运行时行为](./getting-started/04-runtime-behavior.md)

## 项目参考

- [aura_base 运行时与平台适配层](./project-reference/aura-base-runtime.md)
- [YOLO ONNX Runtime 部署、导出与使用指南](./project-reference/yolo-onnx-runtime.md)
- [aura_benchmark 基准计划](./project-reference/benchmark-plan.md)

## 开发参考

- [Action 与 Service](./package-development/actions-and-services.md)
- [Manifest 参考](./package-development/manifest-reference.md)
- [任务引用与依赖](./package-development/task-references-and-dependencies.md)

## 运行维护

- [Observability](./runtime-operations/observability.md)
- [Scheduler 与 Runtime Profile](./runtime-operations/scheduler-and-profiles.md)
- [State Planning](./runtime-operations/state-planning.md)
- [框架打包与外置 Plans](./runtime-operations/packaging-with-external-plans.md)

## 说明

- 这份拆分项目不再把 `backend/`、`FastAPI` 或 `aura_gui/` 视为框架核心。
- 当前仍保留 `plans/`、`plan_name`、`task_ref` 等兼容术语，方便从原项目平滑迁移。
- 面向使用者时，可以把 `plan` 理解为“游戏模块”，把 `task` 理解为“游戏脚本任务”。

# aura_benchmark 基准计划

`plans/aura_benchmark/` 是当前仓库里最适合作为“调度器回归基线”的示例 plan。它没有业务耦合，专门用于验证并发、串行和基础运行统计。

## 组成

### Service

- `benchmark_probe`
  收集一个场景下的并发度、完成数、失败数和耗时统计。

### Actions

- `benchmark_reset`
  重置某个场景的统计状态。
- `benchmark_sleep`
  Sleep 指定毫秒数，并记录一次开始/结束数据。
- `benchmark_snapshot`
  返回场景统计快照。

### Tasks

- `single_sleep`
- `serial_sleep`
- `parallel_sleep`

## 任务定义与意图

### `single_sleep`

文件：`plans/aura_benchmark/tasks/single_sleep.yaml`

行为：

- 执行一次 `benchmark_sleep`
- 返回 `record`

典型用途：

- 验证最小执行链路
- 验证单节点运行耗时
- 验证 run detail 中单节点结果

### `serial_sleep`

文件：`plans/aura_benchmark/tasks/serial_sleep.yaml`

行为：

- `first -> second -> third` 串行执行
- 最后 `collect` 调用 `benchmark_snapshot`
- 返回 `snapshot`

典型用途：

- 验证 `depends_on` 串行链条
- 对比并发与串行的峰值活跃数差异

### `parallel_sleep`

文件：`plans/aura_benchmark/tasks/parallel_sleep.yaml`

行为：

- `first`、`second`、`third` 三个节点彼此无依赖
- `collect` 依赖三者全部完成
- 返回 `snapshot`

典型用途：

- 验证 DAG 中独立节点是否能并发推进
- 验证调度器对 ready node 的处理

## `benchmark_probe` 会记录什么

`BenchmarkProbeService.snapshot()` 当前返回这些关键字段：

- `scenario`
- `started_count`
- `completed_count`
- `failure_count`
- `active_count`
- `peak_active_count`
- `avg_duration_ms`
- `max_duration_ms`
- `min_duration_ms`
- `total_duration_ms`
- `last_started_label`
- `last_completed_label`
- `recent_records`

这组字段足够做两类事情：

- 判断任务有没有按预期完成
- 粗略验证并发特征是否符合预期

## 一个最小对比实验

### 1. 提交串行任务

```json
{
  "plan_name": "aura_benchmark",
  "task_ref": "serial_sleep",
  "inputs": {
    "duration_ms": 100,
    "scenario": "serial_demo"
  }
}
```

### 2. 提交并行任务

```json
{
  "plan_name": "aura_benchmark",
  "task_ref": "parallel_sleep",
  "inputs": {
    "duration_ms": 100,
    "scenario": "parallel_demo"
  }
}
```

### 3. 观察结果

可以通过两种路径看结果：

- `POST /api/v1/tasks/status/batch` 看调度状态
- `GET /api/v1/runs/{cid}` 看最终 `user_data / framework_data / nodes`

对于这两个任务，最值得关注的是返回的 `snapshot`：

- `serial_sleep` 的 `peak_active_count` 应维持在更低水平
- `parallel_sleep` 在允许节点并发时，`peak_active_count` 应高于串行链

这里不建议把文档写成固定阈值，因为最终值仍受当前执行器配置影响。

## 为什么这个 plan 很重要

`aura_benchmark` 的价值不在于“做了什么业务”，而在于它把运行时语义拆得足够纯净：

- 没有设备依赖
- 没有 UI 识别依赖
- 没有 OCR、YOLO、ADB 前置条件
- 只验证调度、依赖和运行记录

因此它非常适合：

- CLI 集成 smoke test
- 本地 SDK 集成 smoke test
- scheduler 改动后的回归验证
- 文档中的示例任务请求

## 建议用法

如果你要验证这套仓库还能不能正常调度，优先跑 `aura_benchmark`：

1. 先用 `cli.py games --all` 确认 `aura_benchmark` 被正确加载。
2. 再用 `cli.py tasks aura_benchmark` 确认任务清单可见。
3. 先跑 `single_sleep` 做基础联通性检查。
4. 再跑 `serial_sleep` 和 `parallel_sleep` 做依赖/并发行为对比。

这比直接上设备型 plan 更容易定位问题到底出在“调度器”还是“目标运行时”。

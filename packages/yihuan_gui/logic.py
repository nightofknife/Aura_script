from __future__ import annotations

from dataclasses import dataclass, field
import html
import json
from typing import Any, Iterable, Mapping


GAME_NAME = "yihuan"
TASK_PLAN_READY = "tasks:bootstrap:plan_ready.yaml"
TASK_PLAN_LOADED = "tasks:checks:plan_loaded.yaml"
TASK_RUNTIME_PROBE = "tasks:checks:runtime_probe.yaml"
TASK_AUTO_LOOP = "tasks:fishing:auto_loop.yaml"
TASK_LIVE_MONITOR = "tasks:fishing:live_monitor.yaml"
TASK_CAFE_AUTO_LOOP = "tasks:cafe:auto_loop.yaml"
TASK_ONE_CAFE_REVENUE_RESTOCK = "tasks:one_cafe/revenue_restock.yaml"
TASK_MAHJONG_AUTO_LOOP = "tasks:mahjong:auto_loop.yaml"

VISIBLE_HISTORY_TASK_REFS = (
    TASK_PLAN_READY,
    TASK_PLAN_LOADED,
    TASK_RUNTIME_PROBE,
    TASK_AUTO_LOOP,
    TASK_CAFE_AUTO_LOOP,
    TASK_ONE_CAFE_REVENUE_RESTOCK,
    TASK_MAHJONG_AUTO_LOOP,
)
DEVELOPER_TASK_REFS = (TASK_PLAN_READY, TASK_PLAN_LOADED, TASK_RUNTIME_PROBE)
TERMINAL_STATUSES = {"success", "error", "failed", "timeout", "cancelled", "partial"}

TASK_DISPLAY_NAMES = {
    TASK_PLAN_READY: "异环方案包信息",
    TASK_PLAN_LOADED: "异环方案包已加载",
    TASK_RUNTIME_PROBE: "异环运行时探针",
    TASK_AUTO_LOOP: "自动循环钓鱼",
    TASK_LIVE_MONITOR: "钓鱼实时监视",
    TASK_CAFE_AUTO_LOOP: "沙威玛",
    TASK_ONE_CAFE_REVENUE_RESTOCK: "一咖舍",
    TASK_MAHJONG_AUTO_LOOP: "自动麻将",
}

STATUS_LABELS = {
    "queued": "排队中",
    "running": "运行中",
    "success": "成功",
    "error": "错误",
    "failed": "失败",
    "timeout": "超时",
    "cancelled": "已取消",
    "partial": "部分成功",
    "skipped": "已跳过",
    "unknown": "未知",
}

STOP_REASON_LABELS = {
    "max_rounds": "达到最大轮数",
    "max_consecutive_failures": "达到最大连续失败次数",
    "duration_reached": "达到运行时长",
    "window_closed": "监视窗口已关闭",
    "max_seconds": "达到最大运行秒数",
    "max_orders": "达到最大订单数",
    "level_end": "关卡结束",
    "failure": "执行失败",
    "exception": "发生异常",
    "phase_timeout": "等待阶段超时",
    "completed": "执行完成",
    "world_return_failed": "返回世界场景失败",
    "unknown_withdraw_state": "收益领取状态未知",
    "unknown_restock_state": "补货状态未知",
}

FAILURE_REASON_LABELS = {
    "hook_timeout": "等待上钩超时",
    "duel_timeout": "对抗超时",
    "result_close_timeout": "结果页关闭超时",
    "post_duel_cleanup_timeout": "对抗结束清理超时",
    "not_ready": "未进入可抛竿状态",
    "duel_lost_to_ready": "对抗结束后回到准备态",
    "capture_failed": "截图失败",
    "level_start_capture_failed": "等待开局时截图失败",
    "level_start_timeout": "等待关卡开始超时",
    "phase_timeout": "等待阶段超时",
    "exception": "发生异常",
    "world_return_failed": "返回世界场景失败",
    "unknown_withdraw_state": "收益领取状态未知",
    "unknown_restock_state": "补货状态未知",
    "none": "无",
}

RECIPE_LABELS = {
    "latte_coffee": "拿铁咖啡",
    "cream_coffee": "奶油咖啡",
    "bacon_bread": "培根面包",
    "egg_croissant": "鸡蛋可颂",
    "jam_cake": "果酱蛋糕",
}

STOCK_LABELS = {
    "bread": "面包",
    "croissant": "可颂",
    "cake": "蛋糕",
    "coffee": "咖啡",
}

MAHJONG_SUIT_LABELS = {
    "wan": "万",
    "tong": "筒",
    "tiao": "条",
}

MAHJONG_TOGGLE_LABELS = {
    "hu": "胡",
    "peng": "碰",
    "discard": "出",
}

CAFE_EVENT_LABELS = {
    "start_game_clicked": "点击开始游戏",
    "level_start_waiting": "等待关卡开始",
    "level_started_confirmed": "确认关卡开始",
    "stock_batch_started": "开始补货",
    "stock_batch_ready": "补货完成",
    "stock_visual_empty_correction": "视觉库存校正",
    "order_pacing_wait": "订单节奏等待",
    "order_guard_blocked_orders": "订单守卫阻挡",
    "order_guard_deferred_all": "订单守卫暂缓全部订单",
    "fake_customer_hammer_clicked": "敲走假顾客",
    "order_selected": "选择订单",
    "order_completed": "完成订单",
    "stock_depleted_after_order": "订单后库存耗尽",
    "level_end_detected": "检测到关卡结束",
    "level_start_timeout": "等待关卡开始超时",
}

ONE_CAFE_RESULT_LABELS = {
    "skipped": "已跳过",
    "no_revenue": "暂无收益",
    "claimed": "已领取",
    "unknown": "未知",
    "inventory_full": "库存已满",
    "delivery_confirmed": "已确认送货",
}

PHASE_LABELS = {
    "ready": "可抛竿",
    "bite": "咬钩",
    "duel": "对抗中",
    "result": "结果页",
    "dingque": "定缺中",
    "playing": "行牌中",
    "open_city_tycoon": "打开都市大亨",
    "one_cafe_entry": "进入一咖舍",
    "withdraw": "领取收益",
    "restock": "补货",
    "exit": "返回世界",
    "unknown": "未知",
    "n/a": "-",
}

TRACE_NOTE_LABELS = {
    "initial": "初始识别",
    "cast": "抛竿",
    "hook_spam": "尝试收钩",
    "release_hold": "松开方向键",
    "post_duel_click": "点击清理结果",
    "post_duel_cleanup_done": "清理完成",
    "after_ready": "准备后等待",
    "after_dingque": "定缺后等待",
    "switch_verify": "验证自动开关",
    "phase_wait": "等待可识别阶段",
    "monitor": "监控结算页",
}

CONTROL_REASON_LABELS = {
    "outside_hold": "区间外长按",
    "edge_hold": "贴边长按",
    "inside_tap": "区间内点按",
    "suspicious_keep_hold": "疑似跳变，短暂保持",
    "suspicious_release": "疑似跳变，释放输入",
    "missing_keep_hold": "短暂丢失，保持输入",
    "not_duel": "非对抗阶段",
    "none": "无需输入",
}

DETECTION_REASON_LABELS = {
    "ok": "正常",
    "zone_missing": "蓝绿区丢失",
    "indicator_missing": "指示线丢失",
    "empty_region": "区域为空",
    "none": "无",
}

RUNTIME_CODE_LABELS = {
    "window_not_found": "未找到符合配置的游戏窗口",
    "window_target_ambiguous": "匹配到了多个候选窗口，请收窄窗口筛选条件",
    "window_target_lost": "之前绑定的游戏窗口已经失效",
    "window_not_visible": "目标窗口当前不可见",
    "window_not_foreground": "目标窗口当前不在前台",
    "window_focus_required": "输入前需要先聚焦目标窗口，但聚焦失败了",
    "provider_unsupported": "当前运行时提供方不受支持",
    "target_config_invalid": "运行时目标配置无效",
    "runtime_family_provider_mismatch": "运行时 family 与 provider 配置不匹配",
    "capture_backend_invalid_for_provider": "当前 provider 不支持该截图后端",
    "input_backend_invalid_for_provider": "当前 provider 不支持该输入后端",
    "input_integrity_mismatch": "当前进程权限级别不足，无法驱动目标窗口输入",
}

EVENT_LABELS = {
    "scheduler.started": "调度器已启动",
    "metrics.update": "运行指标更新",
    "queue.enqueued": "任务已入队",
    "queue.dequeued": "任务开始出队执行",
    "task.started": "任务开始执行",
    "task.finished": "任务执行结束",
    "node.started": "节点开始执行",
    "node.finished": "节点执行结束",
    "node.succeeded": "节点执行成功",
    "node.failed": "节点执行失败",
}

RUNTIME_VALUE_LABELS = {
    "windows": "Windows 桌面",
    "mumu": "MuMu 模拟器",
    "windows_desktop": "Windows 桌面",
    "android_emulator": "安卓模拟器",
    "title": "按窗口标题匹配",
    "process": "按进程匹配",
    "hwnd": "按窗口句柄匹配",
    "gdi": "GDI 截图",
    "dxgi": "DXGI 截图",
    "wgc": "Windows 图形捕获",
    "printwindow": "PrintWindow 截图",
    "sendinput": "SendInput",
    "window_message": "窗口消息输入",
}

SETTING_LABELS = {
    "runtime.target.title_regex": "窗口标题匹配规则",
    "runtime.target.exclude_titles": "排除窗口标题",
    "runtime.target.allow_borderless": "允许无边框窗口",
    "runtime.capture.backend": "截图后端",
    "runtime.input.backend": "输入后端",
    "input.profile": "默认输入档案",
    "gui.history_limit": "历史记录显示条数",
    "gui.auto_runtime_probe_on_startup": "启动时自动执行运行时探针",
    "gui.expand_developer_tools": "默认展开开发者工具",
    "gui.task_start_delay_sec": "任务启动延迟秒数",
    "gui.quick_stop_hotkey": "快捷停止键",
}


@dataclass(frozen=True)
class RuntimeSettings:
    title_regex: str
    exclude_titles: list[str]
    allow_borderless: bool
    capture_backend: str
    input_backend: str
    input_profile: str


@dataclass(frozen=True)
class GuiPreferences:
    history_limit: int = 50
    auto_runtime_probe_on_startup: bool = True
    expand_developer_tools: bool = False
    task_start_delay_sec: int = 5
    quick_stop_hotkey: str = "F8"


@dataclass(frozen=True)
class FishingRunDefaults:
    profile_name: str = "default_1280x720_cn"


@dataclass(frozen=True)
class CafeRunDefaults:
    profile_name: str = "default_1280x720_cn"
    max_seconds: int = 0
    max_orders: int = 0
    start_game: bool = True
    wait_level_started: bool = True
    full_assist_auto_hammer_mode: bool = False
    min_order_interval_sec: float = 0.3
    min_order_duration_sec: float = 0.0


@dataclass(frozen=True)
class OneCafeRunDefaults:
    profile_name: str = "default_1280x720_cn"
    withdraw_enabled: bool = True
    restock_enabled: bool = True
    restock_hours: int = 24


@dataclass(frozen=True)
class MahjongRunDefaults:
    profile_name: str = "default_1280x720_cn"
    max_seconds: int = 0
    start_game: bool = True
    auto_hu: bool = True
    auto_peng: bool = True
    auto_discard: bool = True


@dataclass(frozen=True)
class SettingsField:
    key: str
    label: str
    value: Any
    kind: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class SettingsSection:
    section_id: str
    title: str
    fields: tuple[SettingsField, ...]


@dataclass
class LiveUiState:
    active_runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_event_name: str | None = None
    last_error: str | None = None
    latest_metrics: dict[str, Any] = field(default_factory=dict)
    latest_cid: str | None = None
    latest_status: str | None = None


def build_auto_loop_inputs(max_rounds: Any, defaults: FishingRunDefaults) -> dict[str, Any]:
    return {
        "max_rounds": int(max_rounds),
        "profile_name": str(defaults.profile_name),
    }


def build_cafe_loop_inputs(
    max_seconds: Any,
    max_orders: Any,
    start_game: Any,
    wait_level_started: Any,
    full_assist_auto_hammer_mode: Any,
    defaults: CafeRunDefaults,
) -> dict[str, Any]:
    return {
        "profile_name": str(defaults.profile_name),
        "max_seconds": int(max_seconds),
        "max_orders": int(max_orders),
        "start_game": bool(start_game),
        "wait_level_started": bool(wait_level_started),
        "full_assist_auto_hammer_mode": bool(full_assist_auto_hammer_mode),
        "min_order_interval_sec": float(defaults.min_order_interval_sec),
        "min_order_duration_sec": float(defaults.min_order_duration_sec),
    }


def build_one_cafe_inputs(
    withdraw_enabled: Any,
    restock_enabled: Any,
    restock_hours: Any,
    defaults: OneCafeRunDefaults,
) -> dict[str, Any]:
    return {
        "profile_name": str(defaults.profile_name),
        "withdraw_enabled": bool(withdraw_enabled),
        "restock_enabled": bool(restock_enabled),
        "restock_hours": int(restock_hours),
    }


def build_mahjong_loop_inputs(
    max_seconds: Any,
    start_game: Any,
    auto_hu: Any,
    auto_peng: Any,
    auto_discard: Any,
    defaults: MahjongRunDefaults,
) -> dict[str, Any]:
    return {
        "profile_name": str(defaults.profile_name),
        "max_seconds": int(max_seconds),
        "start_game": bool(start_game),
        "auto_hu": bool(auto_hu),
        "auto_peng": bool(auto_peng),
        "auto_discard": bool(auto_discard),
        "dry_run": False,
        "debug_enabled": False,
    }


def extract_auto_loop_defaults(task_row: Mapping[str, Any] | None) -> FishingRunDefaults:
    default_profile = FishingRunDefaults().profile_name
    for field in (task_row or {}).get("inputs") or []:
        if not isinstance(field, Mapping):
            continue
        if str(field.get("name") or "").strip() != "profile_name":
            continue
        resolved = str(field.get("default") or "").strip()
        if resolved:
            return FishingRunDefaults(profile_name=resolved)
    return FishingRunDefaults(profile_name=default_profile)


def extract_cafe_loop_defaults(task_row: Mapping[str, Any] | None) -> CafeRunDefaults:
    values = {
        "profile_name": CafeRunDefaults().profile_name,
        "max_seconds": CafeRunDefaults().max_seconds,
        "max_orders": CafeRunDefaults().max_orders,
        "start_game": CafeRunDefaults().start_game,
        "wait_level_started": CafeRunDefaults().wait_level_started,
        "full_assist_auto_hammer_mode": CafeRunDefaults().full_assist_auto_hammer_mode,
        "min_order_interval_sec": CafeRunDefaults().min_order_interval_sec,
        "min_order_duration_sec": CafeRunDefaults().min_order_duration_sec,
    }
    for field in (task_row or {}).get("inputs") or []:
        if not isinstance(field, Mapping):
            continue
        name = str(field.get("name") or "").strip()
        if name not in values:
            continue
        values[name] = field.get("default", values[name])
    return CafeRunDefaults(
        profile_name=str(values["profile_name"] or CafeRunDefaults().profile_name),
        max_seconds=int(float(values["max_seconds"] or 0)),
        max_orders=int(float(values["max_orders"] or 0)),
        start_game=_coerce_bool(values["start_game"]),
        wait_level_started=_coerce_bool(values["wait_level_started"]),
        full_assist_auto_hammer_mode=_coerce_bool(values["full_assist_auto_hammer_mode"]),
        min_order_interval_sec=float(values["min_order_interval_sec"] or 0.0),
        min_order_duration_sec=float(values["min_order_duration_sec"] or 0.0),
    )


def extract_one_cafe_defaults(task_row: Mapping[str, Any] | None) -> OneCafeRunDefaults:
    values = {
        "profile_name": OneCafeRunDefaults().profile_name,
        "withdraw_enabled": OneCafeRunDefaults().withdraw_enabled,
        "restock_enabled": OneCafeRunDefaults().restock_enabled,
        "restock_hours": OneCafeRunDefaults().restock_hours,
    }
    for field in (task_row or {}).get("inputs") or []:
        if not isinstance(field, Mapping):
            continue
        name = str(field.get("name") or "").strip()
        if name not in values:
            continue
        values[name] = field.get("default", values[name])
    return OneCafeRunDefaults(
        profile_name=str(values["profile_name"] or OneCafeRunDefaults().profile_name),
        withdraw_enabled=_coerce_bool(values["withdraw_enabled"]),
        restock_enabled=_coerce_bool(values["restock_enabled"]),
        restock_hours=int(float(values["restock_hours"] or OneCafeRunDefaults().restock_hours)),
    )


def extract_mahjong_loop_defaults(task_row: Mapping[str, Any] | None) -> MahjongRunDefaults:
    values = {
        "profile_name": MahjongRunDefaults().profile_name,
        "max_seconds": MahjongRunDefaults().max_seconds,
        "start_game": MahjongRunDefaults().start_game,
        "auto_hu": MahjongRunDefaults().auto_hu,
        "auto_peng": MahjongRunDefaults().auto_peng,
        "auto_discard": MahjongRunDefaults().auto_discard,
    }
    for field in (task_row or {}).get("inputs") or []:
        if not isinstance(field, Mapping):
            continue
        name = str(field.get("name") or "").strip()
        if name not in values:
            continue
        values[name] = field.get("default", values[name])
    return MahjongRunDefaults(
        profile_name=str(values["profile_name"] or MahjongRunDefaults().profile_name),
        max_seconds=int(float(values["max_seconds"] or 0)),
        start_game=_coerce_bool(values["start_game"]),
        auto_hu=_coerce_bool(values["auto_hu"]),
        auto_peng=_coerce_bool(values["auto_peng"]),
        auto_discard=_coerce_bool(values["auto_discard"]),
    )


def build_settings_sections(
    runtime_settings: RuntimeSettings,
    ui_preferences: GuiPreferences,
    available_input_profiles: Iterable[str],
) -> list[SettingsSection]:
    profiles = tuple(str(item) for item in available_input_profiles)
    return [
        SettingsSection(
            section_id="runtime",
            title="运行环境",
            fields=(
                SettingsField(
                    key="runtime.target.title_regex",
                    label=SETTING_LABELS["runtime.target.title_regex"],
                    value=runtime_settings.title_regex,
                    kind="text",
                ),
                SettingsField(
                    key="runtime.target.exclude_titles",
                    label=SETTING_LABELS["runtime.target.exclude_titles"],
                    value="\n".join(runtime_settings.exclude_titles),
                    kind="multiline",
                ),
                SettingsField(
                    key="runtime.target.allow_borderless",
                    label=SETTING_LABELS["runtime.target.allow_borderless"],
                    value=runtime_settings.allow_borderless,
                    kind="bool",
                ),
                SettingsField(
                    key="runtime.capture.backend",
                    label=SETTING_LABELS["runtime.capture.backend"],
                    value=runtime_settings.capture_backend,
                    kind="choice",
                ),
                SettingsField(
                    key="runtime.input.backend",
                    label=SETTING_LABELS["runtime.input.backend"],
                    value=runtime_settings.input_backend,
                    kind="choice",
                ),
                SettingsField(
                    key="input.profile",
                    label=SETTING_LABELS["input.profile"],
                    value=runtime_settings.input_profile,
                    kind="choice",
                    options=profiles,
                ),
            ),
        ),
        SettingsSection(
            section_id="ui",
            title="界面偏好",
            fields=(
                SettingsField(
                    key="gui.history_limit",
                    label=SETTING_LABELS["gui.history_limit"],
                    value=ui_preferences.history_limit,
                    kind="number",
                ),
                SettingsField(
                    key="gui.auto_runtime_probe_on_startup",
                    label=SETTING_LABELS["gui.auto_runtime_probe_on_startup"],
                    value=ui_preferences.auto_runtime_probe_on_startup,
                    kind="bool",
                ),
                SettingsField(
                    key="gui.expand_developer_tools",
                    label=SETTING_LABELS["gui.expand_developer_tools"],
                    value=ui_preferences.expand_developer_tools,
                    kind="bool",
                ),
                SettingsField(
                    key="gui.task_start_delay_sec",
                    label=SETTING_LABELS["gui.task_start_delay_sec"],
                    value=ui_preferences.task_start_delay_sec,
                    kind="number",
                ),
                SettingsField(
                    key="gui.quick_stop_hotkey",
                    label=SETTING_LABELS["gui.quick_stop_hotkey"],
                    value=ui_preferences.quick_stop_hotkey,
                    kind="choice",
                    options=("F6", "F7", "F8", "F9", "F10", "F11", "F12"),
                ),
            ),
        ),
    ]


def is_runtime_interacting_task(task_ref: str | None) -> bool:
    normalized = str(task_ref or "").strip()
    return normalized == TASK_RUNTIME_PROBE or normalized.startswith(
        ("tasks:fishing:", "tasks:cafe:", "tasks:one_cafe/", "tasks:one_cafe:", "tasks:mahjong:")
    )


def task_is_enabled(task_ref: str | None, active_runs: Mapping[str, Mapping[str, Any]]) -> bool:
    if not active_runs:
        return True
    candidate = str(task_ref or "").strip()
    if candidate in {TASK_PLAN_READY, TASK_PLAN_LOADED}:
        return True
    for run in active_runs.values():
        if is_runtime_interacting_task(run.get("task_name")):
            return False
    return True


def task_display_name(task_ref: str | None) -> str:
    normalized = str(task_ref or "").strip()
    return TASK_DISPLAY_NAMES.get(normalized, normalized or "-")


def status_display_name(status: Any) -> str:
    normalized = str(status or "unknown").strip().lower()
    return STATUS_LABELS.get(normalized, normalized or "未知")


def stop_reason_display_name(reason: Any) -> str:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return "-"
    return STOP_REASON_LABELS.get(normalized, normalized)


def failure_reason_display_name(reason: Any) -> str:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return "-"
    return FAILURE_REASON_LABELS.get(normalized, normalized)


def phase_display_name(phase: Any) -> str:
    normalized = str(phase or "").strip().lower()
    if not normalized:
        return "-"
    return PHASE_LABELS.get(normalized, normalized)


def detection_reason_display_name(reason: Any) -> str:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return "-"
    return DETECTION_REASON_LABELS.get(normalized, normalized)


def runtime_code_display_name(code: Any) -> str:
    normalized = str(code or "").strip().lower()
    if not normalized:
        return "-"
    return RUNTIME_CODE_LABELS.get(normalized, normalized)


def event_display_name(name: Any) -> str:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return "-"
    return EVENT_LABELS.get(normalized, normalized)


def runtime_value_display_name(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "-"
    return RUNTIME_VALUE_LABELS.get(normalized, value)


def reduce_live_events(state: LiveUiState, events: Iterable[Mapping[str, Any]]) -> tuple[LiveUiState, list[str]]:
    next_state = LiveUiState(
        active_runs={cid: dict(payload) for cid, payload in state.active_runs.items()},
        last_event_name=state.last_event_name,
        last_error=state.last_error,
        latest_metrics=dict(state.latest_metrics),
        latest_cid=state.latest_cid,
        latest_status=state.latest_status,
    )
    finished_cids: list[str] = []
    for event in events:
        name = str(event.get("name") or "").strip()
        payload = dict(event.get("payload") or {})
        next_state.last_event_name = name or next_state.last_event_name
        if name == "metrics.update":
            next_state.latest_metrics = payload
            continue
        if name == "scheduler.started":
            continue

        game_name = payload.get("game_name")
        if game_name and str(game_name) != GAME_NAME:
            continue

        cid = str(payload.get("cid") or "").strip()
        task_name = str(payload.get("task_name") or "").strip()
        if cid:
            next_state.latest_cid = cid

        if name == "queue.enqueued" and cid:
            next_state.active_runs[cid] = {
                "cid": cid,
                "task_name": task_name,
                "status": "queued",
                "last_event_name": name,
            }
            continue

        if name == "task.started" and cid:
            run = next_state.active_runs.setdefault(cid, {"cid": cid})
            run["task_name"] = task_name or run.get("task_name")
            run["status"] = "running"
            run["last_event_name"] = name
            continue

        if name.startswith("node.") and cid:
            run = next_state.active_runs.setdefault(cid, {"cid": cid, "task_name": task_name})
            run["task_name"] = task_name or run.get("task_name")
            run["last_event_name"] = name
            if payload.get("node_id"):
                run["last_node_id"] = payload.get("node_id")
            continue

        if name == "task.finished" and cid:
            status = str(payload.get("final_status") or payload.get("status") or "unknown").strip().lower()
            next_state.latest_status = status
            run = next_state.active_runs.pop(cid, None)
            if run is not None:
                run["status"] = status
                run["last_event_name"] = name
            finished_cids.append(cid)
            if status not in {"success"}:
                label = task_display_name(task_name or cid)
                next_state.last_error = f"{label}：{status_display_name(status)}"
            continue
    return next_state, finished_cids


def history_row_label(row: Mapping[str, Any]) -> str:
    cid = str(row.get("cid") or "").strip()
    task_name = task_display_name(row.get("task_name"))
    status = status_display_name(row.get("status"))
    return f"{status}  {task_name or '-'}  {cid or '-'}"


def render_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)
    except TypeError:
        return json.dumps({"raw": str(value)}, ensure_ascii=False, indent=2)


def render_overview_plan_info_html(info: Mapping[str, Any] | None) -> str:
    payload = dict(info or {})
    if not payload:
        return "<p>尚未加载方案包信息。</p>"
    lines = [
        _kv_html("游戏名称", payload.get("game_title")),
        _kv_html("方案包", payload.get("plan_name")),
        _kv_html("英文标题", payload.get("official_title_en")),
        _kv_html("窗口匹配规则", payload.get("window_title_regex")),
        _kv_html("默认输入档案", payload.get("default_input_profile")),
    ]
    notes = payload.get("notes") or []
    if notes:
        lines.append("<p><b>说明</b></p><ul>" + "".join(f"<li>{html.escape(str(note))}</li>" for note in notes) + "</ul>")
    return "".join(lines)


def render_runtime_probe_html(probe: Mapping[str, Any] | None) -> str:
    payload = dict(probe or {})
    if not payload:
        return "<p>尚未获取运行时探针结果。</p>"
    target = dict(payload.get("target") or {})
    capture = dict(payload.get("capture") or {})
    input_payload = dict(payload.get("input") or {})
    lines = [
        _kv_html("状态", "正常" if payload.get("ok") else "未就绪"),
        _kv_html("运行时提供方", runtime_value_display_name(payload.get("provider"))),
        _kv_html("运行时家族", runtime_value_display_name(payload.get("family"))),
        _kv_html("目标模式", runtime_value_display_name(target.get("mode"))),
        _kv_html("目标窗口", target.get("title") or target.get("title_regex")),
        _kv_html("截图后端", runtime_value_display_name(capture.get("backend"))),
        _kv_html("输入后端", runtime_value_display_name(input_payload.get("backend"))),
    ]
    if payload.get("code") or payload.get("message"):
        lines.append(_kv_html("错误代码", runtime_code_display_name(payload.get("code"))))
        lines.append(_kv_html("错误信息", payload.get("message") or runtime_code_display_name(payload.get("code"))))
    warnings = payload.get("warnings") or []
    if warnings:
        lines.append(
            "<p><b>警告</b></p><ul>"
            + "".join(f"<li>{html.escape(runtime_code_display_name(item))}</li>" for item in warnings)
            + "</ul>"
        )
    return "".join(lines)


def render_task_result_html(task_ref: str | None, detail: Mapping[str, Any] | None) -> str:
    payload = dict(detail or {})
    task_name = str(task_ref or payload.get("task_name") or "").strip()
    final_result = dict(payload.get("final_result") or {})
    user_data = final_result.get("user_data")

    if task_name == TASK_PLAN_READY:
        info = dict((user_data or {}).get("info") if isinstance(user_data, Mapping) else {})
        return render_overview_plan_info_html(info)

    if task_name == TASK_RUNTIME_PROBE:
        return render_runtime_probe_html(user_data if isinstance(user_data, Mapping) else None)

    if task_name == TASK_PLAN_LOADED:
        return _kv_html("方案包已加载", "是" if bool(user_data) else "否")

    if task_name == TASK_AUTO_LOOP and isinstance(user_data, Mapping):
        return render_auto_loop_result_html(user_data)

    if task_name == TASK_CAFE_AUTO_LOOP and isinstance(user_data, Mapping):
        return render_cafe_loop_result_html(user_data)

    if task_name == TASK_ONE_CAFE_REVENUE_RESTOCK and isinstance(user_data, Mapping):
        return render_one_cafe_result_html(user_data)

    if task_name == TASK_MAHJONG_AUTO_LOOP and isinstance(user_data, Mapping):
        return render_mahjong_loop_result_html(user_data)

    if user_data is not None:
        return f"<pre>{html.escape(render_json(user_data))}</pre>"
    return "<p>当前没有可显示的任务输出。</p>"


def auto_loop_user_data(detail: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    payload = dict(detail or {})
    final_result = dict(payload.get("final_result") or {})
    user_data = final_result.get("user_data")
    return user_data if isinstance(user_data, Mapping) else None


def auto_loop_business_status(detail: Mapping[str, Any] | None) -> str | None:
    user_data = auto_loop_user_data(detail)
    if not user_data:
        return None
    status = str(user_data.get("status") or "").strip().lower()
    return status or None


def cafe_loop_user_data(detail: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    payload = dict(detail or {})
    final_result = dict(payload.get("final_result") or {})
    user_data = final_result.get("user_data")
    return user_data if isinstance(user_data, Mapping) else None


def cafe_loop_business_status(detail: Mapping[str, Any] | None) -> str | None:
    user_data = cafe_loop_user_data(detail)
    if not user_data:
        return None
    status = str(user_data.get("status") or "").strip().lower()
    return status or None


def one_cafe_user_data(detail: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    payload = dict(detail or {})
    final_result = dict(payload.get("final_result") or {})
    user_data = final_result.get("user_data")
    return user_data if isinstance(user_data, Mapping) else None


def one_cafe_business_status(detail: Mapping[str, Any] | None) -> str | None:
    user_data = one_cafe_user_data(detail)
    if not user_data:
        return None
    status = str(user_data.get("status") or "").strip().lower()
    return status or None


def mahjong_loop_user_data(detail: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    payload = dict(detail or {})
    final_result = dict(payload.get("final_result") or {})
    user_data = final_result.get("user_data")
    return user_data if isinstance(user_data, Mapping) else None


def mahjong_loop_business_status(detail: Mapping[str, Any] | None) -> str | None:
    user_data = mahjong_loop_user_data(detail)
    if not user_data:
        return None
    status = str(user_data.get("status") or "").strip().lower()
    return status or None


def render_auto_loop_brief_text(detail: Mapping[str, Any] | None) -> str:
    user_data = auto_loop_user_data(detail)
    if not user_data:
        return "暂无自动钓鱼结果。"
    return (
        f"{status_display_name(user_data.get('status'))} / "
        f"总轮数 {user_data.get('round_count', 0)} / "
        f"成功 {user_data.get('success_count', 0)} / "
        f"失败 {user_data.get('failure_count', 0)} / "
        f"停止原因 {stop_reason_display_name(user_data.get('stopped_reason'))}"
    )


def render_mahjong_loop_brief_text(detail: Mapping[str, Any] | None) -> str:
    user_data = mahjong_loop_user_data(detail)
    if not user_data:
        return "暂无自动麻将结果。"
    counts = dict(user_data.get("hand_suit_counts") or {})
    selected = _mahjong_suit_display_name(user_data.get("selected_missing_suit"))
    return (
        f"{status_display_name(user_data.get('status'))} / "
        f"缺门 {selected} / "
        f"手牌 万{counts.get('wan', 0)} 筒{counts.get('tong', 0)} 条{counts.get('tiao', 0)} / "
        f"停止原因 {stop_reason_display_name(user_data.get('stopped_reason'))} / "
        f"耗时 {_format_seconds(user_data.get('elapsed_sec'))}"
    )


def render_auto_loop_result_html(user_data: Mapping[str, Any]) -> str:
    results = [dict(item) for item in (user_data.get("results") or []) if isinstance(item, Mapping)]
    lines = [
        _kv_html("业务结果", status_display_name(user_data.get("status"))),
        _kv_html("总轮数", user_data.get("round_count")),
        _kv_html("成功轮数", user_data.get("success_count")),
        _kv_html("失败轮数", user_data.get("failure_count")),
        _kv_html("连续失败", user_data.get("consecutive_failures")),
        _kv_html("停止原因", stop_reason_display_name(user_data.get("stopped_reason"))),
    ]
    if results:
        lines.append(_render_auto_loop_rounds_table(results))
        lines.append(_render_round_details(results))
    else:
        lines.append("<p>暂无每轮结果。</p>")
    return "".join(lines)


def render_mahjong_loop_result_html(user_data: Mapping[str, Any]) -> str:
    lines = [
        _kv_html("业务结果", status_display_name(user_data.get("status"))),
        _kv_html("停止原因", stop_reason_display_name(user_data.get("stopped_reason"))),
        _kv_html("失败原因", failure_reason_display_name(user_data.get("failure_reason"))),
        _kv_html("识别档案", user_data.get("profile_name")),
        _kv_html("选择缺门", _mahjong_suit_display_name(user_data.get("selected_missing_suit"))),
        _render_mahjong_counts_table(user_data.get("hand_suit_counts")),
        _render_mahjong_toggle_table(user_data.get("auto_toggles_enabled")),
        _kv_html("总耗时", _format_seconds(user_data.get("elapsed_sec"))),
        _render_mahjong_phase_trace_tail(user_data.get("phase_trace") or []),
    ]
    if user_data.get("failure_message"):
        lines.insert(3, _kv_html("失败信息", user_data.get("failure_message")))
    if user_data.get("planned_actions"):
        lines.append(_kv_html("计划动作", user_data.get("planned_actions")))
    return "".join(lines)


def render_cafe_loop_brief_text(detail: Mapping[str, Any] | None) -> str:
    user_data = cafe_loop_user_data(detail)
    if not user_data:
        return "暂无沙威玛结果。"
    perf_counts = dict(dict(user_data.get("perf_stats") or {}).get("counts") or {})
    return (
        f"{status_display_name(user_data.get('status'))} / "
        f"完成订单 {user_data.get('orders_completed', 0)} / "
        f"驱赶假顾客 {user_data.get('fake_customers_driven', 0)} / "
        f"守卫暂缓 {perf_counts.get('order_guard_deferred_all_count', 0)} / "
        f"停止原因 {stop_reason_display_name(user_data.get('stopped_reason'))} / "
        f"关卡结果 {_cafe_level_outcome_display_name(user_data.get('level_outcome'))} / "
        f"耗时 {_format_seconds(user_data.get('elapsed_sec'))}"
    )


def render_one_cafe_brief_text(detail: Mapping[str, Any] | None) -> str:
    user_data = one_cafe_user_data(detail)
    if not user_data:
        return "暂无一咖舍结果。"
    return (
        f"{status_display_name(user_data.get('status'))} / "
        f"收益 {_one_cafe_result_display_name(user_data.get('withdraw_result'))} / "
        f"补货 {user_data.get('restock_hours', '-')} 小时 "
        f"{_one_cafe_result_display_name(user_data.get('restock_result'))} / "
        f"返回世界 {'是' if _coerce_bool(user_data.get('returned_to_world')) else '否'} / "
        f"停止原因 {stop_reason_display_name(user_data.get('stopped_reason'))}"
    )


def render_one_cafe_result_html(user_data: Mapping[str, Any]) -> str:
    lines = [
        _kv_html("业务结果", status_display_name(user_data.get("status"))),
        _kv_html("停止原因", stop_reason_display_name(user_data.get("stopped_reason"))),
        _kv_html("失败原因", failure_reason_display_name(user_data.get("failure_reason"))),
        _kv_html("识别档案", user_data.get("profile_name")),
        _kv_html("尝试领取收益", "是" if _coerce_bool(user_data.get("withdraw_attempted")) else "否"),
        _kv_html("收益结果", _one_cafe_result_display_name(user_data.get("withdraw_result"))),
        _kv_html("收益确认", "是" if _coerce_bool(user_data.get("withdraw_confirmed")) else "否"),
        _kv_html("尝试补货", "是" if _coerce_bool(user_data.get("restock_attempted")) else "否"),
        _kv_html("补货时长", f"{user_data.get('restock_hours', '-')} 小时"),
        _kv_html("补货结果", _one_cafe_result_display_name(user_data.get("restock_result"))),
        _kv_html("已返回世界", "是" if _coerce_bool(user_data.get("returned_to_world")) else "否"),
        _kv_html("世界 HUD 识别", "是" if _coerce_bool(user_data.get("world_hud_found")) else "否"),
        _render_one_cafe_phase_trace(user_data.get("phase_trace") or []),
    ]
    return "".join(lines)


def render_cafe_loop_result_html(user_data: Mapping[str, Any]) -> str:
    lines = [
        _kv_html("业务结果", status_display_name(user_data.get("status"))),
        _kv_html("停止原因", stop_reason_display_name(user_data.get("stopped_reason"))),
        _kv_html("失败原因", failure_reason_display_name(user_data.get("failure_reason"))),
        _kv_html("完成订单数", user_data.get("orders_completed")),
        _kv_html("检测到假顾客", user_data.get("fake_customers_detected")),
        _kv_html("驱赶假顾客", user_data.get("fake_customers_driven")),
        _render_cafe_order_guard_summary(user_data.get("perf_stats")),
        _kv_html("关卡结果", _cafe_level_outcome_display_name(user_data.get("level_outcome"))),
        _kv_html("未知扫描次数", user_data.get("unknown_scan_count")),
        _kv_html("最小订单间隔", _format_seconds(user_data.get("min_order_interval_sec"))),
        _kv_html("单订单最短耗时", _format_seconds(user_data.get("min_order_duration_sec"))),
        _kv_html("总耗时", _format_seconds(user_data.get("elapsed_sec"))),
        _kv_html("识别档案", user_data.get("profile_name")),
        _render_cafe_counts_table("订单统计", user_data.get("recognized_counts"), RECIPE_LABELS),
        _render_cafe_counts_table("补货统计", user_data.get("batches_made"), STOCK_LABELS),
        _render_cafe_counts_table("剩余库存", user_data.get("stocks_remaining"), STOCK_LABELS),
        _render_pending_batches_table(user_data.get("pending_batches")),
        _render_cafe_perf_stats(user_data.get("perf_stats")),
        _render_cafe_trace_tail(user_data.get("phase_trace") or []),
    ]
    if user_data.get("failure_message"):
        lines.insert(3, _kv_html("失败信息", user_data.get("failure_message")))
    return "".join(lines)


def _one_cafe_result_display_name(result: Any) -> str:
    normalized = str(result or "").strip().lower()
    if not normalized:
        return "-"
    return ONE_CAFE_RESULT_LABELS.get(normalized, normalized)


def _render_one_cafe_phase_trace(trace: Iterable[Mapping[str, Any]]) -> str:
    tail = [dict(item) for item in list(trace)[-12:] if isinstance(item, Mapping)]
    if not tail:
        return "<p><b>阶段轨迹</b>：-</p>"
    rows = [
        "<p><b>阶段轨迹</b></p>"
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<tr><th>阶段</th><th>状态</th><th>结果</th><th>原始数据</th></tr>"
    ]
    for entry in tail:
        phase = phase_display_name(entry.get("phase"))
        status = status_display_name(entry.get("status")) if entry.get("status") is not None else "-"
        result = _one_cafe_phase_result_text(entry)
        rows.append(
            "<tr>"
            f"<td>{html.escape(phase)}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(result)}</td>"
            f"<td><pre>{html.escape(render_json(entry))}</pre></td>"
            "</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _one_cafe_phase_result_text(entry: Mapping[str, Any]) -> str:
    phase = str(entry.get("phase") or "").strip()
    if phase == "one_cafe_entry":
        return "已进入" if _coerce_bool(entry.get("final_detection_found")) else "未确认进入"
    if phase == "withdraw":
        return _one_cafe_result_display_name(entry.get("result"))
    if phase == "restock":
        hours = entry.get("hours", "-")
        return f"{hours} 小时 / {_one_cafe_result_display_name(entry.get('result'))}"
    if phase == "exit":
        return "已返回世界" if _coerce_bool(entry.get("returned_to_world")) else "未确认返回"
    return "-"


def _mahjong_suit_display_name(suit: Any) -> str:
    normalized = str(suit or "").strip().lower()
    if not normalized:
        return "-"
    return MAHJONG_SUIT_LABELS.get(normalized, normalized)


def _render_mahjong_counts_table(values: Any) -> str:
    counts = dict(values or {}) if isinstance(values, Mapping) else {}
    rows = []
    for suit in ("wan", "tong", "tiao"):
        rows.append(
            "<tr>"
            f"<td>{html.escape(_mahjong_suit_display_name(suit))}</td>"
            f"<td>{html.escape(str(counts.get(suit, 0)))}</td>"
            "</tr>"
        )
    return "<h3>定缺手牌统计</h3><table><tr><th>花色</th><th>数量</th></tr>" + "".join(rows) + "</table>"


def _render_mahjong_toggle_table(values: Any) -> str:
    toggles = dict(values or {}) if isinstance(values, Mapping) else {}
    rows = []
    for key in ("hu", "peng", "discard"):
        value = toggles.get(key)
        if value is True:
            state = "已开启"
        elif value is False:
            state = "未确认开启"
        else:
            state = "-"
        rows.append(
            "<tr>"
            f"<td>{html.escape(MAHJONG_TOGGLE_LABELS.get(key, key))}</td>"
            f"<td>{html.escape(state)}</td>"
            "</tr>"
        )
    return "<h3>自动开关</h3><table><tr><th>开关</th><th>状态</th></tr>" + "".join(rows) + "</table>"


def _render_mahjong_phase_trace_tail(trace: Iterable[Mapping[str, Any]]) -> str:
    rows = []
    for entry in list(trace)[-12:]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(_format_seconds(entry.get('t')))}</td>"
            f"<td>{html.escape(phase_display_name(entry.get('phase')))}</td>"
            f"<td>{html.escape(_trace_note_display_name(entry.get('note')))}</td>"
            f"<td>{html.escape(str(entry.get('enabled_switch_count', '-')))}</td>"
            f"<td>{html.escape(str(entry.get('dingque_button_count', '-')))}</td>"
            "</tr>"
        )
    if not rows:
        return "<h3>阶段轨迹</h3><p>暂无阶段轨迹。</p>"
    return (
        "<h3>阶段轨迹</h3>"
        "<table><tr><th>时间</th><th>阶段</th><th>备注</th><th>已开开关</th><th>定缺按钮</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _render_cafe_order_guard_summary(stats: Any) -> str:
    counts = dict(dict(stats or {}).get("counts") or {}) if isinstance(stats, Mapping) else {}
    rows = [
        "<p><b>订单安全守卫</b></p><table border='1' cellspacing='0' cellpadding='4'>",
        "<tr><th>项目</th><th>次数</th></tr>",
    ]
    for key, label in (
        ("order_guard_active_block_count", "活跃阻挡点"),
        ("order_guard_blocked_order_count", "被阻挡订单"),
        ("order_guard_deferred_all_count", "全部订单暂缓"),
        ("order_guard_safe_selection_count", "改选安全订单"),
    ):
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(str(counts.get(key, 0)))}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def _render_cafe_counts_table(title: str, values: Any, labels: Mapping[str, str]) -> str:
    if not isinstance(values, Mapping):
        return f"<p><b>{html.escape(title)}</b>：-</p>"
    rows = [f"<p><b>{html.escape(title)}</b></p><table border='1' cellspacing='0' cellpadding='4'>"]
    rows.append("<tr><th>项目</th><th>数量</th></tr>")
    for key, value in dict(values).items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels.get(str(key), str(key)))}</td>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _render_pending_batches_table(values: Any) -> str:
    if not isinstance(values, Mapping) or not values:
        return "<p><b>未完成补货</b>：无</p>"
    rows = ["<p><b>未完成补货</b></p><table border='1' cellspacing='0' cellpadding='4'>"]
    rows.append("<tr><th>库存</th><th>批量</th><th>剩余时间</th></tr>")
    for stock_id, raw_item in dict(values).items():
        item = dict(raw_item or {}) if isinstance(raw_item, Mapping) else {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(STOCK_LABELS.get(str(stock_id), str(stock_id)))}</td>"
            f"<td>{html.escape(str(item.get('batch_size', '-')))}</td>"
            f"<td>{html.escape(_format_seconds(item.get('ready_in_sec')))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _render_cafe_perf_stats(stats: Any) -> str:
    if not isinstance(stats, Mapping):
        return "<p><b>性能统计</b>：-</p>"
    time_sec = dict(stats.get("time_sec") or {})
    counts = dict(stats.get("counts") or {})
    rows = ["<p><b>性能统计</b></p><table border='1' cellspacing='0' cellpadding='4'>"]
    rows.append("<tr><th>指标</th><th>值</th></tr>")
    for key, value in time_sec.items():
        rows.append(f"<tr><td>{html.escape(_perf_label(key))}</td><td>{html.escape(_format_seconds(value))}</td></tr>")
    for key, value in counts.items():
        rows.append(f"<tr><td>{html.escape(_perf_label(key))}</td><td>{html.escape(str(value))}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def _render_cafe_trace_tail(trace: Iterable[Mapping[str, Any]]) -> str:
    tail = [dict(item) for item in list(trace)[-12:] if isinstance(item, Mapping)]
    if not tail:
        return "<p><b>事件轨迹</b>：-</p>"
    rows = [
        "<p><b>事件轨迹（最后 12 条）</b></p>"
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<tr><th>时间</th><th>事件</th><th>内容</th></tr>"
    ]
    for entry in tail:
        event = str(entry.get("event") or "")
        payload = {key: value for key, value in entry.items() if key not in {"t", "event"}}
        rows.append(
            "<tr>"
            f"<td>{html.escape(_format_seconds(entry.get('t')))}</td>"
            f"<td>{html.escape(CAFE_EVENT_LABELS.get(event, event or '-'))}</td>"
            f"<td><pre>{html.escape(render_json(payload))}</pre></td>"
            "</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _render_auto_loop_rounds_table(results: list[dict[str, Any]]) -> str:
    rows = [
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<tr>"
        "<th>轮次</th><th>结果</th><th>失败原因</th><th>总耗时</th><th>上钩等待</th>"
        "<th>对抗耗时</th><th>对抗识别</th><th>调试截图</th>"
        "</tr>"
    ]
    for result in results:
        timings = dict(result.get("timings") or {})
        detection = _format_detection_compact(result.get("detection_stats"))
        debug_artifact = dict(result.get("duel_debug_artifact") or {})
        debug_path = debug_artifact.get("path") or "-"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result.get('round_index') or '-'))}</td>"
            f"<td>{html.escape(status_display_name(result.get('status')))}</td>"
            f"<td>{html.escape(failure_reason_display_name(result.get('failure_reason')))}</td>"
            f"<td>{html.escape(_format_seconds(timings.get('total_sec')))}</td>"
            f"<td>{html.escape(_format_seconds(timings.get('hook_wait_sec')))}</td>"
            f"<td>{html.escape(_format_seconds(timings.get('duel_sec')))}</td>"
            f"<td>{detection}</td>"
            f"<td>{html.escape(str(debug_path))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "<p><b>每轮摘要</b></p>" + "".join(rows)


def _render_round_details(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for result in results:
        round_index = result.get("round_index") or "-"
        title = (
            f"<p><b>第 {html.escape(str(round_index))} 轮详情</b> "
            f"{html.escape(status_display_name(result.get('status')))}</p>"
        )
        blocks.append(title)
        blocks.append(_render_detection_stats(result.get("detection_stats")))
        blocks.append(_render_phase_trace_tail(result.get("phase_trace") or []))
    return "".join(blocks)


def _render_detection_stats(stats: Any) -> str:
    if not isinstance(stats, Mapping):
        return "<p>对抗识别统计：-</p>"
    reason_sec = dict(stats.get("reason_sec") or {})
    reason_text = " / ".join(
        f"{detection_reason_display_name(reason)} {duration}s" for reason, duration in reason_sec.items()
    )
    return "".join(
        [
            _kv_html("对抗观测时长", _format_seconds(stats.get("observation_sec"))),
            _kv_html("采样次数", stats.get("samples")),
            _kv_html("蓝绿区识别率", _format_ratio((stats.get("zone") or {}).get("detected_ratio"))),
            _kv_html("指示线识别率", _format_ratio((stats.get("indicator") or {}).get("detected_ratio"))),
            _kv_html("原始指示线识别率", _format_ratio((stats.get("indicator_raw") or {}).get("detected_ratio"))),
            _kv_html("识别原因分布", reason_text or "-"),
        ]
    )


def _render_phase_trace_tail(trace: Iterable[Mapping[str, Any]]) -> str:
    tail = [dict(item) for item in list(trace)[-10:] if isinstance(item, Mapping)]
    if not tail:
        return "<p>阶段轨迹：-</p>"
    rows = [
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<tr><th>时间</th><th>阶段</th><th>动作</th><th>误差</th><th>区间</th><th>指示线</th><th>检测</th></tr>"
    ]
    for entry in tail:
        zone = "-"
        if entry.get("zone_left") is not None and entry.get("zone_right") is not None:
            zone = f"{entry.get('zone_left')} - {entry.get('zone_right')}"
        detection = _format_trace_detection(entry)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry.get('t_ms', '-')))} ms</td>"
            f"<td>{html.escape(phase_display_name(entry.get('phase')))}</td>"
            f"<td>{html.escape(_trace_note_display_name(entry.get('note')))}</td>"
            f"<td>{html.escape(str(entry.get('error_px', '-')))}</td>"
            f"<td>{html.escape(zone)}</td>"
            f"<td>{html.escape(str(entry.get('indicator_x', '-')))}</td>"
            f"<td>{html.escape(detection)}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "<p><b>阶段轨迹（最后 10 条）</b></p>" + "".join(rows)


def _format_detection_compact(stats: Any) -> str:
    if not isinstance(stats, Mapping):
        return "-"
    zone = _format_ratio((stats.get("zone") or {}).get("detected_ratio"))
    indicator = _format_ratio((stats.get("indicator") or {}).get("detected_ratio"))
    raw = _format_ratio((stats.get("indicator_raw") or {}).get("detected_ratio"))
    return "<br/>".join(
        [
            f"蓝绿区 {html.escape(zone)}",
            f"指示线 {html.escape(indicator)}",
            f"原始线 {html.escape(raw)}",
        ]
    )


def _format_trace_detection(entry: Mapping[str, Any]) -> str:
    parts: list[str] = []
    if entry.get("duel_reason"):
        parts.append(detection_reason_display_name(entry.get("duel_reason")))
    if entry.get("zone_detected") is not None:
        parts.append(f"蓝绿区 {'有' if entry.get('zone_detected') else '无'}")
    if entry.get("indicator_detected") is not None:
        parts.append(f"指示线 {'有' if entry.get('indicator_detected') else '无'}")
    return " / ".join(parts) if parts else "-"


def _trace_note_display_name(note: Any) -> str:
    normalized = str(note or "").strip()
    if not normalized:
        return "-"
    if normalized.startswith("control_"):
        reason = normalized.removeprefix("control_")
        return f"控制：{CONTROL_REASON_LABELS.get(reason, reason)}"
    return TRACE_NOTE_LABELS.get(normalized, normalized)


def _format_seconds(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return str(value)


def _format_ratio(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _cafe_level_outcome_display_name(outcome: Any) -> str:
    normalized = str(outcome or "").strip().lower()
    if not normalized:
        return "-"
    return {
        "success": "成功",
        "failure": "失败",
        "unknown": "未知",
    }.get(normalized, normalized)


def _perf_label(key: Any) -> str:
    normalized = str(key or "")
    labels = {
        "start_game_click": "点击开始游戏耗时",
        "start_game_click_count": "点击开始游戏次数",
        "start_game_delay": "开始后等待",
        "level_start_wait": "等待关卡开始",
        "capture_before_stock": "补货前截图",
        "capture_before_scan": "扫描前截图",
        "capture_after_order": "订单后截图",
        "capture_count": "截图次数",
        "capture_reuse_before_scan_count": "复用扫描前截图次数",
        "level_end_check_count": "关卡结束检测次数",
        "level_end_check_before_stock": "补货前关卡结束检测",
        "level_end_check_before_scan": "扫描前关卡结束检测",
        "level_end_check_after_order": "订单后关卡结束检测",
        "stock_visual_sync_count": "库存视觉同步次数",
        "stock_visual_correction_count": "库存视觉校正次数",
        "restock_before_scan": "扫描前补货耗时",
        "restock_after_order": "订单后补货耗时",
        "restock_stock_started_count": "启动补货次数",
        "order_scan": "订单扫描",
        "order_scan_count": "订单扫描次数",
        "order_scan_locator": "订单定位",
        "order_scan_classify": "订单分类",
        "order_scan_fallback": "全图兜底扫描",
        "order_scan_fallback_count": "全图兜底次数",
        "order_scan_fallback_skipped_count": "兜底跳过次数",
        "no_order_scan_count": "无订单扫描次数",
        "no_order_sleep": "无订单等待",
        "recipe_execute": "配方执行耗时",
        "level_end_pending_sleep": "关卡结束确认等待",
        "stock_async_wait": "异步补货等待",
        "stock_async_wait_count": "异步补货等待次数",
        "stock_batch_ready_count": "补货完成次数",
        "order_pacing_wait": "订单节奏等待",
        "order_pacing_wait_count": "订单节奏等待次数",
        "min_order_interval_wait_count": "订单间隔等待次数",
        "min_order_duration_wait_count": "订单最短耗时等待次数",
        "order_guard_active_block_count": "订单守卫活跃阻挡点",
        "order_guard_blocked_order_count": "订单守卫阻挡订单",
        "order_guard_deferred_all_count": "订单守卫暂缓全部订单",
        "order_guard_safe_selection_count": "订单守卫改选安全订单",
        "order_guard_sleep": "订单守卫暂缓等待",
        "fake_customer_scan_before_stock": "补货前假顾客扫描",
        "fake_customer_scan_before_order": "订单前假顾客扫描",
        "fake_customer_scan_count": "假顾客扫描次数",
        "fake_customer_detected_count": "假顾客候选数量",
        "fake_customer_hammer_click_count": "敲走假顾客次数",
        "fake_customer_cooldown_skip_count": "假顾客冷却跳过次数",
        "stock_visual_sync_before_scan": "扫描前库存视觉同步",
        "stock_visual_sync_after_order": "订单后库存视觉同步",
        "restock_before_scan_call_count": "扫描前补货调用次数",
        "restock_after_order_call_count": "订单后补货调用次数",
    }
    return labels.get(normalized, normalized)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    return bool(value)


def render_history_summary_html(detail: Mapping[str, Any] | None) -> str:
    payload = dict(detail or {})
    header = [
        _kv_html("运行编号", payload.get("cid")),
        _kv_html("任务", task_display_name(payload.get("task_name"))),
        _kv_html("状态", status_display_name(payload.get("status"))),
        _kv_html("耗时（毫秒）", payload.get("duration_ms")),
    ]
    return "".join(header) + render_task_result_html(str(payload.get("task_name") or ""), payload)


def format_nodes_timeline(nodes: Iterable[Mapping[str, Any]] | None) -> str:
    rows: list[str] = []
    for node in nodes or []:
        node_id = str(node.get("node_id") or "-")
        status = status_display_name(node.get("status"))
        duration_ms = node.get("duration_ms")
        rows.append(f"{node_id:20} {status:10} {duration_ms!s:>8} 毫秒")
    return "\n".join(rows) if rows else "暂无节点时间线。"


def format_event_stream(events: Iterable[Mapping[str, Any]] | None) -> str:
    rows = [render_json(event) for event in events or []]
    return "\n\n".join(rows) if rows else "当前运行没有缓存事件流。"


def _kv_html(label: str, value: Any) -> str:
    if value is None or value == "":
        rendered = "-"
    elif isinstance(value, (dict, list)):
        rendered = f"<pre>{html.escape(render_json(value))}</pre>"
    else:
        rendered = html.escape(str(value))
    return f"<p><b>{html.escape(label)}:</b> {rendered}</p>"

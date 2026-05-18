from __future__ import annotations

from dataclasses import dataclass

from .logic import (
    TASK_AUTO_LOOP,
    TASK_CAFE_AUTO_LOOP,
    TASK_COMBAT_AUTO_LOOP,
    TASK_MAHJONG_AUTO_LOOP,
    TASK_ONE_CAFE_REVENUE_RESTOCK,
    TASK_PIANO_PLAY_MIDI,
    TASK_RHYTHM_AUTO_LOOP,
    TASK_TETROMINOES_AUTO_LOOP,
)


WORKBENCH_TASKS: dict[str, dict[str, str]] = {
    "fishing": {
        "label": "钓鱼",
        "category": "小游戏",
        "kind": "循环采集",
        "summary": "钓鱼 / 卖鱼 / 补饵",
        "task_ref": TASK_AUTO_LOOP,
        "description": "自动循环钓鱼，只保留最大轮数快捷输入。",
    },
    "cafe": {
        "label": "沙威玛",
        "category": "小游戏",
        "kind": "关卡代打",
        "summary": "订单识别 / 制作 / 补货",
        "task_ref": TASK_CAFE_AUTO_LOOP,
        "description": "自动执行沙威玛小游戏，识别订单、补货并制作餐品。",
    },
    "one_cafe": {
        "label": "一咖舍",
        "category": "日常领取",
        "kind": "收益处理",
        "summary": "领取收益 / 自动补货",
        "task_ref": TASK_ONE_CAFE_REVENUE_RESTOCK,
        "description": "领取一咖舍收益，并按配置执行补货，结束后返回世界场景。",
    },
    "mahjong": {
        "label": "麻将",
        "category": "小游戏",
        "kind": "对局辅助",
        "summary": "自动开关 / 结算检测",
        "task_ref": TASK_MAHJONG_AUTO_LOOP,
        "description": "使用异环内置胡、碰、出自动开关完成麻将局，并在结算排行页出现时结束任务。",
    },
    "combat": {
        "label": "战斗",
        "category": "战斗",
        "kind": "自动作战",
        "summary": "锁定目标 / 技能循环 / 闪避",
        "task_ref": TASK_COMBAT_AUTO_LOOP,
        "description": "持续监控敌人血条并自动战斗，可选启用基于音频样本的自动闪避。",
    },
    "tetrominoes": {
        "label": "俄罗斯方块",
        "category": "小游戏",
        "kind": "棋盘求解",
        "summary": "棋盘识别 / 落点规划",
        "task_ref": TASK_TETROMINOES_AUTO_LOOP,
        "description": "自动运行异环俄罗斯方块小游戏，按当前识别档案持续识别棋盘并规划落点。",
    },
    "rhythm": {
        "label": "四键音游",
        "category": "小游戏",
        "kind": "节奏辅助",
        "summary": "四轨识别 / 自动按键",
        "task_ref": TASK_RHYTHM_AUTO_LOOP,
        "description": "自动运行异环鼓组四键下落音游，识别 D/F/J/K 判定线附近音符并按键。",
    },
    "piano": {
        "label": "自动弹钢琴",
        "category": "小游戏",
        "kind": "演奏工具",
        "summary": "MIDI 解析 / 自动按键",
        "task_ref": TASK_PIANO_PLAY_MIDI,
        "description": "解析 MIDI 文件并在异环钢琴小游戏中自动演奏，支持严格模式和滚奏拆分。",
    },
}
WORKBENCH_TASK_REFS = tuple(item["task_ref"] for item in WORKBENCH_TASKS.values())

WORKBENCH_TASK_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("日常领取", ("one_cafe",)),
    ("战斗", ("combat",)),
    ("小游戏", ("fishing", "cafe", "mahjong", "tetrominoes", "rhythm", "piano")),
)


@dataclass(frozen=True)
class AuxiliaryToolSpec:
    label: str
    description: str


AUXILIARY_TOOLS: dict[str, AuxiliaryToolSpec] = {
    "map_overlay": AuxiliaryToolSpec(
        label="大地图点位悬浮层",
        description="在异环大地图上叠加 ZeroLuck 离线点位，辅助找传送点、谜石、货币和收集物。",
    ),
}

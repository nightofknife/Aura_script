"""异环方案包元数据与脚手架辅助服务。"""

from __future__ import annotations

from typing import Any, Dict, List

from packages.aura_core.api import service_info


@service_info(
    alias="yihuan_plan",
    public=True,
    singleton=True,
    description="暴露异环方案包的脚手架元数据和约定信息。",
)
class YihuanPlanService:
    def __init__(self) -> None:
        self._known_titles: List[str] = [
            "异环",
            "Neverness to Everness",
        ]
        self._reserved_states: List[str] = [
            "plan_loaded",
            "launcher",
            "login",
            "world",
            "menu",
            "combat",
            "vehicle",
        ]
        self._reserved_task_groups: List[str] = [
            "bootstrap",
            "checks",
            "commission",
            "daily",
            "navigation",
            "shop",
        ]

    def describe(self, scenario_tag: str = "bootstrap") -> Dict[str, Any]:
        return {
            "plan_name": "yihuan",
            "game_title": "异环",
            "official_title_en": "Neverness to Everness",
            "scenario_tag": scenario_tag,
            "default_input_profile": "default_pc",
            "window_title_regex": "(异环|Neverness to Everness)",
            "known_titles": list(self._known_titles),
            "reserved_states": list(self._reserved_states),
            "reserved_task_groups": list(self._reserved_task_groups),
            "notes": [
                "当前方案包面向 Windows 桌面客户端。",
                "除钓鱼能力外，其余玩法自动化仍会在后续任务流程明确后逐步补齐。",
            ],
        }

    def is_loaded(self) -> bool:
        return True

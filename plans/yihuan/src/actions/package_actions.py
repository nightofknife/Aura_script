"""异环方案包级辅助动作。"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from packages.aura_core.api import action_info, requires_services

from ..services.yihuan_plan_service import YihuanPlanService


@action_info(
    name="yihuan_plan_info",
    public=True,
    read_only=True,
    description="返回异环方案包的脚手架元数据。",
)
@requires_services(yihuan_plan="yihuan_plan")
def yihuan_plan_info(
    scenario_tag: str = "bootstrap",
    yihuan_plan: YihuanPlanService | None = None,
) -> Dict[str, Any]:
    if yihuan_plan is None:
        raise RuntimeError("yihuan_plan 服务不可用。")
    return yihuan_plan.describe(scenario_tag=scenario_tag)


@action_info(
    name="yihuan_plan_loaded",
    public=True,
    read_only=True,
    description="当异环方案包加载成功时返回 True。",
)
@requires_services(yihuan_plan="yihuan_plan")
def yihuan_plan_loaded(
    yihuan_plan: YihuanPlanService | None = None,
) -> bool:
    if yihuan_plan is None:
        raise RuntimeError("yihuan_plan 服务不可用。")
    return bool(yihuan_plan.is_loaded())


def _normalize_runtime_probe_payload(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    data = dict(payload or {})
    return {
        "ok": bool(data.get("ok", False)),
        "provider": data.get("provider"),
        "family": data.get("family"),
        "target": dict(data.get("target") or {}),
        "capture": dict(data.get("capture") or {}),
        "input": dict(data.get("input") or {}),
        "warnings": list(data.get("warnings") or []),
        "code": data.get("code"),
        "message": data.get("message"),
    }


@action_info(
    name="yihuan_runtime_probe",
    public=True,
    read_only=True,
    description="基于当前方案上下文探测异环运行时目标。",
)
@requires_services(screen="plans/aura_base/screen")
def yihuan_runtime_probe(
    screen: Any = None,
) -> Dict[str, Any]:
    if screen is None:
        raise RuntimeError("screen 服务不可用。")

    try:
        payload = screen.self_check()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "ok": False,
            "provider": None,
            "family": None,
            "target": {},
            "capture": {},
            "input": {},
            "warnings": [],
            "code": type(exc).__name__,
            "message": str(exc),
        }
    return _normalize_runtime_probe_payload(payload)

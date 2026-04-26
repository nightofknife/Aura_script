from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._shared import (
    add_common_output_flag,
    add_common_windows_target_args,
    build_overlay_config,
    build_runtime_overlay_from_args,
    maybe_print,
    normalize_payload,
    plan_scope,
    suppress_framework_console_logs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List candidate windows, inspect a specific hwnd, or preview the configured runtime target."
    )
    parser.add_argument("--plan", help="Optional current plan context.")
    parser.add_argument("--hwnd-detail", type=int, help="Describe a specific hwnd.")
    parser.add_argument("--limit", type=int, default=30, help="Maximum listed windows.")
    parser.add_argument("--include-children", action="store_true", help="Include child windows in enumeration.")
    parser.add_argument("--include-empty-title", action="store_true", help="Include empty-title windows.")
    parser.add_argument("--resolve-target", action="store_true", help="Preview the configured/overridden target resolution.")
    add_common_windows_target_args(parser)
    add_common_output_flag(parser)
    return parser


def collect_window_inspection(args: argparse.Namespace) -> dict[str, Any]:
    from plans.aura_base.src.services.target_runtime_service import TargetRuntimeService
    from plans.aura_base.src.services.windows_diagnostics_service import WindowsDiagnosticsService

    overlay = build_runtime_overlay_from_args(args)
    config = build_overlay_config(plan_name=args.plan, overlay=overlay)

    with suppress_framework_console_logs(), plan_scope(args.plan):
        target_runtime = TargetRuntimeService(config)
        diagnostics = WindowsDiagnosticsService(config, target_runtime)
        payload: dict[str, Any] = {
            "listed": diagnostics.list_candidate_windows(
                require_visible=True if args.require_visible is None else bool(args.require_visible),
                include_children=args.include_children or args.allow_child_window,
                include_empty_title=args.include_empty_title or args.allow_empty_title,
                limit=args.limit,
            )
        }
        if args.hwnd_detail is not None:
            payload["detail"] = diagnostics.describe_target_window(args.hwnd_detail)
        if args.resolve_target:
            target_cfg = {}
            if isinstance(overlay, dict):
                target_cfg = dict((overlay.get("runtime") or {}).get("target") or {})
            payload["resolved_target"] = diagnostics.resolve_target_preview(target_overrides=target_cfg)
        return normalize_payload(payload)


def render_text(payload: dict[str, Any]) -> str:
    lines = []
    listed = payload.get("listed") or []
    lines.append(f"Listed windows: {len(listed)}")
    for item in listed:
        lines.append(
            f"- hwnd={item.get('hwnd')} title={item.get('title') or '-'} "
            f"process={item.get('process_name') or '-'} class={item.get('class_name') or '-'} "
            f"client={item.get('client_rect')} fg={item.get('foreground')}"
        )
    detail = payload.get("detail")
    if isinstance(detail, dict):
        lines.append("")
        lines.append("Detail:")
        lines.append(
            f"- hwnd={detail.get('hwnd')} title={detail.get('title') or '-'} "
            f"pid={detail.get('pid')} process={detail.get('process_name') or '-'}"
        )
        lines.append(
            f"- class={detail.get('class_name') or '-'} visible={detail.get('visible')} "
            f"enabled={detail.get('enabled')} foreground={detail.get('foreground')}"
        )
        lines.append(
            f"- client={detail.get('client_rect')} client_screen={detail.get('client_rect_screen')} "
            f"window={detail.get('window_rect_screen')}"
        )
    resolved = payload.get("resolved_target")
    if isinstance(resolved, dict):
        target = resolved.get("target") or {}
        lines.append("")
        lines.append("Resolved target:")
        lines.append(
            f"- hwnd={target.get('hwnd')} title={target.get('title') or '-'} "
            f"process={target.get('process_name') or '-'} class={target.get('class_name') or '-'}"
        )
    return "\n".join(lines)


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = collect_window_inspection(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    maybe_print(payload, as_json=args.json, text_renderer=render_text)
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()

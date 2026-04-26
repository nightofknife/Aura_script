from __future__ import annotations

try:
    from .app import launch_yihuan_gui
except ModuleNotFoundError as exc:
    missing_name = str(getattr(exc, "name", "") or "")
    if missing_name.startswith("PySide6"):
        raise SystemExit(
            "异环 GUI 依赖 PySide6，请先执行：pip install -r requirements/gui.txt"
        ) from exc
    raise


if __name__ == "__main__":
    raise SystemExit(launch_yihuan_gui())

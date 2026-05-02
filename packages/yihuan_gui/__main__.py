from __future__ import annotations

from multiprocessing import freeze_support
import os
from pathlib import Path
import sys


def _infer_release_root() -> Path:
    if getattr(sys, "frozen", False):
        runtime_dir = Path(sys.executable).resolve().parent
        if runtime_dir.name.lower() == "runtime":
            return runtime_dir.parent
        return runtime_dir
    return Path(os.environ.get("AURA_BASE_PATH") or Path.cwd()).resolve()


def _prepare_release_environment() -> Path:
    root = _infer_release_root()
    os.environ.setdefault("AURA_BASE_PATH", str(root))
    if getattr(sys, "frozen", False):
        os.environ.setdefault("PYTHONNOUSERSITE", "1")

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def _load_gui_launcher():
    try:
        from .app import launch_yihuan_gui
    except ModuleNotFoundError as exc:
        missing_name = str(getattr(exc, "name", "") or "")
        if missing_name.startswith("PySide6"):
            raise SystemExit(
                "Yihuan GUI requires PySide6. Install it with: pip install -r requirements/gui.txt"
            ) from exc
        raise
    return launch_yihuan_gui


def _run_self_check() -> int:
    root = _prepare_release_environment()
    try:
        import PySide6  # noqa: F401
        _load_gui_launcher()
    except Exception as exc:  # noqa: BLE001
        print(f"AuraYihuanRuntime self-check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if getattr(sys, "frozen", False):
        required_paths = [
            root / "config.yaml",
            root / "plans",
            root / "models" / "ocr" / "ppocrv5_server" / "ocr.meta.json",
        ]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            for path in missing_paths:
                print(f"AuraYihuanRuntime self-check missing path: {path}", file=sys.stderr)
            return 2

    print(f"AuraYihuanRuntime self-check OK: base={root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _prepare_release_environment()
    if "--self-check" in args:
        return _run_self_check()

    launch_yihuan_gui = _load_gui_launcher()
    return int(launch_yihuan_gui())


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

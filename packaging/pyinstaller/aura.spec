# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for zero-framework-change Aura packaging.

This spec intentionally packages the framework runtime only.
Plans stay external in the assembled release root so users can edit them.
"""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


# PyInstaller executes spec files via `exec`, so `__file__` is not guaranteed.
# The build wrapper runs PyInstaller from the repository root.
ROOT = Path.cwd().resolve()
ENTRYPOINT = ROOT / "cli.py"

# PaddleX performs a connectivity check on import. Disable it during build-time
# metadata collection so packaging stays deterministic and offline-friendly.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

INCLUDE_NVIDIA = os.environ.get("AURA_PKG_INCLUDE_NVIDIA", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _installed_distributions(*names: str) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for name in names:
        normalized = str(name).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        try:
            metadata.version(normalized)
        except metadata.PackageNotFoundError:
            continue
        resolved.append(normalized)
        seen.add(key)
    return resolved


datas = []
binaries = []
hiddenimports = ["win32timezone"]
datas.append((str(ROOT / "docs" / "schemas" / "task-schema.json"), "docs/schemas"))

# Aura uses several lazy package-level exports and importlib-based lookups.
# Packaging the full framework submodule graph is simpler and more reliable
# than chasing each delayed import one by one.
hiddenimports += collect_submodules("packages.aura_core")
hiddenimports += collect_submodules("packages.aura_game")
hiddenimports += [
    "PIL.ImageGrab",
    "win32api",
    "win32con",
    "win32gui",
    "win32ui",
    "win32process",
    "pythoncom",
    "pywintypes",
]


def _collect_optional_package(name: str) -> None:
    global datas, binaries, hiddenimports
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(name)
    except Exception:
        return
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports


for optional_pkg in (
    "cv2",
    "dxcam",
    "screeninfo",
    "av",
    "dotenv",
    "yaml",
    "paddleocr",
    "paddlex",
):
    _collect_optional_package(optional_pkg)

# PaddleOCR 3.x depends on PaddleX package data at runtime.
datas += collect_data_files("paddlex")
datas += collect_data_files("paddleocr")

# Paddle runtime binaries are loaded dynamically and need explicit collection.
binaries += collect_dynamic_libs("paddle")

if INCLUDE_NVIDIA:
    binaries += collect_dynamic_libs("nvidia")

for dist_name in _installed_distributions(
    "paddleocr",
    "paddlex",
    "paddlepaddle-gpu",
    "paddlepaddle",
    "numpy",
    "opencv-python",
    "av",
    "pywin32",
    "screeninfo",
    "psutil",
    "watchdog",
    "cachetools",
    "fastjsonschema",
    "python-dotenv",
    "prompt_toolkit",
):
    datas += copy_metadata(dist_name)


a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "pyinstaller" / "rthook_aura_external_plans.py")],
    excludes=["tests"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="aura",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="aura",
)

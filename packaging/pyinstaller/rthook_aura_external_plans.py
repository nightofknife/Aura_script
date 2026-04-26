"""Optional runtime diagnostics for external plans imports.

Enabled only when AURA_PKG_DEBUG_IMPORT=1 is set.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path


base_path = os.environ.get("AURA_BASE_PATH")
if base_path:
    resolved = str(Path(base_path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

if os.environ.get("AURA_PKG_DEBUG_IMPORT", "").strip().lower() in {"1", "true", "yes", "on"}:
    try:
        module = importlib.import_module("plans.aura_base.src.services.app_provider_service")
        print(f"[aura-pkg-debug] external plan import ok: {getattr(module, '__file__', '<frozen>')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[aura-pkg-debug] external plan import failed: {exc!r}")
        traceback.print_exc()

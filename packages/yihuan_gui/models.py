from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PendingLaunch:
    task_ref: str
    inputs: dict[str, Any]
    selected_task_id: str
    remaining_sec: int
    total_sec: int

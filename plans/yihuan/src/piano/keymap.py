"""Key mapping helpers for Yihuan's piano mini-game."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class PianoRegister(str, Enum):
    HIGH = "high"
    MID = "mid"
    LOW = "low"


class PianoModifier(str, Enum):
    NONE = "none"
    SHIFT = "shift"
    CTRL = "ctrl"


class PianoConflictPolicy(str, Enum):
    STRICT = "strict"
    ROLL = "roll"


@dataclass(frozen=True, slots=True)
class PianoBinding:
    register: PianoRegister
    degree: str
    key: str
    modifier: PianoModifier
    slot_index: int

    @property
    def physical_slot(self) -> str:
        return self.key.lower()

    @property
    def display(self) -> str:
        if self.modifier is PianoModifier.NONE:
            return self.key.upper()
        return f"{self.modifier.value.title()}+{self.key.upper()}"


@dataclass(frozen=True, slots=True)
class PianoNoteEvent:
    register: PianoRegister
    degree: str
    start_ms: int = 0
    duration_ms: int = 0
    source_index: int = 0
    midi_note: int | None = None
    velocity: int | None = None

    def binding(self) -> PianoBinding:
        return get_piano_binding(self.register, self.degree)


@dataclass(frozen=True, slots=True)
class PianoChordConflict:
    physical_slot: str
    notes: tuple[PianoNoteEvent, ...]

    @property
    def display(self) -> str:
        labels = ", ".join(
            f"{note.register.value}:{note.degree}"
            for note in sorted(self.notes, key=lambda item: item.source_index)
        )
        return f"shared key {self.physical_slot.upper()} for [{labels}]"


class UnplayablePianoChordError(ValueError):
    """Raised when a note group cannot be played within one chord onset."""

    def __init__(self, conflicts: Iterable[PianoChordConflict]):
        self.conflicts = tuple(conflicts)
        details = "; ".join(conflict.display for conflict in self.conflicts)
        super().__init__(f"Chord contains mutually exclusive piano notes: {details}")


def _binding(
    register: PianoRegister,
    degree: str,
    key: str,
    modifier: PianoModifier,
    slot_index: int,
) -> PianoBinding:
    return PianoBinding(
        register=register,
        degree=degree,
        key=key,
        modifier=modifier,
        slot_index=slot_index,
    )


PIANO_KEYMAP: dict[tuple[PianoRegister, str], PianoBinding] = {
    (PianoRegister.HIGH, "1"): _binding(PianoRegister.HIGH, "1", "q", PianoModifier.NONE, 0),
    (PianoRegister.HIGH, "#1"): _binding(PianoRegister.HIGH, "#1", "q", PianoModifier.SHIFT, 0),
    (PianoRegister.HIGH, "2"): _binding(PianoRegister.HIGH, "2", "w", PianoModifier.NONE, 1),
    (PianoRegister.HIGH, "b3"): _binding(PianoRegister.HIGH, "b3", "e", PianoModifier.CTRL, 2),
    (PianoRegister.HIGH, "3"): _binding(PianoRegister.HIGH, "3", "e", PianoModifier.NONE, 2),
    (PianoRegister.HIGH, "4"): _binding(PianoRegister.HIGH, "4", "r", PianoModifier.NONE, 3),
    (PianoRegister.HIGH, "#4"): _binding(PianoRegister.HIGH, "#4", "r", PianoModifier.SHIFT, 3),
    (PianoRegister.HIGH, "5"): _binding(PianoRegister.HIGH, "5", "t", PianoModifier.NONE, 4),
    (PianoRegister.HIGH, "#5"): _binding(PianoRegister.HIGH, "#5", "t", PianoModifier.SHIFT, 4),
    (PianoRegister.HIGH, "6"): _binding(PianoRegister.HIGH, "6", "y", PianoModifier.NONE, 5),
    (PianoRegister.HIGH, "b7"): _binding(PianoRegister.HIGH, "b7", "u", PianoModifier.CTRL, 6),
    (PianoRegister.HIGH, "7"): _binding(PianoRegister.HIGH, "7", "u", PianoModifier.NONE, 6),
    (PianoRegister.MID, "1"): _binding(PianoRegister.MID, "1", "a", PianoModifier.NONE, 0),
    (PianoRegister.MID, "#1"): _binding(PianoRegister.MID, "#1", "a", PianoModifier.SHIFT, 0),
    (PianoRegister.MID, "2"): _binding(PianoRegister.MID, "2", "s", PianoModifier.NONE, 1),
    (PianoRegister.MID, "b3"): _binding(PianoRegister.MID, "b3", "d", PianoModifier.CTRL, 2),
    (PianoRegister.MID, "3"): _binding(PianoRegister.MID, "3", "d", PianoModifier.NONE, 2),
    (PianoRegister.MID, "4"): _binding(PianoRegister.MID, "4", "f", PianoModifier.NONE, 3),
    (PianoRegister.MID, "#4"): _binding(PianoRegister.MID, "#4", "f", PianoModifier.SHIFT, 3),
    (PianoRegister.MID, "5"): _binding(PianoRegister.MID, "5", "g", PianoModifier.NONE, 4),
    (PianoRegister.MID, "#5"): _binding(PianoRegister.MID, "#5", "g", PianoModifier.SHIFT, 4),
    (PianoRegister.MID, "6"): _binding(PianoRegister.MID, "6", "h", PianoModifier.NONE, 5),
    (PianoRegister.MID, "b7"): _binding(PianoRegister.MID, "b7", "j", PianoModifier.CTRL, 6),
    (PianoRegister.MID, "7"): _binding(PianoRegister.MID, "7", "j", PianoModifier.NONE, 6),
    (PianoRegister.LOW, "1"): _binding(PianoRegister.LOW, "1", "z", PianoModifier.NONE, 0),
    (PianoRegister.LOW, "#1"): _binding(PianoRegister.LOW, "#1", "z", PianoModifier.SHIFT, 0),
    (PianoRegister.LOW, "2"): _binding(PianoRegister.LOW, "2", "x", PianoModifier.NONE, 1),
    (PianoRegister.LOW, "b3"): _binding(PianoRegister.LOW, "b3", "c", PianoModifier.CTRL, 2),
    (PianoRegister.LOW, "3"): _binding(PianoRegister.LOW, "3", "c", PianoModifier.NONE, 2),
    (PianoRegister.LOW, "4"): _binding(PianoRegister.LOW, "4", "v", PianoModifier.NONE, 3),
    (PianoRegister.LOW, "#4"): _binding(PianoRegister.LOW, "#4", "v", PianoModifier.SHIFT, 3),
    (PianoRegister.LOW, "5"): _binding(PianoRegister.LOW, "5", "b", PianoModifier.NONE, 4),
    (PianoRegister.LOW, "#5"): _binding(PianoRegister.LOW, "#5", "b", PianoModifier.SHIFT, 4),
    (PianoRegister.LOW, "6"): _binding(PianoRegister.LOW, "6", "n", PianoModifier.NONE, 5),
    (PianoRegister.LOW, "b7"): _binding(PianoRegister.LOW, "b7", "m", PianoModifier.CTRL, 6),
    (PianoRegister.LOW, "7"): _binding(PianoRegister.LOW, "7", "m", PianoModifier.NONE, 6),
}


def iter_piano_bindings() -> tuple[PianoBinding, ...]:
    return tuple(PIANO_KEYMAP[key] for key in sorted(PIANO_KEYMAP, key=lambda item: (item[0].value, item[1])))


def get_piano_bindings(register: PianoRegister | str, degree: str) -> tuple[PianoBinding, ...]:
    return (get_piano_binding(register, degree),)


def get_piano_binding(register: PianoRegister | str, degree: str) -> PianoBinding:
    register_value = register if isinstance(register, PianoRegister) else PianoRegister(str(register).strip().lower())
    normalized_degree = str(degree).strip()
    binding = PIANO_KEYMAP.get((register_value, normalized_degree))
    if binding is None:
        raise KeyError(f"Unsupported piano note mapping: register={register_value.value}, degree={normalized_degree}")
    return binding


def detect_chord_conflicts(notes: Iterable[PianoNoteEvent]) -> tuple[PianoChordConflict, ...]:
    grouped: dict[str, list[PianoNoteEvent]] = {}
    for note in notes:
        grouped.setdefault(note.binding().physical_slot, []).append(note)

    conflicts: list[PianoChordConflict] = []
    for physical_slot, grouped_notes in grouped.items():
        unique_labels = {(item.register, item.degree) for item in grouped_notes}
        if len(unique_labels) <= 1:
            continue
        conflicts.append(
            PianoChordConflict(
                physical_slot=physical_slot,
                notes=tuple(sorted(grouped_notes, key=lambda item: item.source_index)),
            )
        )
    return tuple(conflicts)


def ensure_playable_chord(
    notes: Iterable[PianoNoteEvent],
    *,
    policy: PianoConflictPolicy = PianoConflictPolicy.STRICT,
) -> tuple[PianoNoteEvent, ...]:
    resolved_notes = tuple(notes)
    conflicts = detect_chord_conflicts(resolved_notes)
    if not conflicts:
        return resolved_notes
    if policy is PianoConflictPolicy.ROLL:
        return resolved_notes
    raise UnplayablePianoChordError(conflicts)

from __future__ import annotations

import pytest

from plans.yihuan.src.piano.keymap import (
    PianoConflictPolicy,
    PianoNoteEvent,
    PianoRegister,
    UnplayablePianoChordError,
    detect_chord_conflicts,
    ensure_playable_chord,
    get_piano_binding,
    get_piano_bindings,
)


def test_natural_note_bindings_match_visible_keyboard_layout():
    assert get_piano_binding(PianoRegister.HIGH, "1").display == "Q"
    assert get_piano_binding(PianoRegister.HIGH, "7").display == "U"
    assert get_piano_binding(PianoRegister.MID, "1").display == "A"
    assert get_piano_binding(PianoRegister.MID, "7").display == "J"
    assert get_piano_binding(PianoRegister.LOW, "1").display == "Z"
    assert get_piano_binding(PianoRegister.LOW, "7").display == "M"


def test_accidental_notes_use_expected_modifier_bindings():
    assert [binding.display for binding in get_piano_bindings(PianoRegister.HIGH, "#1")] == ["Shift+Q"]
    assert [binding.display for binding in get_piano_bindings(PianoRegister.MID, "#5")] == ["Shift+G"]
    assert [binding.display for binding in get_piano_bindings(PianoRegister.MID, "b3")] == ["Ctrl+D"]
    assert [binding.display for binding in get_piano_bindings(PianoRegister.LOW, "b7")] == ["Ctrl+M"]


def test_natural_and_modified_notes_can_coexist_when_keys_differ():
    notes = (
        PianoNoteEvent(register=PianoRegister.HIGH, degree="1"),
        PianoNoteEvent(register=PianoRegister.HIGH, degree="#4"),
        PianoNoteEvent(register=PianoRegister.HIGH, degree="6"),
    )

    assert detect_chord_conflicts(notes) == ()


def test_same_physical_key_is_reported_as_conflict():
    notes = (
        PianoNoteEvent(register=PianoRegister.HIGH, degree="1", source_index=0),
        PianoNoteEvent(register=PianoRegister.HIGH, degree="#1", source_index=1),
    )

    conflicts = detect_chord_conflicts(notes)

    assert len(conflicts) == 1
    assert conflicts[0].physical_slot == "q"
    assert "high:1" in conflicts[0].display
    assert "high:#1" in conflicts[0].display


def test_same_degree_duplicate_is_not_treated_as_conflict():
    notes = (
        PianoNoteEvent(register=PianoRegister.HIGH, degree="1", source_index=0),
        PianoNoteEvent(register=PianoRegister.HIGH, degree="1", source_index=1),
    )

    assert detect_chord_conflicts(notes) == ()


def test_ensure_playable_chord_raises_in_strict_mode():
    notes = (
        PianoNoteEvent(register=PianoRegister.HIGH, degree="1", source_index=0),
        PianoNoteEvent(register=PianoRegister.HIGH, degree="#1", source_index=1),
    )

    with pytest.raises(UnplayablePianoChordError) as excinfo:
        ensure_playable_chord(notes, policy=PianoConflictPolicy.STRICT)

    assert "mutually exclusive piano notes" in str(excinfo.value)


def test_ensure_playable_chord_allows_conflict_to_be_handled_later_in_roll_mode():
    notes = (
        PianoNoteEvent(register=PianoRegister.LOW, degree="1", source_index=0),
        PianoNoteEvent(register=PianoRegister.LOW, degree="#1", source_index=1),
    )

    resolved = ensure_playable_chord(notes, policy=PianoConflictPolicy.ROLL)

    assert resolved == notes

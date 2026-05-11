from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from plans.yihuan.src.actions.piano_actions import yihuan_piano_play_midi
from plans.yihuan.src.piano.keymap import PianoConflictPolicy, PianoNoteEvent, PianoRegister
from plans.yihuan.src.services.piano_service import YihuanPianoService


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "yihuan" / "piano"


def _vlq(value: int) -> bytes:
    buffer = [value & 0x7F]
    value >>= 7
    while value:
        buffer.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(buffer)


def _make_midi_file(path: Path, events: list[tuple[int, bytes]], *, division: int = 480) -> Path:
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + division.to_bytes(2, "big")
    track_payload = bytearray()
    track_payload.extend(_vlq(0))
    track_payload.extend(b"\xFF\x51\x03\x07\xA1\x20")
    last_tick = 0
    for tick, data in sorted(events, key=lambda item: item[0]):
        track_payload.extend(_vlq(tick - last_tick))
        track_payload.extend(data)
        last_tick = tick
    track_payload.extend(_vlq(0))
    track_payload.extend(b"\xFF\x2F\x00")
    track = b"MTrk" + len(track_payload).to_bytes(4, "big") + bytes(track_payload)
    path.write_bytes(header + track)
    return path


class _FakeApp:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str]] = []

    def focus_with_input(self) -> bool:
        self.actions.append(("focus_with_input", ""))
        return True

    def focus(self) -> bool:
        self.actions.append(("focus", ""))
        return True

    def key_down(self, key: str):
        self.actions.append(("down", key))

    def key_up(self, key: str):
        self.actions.append(("up", key))

    def release_all(self):
        self.actions.append(("release_all", ""))


def test_parse_midi_maps_middle_c_to_mid_register_one():
    service = YihuanPianoService()
    with TemporaryDirectory() as tmpdir:
        midi_path = _make_midi_file(
            Path(tmpdir) / "middle_c.mid",
            [
                (0, bytes([0x90, 60, 100])),
                (480, bytes([0x80, 60, 0])),
            ],
        )

        parsed = service.parse_midi_file(midi_path)

    assert parsed["ok"] is True
    assert parsed["note_count"] == 1
    assert parsed["notes"][0].register.value == "mid"
    assert parsed["notes"][0].degree == "1"


def test_parse_midi_maps_sharp_five_to_shift_layer():
    service = YihuanPianoService()
    with TemporaryDirectory() as tmpdir:
        midi_path = _make_midi_file(
            Path(tmpdir) / "sharp_five.mid",
            [
                (0, bytes([0x90, 68, 100])),
                (240, bytes([0x80, 68, 0])),
            ],
        )

        parsed = service.parse_midi_file(midi_path)

    assert parsed["ok"] is True
    assert parsed["unsupported_note_count"] == 0
    assert parsed["notes"][0].degree == "#5"
    assert parsed["notes"][0].register.value == "mid"


def test_strict_plan_plays_natural_keys_before_modified_keys():
    service = YihuanPianoService()
    with TemporaryDirectory() as tmpdir:
        midi_path = _make_midi_file(
            Path(tmpdir) / "natural_then_shift.mid",
            [
                (0, bytes([0x90, 62, 100])),
                (0, bytes([0x90, 61, 100])),
                (240, bytes([0x80, 62, 0])),
                (240, bytes([0x80, 61, 0])),
            ],
        )

        parsed = service.parse_midi_file(midi_path)
        plan = service.build_playback_plan(parsed["notes"], conflict_policy=PianoConflictPolicy.STRICT, start_delay_ms=0)

    assert plan["ok"] is True
    first_stage = [item for item in plan["action_plan"] if item["t_ms"] == 0]
    second_stage = [item for item in plan["action_plan"] if item["t_ms"] == 1]
    third_stage = [item for item in plan["action_plan"] if item["t_ms"] == 2]
    fourth_stage = [item for item in plan["action_plan"] if item["t_ms"] == 3]
    assert [item["kind"] for item in first_stage] == ["key_down"]
    assert first_stage[0]["key"] == "s"
    assert [item["kind"] for item in second_stage] == ["modifier_down"]
    assert second_stage[0]["key"] == "shift"
    assert [item["kind"] for item in third_stage] == ["key_down"]
    assert third_stage[0]["key"] == "a"
    assert [item["kind"] for item in fourth_stage] == ["modifier_up"]
    assert fourth_stage[0]["key"] == "shift"


def test_strict_plan_rejects_chord_that_shares_one_physical_key():
    service = YihuanPianoService()
    with TemporaryDirectory() as tmpdir:
        midi_path = _make_midi_file(
            Path(tmpdir) / "strict_conflict.mid",
            [
                (0, bytes([0x90, 60, 100])),
                (0, bytes([0x90, 61, 100])),
                (240, bytes([0x80, 60, 0])),
                (240, bytes([0x80, 61, 0])),
            ],
        )

        parsed = service.parse_midi_file(midi_path)
        plan = service.build_playback_plan(parsed["notes"], conflict_policy=PianoConflictPolicy.STRICT, start_delay_ms=0)

    assert plan["ok"] is False
    assert plan["failure_reason"] == "physical_key_conflict"


def test_roll_plan_serializes_conflicting_notes_into_fast_switches():
    service = YihuanPianoService()
    with TemporaryDirectory() as tmpdir:
        midi_path = _make_midi_file(
            Path(tmpdir) / "roll_conflict.mid",
            [
                (0, bytes([0x90, 60, 100])),
                (0, bytes([0x90, 61, 100])),
                (240, bytes([0x80, 60, 0])),
                (240, bytes([0x80, 61, 0])),
            ],
        )

        parsed = service.parse_midi_file(midi_path)
        plan = service.build_playback_plan(
            parsed["notes"],
            conflict_policy=PianoConflictPolicy.ROLL,
            roll_note_ms=30,
            start_delay_ms=0,
        )

    assert plan["ok"] is True
    key_downs = [item for item in plan["action_plan"] if item["kind"] == "key_down"]
    assert [item["t_ms"] for item in key_downs] == [0, 31]
    assert [item["key"] for item in key_downs] == ["a", "a"]


def test_roll_plan_batches_natural_sharp_and_flat_groups_in_order():
    service = YihuanPianoService()
    notes = [
        PianoNoteEvent(register=PianoRegister.MID, degree="1", start_ms=0, duration_ms=500, source_index=0, midi_note=60),
        PianoNoteEvent(register=PianoRegister.MID, degree="2", start_ms=0, duration_ms=500, source_index=1, midi_note=62),
        PianoNoteEvent(register=PianoRegister.MID, degree="#4", start_ms=0, duration_ms=500, source_index=2, midi_note=66),
        PianoNoteEvent(register=PianoRegister.MID, degree="#5", start_ms=0, duration_ms=500, source_index=3, midi_note=68),
        PianoNoteEvent(register=PianoRegister.MID, degree="b3", start_ms=0, duration_ms=500, source_index=4, midi_note=63),
    ]
    plan = service.build_playback_plan(
        notes,
        conflict_policy=PianoConflictPolicy.ROLL,
        roll_note_ms=10,
        start_delay_ms=0,
    )

    assert plan["ok"] is True
    key_downs = [item for item in plan["action_plan"] if item["kind"] == "key_down"]
    assert [(item["t_ms"], item["degree"]) for item in key_downs] == [
        (0, "1"),
        (0, "2"),
        (11, "#4"),
        (11, "#5"),
        (21, "b3"),
    ]


def test_action_dry_run_returns_playback_plan_for_midi_file():
    service = YihuanPianoService()
    app = _FakeApp()
    with TemporaryDirectory() as tmpdir:
        midi_path = _make_midi_file(
            Path(tmpdir) / "play_action.mid",
            [
                (0, bytes([0x90, 60, 100])),
                (480, bytes([0x80, 60, 0])),
            ],
        )

        result = yihuan_piano_play_midi(
            app=app,
            yihuan_piano=service,
            file_path=str(midi_path),
            dry_run=True,
            focus_window=False,
        )

    assert result["status"] == "success"
    assert result["parsed_summary"]["note_count"] == 1
    assert any(item["kind"] == "key_down" and item["key"] == "a" for item in result["action_plan"])
    assert app.actions == [("release_all", "")]


def test_fixture_no_roll_needed_succeeds_in_strict_mode():
    service = YihuanPianoService()

    parsed = service.parse_midi_file(FIXTURE_DIR / "no_roll_needed.mid")
    plan = service.build_playback_plan(
        parsed["notes"],
        conflict_policy=PianoConflictPolicy.STRICT,
        start_delay_ms=0,
    )

    assert parsed["ok"] is True
    assert plan["ok"] is True
    key_down_times = [item["t_ms"] for item in plan["action_plan"] if item["kind"] == "key_down"]
    assert key_down_times.count(0) >= 2


def test_fixture_roll_required_fails_in_strict_and_succeeds_in_roll_mode():
    service = YihuanPianoService()

    parsed = service.parse_midi_file(FIXTURE_DIR / "roll_required.mid")
    strict_plan = service.build_playback_plan(
        parsed["notes"],
        conflict_policy=PianoConflictPolicy.STRICT,
        start_delay_ms=0,
    )
    roll_plan = service.build_playback_plan(
        parsed["notes"],
        conflict_policy=PianoConflictPolicy.ROLL,
        roll_note_ms=30,
        start_delay_ms=0,
    )

    assert parsed["ok"] is True
    assert strict_plan["ok"] is False
    assert strict_plan["failure_reason"] == "physical_key_conflict"
    assert roll_plan["ok"] is True
    key_down_times = [item["t_ms"] for item in roll_plan["action_plan"] if item["kind"] == "key_down"]
    assert key_down_times == [0, 31]


def test_fixture_happy_birthday_uses_three_roll_batches_on_first_chord():
    service = YihuanPianoService()

    parsed = service.parse_midi_file(FIXTURE_DIR / "happy_birthday_tri_batch.mid")
    strict_plan = service.build_playback_plan(
        parsed["notes"],
        conflict_policy=PianoConflictPolicy.STRICT,
        start_delay_ms=0,
    )
    roll_plan = service.build_playback_plan(
        parsed["notes"],
        conflict_policy=PianoConflictPolicy.ROLL,
        roll_note_ms=10,
        start_delay_ms=0,
    )

    assert parsed["ok"] is True
    assert strict_plan["ok"] is True
    assert roll_plan["ok"] is True

    first_key_downs = [item for item in roll_plan["action_plan"] if item["kind"] == "key_down" and item["t_ms"] <= 21]
    grouped = {}
    for item in first_key_downs:
        grouped.setdefault(item["t_ms"], []).append(item["degree"])
    assert {t_ms: sorted(degrees) for t_ms, degrees in grouped.items()} == {
        0: ["1", "2", "5"],
        11: ["#4"],
        21: ["b3"],
    }

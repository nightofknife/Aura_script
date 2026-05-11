"""MIDI parsing and playback planning for Yihuan's piano mini-game."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from packages.aura_core.api import service_info

from ..piano.keymap import (
    PianoBinding,
    PianoConflictPolicy,
    PianoModifier,
    PianoNoteEvent,
    PianoRegister,
    detect_chord_conflicts,
    get_piano_binding,
)


@dataclass(frozen=True, slots=True)
class _RawMidiNote:
    midi_note: int
    velocity: int
    channel: int
    start_tick: int
    end_tick: int
    source_index: int


@dataclass(frozen=True, slots=True)
class _TempoEvent:
    tick: int
    micros_per_quarter: int


@dataclass(slots=True)
class _ScheduledPianoNote:
    event: PianoNoteEvent
    binding: PianoBinding
    start_ms: int
    end_ms: int


@service_info(
    alias="yihuan_piano",
    public=True,
    singleton=True,
    description="Parse MIDI files and build playable keyboard plans for Yihuan's piano mini-game.",
)
class YihuanPianoService:
    """Convert MIDI notes into Yihuan piano key actions."""

    _DEFAULT_BASE_MIDI = 48
    _REGISTER_BY_OCTAVE = {
        0: PianoRegister.LOW,
        1: PianoRegister.MID,
        2: PianoRegister.HIGH,
    }
    _DEGREE_BY_SEMITONE = {
        0: "1",
        1: "#1",
        2: "2",
        3: "b3",
        4: "3",
        5: "4",
        6: "#4",
        7: "5",
        8: "#5",
        9: "6",
        10: "b7",
        11: "7",
    }
    _ACTION_PRIORITY = {
        "key_up": 0,
        "natural_key_down": 1,
        "shift_down": 2,
        "shift_key_down": 3,
        "shift_up": 4,
        "ctrl_down": 5,
        "ctrl_key_down": 6,
        "ctrl_up": 7,
    }
    _ROLL_BATCH_ORDER = (
        PianoModifier.NONE,
        PianoModifier.SHIFT,
        PianoModifier.CTRL,
    )

    def resolve_input_path(self, file_path: str | Path) -> Path:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    def parse_midi_file(
        self,
        file_path: str | Path,
        *,
        transpose_semitones: int = 0,
        tempo_scale: float = 1.0,
        velocity_threshold: int = 1,
        skip_percussion_channel: bool = True,
    ) -> dict[str, Any]:
        resolved_path = self.resolve_input_path(file_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"MIDI file not found: {resolved_path}")

        payload = resolved_path.read_bytes()
        format_type, track_count, division, track_payloads = self._parse_midi_chunks(payload)
        if division <= 0:
            raise ValueError("SMPTE MIDI timing is not supported yet; only ticks-per-quarter-note MIDI files are supported.")

        raw_notes: list[_RawMidiNote] = []
        tempo_events: list[_TempoEvent] = []
        for track_index, track_bytes in enumerate(track_payloads):
            notes, tempos = self._parse_track_events(
                track_bytes,
                track_index=track_index,
                skip_percussion_channel=skip_percussion_channel,
            )
            raw_notes.extend(notes)
            tempo_events.extend(tempos)

        tempo_map = self._build_tempo_map(tempo_events, division=division)
        transpose = int(transpose_semitones)
        scale = max(float(tempo_scale), 0.0001)
        min_velocity = max(int(velocity_threshold), 0)

        playable_events: list[PianoNoteEvent] = []
        unsupported_notes: list[dict[str, Any]] = []
        source_index = 0

        for raw_note in sorted(raw_notes, key=lambda item: (item.start_tick, item.midi_note, item.source_index)):
            if raw_note.velocity < min_velocity:
                continue
            start_ms = round(self._ticks_to_microseconds(raw_note.start_tick, tempo_map, division) / 1000.0 / scale)
            end_ms = round(self._ticks_to_microseconds(raw_note.end_tick, tempo_map, division) / 1000.0 / scale)
            duration_ms = max(end_ms - start_ms, 1)
            translated = self._translate_midi_note(raw_note.midi_note + transpose)
            if not translated["ok"]:
                unsupported_notes.append(
                    {
                        "midi_note": raw_note.midi_note,
                        "transposed_midi_note": raw_note.midi_note + transpose,
                        "velocity": raw_note.velocity,
                        "start_ms": int(start_ms),
                        "duration_ms": int(duration_ms),
                        "reason": translated["reason"],
                        "degree": translated.get("degree"),
                        "register": translated.get("register"),
                    }
                )
                continue

            try:
                get_piano_binding(PianoRegister(translated["register"]), str(translated["degree"]))
            except KeyError:
                unsupported_notes.append(
                    {
                        "midi_note": raw_note.midi_note,
                        "transposed_midi_note": raw_note.midi_note + transpose,
                        "velocity": raw_note.velocity,
                        "start_ms": int(start_ms),
                        "duration_ms": int(duration_ms),
                        "reason": "degree_unmapped",
                        "degree": translated["degree"],
                        "register": translated["register"],
                    }
                )
                continue

            playable_events.append(
                PianoNoteEvent(
                    register=PianoRegister(str(translated["register"])),
                    degree=str(translated["degree"]),
                    start_ms=int(start_ms),
                    duration_ms=int(duration_ms),
                    source_index=source_index,
                    midi_note=int(raw_note.midi_note + transpose),
                    velocity=int(raw_note.velocity),
                )
            )
            source_index += 1

        return {
            "ok": len(unsupported_notes) == 0,
            "file_path": str(resolved_path),
            "format_type": int(format_type),
            "track_count": int(track_count),
            "division": int(division),
            "tempo_changes": len(tempo_map),
            "transpose_semitones": transpose,
            "tempo_scale": scale,
            "note_count": len(playable_events),
            "unsupported_note_count": len(unsupported_notes),
            "notes": playable_events,
            "unsupported_notes": unsupported_notes,
        }

    def build_playback_plan(
        self,
        notes: Iterable[PianoNoteEvent],
        *,
        conflict_policy: PianoConflictPolicy | str = PianoConflictPolicy.STRICT,
        roll_note_ms: int = 35,
        start_delay_ms: int = 250,
    ) -> dict[str, Any]:
        policy = (
            conflict_policy
            if isinstance(conflict_policy, PianoConflictPolicy)
            else PianoConflictPolicy(str(conflict_policy).strip().lower())
        )
        roll_step_ms = max(int(roll_note_ms), 1)
        delay_ms = max(int(start_delay_ms), 0)

        grouped = self._group_notes_by_start(notes)
        scheduled_notes: list[_ScheduledPianoNote] = []
        conflicts: list[dict[str, Any]] = []

        for start_ms, group_notes in grouped:
            if policy is PianoConflictPolicy.ROLL:
                roll_batches = self._partition_roll_groups(group_notes)
                if len(roll_batches) > 1:
                    for batch_index, subgroup in enumerate(roll_batches):
                        subgroup_start = int(start_ms) + batch_index * roll_step_ms
                        batch_notes = self._schedule_note_group(subgroup, start_ms=subgroup_start)
                        if batch_index < len(roll_batches) - 1:
                            next_batch_start = int(start_ms) + (batch_index + 1) * roll_step_ms
                            for scheduled in batch_notes:
                                scheduled.end_ms = min(scheduled.end_ms, next_batch_start)
                                if scheduled.end_ms <= scheduled.start_ms:
                                    scheduled.end_ms = scheduled.start_ms + 1
                        scheduled_notes.extend(batch_notes)
                    continue

            chord_conflicts = detect_chord_conflicts(group_notes)
            if chord_conflicts and policy is PianoConflictPolicy.STRICT:
                for conflict in chord_conflicts:
                    conflicts.append(
                        {
                            "reason": "physical_key_conflict",
                            "at_ms": int(start_ms),
                            "physical_slot": conflict.physical_slot,
                            "notes": [self._serialize_note(note) for note in conflict.notes],
                        }
                    )
                continue

            scheduled_notes.extend(self._schedule_note_group(group_notes, start_ms=start_ms))

        if conflicts:
            return {
                "ok": False,
                "failure_reason": "physical_key_conflict",
                "conflicts": conflicts,
                "scheduled_notes": [],
                "action_plan": [],
            }

        overlap_resolution = self._resolve_key_overlaps(
            scheduled_notes,
            policy=policy,
        )
        scheduled_notes = overlap_resolution["scheduled_notes"]
        conflicts = overlap_resolution["conflicts"]
        if conflicts:
            return {
                "ok": False,
                "failure_reason": "key_overlap",
                "conflicts": conflicts,
                "scheduled_notes": [self._serialize_scheduled_note(item) for item in scheduled_notes],
                "action_plan": [],
            }

        action_plan = self._build_action_plan(scheduled_notes, start_delay_ms=delay_ms)
        return {
            "ok": True,
            "failure_reason": None,
            "conflicts": [],
            "scheduled_note_count": len(scheduled_notes),
            "scheduled_notes": [self._serialize_scheduled_note(item) for item in scheduled_notes],
            "action_plan": action_plan,
            "roll_note_ms": roll_step_ms,
            "start_delay_ms": delay_ms,
        }

    def _parse_midi_chunks(self, payload: bytes) -> tuple[int, int, int, list[bytes]]:
        if len(payload) < 14 or payload[:4] != b"MThd":
            raise ValueError("Invalid MIDI header chunk.")
        header_length = int.from_bytes(payload[4:8], "big")
        if header_length < 6:
            raise ValueError("Unsupported MIDI header length.")
        header_data = payload[8 : 8 + header_length]
        format_type = int.from_bytes(header_data[0:2], "big")
        track_count = int.from_bytes(header_data[2:4], "big")
        division = int.from_bytes(header_data[4:6], "big", signed=True)
        cursor = 8 + header_length
        tracks: list[bytes] = []
        for _ in range(track_count):
            if cursor + 8 > len(payload) or payload[cursor : cursor + 4] != b"MTrk":
                raise ValueError("Invalid MIDI track chunk.")
            track_length = int.from_bytes(payload[cursor + 4 : cursor + 8], "big")
            cursor += 8
            track_data = payload[cursor : cursor + track_length]
            if len(track_data) != track_length:
                raise ValueError("Truncated MIDI track data.")
            tracks.append(track_data)
            cursor += track_length
        return format_type, track_count, division, tracks

    def _parse_track_events(
        self,
        payload: bytes,
        *,
        track_index: int,
        skip_percussion_channel: bool,
    ) -> tuple[list[_RawMidiNote], list[_TempoEvent]]:
        cursor = 0
        absolute_tick = 0
        running_status: int | None = None
        active_notes: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        raw_notes: list[_RawMidiNote] = []
        tempo_events: list[_TempoEvent] = []
        source_index = 0

        while cursor < len(payload):
            delta_tick, cursor = self._read_variable_length_quantity(payload, cursor)
            absolute_tick += delta_tick
            if cursor >= len(payload):
                break

            status = payload[cursor]
            if status < 0x80:
                if running_status is None:
                    raise ValueError(f"Running status encountered without previous status in track {track_index}.")
                status = running_status
            else:
                cursor += 1
                if status < 0xF0:
                    running_status = status

            if status == 0xFF:
                if cursor >= len(payload):
                    break
                meta_type = payload[cursor]
                cursor += 1
                length, cursor = self._read_variable_length_quantity(payload, cursor)
                meta_data = payload[cursor : cursor + length]
                cursor += length
                if meta_type == 0x51 and len(meta_data) == 3:
                    tempo_events.append(
                        _TempoEvent(
                            tick=absolute_tick,
                            micros_per_quarter=int.from_bytes(meta_data, "big"),
                        )
                    )
                continue

            if status in (0xF0, 0xF7):
                length, cursor = self._read_variable_length_quantity(payload, cursor)
                cursor += length
                continue

            event_type = status & 0xF0
            channel = status & 0x0F
            data_len = 1 if event_type in (0xC0, 0xD0) else 2
            data = payload[cursor : cursor + data_len]
            if len(data) != data_len:
                raise ValueError(f"Truncated MIDI channel event in track {track_index}.")
            cursor += data_len

            if event_type not in (0x80, 0x90):
                continue

            if skip_percussion_channel and channel == 9:
                continue

            midi_note = int(data[0])
            velocity = int(data[1]) if data_len > 1 else 0
            note_key = (channel, midi_note)

            if event_type == 0x90 and velocity > 0:
                active_notes.setdefault(note_key, []).append((absolute_tick, velocity, source_index))
                source_index += 1
                continue

            stack = active_notes.get(note_key) or []
            if not stack:
                continue
            start_tick, start_velocity, note_source_index = stack.pop()
            raw_notes.append(
                _RawMidiNote(
                    midi_note=midi_note,
                    velocity=start_velocity,
                    channel=channel,
                    start_tick=int(start_tick),
                    end_tick=max(int(absolute_tick), int(start_tick + 1)),
                    source_index=note_source_index,
                )
            )
            if not stack:
                active_notes.pop(note_key, None)

        if active_notes:
            for (channel, midi_note), stack in active_notes.items():
                for start_tick, velocity, note_source_index in stack:
                    raw_notes.append(
                        _RawMidiNote(
                            midi_note=int(midi_note),
                            velocity=int(velocity),
                            channel=int(channel),
                            start_tick=int(start_tick),
                            end_tick=max(int(absolute_tick), int(start_tick + 1)),
                            source_index=int(note_source_index),
                        )
                    )
        return raw_notes, tempo_events

    def _build_tempo_map(self, tempo_events: Iterable[_TempoEvent], *, division: int) -> list[dict[str, Any]]:
        cleaned: dict[int, int] = {0: 500000}
        for event in tempo_events:
            cleaned[int(event.tick)] = int(event.micros_per_quarter)

        sorted_events = sorted(cleaned.items(), key=lambda item: item[0])
        tempo_map: list[dict[str, Any]] = []
        cumulative_micros = 0.0
        previous_tick = 0
        previous_tempo = sorted_events[0][1]
        for tick, tempo in sorted_events:
            if tempo_map:
                cumulative_micros += ((tick - previous_tick) * previous_tempo) / float(division)
            tempo_map.append(
                {
                    "tick": int(tick),
                    "tempo": int(tempo),
                    "cumulative_micros": float(cumulative_micros),
                }
            )
            previous_tick = tick
            previous_tempo = tempo
        return tempo_map

    def _ticks_to_microseconds(self, tick: int, tempo_map: list[dict[str, Any]], division: int) -> float:
        selected = tempo_map[0]
        for event in tempo_map:
            if int(event["tick"]) > int(tick):
                break
            selected = event
        return float(selected["cumulative_micros"]) + ((int(tick) - int(selected["tick"])) * int(selected["tempo"])) / float(division)

    def _translate_midi_note(self, midi_note: int) -> dict[str, Any]:
        relative = int(midi_note) - self._DEFAULT_BASE_MIDI
        if relative < 0:
            return {"ok": False, "reason": "below_supported_range"}
        octave_index, semitone = divmod(relative, 12)
        register = self._REGISTER_BY_OCTAVE.get(octave_index)
        if register is None:
            return {"ok": False, "reason": "above_supported_range"}
        degree = self._DEGREE_BY_SEMITONE.get(semitone)
        if degree is None:
            return {"ok": False, "reason": "unsupported_semitone"}
        return {
            "ok": True,
            "register": register.value,
            "degree": degree,
        }

    def _group_notes_by_start(self, notes: Iterable[PianoNoteEvent]) -> list[tuple[int, list[PianoNoteEvent]]]:
        grouped: dict[int, list[PianoNoteEvent]] = {}
        for note in notes:
            grouped.setdefault(int(note.start_ms), []).append(note)
        return [
            (
                start_ms,
                sorted(group_notes, key=lambda item: (item.midi_note if item.midi_note is not None else 0, item.source_index)),
            )
            for start_ms, group_notes in sorted(grouped.items(), key=lambda item: item[0])
        ]

    def _schedule_note_group(
        self,
        notes: Iterable[PianoNoteEvent],
        *,
        start_ms: int,
    ) -> list[_ScheduledPianoNote]:
        scheduled: list[_ScheduledPianoNote] = []
        for note in notes:
            binding = get_piano_binding(note.register, note.degree)
            scheduled.append(
                _ScheduledPianoNote(
                    event=note,
                    binding=binding,
                    start_ms=int(start_ms),
                    end_ms=int(start_ms) + max(int(note.duration_ms), 1),
                )
            )
        return scheduled

    def _partition_roll_groups(
        self,
        notes: Iterable[PianoNoteEvent],
    ) -> list[list[PianoNoteEvent]]:
        grouped: dict[PianoModifier, list[PianoNoteEvent]] = {
            PianoModifier.NONE: [],
            PianoModifier.SHIFT: [],
            PianoModifier.CTRL: [],
        }
        for note in notes:
            grouped[get_piano_binding(note.register, note.degree).modifier].append(note)

        result: list[list[PianoNoteEvent]] = []
        for modifier in self._ROLL_BATCH_ORDER:
            modifier_notes = grouped[modifier]
            if not modifier_notes:
                continue
            result.append(
                sorted(
                    modifier_notes,
                    key=lambda item: (
                        item.midi_note if item.midi_note is not None else 0,
                        item.source_index,
                    ),
                )
            )
        return result

    def _resolve_key_overlaps(
        self,
        scheduled_notes: list[_ScheduledPianoNote],
        *,
        policy: PianoConflictPolicy,
    ) -> dict[str, Any]:
        grouped: dict[str, list[_ScheduledPianoNote]] = {}
        for note in scheduled_notes:
            grouped.setdefault(note.binding.key, []).append(note)

        conflicts: list[dict[str, Any]] = []
        resolved: list[_ScheduledPianoNote] = []

        for key, key_notes in grouped.items():
            ordered = sorted(
                key_notes,
                key=lambda item: (item.start_ms, item.event.source_index, item.end_ms),
            )
            merged: list[_ScheduledPianoNote] = []
            for note in ordered:
                if (
                    merged
                    and merged[-1].start_ms == note.start_ms
                    and merged[-1].binding.modifier == note.binding.modifier
                    and merged[-1].event.degree == note.event.degree
                    and merged[-1].event.register == note.event.register
                ):
                    merged[-1].end_ms = max(merged[-1].end_ms, note.end_ms)
                    continue
                merged.append(note)

            for index in range(len(merged) - 1):
                current = merged[index]
                nxt = merged[index + 1]
                if nxt.start_ms < current.end_ms:
                    if policy is PianoConflictPolicy.STRICT:
                        conflicts.append(
                            {
                                "reason": "key_overlap",
                                "key": key,
                                "at_ms": int(nxt.start_ms),
                                "notes": [
                                    self._serialize_scheduled_note(current),
                                    self._serialize_scheduled_note(nxt),
                                ],
                            }
                        )
                    else:
                        current.end_ms = max(current.start_ms + 1, nxt.start_ms)
            resolved.extend(merged)

        return {
            "scheduled_notes": sorted(
                resolved,
                key=lambda item: (item.start_ms, item.event.source_index, item.binding.key),
            ),
            "conflicts": conflicts,
        }

    def _build_action_plan(
        self,
        scheduled_notes: list[_ScheduledPianoNote],
        *,
        start_delay_ms: int,
    ) -> list[dict[str, Any]]:
        onset_groups: dict[int, list[_ScheduledPianoNote]] = {}
        for note in scheduled_notes:
            onset_groups.setdefault(int(note.start_ms), []).append(note)

        actions: list[dict[str, Any]] = []
        for start_ms, group_notes in sorted(onset_groups.items(), key=lambda item: item[0]):
            base_action_time_ms = int(start_delay_ms) + int(start_ms)
            naturals = sorted(
                (note for note in group_notes if note.binding.modifier is PianoModifier.NONE),
                key=lambda item: (item.event.midi_note if item.event.midi_note is not None else 0, item.binding.key),
            )
            shifts = sorted(
                (note for note in group_notes if note.binding.modifier is PianoModifier.SHIFT),
                key=lambda item: (item.event.midi_note if item.event.midi_note is not None else 0, item.binding.key),
            )
            ctrls = sorted(
                (note for note in group_notes if note.binding.modifier is PianoModifier.CTRL),
                key=lambda item: (item.event.midi_note if item.event.midi_note is not None else 0, item.binding.key),
            )

            stage_offset_ms = 0
            for note in naturals:
                actions.append(
                    {
                        "t_ms": base_action_time_ms + stage_offset_ms,
                        "kind": "key_down",
                        "key": note.binding.key,
                        "priority": self._ACTION_PRIORITY["natural_key_down"],
                        "register": note.event.register.value,
                        "degree": note.event.degree,
                        "display": note.binding.display,
                    }
                )
            if naturals:
                stage_offset_ms += 1

            if shifts:
                actions.append(
                    {
                        "t_ms": base_action_time_ms + stage_offset_ms,
                        "kind": "modifier_down",
                        "key": PianoModifier.SHIFT.value,
                        "priority": self._ACTION_PRIORITY["shift_down"],
                    }
                )
                for note in shifts:
                    actions.append(
                        {
                            "t_ms": base_action_time_ms + stage_offset_ms + 1,
                            "kind": "key_down",
                            "key": note.binding.key,
                            "priority": self._ACTION_PRIORITY["shift_key_down"],
                            "register": note.event.register.value,
                            "degree": note.event.degree,
                            "display": note.binding.display,
                        }
                    )
                actions.append(
                    {
                        "t_ms": base_action_time_ms + stage_offset_ms + 2,
                        "kind": "modifier_up",
                        "key": PianoModifier.SHIFT.value,
                        "priority": self._ACTION_PRIORITY["shift_up"],
                    }
                )
                stage_offset_ms += 3

            if ctrls:
                actions.append(
                    {
                        "t_ms": base_action_time_ms + stage_offset_ms,
                        "kind": "modifier_down",
                        "key": PianoModifier.CTRL.value,
                        "priority": self._ACTION_PRIORITY["ctrl_down"],
                    }
                )
                for note in ctrls:
                    actions.append(
                        {
                            "t_ms": base_action_time_ms + stage_offset_ms + 1,
                            "kind": "key_down",
                            "key": note.binding.key,
                            "priority": self._ACTION_PRIORITY["ctrl_key_down"],
                            "register": note.event.register.value,
                            "degree": note.event.degree,
                            "display": note.binding.display,
                        }
                    )
                actions.append(
                    {
                        "t_ms": base_action_time_ms + stage_offset_ms + 2,
                        "kind": "modifier_up",
                        "key": PianoModifier.CTRL.value,
                        "priority": self._ACTION_PRIORITY["ctrl_up"],
                    }
                )

        for note in scheduled_notes:
            actions.append(
                {
                    "t_ms": int(start_delay_ms) + int(note.end_ms),
                    "kind": "key_up",
                    "key": note.binding.key,
                    "priority": self._ACTION_PRIORITY["key_up"],
                    "register": note.event.register.value,
                    "degree": note.event.degree,
                    "display": note.binding.display,
                }
            )

        return sorted(
            actions,
            key=lambda item: (
                int(item["t_ms"]),
                int(item["priority"]),
                str(item["key"]),
                str(item.get("degree") or ""),
            ),
        )

    def _serialize_note(self, note: PianoNoteEvent) -> dict[str, Any]:
        return {
            "register": note.register.value,
            "degree": note.degree,
            "start_ms": int(note.start_ms),
            "duration_ms": int(note.duration_ms),
            "midi_note": note.midi_note,
            "velocity": note.velocity,
            "binding": note.binding().display,
        }

    def _serialize_scheduled_note(self, note: _ScheduledPianoNote) -> dict[str, Any]:
        return {
            "register": note.event.register.value,
            "degree": note.event.degree,
            "start_ms": int(note.start_ms),
            "end_ms": int(note.end_ms),
            "duration_ms": int(max(note.end_ms - note.start_ms, 1)),
            "midi_note": note.event.midi_note,
            "velocity": note.event.velocity,
            "key": note.binding.key,
            "modifier": note.binding.modifier.value,
            "display": note.binding.display,
        }

    def _read_variable_length_quantity(self, payload: bytes, cursor: int) -> tuple[int, int]:
        value = 0
        while True:
            if cursor >= len(payload):
                raise ValueError("Unexpected end of MIDI data while reading VLQ.")
            byte = payload[cursor]
            cursor += 1
            value = (value << 7) | (byte & 0x7F)
            if (byte & 0x80) == 0:
                break
        return value, cursor

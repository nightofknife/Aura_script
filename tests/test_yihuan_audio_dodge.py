from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from plans.yihuan.src.audio_dodge import AudioDodgeRuntime, AudioDodgeSettings


class TestAudioDodgeRuntime(unittest.TestCase):
    def test_process_frame_detects_matching_waveform(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = Path(tmp_dir) / "dodge_sample.wav"
            signal = _make_sine_wave(sample_rate=8000, duration_sec=0.1, frequency_hz=1200)
            _write_wav(sample_path, signal, sample_rate=8000)

            runtime = AudioDodgeRuntime(
                AudioDodgeSettings(
                    enabled=True,
                    sample_path=sample_path,
                    sample_rate=8000,
                    channels=2,
                    chunk_size=400,
                    threshold=0.05,
                    ratio=1.0,
                    allow_repeat=False,
                    cooldown_sec=0.5,
                    window_sec=0.1,
                    trigger_max_age_sec=1.0,
                    reconnect_delay_sec=0.1,
                    pre_emphasis=0.0,
                    dodge_pause_ms=400,
                    right_hold_ms=40,
                    post_right_delay_ms=30,
                    shift_hold_ms=40,
                    dodge_key="shift",
                    dodge_mouse_button="right",
                )
            )

            score = runtime.process_frame(signal)
            event = runtime.consume_trigger(max_age_sec=1.0)

            self.assertGreater(score, 0.05)
            self.assertIsNotNone(event)
            self.assertGreater(event["score"], 0.05)

    def test_missing_sample_marks_runtime_unavailable(self):
        runtime = AudioDodgeRuntime(
            AudioDodgeSettings(
                enabled=True,
                sample_path=Path("Z:/missing/dodge.wav"),
                sample_rate=8000,
                channels=2,
                chunk_size=400,
                threshold=0.05,
                ratio=1.0,
                allow_repeat=False,
                cooldown_sec=0.5,
                window_sec=0.1,
                trigger_max_age_sec=1.0,
                reconnect_delay_sec=0.1,
                pre_emphasis=0.0,
                dodge_pause_ms=400,
                right_hold_ms=40,
                post_right_delay_ms=30,
                shift_hold_ms=40,
                dodge_key="shift",
                dodge_mouse_button="right",
            )
        )

        status = runtime.status()
        self.assertEqual(status["status"], "unavailable")
        self.assertEqual(status["error"], "sample_missing")


def _make_sine_wave(*, sample_rate: int, duration_sec: float, frequency_hz: float) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float64) / float(sample_rate)
    return (0.8 * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float64)


def _write_wav(path: Path, signal: np.ndarray, *, sample_rate: int) -> None:
    pcm = np.clip(signal * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

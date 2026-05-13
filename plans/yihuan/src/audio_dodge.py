"""Audio-triggered dodge helper for Yihuan combat."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import wave
from typing import Any, Mapping

import numpy as np

from packages.aura_core.observability.logging.core_logger import logger

try:
    import soundcard as sc
except Exception:  # noqa: BLE001
    sc = None


@dataclass(frozen=True)
class AudioDodgeSettings:
    enabled: bool
    sample_path: Path
    sample_rate: int
    channels: int
    chunk_size: int
    threshold: float
    ratio: float
    allow_repeat: bool
    cooldown_sec: float
    window_sec: float
    trigger_max_age_sec: float
    reconnect_delay_sec: float
    pre_emphasis: float
    dodge_pause_ms: int
    right_hold_ms: int
    post_right_delay_ms: int
    shift_hold_ms: int
    dodge_key: str
    dodge_mouse_button: str


class AudioDodgeRuntime:
    """Background loopback matcher that turns dodge audio into queued trigger events."""

    def __init__(self, settings: AudioDodgeSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._events: deque[dict[str, Any]] = deque()
        self._last_score = 0.0
        self._last_trigger_at = 0.0
        self._ready = True
        self._active = False
        self._error: str | None = None
        self._status = "disabled" if not settings.enabled else "idle"

        self._sample = np.array([], dtype=np.float64)
        self._ref = np.array([], dtype=np.float64)
        self._ref_fft = np.array([], dtype=np.complex128)
        self._fft_n = 0
        self._buf = np.array([], dtype=np.float64)
        self._buf_size = 0
        self._pos = 0
        self._filled = 0

        if self.settings.enabled:
            self._prepare_reference()

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any], *, enabled: bool) -> "AudioDodgeRuntime":
        payload = dict(profile.get("audio_dodge") or {})
        sample_path = Path(str(payload.get("sample_path") or "")).expanduser()
        if not sample_path.is_absolute():
            plan_root = Path(__file__).resolve().parents[2]
            sample_path = (plan_root / sample_path).resolve()
        settings = AudioDodgeSettings(
            enabled=bool(enabled and payload.get("enabled", False)),
            sample_path=sample_path,
            sample_rate=max(int(payload.get("sample_rate", 32000)), 8000),
            channels=max(int(payload.get("channels", 2)), 1),
            chunk_size=max(int(payload.get("chunk_size", 1600)), 128),
            threshold=max(float(payload.get("threshold", 0.13)), 0.0),
            ratio=max(float(payload.get("ratio", 1.0)), 0.01),
            allow_repeat=bool(payload.get("allow_repeat", False)),
            cooldown_sec=max(float(payload.get("cooldown_sec", 0.75)), 0.0),
            window_sec=max(float(payload.get("window_sec", 0.5)), 0.1),
            trigger_max_age_sec=max(float(payload.get("trigger_max_age_sec", 0.6)), 0.05),
            reconnect_delay_sec=max(float(payload.get("reconnect_delay_sec", 2.0)), 0.1),
            pre_emphasis=min(max(float(payload.get("pre_emphasis", 0.97)), 0.0), 0.999),
            dodge_pause_ms=max(int(payload.get("dodge_pause_ms", 420)), 0),
            right_hold_ms=max(int(payload.get("right_hold_ms", 40)), 0),
            post_right_delay_ms=max(int(payload.get("post_right_delay_ms", 35)), 0),
            shift_hold_ms=max(int(payload.get("shift_hold_ms", 40)), 0),
            dodge_key=str(payload.get("dodge_key") or "shift"),
            dodge_mouse_button=str(payload.get("dodge_mouse_button") or "right"),
        )
        return cls(settings)

    def start(self) -> None:
        if not self.settings.enabled:
            self._status = "disabled"
            logger.info("AudioDodge[start] disabled")
            return
        if sc is None:
            self._set_error("soundcard_unavailable")
            logger.warning("AudioDodge[start] unavailable error=soundcard_unavailable")
            return
        if self._error is not None:
            logger.warning("AudioDodge[start] unavailable error=%s sample_path=%s", self._error, self.settings.sample_path)
            return
        if self._thread is not None and self._thread.is_alive():
            logger.info("AudioDodge[start] already_running status=%s", self._status)
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="yihuan-audio-dodge", daemon=True)
        self._thread.start()
        logger.info(
            "AudioDodge[start] thread_started sample_path=%s sample_rate=%s channels=%s chunk_size=%s threshold=%.4f cooldown_sec=%.3f",
            self.settings.sample_path,
            int(self.settings.sample_rate),
            int(self.settings.channels),
            int(self.settings.chunk_size),
            float(self.settings.threshold),
            float(self.settings.cooldown_sec),
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._active = False
        if self._status == "running":
            self._status = "stopped"

    def consume_trigger(self, *, max_age_sec: float | None = None) -> dict[str, Any] | None:
        now = time.monotonic()
        limit = float(max_age_sec if max_age_sec is not None else self.settings.trigger_max_age_sec)
        with self._lock:
            while self._events:
                item = self._events.popleft()
                if now - float(item["t"]) <= limit:
                    return item
        return None

    def process_frame(self, mono_frame: np.ndarray) -> float:
        frame = np.asarray(mono_frame, dtype=np.float64).reshape(-1)
        if frame.size == 0 or self._buf_size <= 0:
            return 0.0
        processed = self._preprocess(frame)
        self._write_ring(processed)
        segment = self._current_segment()
        if segment.size == 0:
            return 0.0
        score = self._match(self._normalize(segment)) * self.settings.ratio
        with self._lock:
            self._last_score = float(score)
        self._handle_score(score)
        return float(score)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self.settings.enabled),
                "active": bool(self._active),
                "status": str(self._status),
                "error": self._error,
                "last_score": round(float(self._last_score), 5),
                "queued_triggers": int(len(self._events)),
                "sample_path": str(self.settings.sample_path),
                "threshold": round(float(self.settings.threshold), 5),
                "cooldown_sec": round(float(self.settings.cooldown_sec), 3),
            }

    def _prepare_reference(self) -> None:
        if not self.settings.sample_path.is_file():
            self._set_error("sample_missing")
            return
        try:
            sample = self._load_sample(self.settings.sample_path, target_sr=self.settings.sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AudioDodge[sample] failed to load %s: %s", self.settings.sample_path, exc)
            self._set_error("sample_load_failed")
            return
        if sample.size == 0:
            self._set_error("sample_empty")
            return

        processed = self._preprocess(sample)
        self._sample = processed
        self._ref = self._normalize(processed)
        sample_sec = len(processed) / float(self.settings.sample_rate)
        self._buf_size = max(int(max(sample_sec, self.settings.window_sec) * self.settings.sample_rate), len(self._ref))
        self._buf = np.zeros(self._buf_size, dtype=np.float64)
        self._pos = 0
        self._filled = 0
        self._fft_n = 1 << ((self._buf_size + len(self._ref) - 1).bit_length())
        self._ref_fft = np.fft.rfft(self._ref, n=self._fft_n).conj()
        self._status = "ready"
        logger.info(
            "AudioDodge[sample] ready sample_path=%s sample_frames=%s sample_rate=%s window_sec=%.3f threshold=%.4f",
            self.settings.sample_path,
            int(len(self._ref)),
            int(self.settings.sample_rate),
            float(self.settings.window_sec),
            float(self.settings.threshold),
        )

    def _run_loop(self) -> None:
        self._active = True
        self._status = "running"
        while not self._stop_event.is_set():
            try:
                default_speaker = sc.default_speaker()
                speaker = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
                logger.info("AudioDodge[loopback] recording speaker=%s loopback=%s", default_speaker, speaker)
                with speaker.recorder(
                    samplerate=self.settings.sample_rate,
                    channels=self.settings.channels,
                ) as recorder:
                    while not self._stop_event.is_set():
                        data = recorder.record(numframes=self.settings.chunk_size)
                        mono = self._to_mono(data)
                        self.process_frame(mono)
            except Exception as exc:  # noqa: BLE001
                if self._stop_event.is_set():
                    break
                logger.warning("AudioDodge[loopback] error: %s", exc)
                self._status = "reconnecting"
                time.sleep(self.settings.reconnect_delay_sec)
        self._active = False
        if self._status == "running":
            self._status = "stopped"

    def _handle_score(self, score: float) -> None:
        now = time.monotonic()
        hit = float(score) >= self.settings.threshold
        if hit and self._ready:
            if self.settings.allow_repeat or now - self._last_trigger_at >= self.settings.cooldown_sec:
                with self._lock:
                    self._events.append({"t": now, "score": float(score)})
                self._last_trigger_at = now
                logger.info("AudioDodge[trigger] score=%.5f threshold=%.5f", float(score), float(self.settings.threshold))
            self._ready = False
            return
        if not hit:
            self._ready = True

    def _write_ring(self, frame: np.ndarray) -> None:
        count = int(frame.shape[0])
        if count >= self._buf_size:
            self._buf[:] = frame[-self._buf_size :]
            self._pos = 0
            self._filled = self._buf_size
            return
        end = self._pos + count
        if end <= self._buf_size:
            self._buf[self._pos : end] = frame
        else:
            first = self._buf_size - self._pos
            self._buf[self._pos :] = frame[:first]
            self._buf[: end - self._buf_size] = frame[first:]
        self._pos = end % self._buf_size
        self._filled = min(self._filled + count, self._buf_size)

    def _current_segment(self) -> np.ndarray:
        if self._filled <= 0:
            return np.array([], dtype=np.float64)
        if self._filled < self._buf_size:
            return self._buf[: self._filled]
        if self._pos == 0:
            return self._buf
        return np.concatenate((self._buf[self._pos :], self._buf[: self._pos]))

    def _match(self, segment: np.ndarray) -> float:
        if segment.size == 0 or self._ref.size == 0:
            return 0.0
        n = len(segment) + len(self._ref) - 1
        if n > self._fft_n:
            self._fft_n = 1 << n.bit_length()
            self._ref_fft = np.fft.rfft(self._ref, n=self._fft_n).conj()
        freq = np.fft.rfft(segment, n=self._fft_n)
        corr = np.fft.irfft(freq * self._ref_fft)[:n]
        return float(np.max(corr) / max(len(segment), len(self._ref), 1))

    def _normalize(self, wf: np.ndarray) -> np.ndarray:
        if wf.size == 0:
            return wf
        rms = np.sqrt(np.mean(np.square(wf), dtype=np.float64) + 1e-6)
        return wf / rms

    def _preprocess(self, wf: np.ndarray) -> np.ndarray:
        if wf.size == 0:
            return wf.astype(np.float64)
        frame = wf.astype(np.float64)
        frame = frame - np.mean(frame, dtype=np.float64)
        coeff = self.settings.pre_emphasis
        if coeff > 0.0 and frame.size > 1:
            emphasized = np.empty_like(frame)
            emphasized[0] = frame[0]
            emphasized[1:] = frame[1:] - coeff * frame[:-1]
            frame = emphasized
        return frame

    def _to_mono(self, data: Any) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                return arr[:, 0]
            return np.mean(arr, axis=1, dtype=np.float64)
        return arr.reshape(-1)

    def _load_sample(self, path: Path, *, target_sr: int) -> np.ndarray:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
            data = (data - 128.0) / 128.0
        elif sample_width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
        elif sample_width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        if channels > 1:
            data = data.reshape(-1, channels)
            data = np.mean(data, axis=1, dtype=np.float64)
        if sample_rate != target_sr:
            data = self._resample(data, source_sr=sample_rate, target_sr=target_sr)
        return data.astype(np.float64)

    def _resample(self, data: np.ndarray, *, source_sr: int, target_sr: int) -> np.ndarray:
        if data.size == 0 or source_sr == target_sr:
            return data
        source_positions = np.arange(data.size, dtype=np.float64)
        target_length = max(int(round(data.size * target_sr / float(source_sr))), 1)
        target_positions = np.linspace(0.0, max(data.size - 1, 0), target_length, dtype=np.float64)
        return np.interp(target_positions, source_positions, data).astype(np.float64)

    def _set_error(self, value: str) -> None:
        self._error = value
        self._status = "unavailable"

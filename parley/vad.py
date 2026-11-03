"""Streaming voice-activity segmentation.

A meeting interpreter cannot wait for a 5-second file to end: it must cut speech
into utterances at natural pauses, and it must not mistake a fan or a keyboard for
a sentence. So the threshold adapts to the room's noise floor instead of being a
fixed number, and a short pre-roll keeps the first phoneme (the one the VAD often
eats) in the clip.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

SR = 16000


def db(rms: float) -> float:
    return 20.0 * float(np.log10(max(rms, 1e-9)))


@dataclass
class Utterance:
    audio: np.ndarray
    t_start: float
    t_end: float
    forced_cut: bool = False

    @property
    def seconds(self) -> float:
        return len(self.audio) / SR


class Segmenter:
    """Feed 16 kHz frames in, get :class:`Utterance` objects out."""

    def __init__(self, threshold_db: float = -38.0, silence_ms: int = 420, min_speech_ms: int = 280,
                 max_s: float = 14.0, preroll_ms: int = 200, block_ms: int = 30):
        self.base_threshold_db = threshold_db
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_s = max_s
        self.block_ms = block_ms
        self._preroll = deque(maxlen=max(1, preroll_ms // block_ms))
        self._buffer: list[np.ndarray] = []
        self._speaking = False
        self._silence_run_ms = 0
        self._speech_run_ms = 0
        self._speech_ms = 0
        self._t_start = 0.0
        self._now = 0.0
        self._noise_db = -60.0

    # -- helpers -------------------------------------------------------------
    @property
    def threshold_db(self) -> float:
        """Never lower than 9 dB above the measured noise floor."""
        return max(self.base_threshold_db, self._noise_db + 9.0)

    def _learn_noise(self, level: float) -> None:
        if not self._speaking:
            self._noise_db = 0.95 * self._noise_db + 0.05 * level

    # -- main entry ----------------------------------------------------------
    def push(self, frame: np.ndarray, now: float | None = None) -> list[Utterance]:
        now = now if now is not None else time.time()
        self._now = now
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        if frame.size == 0:
            return []
        level = db(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))
        self._learn_noise(level)
        loud = level > self.threshold_db
        out: list[Utterance] = []

        if not self._speaking:
            self._preroll.append(frame)
            if loud:
                self._speech_run_ms += self.block_ms
                if self._speech_run_ms >= 2 * self.block_ms:
                    self._speaking = True
                    self._buffer = list(self._preroll)
                    self._preroll.clear()
                    self._speech_ms = self._speech_run_ms
                    self._t_start = now - self._speech_ms / 1000.0
                    self._silence_run_ms = 0
            else:
                self._speech_run_ms = 0
            return out

        # speaking
        self._buffer.append(frame)
        if loud:
            self._speech_ms += self.block_ms
            self._silence_run_ms = 0
        else:
            self._silence_run_ms += self.block_ms

        duration = len(self._buffer) * self.block_ms / 1000.0
        endpoint = self._silence_run_ms >= self.silence_ms and self._speech_ms >= self.min_speech_ms
        forced = duration >= self.max_s
        if endpoint or forced:
            audio = np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.float32)
            # drop the trailing silence we waited for, keep 120 ms of tail
            keep_ms = max(self.silence_ms - 120, 80)
            if endpoint and not forced:
                trim = min(len(audio), int(keep_ms / 1000 * SR))
                if trim:
                    audio = audio[:-trim]
            out.append(Utterance(audio=audio, t_start=self._t_start, t_end=now, forced_cut=forced))
            self._speaking = False
            self._buffer = []
            self._silence_run_ms = 0
            self._speech_run_ms = 0
            self._speech_ms = 0
        return out

    def in_progress(self) -> tuple[np.ndarray, float, bool]:
        """(audio so far, t_start, speaking) — lets the pipeline decode partials."""
        if not self._speaking or not self._buffer:
            return np.zeros(0, dtype=np.float32), self._t_start, False
        return np.concatenate(self._buffer), self._t_start, True

    def flush(self) -> list[Utterance]:
        if self._speaking and self._buffer and self._speech_ms >= self.min_speech_ms:
            audio = np.concatenate(self._buffer)
            self._speaking = False
            self._buffer = []
            return [Utterance(audio=audio, t_start=self._t_start, t_end=self._now, forced_cut=True)]
        self._speaking = False
        self._buffer = []
        return []


def segment_file(audio: np.ndarray, **kwargs) -> list[Utterance]:
    """Offline segmentation of a file with the same state machine (deterministic timing)."""
    block_ms = kwargs.pop("block_ms", 30)
    seg = Segmenter(block_ms=block_ms, **kwargs)
    step = int(SR * block_ms / 1000)
    out: list[Utterance] = []
    t = 0.0
    for i in range(0, len(audio), step):
        frame = audio[i:i + step]
        if frame.size < step:
            frame = np.pad(frame, (0, step - frame.size))
        t = i / SR
        out.extend(seg.push(frame, now=t))
    for utt in seg.flush():
        out.append(utt)
    return out

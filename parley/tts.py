"""Speech synthesis stage.

Piper runs on the **CPU**, on purpose. On a 6 GB card the GPU is worth more to the
recogniser and the translator than to a vocoder that already sustains several
times real time on a laptop CPU — and keeping TTS off the GPU means the resident
set stays a pair of models, not three.

Quality is handled where Piper actually has levers: the high-quality voice
variants, sentence-level chunking (so prosody resets at punctuation instead of
running one flat breath through a whole paragraph), and an explicit length scale
so the translated voice speaks at a listening pace rather than a dictation pace.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|(?<=[;:])\s+")
MAX_CHUNK_CHARS = 220


def split_for_speech(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Break translated text into speakable clauses, keeping punctuation."""
    text = " ".join(text.split())
    if not text:
        return []
    pieces: list[str] = []
    for sentence in SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        # long sentence: prefer commas, then hard-wrap
        buffer = ""
        for part in re.split(r"(?<=,)\s+", sentence):
            if len(buffer) + len(part) + 1 <= max_chars:
                buffer = f"{buffer} {part}".strip()
            else:
                if buffer:
                    pieces.append(buffer)
                while len(part) > max_chars:
                    cut = part.rfind(" ", 0, max_chars)
                    cut = cut if cut > 40 else max_chars
                    pieces.append(part[:cut].strip())
                    part = part[cut:].strip()
                buffer = part
        if buffer:
            pieces.append(buffer)
    return [p for p in pieces if p]


class PiperTts:
    """Piper voice on CPU. Zero VRAM."""

    estimated_mb = 0

    def __init__(self, voice_path: str | Path, length_scale: float = 1.05, noise_scale: float = 0.667,
                 noise_w: float = 0.8, speaker_id: int | None = None):
        self.voice_path = Path(voice_path)
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.speaker_id = speaker_id
        self.voice = None
        self.sample_rate = 22050
        self.load_s = 0.0

    def load(self) -> None:
        if self.voice is not None:
            return
        from piper import PiperVoice

        if not self.voice_path.exists():
            raise SystemExit(
                f"TTS voice missing at {self.voice_path}. Run:\n"
                f"  python scripts/fetch_models.py --voices de en")
        started = time.perf_counter()
        self.voice = PiperVoice.load(str(self.voice_path))
        self.load_s = time.perf_counter() - started

    def unload(self) -> None:
        self.voice = None

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (float32 mono audio, sample_rate)."""
        assert self.voice is not None, "call load() first"
        from piper.config import SynthesisConfig

        config = SynthesisConfig(
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w,
            speaker_id=self.speaker_id,
        )
        parts: list[np.ndarray] = []
        rate = self.sample_rate
        for chunk in self.voice.synthesize(text, syn_config=config):
            rate = chunk.sample_rate
            parts.append(np.asarray(chunk.audio_float_array, dtype=np.float32))
        if not parts:
            return np.zeros(0, dtype=np.float32), rate
        return np.concatenate(parts), rate

    def synthesize_stream(self, text: str):
        """Yield audio as Piper produces it — this is what removes the TTS wait."""
        assert self.voice is not None, "call load() first"
        from piper.config import SynthesisConfig

        config = SynthesisConfig(
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w,
            speaker_id=self.speaker_id,
        )
        for chunk in self.voice.synthesize(text, syn_config=config):
            yield np.asarray(chunk.audio_float_array, dtype=np.float32), chunk.sample_rate

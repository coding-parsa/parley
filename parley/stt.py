"""Speech recognition stage.

Two responsibilities beyond calling faster-whisper:

* **The language gate.** Whisper's language token changes the language of its
  *output*: decoding non-English audio while the token says ``de`` makes it
  translate rather than transcribe (measured in ``docs/whisper-language-token.md``).
  In a meeting that failure mode is not a typo — it is an unauthorised translation
  spoken into the room. So every utterance is language-checked against the pair
  and rejected when it does not match.
* **The commit policy.** Whisper rewrites its own partial hypotheses. Speaking a
  partial says the sentence twice or reads garbage, so only text up to a stable
  clause boundary is ever handed to the translator.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLAUSE_END = re.compile(r"[.!?;:…](?=\s|$)|\n")
SOFT_END = re.compile(r",(?=\s)")


@dataclass
class Transcript:
    text: str
    language: str
    language_prob: float
    avg_logprob: float
    no_speech_prob: float
    seconds: float
    decode_s: float
    chunks: list[dict] = field(default_factory=list)


class WhisperStt:
    """faster-whisper wrapper with a single model resident at a time."""

    def __init__(self, model_dir: str | Path, device: str = "cuda", compute_type: str = "int8_float16",
                 beam_size: int = 2, estimated_mb: int = 1500):
        self.model_dir = Path(model_dir)
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.estimated_mb = estimated_mb
        self.model = None
        self.load_s = 0.0

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        if self.model is not None:
            return
        from faster_whisper import WhisperModel

        if not self.model_dir.exists():
            raise SystemExit(
                f"STT model missing at {self.model_dir}. Run:\n"
                f"  python scripts/fetch_models.py --stt turbo")
        started = time.perf_counter()
        self.model = WhisperModel(str(self.model_dir), device=self.device, compute_type=self.compute_type)
        self.load_s = time.perf_counter() - started

    def unload(self) -> None:
        self.model = None
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001
            pass

    # -- inference -----------------------------------------------------------
    def detect(self, audio: np.ndarray) -> tuple[str, float]:
        assert self.model is not None, "call load() first"
        result = self.model.detect_language(audio=audio, language_detection_segments=1)
        if isinstance(result, tuple) and len(result) >= 2:
            return str(result[0]), float(result[1])
        return str(result), 0.0

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        assert self.model is not None, "call load() first"
        started = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        pieces = []
        logprobs = []
        no_speech = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                pieces.append(text)
            logprobs.append(float(getattr(seg, "avg_logprob", 0.0)))
            no_speech.append(float(getattr(seg, "no_speech_prob", 0.0)))
        return Transcript(
            text=" ".join(pieces).strip(),
            language=getattr(info, "language", language or "?"),
            language_prob=float(getattr(info, "language_probability", 0.0)),
            avg_logprob=float(np.mean(logprobs)) if logprobs else -9.0,
            no_speech_prob=float(np.max(no_speech)) if no_speech else 0.0,
            seconds=len(audio) / 16000,
            decode_s=time.perf_counter() - started,
        )


class LanguageGate:
    """Keeps the pipeline inside the configured pair."""

    def __init__(self, allowed: set[str], min_prob: float = 0.60, strict: bool = True):
        self.allowed = {a.lower() for a in allowed}
        self.min_prob = min_prob
        self.strict = strict
        self.rejected: list[tuple[str, float]] = []

    def check(self, lang: str, prob: float) -> tuple[bool, str]:
        if lang.lower() in self.allowed:
            return True, ""
        self.rejected.append((lang, prob))
        if prob < self.min_prob:
            return True, ""      # low confidence: do not trust the detector enough to drop speech
        if self.strict:
            return False, f"language {lang} ({prob:.2f}) is outside the {sorted(self.allowed)} pair"
        return True, ""


class Committer:
    """Turns a stream of partial hypotheses into text that is safe to speak.

    A chunk is committed when it ends at a hard clause boundary and the same
    boundary survived at least one previous hypothesis (LocalAgreement, the idea
    behind ufal's whisper_streaming). Nothing is ever committed twice.
    """

    def __init__(self, min_chars: int = 3, soft_commit: bool = True, max_buffer_chars: int = 400):
        self.min_chars = min_chars
        self.soft_commit = soft_commit
        self.max_buffer_chars = max_buffer_chars
        self._committed = ""
        self._last_hypothesis = ""

    @property
    def committed(self) -> str:
        return self._committed

    def reset(self) -> None:
        self._committed = ""
        self._last_hypothesis = ""

    def update(self, hypothesis: str, final: bool = False) -> str:
        """Return the newly committed text ('' when nothing new is safe yet)."""
        hypothesis = hypothesis.strip()
        if not hypothesis:
            return ""
        previous = self._last_hypothesis
        self._last_hypothesis = hypothesis
        already = self._committed
        if hypothesis.startswith(already):
            candidate = hypothesis[len(already):]
        else:                                   # the model rewrote the committed prefix
            candidate = hypothesis
            already = ""

        cut = 0
        for match in CLAUSE_END.finditer(candidate):
            if match.end() >= self.min_chars:
                cut = match.end()
        if cut == 0 and self.soft_commit:
            for match in SOFT_END.finditer(candidate):
                end = match.end()
                if end >= max(self.min_chars, 12) and candidate[:end] in previous:
                    cut = end

        if final:
            commit = candidate.strip()
        elif cut:
            commit = candidate[:cut].strip()
        else:
            return ""
        if len(commit) < self.min_chars:
            return ""
        self._committed = (already + " " + commit).strip() if already else commit
        return commit

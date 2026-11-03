"""The interpreter loop: audio in, translated speech and captions out.

Four stages run as three threads so a partial hypothesis never blocks the
microphone:

    capture thread   →  Segmenter  →  worker (STT → language gate → MT)  →  TTS thread

Everything a caller needs to see (captions, blind spots, timings) is delivered as
an event dict through ``on_event``, which is what the web UI and the CLI renderer
subscribe to.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import AudioSource, Speaker, read_wav, write_wav
from .config import Config, language_name
from .mt import Glossary, IdentityMt, NllbMt
from .stt import Committer, LanguageGate, WhisperStt
from .tts import PiperTts, split_for_speech
from .vad import SR, Segmenter, Utterance, segment_file
from .vram import Budget, GpuSampler, VramError

Event = dict


@dataclass
class Engines:
    """Engines shared by two directions (bidirectional meetings).

    One recogniser and one multilingual translator serve both directions, so a
    meeting costs the same VRAM as a single direction — the pair is what is
    resident, not the direction.
    """

    budget: Budget
    stt: "WhisperStt | None" = None
    mt: "NllbMt | None" = None
    warm: bool = False


@dataclass
class SegmentResult:
    index: int
    source_text: str = ""
    translated_text: str = ""
    src_lang: str = ""
    lang_prob: float = 0.0
    audio_s: float = 0.0
    stt_s: float = 0.0
    mt_s: float = 0.0
    tts_s: float = 0.0
    lag_s: float = 0.0            # end of speech → first translated audio
    first_out_s: float = 0.0      # start of speech → first translated audio
    dropped: str = ""
    partial: bool = False
    audio_path: str = ""

    def row(self) -> dict[str, object]:
        return asdict(self)


class Interpreter:
    def __init__(self, cfg: Config, on_event: Callable[[Event], None] | None = None,
                 budget: Budget | None = None, engines: "Engines | None" = None):
        self.cfg = cfg
        self.on_event = on_event or (lambda event: None)
        self.engines = engines
        self.budget = (engines.budget if engines is not None else None) or budget or Budget(
            cfg.vram_ceiling_mb, log=self._log)
        self.stt: WhisperStt | None = None
        self.mt: NllbMt | IdentityMt | None = None
        self.tts: PiperTts | None = None
        self.glossary = Glossary.load(cfg.glossary_path)
        self.gate = LanguageGate({cfg.source}, cfg.min_lang_prob, cfg.strict_language)
        self.results: list[SegmentResult] = []
        self._lock = threading.Lock()
        self._jobs: "queue.Queue[tuple]" = queue.Queue()
        self._out: "queue.Queue[tuple]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._speaker_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._committer = Committer(cfg.min_commit_chars)
        self._utt_index = 0

    # -- plumbing ------------------------------------------------------------
    def _log(self, message: str) -> None:
        self.on_event({"type": "log", "message": message})

    def _emit(self, event: Event) -> None:
        self.on_event(event)

    # -- model lifecycle -----------------------------------------------------
    def warm(self) -> None:
        """Load the pair. With shared :class:`Engines` the STT/MT are already resident
        (that is how bidirectional mode stays inside one pair's worth of VRAM)."""
        prof = self.cfg.prof
        if self.engines is not None and self.engines.warm:
            self.stt = self.engines.stt
            self.mt = self.engines.mt
            self.budget = self.engines.budget
        else:
            stt_mb = 900 if "turbo" in prof.stt_model else 1600
            self.stt = WhisperStt(self.cfg.stt_path(), self.cfg.device, self.cfg.compute_type,
                                  beam_size=prof.beam_size, estimated_mb=stt_mb)
            mt_mb = 700 if prof.mt_size == "600m" else 1500
            self.mt = NllbMt(self.cfg.mt_path(), self.cfg.device, self.cfg.compute_type,
                             beam_size=prof.beam_size, estimated_mb=mt_mb)
            for name, model in (("stt", self.stt), ("mt", self.mt)):
                if model is None:
                    continue
                try:
                    self.budget.admit(name, getattr(model, "estimated_mb", 0), unload=model.unload)
                    self.budget.measure(name, model.load)
                except VramError as exc:
                    raise SystemExit(f"{exc}") from exc
                self._emit({"type": "model", "name": name, "vram_mb": self.budget.snapshot().get(name, 0),
                            "load_s": round(getattr(model, "load_s", 0.0), 2)})
            if self.engines is not None:
                self.engines.stt, self.engines.mt = self.stt, self.mt
                self.engines.warm = True

        if self.cfg.speak:
            key = f"tts:{self.cfg.target}"
            self.tts = PiperTts(self.cfg.voice_path(self.cfg.target))
            if not self.budget.has(key):
                self.budget.admit(key, 0, unload=self.tts.unload)
                self.tts.load()
                self._emit({"type": "model", "name": key, "vram_mb": 0,
                            "load_s": round(self.tts.load_s, 2)})

        self.budget.pin("stt", True)
        self.budget.pin("mt", True)
        self._emit({"type": "ready", "config": self.cfg.describe(), "vram": self.budget.report()})

    def close(self) -> None:
        self._stop.set()
        for name in ("tts", "mt", "stt"):
            self.budget.release(name)

    # -- one chunk: STT → gate → MT ------------------------------------------
    def _decode(self, audio: np.ndarray, final: bool, utt_end: float) -> str:
        """Run STT + gate + commit; return newly committed source text."""
        assert self.stt is not None and self.mt is not None
        lang, prob = self.stt.detect(audio)
        ok, why = self.gate.check(lang, prob)
        if not ok:
            self._emit({"type": "rejected", "language": lang, "prob": round(prob, 2), "reason": why})
            return ""
        transcript = self.stt.transcribe(audio, language=lang if prob >= 0.5 else self.cfg.source)
        if not transcript.text:
            return ""
        committed = self._committer.update(transcript.text, final=final)
        if not committed:
            self._emit({"type": "partial", "text": transcript.text, "lang": transcript.language})
            return ""
        return committed

    def _translate(self, text: str) -> tuple[str, float]:
        assert self.mt is not None
        started = time.perf_counter()
        translated = self.mt.translate(text, self.cfg.source, self.cfg.target)
        elapsed = time.perf_counter() - started
        translated = self.glossary.repair(text, translated)
        return translated, elapsed

    # -- utterance handling --------------------------------------------------
    def _handle_utterance(self, utt: Utterance, t_end: float) -> SegmentResult:
        assert self.stt is not None
        lang, prob = self.stt.detect(utt.audio)
        ok, why = self.gate.check(lang, prob)
        result = SegmentResult(index=self._utt_index, src_lang=lang, lang_prob=prob,
                               audio_s=utt.seconds)
        self._utt_index += 1
        if not ok:
            result.dropped = why
            self._emit({"type": "rejected", "language": lang, "prob": round(prob, 2), "reason": why})
            return result

        self._committer.reset()
        started = time.perf_counter()
        transcript = self.stt.transcribe(utt.audio, language=lang if prob >= 0.5 else self.cfg.source)
        result.stt_s = time.perf_counter() - started
        if not transcript.text:
            result.dropped = "no speech recognised"
            return result

        text = transcript.text
        result.source_text = text
        translated, mt_s = self._translate(text)
        result.translated_text = translated
        result.mt_s = mt_s
        result.lag_s = self._speech_to_output(utt, t_end)
        return result

    def _speech_to_output(self, utt: Utterance, t_end: float) -> float:
        return max(0.0, time.time() - t_end)

    # -- workers -------------------------------------------------------------
    def _start_workers(self, speaker: Speaker | None) -> None:
        self._worker = threading.Thread(target=self._work_loop, name="stt-mt", daemon=True)
        self._worker.start()
        self._speaker_thread = threading.Thread(target=self._speak_loop, args=(speaker,),
                                               name="tts", daemon=True)
        self._speaker_thread.start()

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            job = self._jobs.get()
            if job is None:
                break
            kind = job[0]
            try:
                if kind == "final":
                    _, utt, t_end = job
                    self._committer.reset()
                    result = self._handle_utterance(utt, t_end)
                    if result.translated_text:
                        self._out.put(("speak", result.translated_text, result.index))
                    with self._lock:
                        self.results.append(result)
                    self._emit({"type": "segment", **result.row()})
                elif kind == "partial":
                    _, audio, t_end = job
                    started = time.perf_counter()
                    committed = self._decode(audio, final=False, utt_end=t_end)
                    if committed:
                        translated, mt_s = self._translate(committed)
                        latency = time.perf_counter() - started
                        self._out.put(("speak", translated, None))
                        self._emit({"type": "partial_commit", "source": committed,
                                    "translated": translated, "latency_s": round(latency, 2)})
            except Exception as exc:                      # noqa: BLE001 - a bad chunk must not kill the call
                self._emit({"type": "error", "where": kind, "message": f"{type(exc).__name__}: {exc}"})

    def _speak_loop(self, speaker: Speaker | None) -> None:
        while not self._stop.is_set():
            item = self._out.get()
            if item is None:
                break
            _, text, index = item
            if not self.cfg.speak or self.tts is None:
                self._emit({"type": "caption", "translated": text})
                continue
            for clause in split_for_speech(text):
                started = time.perf_counter()
                chunks = []
                first_audio: float | None = None
                for audio, rate in self.tts.synthesize_stream(clause):
                    chunks.append(audio)
                    if first_audio is None:
                        first_audio = time.perf_counter() - started
                    if speaker is not None:
                        speaker.enqueue(audio, rate)
                if not chunks:
                    continue
                self._emit({"type": "caption", "translated": clause,
                            "tts_first_audio_s": round(first_audio or 0.0, 2)})
                if index is not None:
                    with self._lock:
                        for result in reversed(self.results):
                            if result.index == index and not result.tts_s:
                                result.tts_s = first_audio or 0.0
                                break

    # -- public entry points -------------------------------------------------
    def run_live(self, source: AudioSource, speaker: Speaker | None = None,
                 stop: threading.Event | None = None, partials: bool = True) -> list[SegmentResult]:
        stop = stop or threading.Event()
        self._stop.clear()
        self._start_workers(speaker)
        segmenter = Segmenter(
            threshold_db=self.cfg.vad_threshold_db,
            silence_ms=self.cfg.silence_ms,
            min_speech_ms=self.cfg.min_speech_ms,
            max_s=self.cfg.max_utterance_s,
            block_ms=self.cfg.block_ms,
        )
        last_partial = 0.0
        self._emit({"type": "listening"})
        try:
            with GpuSampler(self.budget):
                for frame in source.frames():
                    if stop.is_set():
                        break
                    now = time.time()
                    for utt in segmenter.push(frame, now=now):
                        self._jobs.put(("final", utt, now))
                    if not partials:
                        continue
                    audio, t_start, speaking = segmenter.in_progress()
                    if (speaking and len(audio) / SR >= 1.2
                            and (now - last_partial) * 1000 >= self.cfg.partial_interval_ms
                            and self._jobs.empty()):
                        last_partial = now
                        self._jobs.put(("partial", audio, now))
                for utt in segmenter.flush():
                    self._jobs.put(("final", utt, time.time()))
        finally:
            self._jobs.put(None)
            self._jobs.put(None)
            if self._worker:
                self._worker.join(timeout=60)
            self._out.put(None)
            if self._speaker_thread:
                self._speaker_thread.join(timeout=120)
            if speaker is not None:
                speaker.flush()
        return list(self.results)

    def translate_file(self, path: str | Path, out_dir: str | Path | None = None,
                       speak_each: bool = False) -> list[SegmentResult]:
        """Offline: segment a file and translate every utterance, measuring real timings."""
        audio, sr = read_wav(path)
        if sr != SR:
            from .audio import resample

            audio = resample(audio, sr, SR)
        utterances = segment_file(audio, threshold_db=self.cfg.vad_threshold_db,
                                  silence_ms=self.cfg.silence_ms, min_speech_ms=self.cfg.min_speech_ms,
                                  max_s=self.cfg.max_utterance_s, block_ms=self.cfg.block_ms)
        out_dir = Path(out_dir) if out_dir else None
        results: list[SegmentResult] = []
        self._emit({"type": "listening"})
        if not utterances:
            self._emit({"type": "error", "where": "file", "message": "no speech found in file"})
            return results
        # one batched MT pass keeps file mode fast without changing the live path
        texts: list[str] = []
        transcripts = []
        assert self.stt is not None
        for utt in utterances:
            lang, prob = self.stt.detect(utt.audio)
            ok, why = self.gate.check(lang, prob)
            if not ok:
                transcripts.append(None)
                texts.append("")
                continue
            t0 = time.perf_counter()
            transcript = self.stt.transcribe(utt.audio, language=lang if prob >= 0.5 else self.cfg.source)
            elapsed = time.perf_counter() - t0
            transcripts.append((transcript, lang, prob, elapsed))
            texts.append(transcript.text)

        started = time.perf_counter()
        assert self.mt is not None
        translated = self.mt.translate_many([t for t in texts], self.cfg.source, self.cfg.target)
        batch_s = time.perf_counter() - started

        for i, (utt, item, tgt_text) in enumerate(zip(utterances, transcripts, translated)):
            if item is None:
                results.append(SegmentResult(index=i, audio_s=utt.seconds,
                                             dropped="language outside the pair"))
                continue
            transcript, lang, prob, elapsed = item
            result = SegmentResult(
                index=i, source_text=transcript.text,
                translated_text=self.glossary.repair(transcript.text, tgt_text),
                src_lang=lang, lang_prob=prob, audio_s=utt.seconds, stt_s=elapsed,
                mt_s=batch_s / max(1, len([t for t in texts if t])),
            )
            if self.cfg.speak and self.tts is not None and out_dir is not None and result.translated_text:
                t0 = time.perf_counter()
                audio_out = []
                rate = 22050
                for clause in split_for_speech(result.translated_text):
                    chunk, rate = self.tts.synthesize(clause)
                    audio_out.append(chunk)
                result.tts_s = time.perf_counter() - t0
                if audio_out:
                    path_out = out_dir / f"{Path(path).stem}.{i:03d}.wav"
                    write_wav(path_out, np.concatenate(audio_out), rate)
                    result.audio_path = str(path_out)
            result.lag_s = result.stt_s + result.mt_s + result.tts_s
            result.first_out_s = result.lag_s
            results.append(result)
            self._emit({"type": "segment", **result.row()})
        return results

    def stats(self) -> dict[str, object]:
        with self._lock:
            rows = [r for r in self.results if r.translated_text]
        lags = sorted(r.lag_s for r in rows)
        stt = sorted(r.stt_s for r in rows)
        mt = sorted(r.mt_s for r in rows)

        def pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            idx = min(len(values) - 1, int(round((len(values) - 1) * p)))
            return values[idx]

        return {
            "segments": len(self.results),
            "translated": len(rows),
            "dropped": len([r for r in self.results if r.dropped]),
            "lag_p50": pct(lags, 0.5),
            "lag_p90": pct(lags, 0.9),
            "stt_p50": pct(stt, 0.5),
            "mt_p50": pct(mt, 0.5),
            "vram": self.budget.report(),
            "gate_rejections": self.gate.rejected,
            "glossary_misses": self.glossary.audit(),
        }

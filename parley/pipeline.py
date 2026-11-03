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
from .text import split_sentences
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
    t_end_wall: float = 0.0       # wall clock when the utterance ended
    t_start_wall: float = 0.0

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
        self._jobs: "queue.PriorityQueue[tuple]" = queue.PriorityQueue()
        self._job_seq = 0
        self._out: "queue.Queue[tuple]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._speaker_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._committer = Committer(cfg.min_commit_chars,
                                    require_agreement=(cfg.commit_mode != "eager"))
        self._utt_index = 0
        self._utt_key: float | None = None
        self._utt_pieces: list[str] = []
        self._utt_lang = ""
        self._first_audio_wall: dict[int, float] = {}
        self._requested_profile = ""

    # -- plumbing ------------------------------------------------------------
    def _log(self, message: str) -> None:
        self.on_event({"type": "log", "message": message})

    def _emit(self, event: Event) -> None:
        self.on_event(event)

    # -- model lifecycle -----------------------------------------------------
    FALLBACKS = {"quality": ("quality", "balanced", "fast"),
                 "balanced": ("balanced", "fast"),
                 "fast": ("fast",)}

    def warm(self) -> None:
        """Load the pair, stepping down a profile if the card cannot fit it.

        A 6 GB laptop shares its VRAM with the browser and the desktop. Refusing to
        start because a tab is holding 2 GB is technically correct and practically
        useless, so parley tries the next smaller profile and says so.
        """
        last_error: Exception | None = None
        requested = self.cfg.profile
        self._requested_profile = requested
        for name in self.FALLBACKS.get(requested, ("balanced", "fast")):
            try:
                self._load_profile(name)
                if name != requested:
                    self._emit({"type": "fallback", "from": requested, "to": name})
                return
            except VramError as exc:
                last_error = exc
                self._emit({"type": "log", "message": f"{name} does not fit: {exc}"})
                continue
        raise SystemExit(str(last_error) if last_error else "could not load any profile")

    def _load_profile(self, name: str) -> None:
        self.cfg.profile = name
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
                    self.budget.ensure_free(name, getattr(model, "estimated_mb", 0))
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

    # -- one chunk: STT → gate → commit --------------------------------------
    def _detect(self, audio: np.ndarray) -> tuple[str, float]:
        """Language detection, except where it cannot be trusted.

        Whisper's language ID needs roughly a second of speech; below that it guesses,
        and a confident wrong guess makes the gate refuse the user's own sentence. For
        very short clips the configured source language is the better prior.
        """
        assert self.stt is not None
        if len(audio) / SR < 0.9:
            return self.cfg.source, 1.0
        return self.stt.detect(audio)

    def _decode(self, audio: np.ndarray, final: bool) -> tuple[str, str]:
        """Run STT + gate + commit; return (full transcript, newly committed text).

        Partials decode greedily (beam 1): they are speculative, only their agreed
        prefix is ever spoken, and the seconds they save are the seconds the listener
        waits.
        """
        assert self.stt is not None
        lang, prob = self._detect(audio)
        ok, why = self.gate.check(lang, prob)
        if not ok:
            self._emit({"type": "rejected", "language": lang, "prob": round(prob, 2), "reason": why})
            return "", ""
        self._utt_lang = lang
        transcript = self.stt.transcribe(audio, language=lang if prob >= 0.5 else self.cfg.source,
                                         beam_size=None if final else 1)
        if not transcript.text:
            return "", ""
        committed = self._committer.update(transcript.text, final=final)
        return transcript.text, committed

    def _translate(self, text: str) -> tuple[str, float]:
        """Translate sentence by sentence.

        NLLB returns only the last sentence of a multi-sentence input (measured),
        so the source is always split first — which also gives the TTS stage
        clause-sized units to speak as they arrive.
        """
        assert self.mt is not None
        prepared = self.glossary.inject(text)
        sentences = split_sentences(prepared)
        started = time.perf_counter()
        if len(sentences) <= 1:
            translated = self.mt.translate(prepared, self.cfg.source, self.cfg.target)
        else:
            translated = " ".join(self.mt.translate_many(sentences, self.cfg.source, self.cfg.target))
        elapsed = time.perf_counter() - started
        return self.glossary.repair(text, translated), elapsed

    # -- workers -------------------------------------------------------------
    def _start_workers(self, speaker: Speaker | None) -> None:
        self._worker = threading.Thread(target=self._work_loop, name="stt-mt", daemon=True)
        self._worker.start()
        self._speaker_thread = threading.Thread(target=self._speak_loop, args=(speaker,),
                                               name="tts", daemon=True)
        self._speaker_thread.start()

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            entry = self._jobs.get()
            if entry is None:
                break
            _, _, job = entry
            if job is None:
                break
            kind = job[0]
            try:
                if kind == "partial":
                    _, audio, key, t_end = job
                    self._begin(key, t_end)
                    transcript_text, committed = self._decode(audio, final=False)
                    if transcript_text and not committed:
                        self._emit({"type": "partial", "text": transcript_text,
                                    "audio_s": round(len(audio) / SR, 2)})
                    if committed:
                        started = time.perf_counter()
                        translated, _mt_s = self._translate(committed)
                        self._utt_pieces.append(translated)
                        self._out.put(("speak", translated, self._utt_index, False))
                        self._emit({"type": "partial_commit", "source": committed,
                                    "translated": translated,
                                    "latency_s": round(time.perf_counter() - started, 2)})
                elif kind == "final":
                    _, utt, key, t_end = job
                    self._begin(key, t_end)
                    t0 = time.perf_counter()
                    transcript_text, committed = self._decode(utt.audio, final=True)
                    stt_s = time.perf_counter() - t0
                    translated = ""
                    mt_s = 0.0
                    if committed:
                        translated, mt_s = self._translate(committed)
                        self._utt_pieces.append(translated)
                    result = SegmentResult(
                        index=self._utt_index, source_text=transcript_text,
                        translated_text=" ".join(p for p in self._utt_pieces if p),
                        src_lang=self._utt_lang,
                        audio_s=utt.seconds, stt_s=stt_s, mt_s=mt_s,
                        t_end_wall=t_end, t_start_wall=t_end - utt.seconds,
                        first_out_s=self._first_out(self._utt_index, t_end - utt.seconds),
                    )
                    if not result.translated_text:
                        result.dropped = "no speech recognised"
                    if not self.cfg.speak:               # captions-only: lag ends when the text is ready
                        result.lag_s = max(0.0, time.time() - t_end)
                    with self._lock:
                        self.results.append(result)
                    self._utt_index += 1
                    self._emit({"type": "segment", **result.row()})
                    if translated:                       # enqueue after the row exists, so the
                        self._out.put(("speak", translated, result.index, True))  # lag stamp can find it
            except Exception as exc:                      # noqa: BLE001 - a bad chunk must not kill the call
                self._emit({"type": "error", "where": kind, "message": f"{type(exc).__name__}: {exc}"})

    def _begin(self, key: float, t_end: float) -> None:
        """Start a new utterance (reset commit state) or continue the current one."""
        if getattr(self, "_utt_key", None) != key:
            self._utt_key = key
            self._committer.reset()
            self._utt_pieces = []

    def _enqueue_job(self, job: tuple, priority: int = 1) -> None:
        """Finals jump ahead of partials: a partial must never delay the sentence end."""
        self._job_seq += 1
        self._jobs.put((priority, self._job_seq, job))

    def _speak_loop(self, speaker: Speaker | None) -> None:
        while not self._stop.is_set():
            item = self._out.get()
            if item is None:
                break
            _, text, index, is_final = item
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
                        if is_final:
                            self._mark_output(index)
                        else:
                            self._note_first(index)
                    if speaker is not None:
                        speaker.enqueue(audio, rate)
                if not chunks:
                    continue
                self._emit({"type": "caption", "translated": clause,
                            "tts_first_audio_s": round(first_audio or 0.0, 2)})
                if index is not None and is_final:
                    with self._lock:
                        for result in reversed(self.results):
                            if result.index == index and not result.tts_s:
                                result.tts_s = first_audio or 0.0
                                break

    def _note_first(self, index: int | None) -> None:
        """Record the wall clock when this utterance's first translated audio existed."""
        if index is None or index in self._first_audio_wall:
            return
        self._first_audio_wall[index] = time.time()

    def _first_out(self, index: int, t_start_wall: float) -> float:
        stamp = self._first_audio_wall.get(index)
        return max(0.0, stamp - t_start_wall) if stamp else 0.0

    def _mark_output(self, index: int | None) -> None:
        """Stamp the moment translated audio actually reached the device."""
        if index is None:
            return
        now = time.time()
        with self._lock:
            for result in reversed(self.results):
                if result.index == index and not result.lag_s:
                    result.lag_s = max(0.0, now - result.t_end_wall)
                    result.first_out_s = max(0.0, now - result.t_start_wall)
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
                        self._enqueue_job(("final", utt, utt.t_start, now), priority=0)
                    if not partials:
                        continue
                    audio, t_start, speaking = segmenter.in_progress()
                    if (speaking and len(audio) / SR >= 1.0
                            and (now - last_partial) * 1000 >= self.cfg.partial_interval_ms
                            and self._jobs.empty()):
                        last_partial = now
                        self._enqueue_job(("partial", audio, t_start, now), priority=1)
                for utt in segmenter.flush():
                    self._enqueue_job(("final", utt, utt.t_start, time.time()), priority=0)
        finally:
            self._jobs.put((0, 0, None))
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
        # speech recognition per utterance (language gate included)
        transcripts: list[tuple | None] = []
        assert self.stt is not None
        for utt in utterances:
            lang, prob = self._detect(utt.audio)
            ok, why = self.gate.check(lang, prob)
            if not ok:
                self._emit({"type": "rejected", "language": lang, "prob": round(prob, 2), "reason": why})
                transcripts.append(None)
                continue
            t0 = time.perf_counter()
            transcript = self.stt.transcribe(utt.audio, language=lang if prob >= 0.5 else self.cfg.source)
            elapsed = time.perf_counter() - t0
            transcripts.append((transcript, lang, prob, elapsed))

        # translate every sentence of every utterance in one batched pass; NLLB
        # drops all but the last sentence of a multi-sentence input, so sentences
        # are the unit of work here too.
        jobs: list[tuple[int, str]] = []
        for idx, item in enumerate(transcripts):
            if item is None:
                continue
            for sentence in split_sentences(self.glossary.inject(item[0].text)):
                if sentence.strip():
                    jobs.append((idx, sentence))
        started = time.perf_counter()
        assert self.mt is not None
        pieces = self.mt.translate_many([s for _, s in jobs], self.cfg.source, self.cfg.target)
        batch_s = time.perf_counter() - started
        per_sentence = batch_s / max(1, len(jobs))
        assembled: dict[int, list[str]] = {}
        for (idx, _), piece in zip(jobs, pieces):
            assembled.setdefault(idx, []).append(piece)

        for i, (utt, item) in enumerate(zip(utterances, transcripts)):
            if item is None:
                results.append(SegmentResult(index=i, audio_s=utt.seconds,
                                             dropped="language outside the pair"))
                continue
            transcript, lang, prob, elapsed = item
            joined = " ".join(assembled.get(i, []))
            result = SegmentResult(
                index=i, source_text=transcript.text,
                translated_text=self.glossary.repair(transcript.text, joined),
                src_lang=lang, lang_prob=prob, audio_s=utt.seconds, stt_s=elapsed,
                mt_s=per_sentence * max(1, len(assembled.get(i, []))),
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
            for row in rows:                      # first audio may have arrived after the row was built
                stamp = self._first_out(row.index, row.t_start_wall)
                if stamp:
                    row.first_out_s = stamp
        lags = sorted(r.lag_s for r in rows)
        firsts = sorted(r.first_out_s for r in rows if r.first_out_s)
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
            "first_out_p50": pct(firsts, 0.5),
            "stt_p50": pct(stt, 0.5),
            "mt_p50": pct(mt, 0.5),
            "vram": self.budget.report(),
            "gate_rejections": self.gate.rejected,
            "glossary_misses": self.glossary.audit(),
        }

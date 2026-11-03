# Architecture

## Threads

```
                 ┌────────────── capture (PortAudio callback) ──────────────┐
  mic / loopback │ 16 kHz mono blocks, 30 ms                                 │
                 └───────────────────────────┬──────────────────────────────┘
                                             ▼
                                  Segmenter (VAD, adaptive)
                       speech span start ── partial decode every 700 ms
                       silence 420 ms    ── final utterance
                                             ▼
                        worker thread: Whisper → language gate → NLLB
                                    (Committer decides what is speakable)
                                             ▼
                        TTS thread: clause split → Piper (CPU) → Speaker
```

Three threads, two queues. The capture path never blocks on inference: a partial
hypothesis that is still being decoded cannot stall the microphone, and the speaker
thread owns playback so audio never stutters while the translator is busy.

## Why the commit policy exists

Whisper is not a streaming model. It re-decodes a growing window and rewrites what it
already said. Naively piping partials into TTS produces the sentence twice, or a
half-sentence followed by its correction, read aloud. `Committer` therefore:

1. only commits text ending at a hard clause boundary (`. ! ? ; : …`);
2. requires the boundary to have survived the previous hypothesis (LocalAgreement —
   the idea behind `ufal/whisper_streaming`);
3. never commits the same span twice, even when the hypothesis rewrites its prefix;
4. flushes the tail on the final hypothesis of an utterance.

## Why the language gate is before translation

Measured behaviour of Whisper: the language token selects the **output** language.
Pinning it to `de` while the room speaks Persian yields fluent German — a silent
translation, with no error and no flag. See
[whisper-language-token.md](whisper-language-token.md).

So `LanguageGate` runs on every utterance:

* detect language + probability,
* accept only languages in the session's pair,
* refuse when the detector is confident about a foreign language (≥ 0.60),
* pass through when the detector is unsure (a 0.3 detection is not evidence),
* emit `{"type": "rejected", ...}` so the operator sees the refusal.

## The VRAM budget

`Budget` is an LRU of resident models with a hard ceiling:

```python
budget.admit("stt", 900, unload=stt.unload)   # evicts if needed, refuses if impossible
budget.measure("stt", stt.load)               # real GPU delta from nvidia-smi
```

Estimates are replaced by measurements right after each load, so `README` figures
and `parley doctor` output come from the card. `GpuSampler` polls during a run to
record the peak.

Two consequences worth stating plainly:

* **A pair, not a language list.** NLLB holds 200 languages in one file, so switching
  from en→de to en→fr costs nothing. A per-pair model design (OPUS-MT style) would
  need a new 300 MB model per pair; the LRU exists so that path can be added later
  without changing callers.
* **Bidirectional is not double.** `Engines` shares the recogniser and translator
  between the two directions; only the voice (CPU) differs.

## Failure behaviour

| Failure | What parley does |
|---|---|
| foreign language detected | refuses, emits `rejected`, keeps listening |
| no speech in a span | drops it, no TTS |
| model missing | `parley doctor` fails early with the exact fetch command |
| model bigger than ceiling | `VramError` at startup, profile named in the message |
| STT/MT exception mid-run | logged as an `error` event, loop continues |
| loopback device missing | clear instruction (enable Stereo Mix / install VB-Cable) |

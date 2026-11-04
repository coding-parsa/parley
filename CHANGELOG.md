# Changelog

## 0.3.2 — Start actually starts

The app appeared to hang on Start and then stick on "stopping…" forever.

**Cause:** `resample()` imported `scipy.signal` *inside* the capture loop. The first
audio block therefore triggered a multi-second import while holding the read path:
no frames reached the segmenter, the device buffer overflowed, and the stop flag was
never reached because the thread was inside the import. In the frozen build it was
long enough to look like a dead application.

- Resolution is now numpy-only: a cached windowed-sinc low-pass plus decimation
  (`48k→16k`, `32k→16k`, `96k→16k` integer ratios; interpolated path for
  `44.1k/22.05k→16k`). 0.5 s of audio converts in **under 2 ms**, and scipy is gone
  from the bundle entirely.
- Capture now asks the driver for **16 kHz directly**, so WASAPI does the conversion
  in the mixer and the loop does none. It falls back to the device's native rate if
  the driver refuses.
- Stop has a 5 s watchdog: the window returns to `ready` even if a decode or a
  device call outlives the request, instead of showing "stopping…" indefinitely.

Also new: `parley.exe --live-test 10` records ten seconds from the real microphone in
the packaged build and writes what it heard to `%LOCALAPPDATA%\parley\selftest.log` —
the check that would have caught this.

## 0.3.1 — the build that actually speaks

Four bugs stood between 0.3.0 and a usable app. All four are fixed and verified
inside the packaged `.exe`, not just in the source tree.

### Speech output was silent
`Speaker` played through a PortAudio callback that resampled with swapped
arguments and swallowed its own exceptions: captions were perfect, the room was
quiet, and nothing anywhere reported a failure.

It is now a blocking writer thread. `write()` returns only when the card has taken
the samples, so "did it play" is measurable: `frames_written` and `last_error` are
readable, `flush()` waits for the item in flight instead of merely an empty queue,
and any playback error reaches the UI as a message.

Verified: `parley.exe --selftest` plays 0.4 s and reports
`17640 frames to device` — exactly 0.4 s at 44.1 kHz.

### The second Start failed
The GUI built a new `Interpreter` for every session and never released the old
one's models, so the second session loaded another 1.8 GB on top of the first and
ran out of VRAM. It also carried the previous session's transcripts.

Now the interpreter is loaded once and reused: `reset_session()` clears the
transcript, committer, and counters between runs, and model memory stays constant.

Verified: three consecutive sessions report `resident 1821 MB` each, no errors.

### The language gate refused the user's own speech
Whisper labels short or accented English as another language often enough that a
hard rejection turned into "the app refused to translate me". The detection
probability threshold alone cannot fix this.

The gate is now advisory: when the detector disagrees, the audio is decoded as the
configured source language anyway, and it is only dropped when that decode is
nonsense (`avg_logprob < -0.8`). Genuinely foreign audio still gets refused —
German speech in an English session is rejected with `de (1.00)`.

### Console windows flashed
`vram.py` shelled out to `nvidia-smi` from a windowed process with no
`CREATE_NO_WINDOW` and no caching, so the VRAM poller flickered consoles
continuously.

All subprocess calls now pass `creationflags=CREATE_NO_WINDOW`, and the query is
cached for 1.5 s.

Verified by enumerating visible `ConsoleWindowClass` windows during a full demo
run: peak **0**.

### Also
- The self-test now proves playback, phonemisation, and a real translation inside
  the frozen build, and prints the transcript it actually heard.
- Temporary nvidia-smi failures no longer kill a session: `VramError` propagates
  instead of being swallowed by `SystemExit`.
- A TTS failure is reported per clause instead of silently ending the voice thread.
- `NllbMt.translate()` splits sentences internally, so no caller can trigger
  NLLB's silent "only the last sentence" truncation.
- `parley.exe --demo` runs the demo automatically, for screenshots and smoke tests.

## 0.3.0

First packaged Windows build: STT → language gate → NLLB → Piper, VRAM budgeting
with profile fallback, GUI, CLI, and benchmarks (balanced: BLEU 48.8, lag 1.17 s;
quality: BLEU 56.1, lag 2.01 s).

"""Entry point for the frozen desktop app (PyInstaller).

Also hosts `--selftest`, which exercises every native dependency and one real
translation *inside the packaged build*. A windowed exe has nowhere to print, so the
result goes to %LOCALAPPDATA%\\parley\\selftest.log and the exit code says pass/fail.
That is what makes "it works on my machine" checkable after freezing.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path


def log_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home()))
    return base / "parley" / "selftest.log"


def selftest() -> int:
    lines: list[str] = []
    ok = True

    def step(name: str, fn):
        nonlocal ok
        started = time.perf_counter()
        try:
            detail = fn()
            lines.append(f"PASS  {name}  ({time.perf_counter() - started:.2f}s)  {detail or ''}")
        except Exception as exc:                      # noqa: BLE001
            ok = False
            lines.append(f"FAIL  {name}  {type(exc).__name__}: {exc}")
            lines.append(traceback.format_exc())

    def imports():
        import ctranslate2, faster_whisper, huggingface_hub, numpy, piper  # noqa: F401
        import sentencepiece, sounddevice, soundfile, tkinter  # noqa: F401
        return "all native modules import"

    def models():
        from parley.config import Config

        cfg = Config()
        missing = [str(p) for p in (cfg.stt_path(), cfg.mt_path()) if not (p / "model.bin").exists()]
        if missing:
            raise RuntimeError(f"models missing: {missing}")
        return f"models at {cfg.models_dir}"

    def translate():
        import numpy as np

        from parley.audio import write_wav
        from parley.config import Config
        from parley.pipeline import Interpreter

        cfg = Config(source="en", target="de", profile="balanced", speak=True)
        interpreter = Interpreter(cfg, on_event=lambda _e: None)
        interpreter.warm()

        from parley.config import MODELS_DIR
        from parley.tts import PiperTts

        # speak the test sentence with the SOURCE voice — the interpreter's own TTS is
        # the target-language voice, and feeding it English proves nothing
        source_voice = MODELS_DIR / "piper" / "en_US-hfc_male-medium.onnx"
        tts = PiperTts(source_voice)
        tts.load()
        audio, rate = tts.synthesize("Thanks everyone for joining. The budget is ready for review.")
        seconds = len(audio) / max(rate, 1)
        if len(audio) == 0:
            raise RuntimeError("speech synthesis produced no audio (missing espeak-ng data in the bundle?)")
        sample_dir = Path(os.environ.get("TEMP", ".")) / "parley-selftest"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample = sample_dir / "selftest.wav"
        write_wav(sample, audio, rate)
        from parley.audio import read_wav

        back, back_rate = read_wav(sample)
        from parley.audio import resample

        detected = interpreter._detect(resample(back, back_rate, 16000))
        results = interpreter.translate_file(sample, out_dir=sample_dir)
        text = " ".join(r.translated_text for r in results).strip()
        detail = [(round(r.audio_s, 2), r.src_lang, r.dropped) for r in results]
        if not text:
            raise RuntimeError(
                f"no translation produced · wav {len(back)/back_rate:.2f}s @ {back_rate} Hz · "
                f"synth {seconds:.2f}s @ {rate} Hz · detected {detected} · results={detail}")
        return f"→ {text[:60]}  [{seconds:.1f}s audio, detected {detected[0]}/{detected[1]:.2f}, " \
               f"heard {results[0].source_text[:50]!r}]"

    def phonemes():
        """Phonemisation is where a frozen build can silently break: espeak data and
        the voice language must both resolve, or the voice mispronounces everything."""
        from piper.phonemize_espeak import ESPEAK_DATA_DIR, EspeakPhonemizer

        phonemizer = EspeakPhonemizer()
        sample = phonemizer.phonemize("en-us", "Thanks everyone for joining.")[0]
        return f"{ESPEAK_DATA_DIR.exists()=} {' '.join(sample)[:90]}"

    def playback():
        import numpy as np

        from parley.audio import Speaker

        speaker = Speaker()
        speaker.open()
        t = np.arange(int(16000 * 0.4)) / 16000.0
        tone = (0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        before = speaker.frames_written
        speaker.enqueue(tone, 16000)
        speaker.flush(timeout=20)
        written = speaker.frames_written - before
        error = speaker.last_error
        speaker.close()
        if written <= 0:
            raise RuntimeError(f"nothing reached the sound card (error={error})")
        return f"{written} frames to device {speaker.device}"

    def audio_devices():
        import sounddevice as sd

        inputs = [d["name"] for d in sd.query_devices() if d["max_input_channels"]]
        outputs = [d["name"] for d in sd.query_devices() if d["max_output_channels"]]
        if not outputs:
            raise RuntimeError("no output device")
        return f"{len(inputs)} inputs, {len(outputs)} outputs"

    step("imports", imports)
    step("audio devices", audio_devices)
    step("playback", playback)
    step("phonemes", phonemes)
    step("models", models)
    step("translate", translate)

    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n") + "\n".join(lines) + "\n", encoding="utf-8")
    try:
        print("\n".join(lines))
        print("SELFTEST", "PASS" if ok else "FAIL")
    except Exception:                                  # noqa: BLE001 - windowed exe has no stdout
        pass
    return 0 if ok else 1


def live_check(seconds: float = 10.0) -> int:
    """Record from the real microphone for N seconds and report what arrived.

    The frozen build's live path cannot be clicked from a script, and "it works" is
    not something to assume about audio devices: this counts frames, overflows and
    any transcript, then writes it to the selftest log.
    """
    import threading

    from parley.audio import open_source
    from parley.config import Config
    from parley.pipeline import Interpreter

    cfg = Config(source="en", target="de", profile="fast", speak=False)
    interpreter = Interpreter(cfg, on_event=lambda _e: None)
    lines = []
    try:
        interpreter.warm()
        source = open_source("", loopback=False, block_ms=cfg.block_ms)
        stop = threading.Event()
        threading.Timer(seconds, stop.set).start()
        with source:
            interpreter.run_live(source, None, stop, partials=False)
        stats = interpreter.stats()
        lines.append(f"device={source.samplerate} Hz · overflows={source.overflow} · "
                     f"segments={stats['segments']} · translated={stats['translated']}")
        for result in interpreter.results[:3]:
            lines.append(f"  heard {result.source_text[:60]!r} → {result.translated_text[:60]!r}")
        if stats["segments"] == 0:
            lines.append("  (silence: no speech detected in the window — that is fine, "
                         "but say something next time)")
    except Exception as exc:                            # noqa: BLE001
        lines.append(f"FAILED {type(exc).__name__}: {exc}")
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n") + "LIVE CHECK\n" +
                    "\n".join(lines) + "\n", encoding="utf-8")
    try:
        print("\n".join(lines))
    except Exception:                                   # noqa: BLE001
        pass
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if "--live-test" in sys.argv:
        seconds = 10.0
        index = sys.argv.index("--live-test")
        if len(sys.argv) > index + 1:
            try:
                seconds = float(sys.argv[index + 1])
            except ValueError:
                pass
        return live_check(seconds)
    if "--demo" in sys.argv:
        os.environ["PARLEY_AUTODEMO"] = "1"

    from parley.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())

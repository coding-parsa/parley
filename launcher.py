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
        import ctranslate2, faster_whisper, huggingface_hub, numpy, piper, scipy.signal  # noqa: F401
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
        tts = interpreter.tts
        audio, rate = tts.synthesize("Thanks everyone for joining. The budget is ready for review.")
        seconds = len(audio) / max(rate, 1)
        if len(audio) == 0:
            raise RuntimeError("speech synthesis produced no audio (missing espeak-ng data in the bundle?)")
        sample_dir = Path(os.environ.get("TEMP", ".")) / "parley-selftest"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample = sample_dir / "selftest.wav"
        write_wav(sample, audio, rate)
        results = interpreter.translate_file(sample, out_dir=sample_dir)
        text = " ".join(r.translated_text for r in results).strip()
        if not text:
            detail = [(r.audio_s, r.src_lang, r.dropped) for r in results] or "no utterances segmented"
            raise RuntimeError(f"no translation produced ({seconds:.1f}s of audio); results={detail}")
        return f"→ {text[:70]}  [{seconds:.1f}s audio]"

    def audio_devices():
        import sounddevice as sd

        inputs = [d["name"] for d in sd.query_devices() if d["max_input_channels"]]
        outputs = [d["name"] for d in sd.query_devices() if d["max_output_channels"]]
        if not outputs:
            raise RuntimeError("no output device")
        return f"{len(inputs)} inputs, {len(outputs)} outputs"

    step("imports", imports)
    step("audio devices", audio_devices)
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


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()

    from parley.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())

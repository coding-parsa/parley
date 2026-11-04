"""Prove audio reaches the sound card, without needing a loopback device.

`Speaker` writes through PortAudio with a blocking writer: if the card takes the
samples, playing one second of audio takes about one second and `frames_written`
matches. If the old callback swallowed its exception, this reports zero frames in
milliseconds — exactly the "text but no speech" failure.
"""

import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\parsa\parley")

from parley.audio import Speaker
from parley.config import MODELS_DIR
from parley.tts import PiperTts


def tone(seconds: float = 1.0, freq: float = 440.0, amp: float = 0.15) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000.0
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def play(speaker: Speaker, audio: np.ndarray, rate: int, label: str) -> bool:
    before = speaker.frames_written
    started = time.perf_counter()
    speaker.enqueue(audio, rate)
    speaker.flush(timeout=30)
    elapsed = time.perf_counter() - started
    written = speaker.frames_written - before
    ok = written > 0 and elapsed > 0.5 * len(audio) / rate
    print(f"{label:24s} {len(audio)/rate:4.2f}s audio · wrote {written} frames "
          f"({written/speaker.samplerate:.2f}s) in {elapsed:.2f}s · error={speaker.last_error}")
    return ok


def main() -> int:
    speaker = Speaker()
    print(f"device={speaker.device} samplerate={speaker.samplerate}")
    speaker.open()
    results = [play(speaker, tone(), 16000, "440 Hz tone")]

    tts = PiperTts(MODELS_DIR / "piper" / "de_DE-thorsten-high.onnx")
    tts.load()
    german, rate = tts.synthesize("Die Besprechung findet morgen um zehn Uhr statt.")
    results.append(play(speaker, german, rate, "German TTS sentence"))

    speaker.close()
    print("RESULT:", "AUDIO PLAYED" if all(results) else "NO AUDIO — investigate")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

"""How fast does Piper actually start speaking? First-chunk latency by clause length."""

import time
from pathlib import Path

import numpy as np

from parley.tts import PiperTts

CACHE = Path(r"C:\Users\parsa\AppData\Local\hermes\cache\piper-voices")

samples = {
    "short (35c)": "Die Sitzung beginnt morgen früh.",
    "medium (80c)": "Bitte senden Sie mir den Bericht vorher, damit ich ihn noch lesen kann.",
    "long (150c)": ("Wir haben heute über das Projekt gesprochen und die wichtigsten Punkte "
                    "zusammengefasst, damit alle Beteiligten den gleichen Stand haben."),
}

for name, voice in (("de high", "de_DE-thorsten-high.onnx"), ("en med", "en_US-hfc_male-medium.onnx")):
    tts = PiperTts(CACHE / voice)
    tts.load()
    tts.synthesize("Hallo.")  # warm phonemizer
    print(f"\n=== {name} ===")
    for label, text in samples.items():
        t0 = time.perf_counter()
        first = None
        total_audio = 0
        rate = 22050
        for chunk, sr in tts.synthesize_stream(text):
            if first is None:
                first = time.perf_counter() - t0
            total_audio += len(chunk)
            rate = sr
        wall = time.perf_counter() - t0
        print(f"{label:13s} first audio {first*1000:5.0f} ms · audio {total_audio/rate:4.2f}s · "
              f"wall {wall:4.2f}s · rtf {wall/max(total_audio/rate,1e-6):.2f}")

"""Smoke test: Piper speaks, Whisper hears it back, timings printed."""

import time
from pathlib import Path

import numpy as np

from parley.audio import write_wav
from parley.stt import WhisperStt
from parley.tts import PiperTts, split_for_speech

CACHE = Path(r"C:\Users\parsa\AppData\Local\hermes\cache\piper-voices")
TURBO = Path(r"C:\Users\parsa\models\faster-whisper-turbo")

text = ("Die Besprechung findet morgen um zehn Uhr statt. "
        "Bitte schicken Sie mir den Finanzbericht vorher, damit ich ihn lesen kann.")

tts = PiperTts(CACHE / "de_DE-thorsten-high.onnx")
t0 = time.perf_counter()
tts.load()
print(f"tts load {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
audio = []
rate = 22050
first = None
for clause in split_for_speech(text):
    for chunk, sr in tts.synthesize_stream(clause):
        if first is None:
            first = time.perf_counter() - t0
        audio.append(chunk)
        rate = sr
full = np.concatenate(audio)
print(f"tts {len(text)} chars -> {len(full)/rate:.2f}s audio, first chunk in {first:.2f}s, "
      f"total {time.perf_counter()-t0:.2f}s, peak {np.abs(full).max():.2f}")
out = write_wav("C:/Users/parsa/parley/tests/audio/de_reference.wav", full, rate)
print("wrote", out)

stt = WhisperStt(TURBO, device="cuda", compute_type="int8_float16", beam_size=2)
t0 = time.perf_counter()
stt.load()
print(f"stt load {time.perf_counter()-t0:.2f}s")

lang, prob = stt.detect(full if rate == 16000 else full)
print("detect:", lang, round(prob, 2))
tr = stt.transcribe(full, language="de")
print(f"stt {tr.seconds:.1f}s audio in {tr.decode_s:.2f}s -> {tr.text}")

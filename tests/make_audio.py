"""Build the test corpus: Piper speaks the script, references are written next to it.

The references are author-written translations, so the BLEU/chrF from `parley bench`
measures agreement with a human reference — not ground truth. They exist to catch
regressions, which is all a project this size can honestly claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from parley.audio import write_wav
from parley.tts import PiperTts, split_for_speech

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio"
VOICES = ROOT.parent / "models" / "piper"

CLIPS = [
    {
        "name": "en_meeting_01",
        "lang": "en",
        "voice": "en_US-hfc_male-medium.onnx",
        "text": ("Thanks everyone for joining. Today we need to review the deployment schedule "
                 "and the budget for the next quarter."),
        "ref": {"de": "Danke, dass alle dabei sind. Heute müssen wir den Zeitplan für die "
                       "Bereitstellung und das Budget für das nächste Quartal prüfen."},
    },
    {
        "name": "en_meeting_02",
        "lang": "en",
        "voice": "en_US-hfc_male-medium.onnx",
        "text": ("The client asked for a demo on Thursday, so the release has to be ready "
                 "by Wednesday evening."),
        "ref": {"de": "Der Kunde hat für Donnerstag eine Demo verlangt, also muss die "
                       "Veröffentlichung bis Mittwochabend fertig sein."},
    },
    {
        "name": "en_meeting_03",
        "lang": "en",
        "voice": "en_US-hfc_male-medium.onnx",
        "text": ("I think we should postpone the migration until the new database is stable."),
        "ref": {"de": "Ich denke, wir sollten die Migration verschieben, bis die neue "
                       "Datenbank stabil ist."},
    },
    {
        "name": "de_meeting_01",
        "lang": "de",
        "voice": "de_DE-thorsten-high.onnx",
        "text": ("Die Besprechung findet morgen um zehn Uhr statt. Bitte schicken Sie mir den "
                 "Finanzbericht vorher."),
        "ref": {"en": "The meeting takes place tomorrow at ten o'clock. Please send me the "
                       "financial report beforehand."},
    },
    {
        "name": "de_meeting_02",
        "lang": "de",
        "voice": "de_DE-thorsten-high.onnx",
        "text": ("Wir müssen die Kosten für den neuen Server reduzieren, sonst wird das "
                 "Projekt zu teuer."),
        "ref": {"en": "We have to reduce the cost of the new server, otherwise the project "
                       "becomes too expensive."},
    },
]


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        tts = PiperTts(VOICES / clip["voice"])
        tts.load()
        chunks = []
        rate = 22050
        for clause in split_for_speech(clip["text"]):
            audio, rate = tts.synthesize(clause)
            chunks.append(audio)
        wav = write_wav(AUDIO / f"{clip['name']}.wav", np.concatenate(chunks), rate)
        for lang, ref in clip["ref"].items():
            (AUDIO / f"{clip['name']}.{lang}.txt").write_text(ref, encoding="utf-8")
        print(f"{wav.name}: {len(np.concatenate(chunks))/rate:.1f}s")


if __name__ == "__main__":
    main()

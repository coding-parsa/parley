#!/usr/bin/env python
"""Fetch everything parley needs into the model store.

    python scripts/fetch_models.py --mt 600m --voices de en --stt turbo

Models land in $PARLEY_MODELS (default: ./models). Every download is resumable;
re-running the same command only fetches what is missing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = Path(os.environ.get("PARLEY_MODELS", ROOT / "models"))

MT_REPOS = {
    "600m": "entai2965/nllb-200-distilled-600M-ctranslate2",
    "1.3b": "entai2965/nllb-200-distilled-1.3B-ctranslate2",
}

# Piper voice id -> path inside rhasspy/piper-voices
VOICES = {
    "en": "en/en_US/hfc_male/medium/en_US-hfc_male-medium",
    "en_female": "en/en_US/amy/medium/en_US-amy-medium",
    "en_gb": "en/en_GB/alan/medium/en_GB-alan-medium",
    "de": "de/de_DE/thorsten/high/de_DE-thorsten-high",
    "de_medium": "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
    "fa": "fa/fa_IR/gyro/medium/fa_IR-gyro-medium",
}

WHISPER_REPOS = {
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def _hub():
    from huggingface_hub import snapshot_download

    return snapshot_download


def fetch_mt(size: str) -> Path:
    repo = MT_REPOS[size]
    target = STORE / "mt" / f"nllb-{size}"
    print(f"[mt] {repo} -> {target}", flush=True)
    _hub()(repo_id=repo, local_dir=str(target), allow_patterns=["*.bin", "*.json", "*.txt", "*.model", "*.spm", "shared_vocabulary*", "sentencepiece*"])
    return target


def fetch_voice(name: str) -> Path:
    key = VOICES[name]
    target = STORE / "piper"
    target.mkdir(parents=True, exist_ok=True)
    allow = [f"{key}.onnx", f"{key}.onnx.json"]
    print(f"[tts] {key} -> {target}", flush=True)
    _hub()(repo_id="rhasspy/piper-voices", local_dir=str(target), allow_patterns=allow)
    return target


def fetch_stt(size: str) -> Path:
    repo = WHISPER_REPOS[size]
    target = STORE / "stt" / f"whisper-{size}"
    print(f"[stt] {repo} -> {target}", flush=True)
    _hub()(repo_id=repo, local_dir=str(target), allow_patterns=["*.bin", "*.json", "*.txt", "vocabulary*", "tokenizer*"])
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt", action="append", choices=sorted(MT_REPOS), default=[])
    parser.add_argument("--voices", nargs="*", choices=sorted(VOICES), default=[])
    parser.add_argument("--stt", action="append", choices=sorted(WHISPER_REPOS), default=[])
    args = parser.parse_args()

    if not (args.mt or args.voices or args.stt):
        parser.error("nothing requested; pass --mt/--voices/--stt")

    for size in args.mt:
        fetch_mt(size)
    for voice in args.voices:
        fetch_voice(voice)
    for size in args.stt:
        fetch_stt(size)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

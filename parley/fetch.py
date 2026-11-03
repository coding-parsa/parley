"""Model downloads, shared by the CLI script and the desktop app.

Kept in the package so a frozen .exe can fetch models too — the app cannot assume
`scripts/fetch_models.py` exists next to the binary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

MT_REPOS = {
    "600m": "entai2965/nllb-200-distilled-600M-ctranslate2",
    "1.3b": "entai2965/nllb-200-distilled-1.3B-ctranslate2",
}

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

Progress = Callable[[str, float], None]


def _snapshot(repo: str, target: Path, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(target), allow_patterns=patterns)
    return target


def fetch_stt(size: str, models_dir: Path, progress: Progress | None = None) -> Path:
    report = progress or (lambda _msg, _frac: None)
    report(f"Speech recognition ({size})", 0.02)
    target = models_dir / "stt" / f"whisper-{size}"
    _snapshot(WHISPER_REPOS[size], target, ["*.bin", "*.json", "*.txt", "vocabulary*", "tokenizer*"])
    report(f"Speech recognition ({size}) ready", 0.45)
    return target


def fetch_mt(size: str, models_dir: Path, progress: Progress | None = None) -> Path:
    report = progress or (lambda _msg, _frac: None)
    report(f"Translator (NLLB {size})", 0.45)
    target = models_dir / "mt" / f"nllb-{size}"
    _snapshot(MT_REPOS[size], target,
              ["*.bin", "*.json", "*.txt", "*.model", "*.spm", "shared_vocabulary*", "sentencepiece*"])
    report(f"Translator (NLLB {size}) ready", 0.9)
    return target


def fetch_voice(name: str, models_dir: Path, progress: Progress | None = None) -> Path:
    report = progress or (lambda _msg, _frac: None)
    key = VOICES[name]
    report(f"Voice ({key.rsplit('/', 1)[-1]})", 0.9)
    target = models_dir / "piper"
    _snapshot("rhasspy/piper-voices", target, [f"{key}.onnx", f"{key}.onnx.json"])
    report("Voice ready", 1.0)
    return target


def needs_for(source: str, target: str, profile: str) -> list[tuple[str, str]]:
    """Which artifacts a given pair + profile requires, in download order."""
    from .config import PROFILES

    prof = PROFILES[profile]
    items: list[tuple[str, str]] = [
        ("stt", prof.stt_model.replace("whisper-", "")),
        ("mt", prof.mt_size),
    ]
    for lang in (target, source):                 # the voice we speak, then the one echoed back
        key = lang if lang in VOICES else "en"
        if ("voice", key) not in items:
            items.append(("voice", key))
    return items


def fetch_all(source: str, target: str, profile: str, models_dir: Path,
              progress: Progress | None = None) -> None:
    report = progress or (lambda _msg, _frac: None)
    stages = needs_for(source, target, profile)
    span = 1.0 / max(1, len(stages))
    for index, (kind, name) in enumerate(stages):

        def scaled(message: str, fraction: float, _i=index) -> None:
            report(message, min(1.0, _i * span + fraction * span))

        if kind == "stt":
            fetch_stt(name, models_dir, scaled)
        elif kind == "mt":
            fetch_mt(name, models_dir, scaled)
        else:
            fetch_voice(name, models_dir, scaled)
    report("Everything is ready", 1.0)

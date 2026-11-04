"""Configuration: language pairs, quality profiles, VRAM ceiling.

Everything the pipeline needs is decided here, once, from a profile + a pair. The
central rule of the project lives in this file: **the resident set is one language
pair at a time**, bounded by a VRAM ceiling. See :mod:`parley.vram`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def default_models_dir() -> Path:
    """Where the models live.

    Override with ``PARLEY_MODELS``. When the app is frozen into an .exe the
    models cannot sit next to the binary (13 GB), so they default to
    ``%LOCALAPPDATA%\\parley\\models``; a source checkout keeps them in ``./models``.
    """
    env = os.environ.get("PARLEY_MODELS")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "parley" / "models"
    return ROOT / "models"


MODELS_DIR = default_models_dir()

# --- Language codes -------------------------------------------------------------
# parley speaks Whisper codes on one side (2-letter) and NLLB codes on the other
# (script-tagged). One table, so a pair is always expressed the same way.

WHISPER_TO_NLLB: dict[str, str] = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "az": "azj_Latn",
    "be": "bel_Cyrl", "bg": "bul_Cyrl", "bn": "ben_Beng", "bs": "bos_Latn",
    "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn",
    "et": "est_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn",
    "gl": "glg_Latn", "gu": "guj_Gujr", "he": "heb_Hebr", "hi": "hin_Deva",
    "hr": "hrv_Latn", "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn",
    "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "ka": "kat_Geor",
    "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang",
    "lt": "lit_Latn", "lv": "lvs_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym",
    "mn": "khk_Cyrl", "mr": "mar_Deva", "ms": "zsm_Latn", "mt": "mlt_Latn",
    "nl": "nld_Latn", "no": "nob_Latn", "pa": "pan_Guru", "pl": "pol_Latn",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sk": "slk_Latn",
    "sl": "slv_Latn", "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai",
    "tl": "tgl_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab",
    "uz": "uzn_Latn", "vi": "vie_Latn", "zh": "zho_Hans", "zu": "zul_Latn",
}

NLLB_TO_WHISPER = {v: k for k, v in WHISPER_TO_NLLB.items()}

LANG_NAMES = {
    "en": "English", "de": "German", "fa": "Persian", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "tr": "Turkish",
    "ar": "Arabic", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
    "pl": "Polish", "sv": "Swedish", "uk": "Ukrainian", "ro": "Romanian", "cs": "Czech",
}


def language_name(code: str) -> str:
    return LANG_NAMES.get(code, code.upper() if len(code) == 2 else code)


def nllb_code(code: str) -> str:
    if code in WHISPER_TO_NLLB:
        return WHISPER_TO_NLLB[code]
    if code in NLLB_TO_WHISPER:
        return code
    raise SystemExit(f"unsupported language {code!r} — see parley.config.WHISPER_TO_NLLB")


def whisper_code(code: str) -> str:
    return NLLB_TO_WHISPER.get(code, code)


# --- Profiles -------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    """A quality/latency trade-off that is known to fit the VRAM ceiling."""

    name: str
    stt_model: str
    mt_size: str
    tts_voice: dict[str, str]
    note: str
    beam_size: int = 2

    def voice_for(self, lang: str) -> str:
        return self.tts_voice.get(lang, self.tts_voice.get("default", ""))


PROFILES: dict[str, Profile] = {
    "fast": Profile(
        name="fast",
        stt_model="whisper-turbo",
        mt_size="600m",
        beam_size=1,
        tts_voice={"en": "en_US-hfc_male-medium", "de": "de_DE-thorsten-medium", "default": "en_US-hfc_male-medium"},
        note="lowest lag (~1.0 s), tiny quality cost; use when the room is noisy and fast",
    ),
    "balanced": Profile(
        name="balanced",
        stt_model="whisper-turbo",
        mt_size="600m",
        beam_size=2,
        tts_voice={"en": "en_US-hfc_male-medium", "de": "de_DE-thorsten-high", "default": "en_US-hfc_male-medium"},
        note="default: turbo STT + NLLB-600M + high-quality Piper voice, 1.8 GB measured",
    ),
    "quality": Profile(
        name="quality",
        stt_model="whisper-large-v3",
        mt_size="1.3b",
        beam_size=2,
        tts_voice={"en": "en_US-hfc_male-medium", "de": "de_DE-thorsten-high", "default": "en_US-hfc_male-medium"},
        note="Large-v3 + NLLB-1.3B, 3.5 GB measured, 2.1 s lag — best translation, still under 6 GB",
    ),
}


# --- Runtime config -------------------------------------------------------------
@dataclass
class Config:
    source: str = "en"
    target: str = "de"
    profile: str = "balanced"

    # hardware
    device: str = "cuda"
    compute_type: str = "int8_float16"
    vram_ceiling_mb: int = int(os.environ.get("PARLEY_VRAM_CEILING_MB", "5600"))

    # audio
    sample_rate: int = 16000
    block_ms: int = 30
    input_device: str | int | None = None
    output_device: str | int | None = None

    # segmentation / commit policy
    vad_threshold_db: float = -38.0
    silence_ms: int = 420          # end-of-utterance silence
    min_speech_ms: int = 280
    max_utterance_s: float = 14.0
    min_commit_chars: int = 3
    partial_interval_ms: int = 500  # re-decode cadence while the speaker talks
    commit_mode: str = "strict"     # strict = two decodes must agree; eager = first clause wins

    # language gate
    min_lang_prob: float = 0.85
    strict_language: bool = True    # refuse spans whose detected language is not the pair

    # translation
    glossary_path: str | None = None
    mt_max_batch: int = 8

    # output
    speak: bool = True
    captions: bool = True
    playback_gain: float = 1.0

    # models
    models_dir: Path = MODELS_DIR

    @property
    def prof(self) -> Profile:
        return PROFILES[self.profile]

    def stt_path(self) -> Path:
        return self.models_dir / "stt" / self.prof.stt_model

    def mt_path(self) -> Path:
        return self.models_dir / "mt" / f"nllb-{self.prof.mt_size}"

    def voice_path(self, lang: str) -> Path:
        voice = self.prof.voice_for(lang)
        return self.models_dir / "piper" / f"{voice}.onnx"

    def describe(self) -> str:
        prof = self.prof
        return (f"profile={prof.name} stt={prof.stt_model} mt=nllb-{prof.mt_size} "
                f"tts={prof.voice_for(self.target)} ceiling={self.vram_ceiling_mb}MB")

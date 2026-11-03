"""Unit tests that need no GPU and no model downloads.

Run with:  ./.venv/Scripts/python.exe -m pytest tests -q
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from parley.config import PROFILES, Config, nllb_code, whisper_code
from parley.mt import Glossary, GlossaryEntry
from parley.stt import Committer, LanguageGate
from parley.tts import split_for_speech
from parley.vad import Segmenter, segment_file
from parley.vram import Budget, VramError


# --- language plumbing ------------------------------------------------------------
def test_language_codes_round_trip():
    for whisper, nllb in (("en", "eng_Latn"), ("de", "deu_Latn"), ("fa", "pes_Arab")):
        assert nllb_code(whisper) == nllb
        assert whisper_code(nllb) == whisper


def test_unknown_language_is_a_clear_error():
    with pytest.raises(SystemExit):
        nllb_code("xx")


# --- the language gate ------------------------------------------------------------
def test_gate_accepts_only_the_pair():
    gate = LanguageGate({"en"}, min_prob=0.6)
    assert gate.check("en", 0.99)[0]
    ok, reason = gate.check("de", 0.95)
    assert not ok and "outside" in reason


def test_gate_still_accepts_low_confidence_foreign_audio():
    """A 0.3-confidence detection is not trustworthy enough to drop speech."""
    gate = LanguageGate({"en"}, min_prob=0.6)
    assert gate.check("de", 0.3)[0]


# --- commit policy ----------------------------------------------------------------
def test_committer_waits_for_agreement_then_a_clause_boundary():
    """One unstable hypothesis is never enough: two decodes must agree."""
    committer = Committer(min_chars=3)
    assert committer.update("The meeting is tomorrow") == ""
    assert committer.update("The meeting is tomorrow. Please send") == "The meeting is tomorrow."
    # the second clause is not yet agreed by two decodes, so it waits…
    assert committer.update("The meeting is tomorrow. Please send the report.") == ""
    # …and is flushed when the utterance ends
    assert committer.update("The meeting is tomorrow. Please send the report.", final=True) == \
        "Please send the report."


def test_committer_ignores_a_hypothesis_that_changes_its_mind():
    committer = Committer(min_chars=3)
    assert committer.update("Vielen Dank, dass Sie uns begleitet haben.") == ""
    # the model rewrote the beginning; nothing may be spoken yet
    assert committer.update("Sie sind dabei.") == ""
    assert committer.update("Sie sind dabei. Heute müssen wir planen.") == "Sie sind dabei."


def test_committer_does_not_repeat_itself():
    committer = Committer(min_chars=3)
    committer.update("First sentence.", final=False)
    first = committer.update("First sentence. Second", final=False)
    second = committer.update("First sentence. Second", final=False)
    assert first == "First sentence."
    assert second == ""


def test_committer_final_flushes_the_remainder():
    committer = Committer(min_chars=3)
    committer.update("No punctuation here", final=False)
    assert committer.update("No punctuation here", final=True) == "No punctuation here"


# --- speech splitting -------------------------------------------------------------
def test_long_text_is_split_at_sentences():
    text = "First sentence. Second sentence! Third question? " + "x " * 200
    parts = split_for_speech(text, max_chars=120)
    assert len(parts) > 2
    assert all(len(p) <= 130 for p in parts)


def test_empty_text_yields_nothing():
    assert split_for_speech("   ") == []


# --- glossary ---------------------------------------------------------------------
def test_glossary_repairs_and_reports(tmp_path):
    path = tmp_path / "glossary.tsv"
    path.write_text("Deployment\tBereitstellung\tdas Geld;die Bereitstellung\n", encoding="utf-8")
    glossary = Glossary.load(path)
    fixed = glossary.repair("We finished the deployment.", "Wir haben das Geld abgeschlossen.")
    assert "Bereitstellung" in fixed
    assert glossary.audit() == []


def test_glossary_flags_a_missed_term():
    glossary = Glossary([GlossaryEntry("deployment", "Bereitstellung")])
    glossary.repair("The deployment is done.", "Es ist fertig.")
    assert glossary.audit() == [("deployment", "Bereitstellung")]


# --- segmentation ----------------------------------------------------------------
def _tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_segmenter_cuts_at_silence():
    audio = np.concatenate([_tone(1.0), np.zeros(16000, dtype=np.float32), _tone(1.0)])
    utts = segment_file(audio, silence_ms=300, min_speech_ms=200)
    assert len(utts) == 2
    assert all(0.5 < u.seconds < 2.0 for u in utts)


def test_segmenter_ignores_quiet_room_noise():
    noise = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.002).astype(np.float32)
    utts = segment_file(noise, silence_ms=300, min_speech_ms=200)
    assert utts == []


def test_segmenter_forces_a_cut_on_long_speech():
    utts = segment_file(_tone(20.0), silence_ms=300, min_speech_ms=200, max_s=5.0)
    assert len(utts) >= 3


# --- VRAM budget ------------------------------------------------------------------
class _Fake:
    def __init__(self):
        self.unloaded = False

    def unload(self):
        self.unloaded = True


def test_budget_evicts_lru_to_make_room():
    budget = Budget(ceiling_mb=2000)
    a, b = _Fake(), _Fake()
    budget.admit("a", 1400, unload=a.unload)
    budget.admit("b", 1500, unload=b.unload)          # forces eviction of "a"
    assert a.unloaded and not b.unloaded
    assert set(budget.keys()) == {"b"}


def test_budget_refuses_a_model_larger_than_the_ceiling():
    budget = Budget(ceiling_mb=1000)
    with pytest.raises(VramError):
        budget.admit("huge", 1500)


def test_budget_never_evicts_pinned_models():
    budget = Budget(ceiling_mb=2000)
    budget.admit("stt", 1200)
    budget.pin("stt", True)
    with pytest.raises(VramError):
        budget.admit("mt", 1500)


def test_pinned_resident_set_stays_under_ceiling():
    budget = Budget(ceiling_mb=5600)
    budget.admit("stt", 900)
    budget.admit("mt", 700)
    assert budget.resident_mb + 200 <= 5600


# --- config -----------------------------------------------------------------------
def test_profiles_fit_the_six_gig_card():
    for name, prof in PROFILES.items():
        stt = 900 if "turbo" in prof.stt_model else 1600
        mt = 700 if prof.mt_size == "600m" else 1500
        assert stt + mt < 5600, f"profile {name} would not fit"


def test_duplex_shares_one_pair():
    from parley.duplex import Duplex, DuplexConfig

    duplex = Duplex(DuplexConfig(a="en", b="de"))
    assert duplex.outbound.engines is duplex.inbound.engines
    assert duplex.outbound.engines.budget is duplex.inbound.engines.budget


def test_voice_path_tracks_the_target_language():
    cfg = Config(source="en", target="de", profile="balanced")
    assert cfg.voice_path("de").name.startswith("de_DE")
    assert cfg.voice_path("en").name.startswith("en_US")

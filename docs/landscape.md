# Landscape: what already exists, as of November 2026

Research snapshot taken while deciding whether to build this. Re-verify star counts
and features before quoting them again — this field moves in weeks.

| Project | What it is | Local | Directions | Speech out | The catch |
|---|---|---|---|---|---|
| **Kyutai Hibiki / Hibiki-Zero** | end-to-end *simultaneous* speech→speech, 3B, voice transfer | yes, on-device | FR/ES/PT/DE → **EN only** | yes | no EN→DE; one speaker per session |
| **Meta SeamlessStreaming / M4T v2** | any-to-any, ~2 s latency, 100 text / 36 speech languages | yes, ~8 GB VRAM | both ways | yes | too heavy and too slow for 6 GB |
| **WhisperLiveKit** (~11k★) | streaming ASR + simultaneous MT (NLLB-600M CTranslate2) + diarization | yes | 200 languages, text | **no TTS** | stops at text |
| **PolyTalk** (~80★) | whisper + LLM translation + Piper TTS, Docker stack | yes | via LLM | yes | needs an LLM for MT (VRAM fight), browser audio, 0.8–5 s flush |
| **parlIAmo** (~0★) | Whisper + NLLB + Kokoro/Piper + OmniVoice/RVC cloning, 8 GB | yes | 7 languages both ways | yes | one-person demo, no meeting integration |
| **Whisperr** (commercial) | participant-side live translation into your earbuds | no, cloud | 100+ | yes | closed, paid, audio leaves the machine |
| **parley** (this) | cascade, one pair resident, meeting plumbing, measured 6 GB budget | yes | 65 languages, any pair | yes, CPU TTS | quality ceiling of NLLB + Piper; no cloning yet |

## The gap this project targets

1. **Meeting plumbing.** Virtual microphone output so Zoom/Teams sees the translated
   voice as your mic; loopback capture for the inbound direction; visible captions.
2. **A published 6 GB latency budget.** Every project above assumes 8–12 GB or
   offloads MT to a local LLM. The numbers here come from the card.
3. **A commit policy.** Whisper rewrites its own partials; speaking a partial either
   repeats the sentence or reads garbage. Only text that survived to a clause
   boundary is spoken.
4. **A language gate.** A wrong detection is not a typo — it is a translation into
   the wrong language, spoken aloud into a meeting.
5. **A benchmark harness** with BLEU/chrF and lag, so the claims are checkable.

## Why a cascade and not an end-to-end model

End-to-end speech translation (Hibiki-Zero, Seamless) is the right long-term shape:
lower lag, voice transfer, no text bottleneck. Today it fails one hard requirement —
it translates *into English*. A cascade over NLLB covers 65 languages in both
directions with a 600 MB model, which on a 6 GB card is the difference between
"works for my meeting" and "works for some meetings in one direction".

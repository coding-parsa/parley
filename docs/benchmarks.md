# Benchmarks

Every number on this page comes from `parley` itself, on one machine. Nothing here
is estimated, and the commands are included so they can be re-run.

**Machine:** RTX 4050 Laptop, 6 GB VRAM (6141 MiB), 15 GB RAM, Windows 11, CUDA 12.9,
`int8_float16` compute for both STT and MT, Piper on CPU.

**Corpus:** `tests/audio/*.wav` — five synthetic meeting turns spoken by Piper
(three English, two German), 4–17 s each. References are author-written translations,
so BLEU/chrF measure agreement with a human reference, not ground truth. Synthetic
audio flatters every model; a real meeting corpus would score lower.

## Translation quality

```bash
parley bench tests/audio --from en --to de --profile balanced
parley bench tests/audio --from en --to de --profile quality
parley bench tests/audio --from en --to de --profile quality --glossary examples/glossary.tsv
```

| profile | STT | MT | glossary | BLEU | chrF | own VRAM | peak GPU |
|---|---|---|---|---|---|---|---|
| balanced | large-v3-turbo | NLLB-600M | – | 48.8 | 71.1 | 1.8 GB | 2.3 GB |
| quality | large-v3 | NLLB-1.3B | – | 53.2 | 73.1 | 3.5 GB | 3.8 GB |
| quality | large-v3 | NLLB-1.3B | yes | **54.5** | **77.6** | 3.5 GB | 4.7 GB |

The glossary row is the interesting one: two repair entries
(`Einsatzplan → Bereitstellungsplan`, `Haushalt → Budget`) bought more chrF than the
ten-fold larger MT model. Terminology is where meeting translation actually fails.

## Latency

```bash
parley simulate tests/audio/en_meeting_long.wav --from en --to de --profile balanced --no-speak
parley simulate tests/audio/en_meeting_long.wav --from en --to de --profile quality  --no-speak
```

`--no-speak` isolates recognition + translation. Speaker output adds the first Piper
chunk (see below).

| profile | STT p50 | MT p50 | lag p50 (end of speech → text ready) | peak GPU |
|---|---|---|---|---|
| balanced | 0.63 s | 0.25 s | **1.17 s** | 2.3 GB |
| quality | 1.25 s | 0.46 s | 2.12 s | 5.3 GB |

`lag` includes the VAD endpoint wait (420 ms of silence before an utterance is
declared finished). With speech enabled, add the time to the first synthesised audio
chunk.

## Speech synthesis (Piper, CPU, one thread of work per clause)

| voice | first audio (35 chars) | first audio (80 chars) | real-time factor |
|---|---|---|---|
| `de_DE-thorsten-high` | 332 ms | 667 ms | 0.20 (5× faster than real time) |
| `en_US-hfc_male-medium` | 101 ms | 201 ms | 0.04 (25× faster) |

German `high` is the slow one and still keeps up: 6.8 s of speech synthesised in
1.33 s. Clause-splitting is what protects the first-chunk figure — a 150-character
clause takes 1.3 s before the first sample.

## End-to-end budget (balanced, measured parts)

```
VAD endpoint        0.42 s   (silence_ms = 420)
STT (turbo)         0.63 s
MT (NLLB-600M)      0.25 s
TTS first chunk     0.33 s   (German high, short clause)
────────────────────────────
≈ 1.6 s from end of speech to the listener hearing German
```

For scale: Meta SeamlessStreaming reports ~2 s, Kyutai Hibiki-Zero averages 6.1 s lag,
and a human simultaneous interpreter runs 2–4 s behind the speaker.

## What did **not** work

- **NLLB-600M with multi-sentence input** returns only the last sentence. parley
  splits into sentences before translating; the effect on BLEU was +8 points.
- **Beam search 5 on the MT stage**: 59.0 BLEU vs 61.1 at beam 2 on the sentence
  set, and slower. Beam 2 is the default; `fast` uses 1.
- **Whisper's forced-language decoding as the MT stage**: it does translate
  (see [whisper-language-token.md](whisper-language-token.md)) but wrecks
  terminology and gender, and can loop. Kept only as a baseline probe.
- **Agreement-free commits (`commit_mode=eager`)**: faster on paper, measurably worse
  in practice — the model rewrites its own prefix and the listener hears a clause
  twice. Strict mode is the default.

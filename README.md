# parley

**Offline live interpretation that fits in 6 GB of VRAM.** Speak English, the meeting
hears German — in your own laptop, with no cloud, no API keys and no audio leaving
the machine.

```
microphone ──► VAD ──► Whisper ──► language gate ──► NLLB-200 ──► Piper ──► virtual cable ──► meeting
                          ▲                                                          │
                          └────────────── the other side, captured via loopback ◄────┘
```

> Status: **v0.3** — single-direction live mode, file mode, bidirectional meeting
> mode and a caption console all work on a 6 GB RTX 4050. Numbers below are measured
> on that card, not estimated.

---

## Why another one

The pieces exist. What does not: an assembled, tuned interpreter that runs in 6 GB,
only keeps *one language pair* resident, refuses to speak the wrong language, and
plugs into a real meeting as a microphone. See [docs/landscape.md](docs/landscape.md)
for the inventory and the gap.

Three things this project does that the wrappers do not:

1. **A published 6 GB budget.** Every stage is measured and admitted through a VRAM
   manager that evicts a pair before it would exceed the ceiling.
2. **A language gate.** Whisper's language token can *translate* — forcing it on
   foreign speech turns transcription into translation
   ([measured here](docs/whisper-language-token.md)). A wrong detection in a meeting
   is not a typo, it is the wrong language spoken into the room. parley detects per
   utterance, refuses anything outside the pair, and reports the refusal.
3. **A commit policy.** Whisper rewrites its own partials. parley speaks only text
   that reached a clause boundary and survived a second hypothesis — so it does not
   say the sentence twice.

## Measured

<!-- bench-table -->

## Install

```bash
git clone <your-fork> parley && cd parley
python -m venv .venv && .venv/Scripts/activate      # Windows; source .venv/bin/activate elsewhere
pip install -e .

python scripts/fetch_models.py --stt turbo --mt 600m --voices de en
parley doctor
```

`doctor` tells you what is missing before you find out mid-meeting.

## Use

```bash
# translate a file, report every stage's timing
parley file tests/audio/en_meeting_01.wav --from en --to de --out out/

# live, from the microphone, spoken into your speakers
parley run --from en --to de

# live captions in a browser console (lag, VRAM, refusals)
parley serve --from en --to de
```

### In a real meeting

Install [VB-Cable](https://vb-audio.com/Cable/) (free) once. Then:

```bash
parley devices          # find: your mic, "Stereo Mix"/loopback, "CABLE Input"

parley meet --a en --b de \
    --mic "Headset Microphone" \
    --loopback-device "Stereo Mix" \
    --cable "CABLE Input" \
    --headphones "Headphone"
```

Then in Zoom/Teams: **microphone = CABLE Output**. The room hears your translated
voice; you hear theirs translated in your headphones. `--captions-only` keeps the
inbound direction silent and on screen.

## Profiles and the VRAM ceiling

One pair resident, always. The budget manager evicts least-recently-used models and
refuses to load a profile that cannot fit.

| profile | STT | MT | ~VRAM | lag | when |
|---|---|---|---|---|---|
| `fast` | large-v3-turbo | NLLB-600M | ~1.0 GB | lowest | noisy rooms, long calls |
| `balanced` | large-v3-turbo | NLLB-600M | ~1.6 GB | ~1.3 s | default |
| `quality` | large-v3 | NLLB-1.3B | ~3.3 GB | ~2 s | best translation, still < 6 GB |

TTS runs on the **CPU** on purpose: on a small card the GPU belongs to the
recogniser and translator, and Piper already runs 5–25× faster than real time on a
laptop CPU. That is also why bidirectional mode costs the same VRAM as one
direction — the pair is resident, not the direction.

## Languages

65 languages in any direction (`parley.config.WHISPER_TO_NLLB`), because NLLB covers
them in one 600 MB model: en, de, fa, fr, es, it, pt, nl, ru, tr, ar, zh, ja, ko, hi,
pl, uk, sv, cs, ro, and the rest of NLLB's high-resource set. The pair is chosen at
startup, so nothing else is loaded.

## Glossary

Names and product words are where MT embarrasses itself. A TSV repairs and audits
them:

```tsv
# source<TAB>approved target<TAB>bad variants (;)
deployment	Bereitstellung	das Geld;die Einsatz
```

```bash
parley file meeting.wav --from en --to de --glossary glossary.tsv
```

parley replaces known bad renderings and reports every term it could not confirm
rather than pretending.

## Honest limits

- **NLLB + Piper is not a human interpreter.** Short, literal, meeting-shaped
  sentences come out well; idioms and jokes do not.
- **~1.3 s behind, not simultaneous.** A human interpreter is 2–4 s behind; Hibiki-Zero
  averages ~6 s lag but only translates *into English*.
- **NLLB is CC-BY-NC-4.0.** Personal and research use is fine; selling interpretation
  needs a different MT model. The MIT licence here covers the code, not the weights.
- **No voice cloning.** Preserving *your* voice is a fine feature; cloning another
  participant is a legal problem (EU AI Act, consent records). Roadmap, with
  consent gates, not a v1 claim.

## Roadmap

- **v0.4** — streaming TTS per clause boundary (already written, needs measurement),
  per-session glossary UI in the console.
- **v0.5** — optional MT backends behind the same interface (MADLAD-400 for
  terminology-heavy work) with the VRAM manager choosing one.
- **v1.0** — Windows installer, one-click model download, demo recording.

## Credits

Whisper (OpenAI) · faster-whisper (SYSTRAN) · CTranslate2 · NLLB-200 (Meta) ·
Piper (rhasspy) · Silero VAD. Built on a laptop in Iran, where every cloud API is one
DNS block away from ruining a meeting.

MIT — see [LICENSE](LICENSE).

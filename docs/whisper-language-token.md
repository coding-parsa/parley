# Why Whisper appeared to translate into German

A measurement note, not a guess: this is the investigation that started the project.
Every number below was produced on an RTX 4050 Laptop (6 GB) with
`faster-whisper` `large-v3`, `int8_float16`.

## The observation

During a voice call, Persian and English speech were transcribed **as German**:

> "Wir haben ein kleines Problem und das ist in einer interessanten Weise, wenn ich
> Englisch spreche oder manchmal Englisch und immer Persian, das wird in der
> deutschen Sprache übersetzt."

No translation service was in the loop. The STT was pinned to `language=de` at the
time, so that German practice sentences would decode cleanly.

## The mechanism

Whisper's language token selects the language of the **output**, not only the input.
Its training data mixed transcription with translation, so when the token points at
a language that differs from the audio, the decoder can translate instead of
transcribe.

The asymmetry is the interesting part. Measured, same clips, same model:

| Audio | forced token | output |
|---|---|---|
| Persian | `de` | *"Gestern habe ich den Mikrofon getestet und habe bemerkt, dass alles in einer anderen Sprache geschrieben wird."* — fluent German |
| Persian | `en` | English translation — same trick, other direction |
| German | `en` | *"The meeting will take place tomorrow at 10 o'clock. Please send me the financial report beforehand."* |
| English | `de` | **unchanged English** — token ignored |
| English | `de` + `task="translate"` | unchanged English |

Whisper's multilingual corpus uses English as the pivot: much of the paired text for
non-English audio was already English. The decoder therefore learned
"foreign speech → text in the requested language", while English audio keeps its
verbatim-transcription behaviour and the token is mostly inert.

## Why it is not a translator

| Input (Persian) | Forced-`de` output | Problem |
|---|---|---|
| گزارش مالی (financial report) | *das Geld* (the money) | terminology lost |
| پروژه ترجمه زنده (live translation project) | *Projektsprüfungsprojekt* | nonsense compound |
| بودجه (budget) | *der Budget* | wrong gender |

It also degenerates: one clip ended in a repetition loop, and decoding slows down
when the token fights the audio (4.6 s vs 0.95 s for the same duration).

## The second, quieter source of German

On silence or microphone noise, Whisper hallucinates from subtitle training data —
in the language of the token:

```
silence 3s  [de]   -> 'Vielen Dank.'
noise 0.02  [de]   -> 'Vielen Dank.'
silence 3s  [auto] -> 'Teksting av Nicolai Winther'   (Norwegian, detected nn at 0.35)
```

The auto-detect path is not safe by itself either: 0.35 confidence for a language
nobody in the room was speaking.

## What parley does about it

1. **The STT language is never pinned.** Every utterance is detected individually.
2. **A language gate** rejects anything outside the configured pair with a
   confidence floor of 0.60 (not 0.45 — the 0.35–0.5 band is where the detector
   invents languages).
3. **A rejection is an event, not a silent drop.** `{"type": "rejected"}` carries the
   detected language and probability, so the operator sees that the interpreter
   refused to speak rather than hearing the wrong language in the room.

The forced-token trick survives only as a **baseline row** in the benchmark: it is a
free second translator to compare against NLLB on the same audio.

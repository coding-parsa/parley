"""Text shaping shared by the translation and speech stages.

Two different jobs that both look like "split the text":

* :func:`split_sentences` — before translation. NLLB silently drops everything but
  the last sentence of a multi-sentence input (measured: a two-sentence utterance
  came back as one), so the source is always cut into sentences first.
* :func:`speakable_chunks` — after translation, before synthesis. Short clauses mean
  the first audio arrives sooner and the prosody resets at punctuation instead of
  running one flat breath through a paragraph.
"""

from __future__ import annotations

import re

SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
SOFT_SPLIT = re.compile(r"(?<=[;:])\s+")

MAX_CHUNK_CHARS = 220


def split_sentences(text: str) -> list[str]:
    """Cut a transcript into sentences, keeping the terminal punctuation."""
    text = " ".join(text.split())
    if not text:
        return []
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or [text]


def speakable_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    pieces: list[str] = []
    for sentence in SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        buffer = ""
        for part in re.split(r"(?<=,)\s+", sentence):
            if len(buffer) + len(part) + 1 <= max_chars:
                buffer = f"{buffer} {part}".strip()
            else:
                if buffer:
                    pieces.append(buffer)
                while len(part) > max_chars:
                    cut = part.rfind(" ", 0, max_chars)
                    cut = cut if cut > 40 else max_chars
                    pieces.append(part[:cut].strip())
                    part = part[cut:].strip()
                buffer = part
        if buffer:
            pieces.append(buffer)
    return [p for p in pieces if p]

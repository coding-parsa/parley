"""Translation stage.

NLLB-200 (distilled) in CTranslate2 is the sweet spot on a 6 GB card: one model
covers 200 languages, so a pair costs nothing extra, and the 600M/1.3B variants
leave room for the recogniser and the synthesiser at the same time.

Deliberately *not* an LLM. A 4B chat model would translate more idiomatically but
it would take the VRAM the STT needs, and on a 6 GB laptop that trade is a loss —
see docs/landscape.md for the measurements behind that call.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import nllb_code


class ModelFilesMissing(SystemExit):
    pass


@dataclass
class GlossaryEntry:
    source: str
    target: str
    variants: tuple[str, ...] = ()


class Glossary:
    """Terminology lock for names, product words and numbers.

    Two jobs, both honest about what a plain MT model can and cannot do:

    * *repair* — replace known bad renderings with the approved term;
    * *report* — flag when the source term appeared but the approved target term
      did not, so the operator sees it instead of shipping a wrong word.

    It does not try to force the model's decoder; NLLB offers no such hook, and
    pretending otherwise is how glossaries get a bad name.
    """

    def __init__(self, entries: list[GlossaryEntry] | None = None):
        self.entries = entries or []
        self.misses: list[tuple[str, str]] = []

    @classmethod
    def load(cls, path: str | Path | None) -> "Glossary":
        if not path:
            return cls()
        file = Path(path)
        if not file.exists():
            raise SystemExit(f"glossary not found: {file}")
        entries: list[GlossaryEntry] = []
        for raw in file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in re.split(r"\t|\s*\|\s*", line)]
            if len(parts) < 2:
                continue
            variants = tuple(v for v in (parts[2].split(";") if len(parts) > 2 else []) if v)
            entries.append(GlossaryEntry(parts[0], parts[1], variants))
        return cls(entries)

    def repair(self, source_text: str, target_text: str) -> str:
        fixed = target_text
        for entry in self.entries:
            if entry.source.lower() not in source_text.lower():
                continue
            for variant in entry.variants:
                fixed = re.sub(re.escape(variant), entry.target, fixed, flags=re.IGNORECASE)
            if entry.target.lower() not in fixed.lower():
                self.misses.append((entry.source, entry.target))
        return fixed

    def audit(self) -> list[tuple[str, str]]:
        return list(self.misses)


def _find_sentencepiece(path: Path) -> Path:
    for pattern in ("*.bpe.model", "sentencepiece*.model", "*.model", "spm*.model"):
        for hit in sorted(path.glob(pattern)):
            if hit.is_file():
                return hit
    raise ModelFilesMissing(
        f"no sentencepiece model inside {path}; the NLLB CTranslate2 export should ship one — "
        f"re-run scripts/fetch_models.py --mt 600m")


class NllbMt:
    """NLLB-200 distilled in CTranslate2, one model resident for all pairs."""

    def __init__(self, model_dir: str | Path, device: str = "cuda", compute_type: str = "int8_float16",
                 beam_size: int = 2, estimated_mb: int = 700):
        self.model_dir = Path(model_dir)
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.estimated_mb = estimated_mb
        self.translator = None
        self.sp = None
        self.load_s = 0.0

    def load(self) -> None:
        if self.translator is not None:
            return
        import ctranslate2
        import sentencepiece as spm

        if not (self.model_dir / "model.bin").exists():
            raise ModelFilesMissing(
                f"NLLB model missing at {self.model_dir} (no model.bin). Run:\n"
                f"  python scripts/fetch_models.py --mt 600m")
        started = time.perf_counter()
        self.translator = ctranslate2.Translator(
            str(self.model_dir), device=self.device, compute_type=self.compute_type,
            inter_threads=1, intra_threads=2,
        )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(_find_sentencepiece(self.model_dir)))
        self.load_s = time.perf_counter() - started

    def unload(self) -> None:
        self.translator = None
        self.sp = None
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001
            pass

    def translate(self, text: str, source: str, target: str) -> str:
        assert self.translator is not None and self.sp is not None, "call load() first"
        text = text.strip()
        if not text:
            return ""
        src, tgt = nllb_code(source), nllb_code(target)
        if src == tgt:
            return text
        tokens = [src] + list(self.sp.encode(text, out_type=str))
        t0 = time.perf_counter()
        results = self.translator.translate_batch(
            [tokens],
            target_prefix=[[tgt]],
            beam_size=self.beam_size,
            max_batch_size=1,
            max_decoding_length=512,
            replace_unknowns=True,
        )
        elapsed = time.perf_counter() - t0
        hypothesis = list(results[0].hypotheses[0])
        if hypothesis and hypothesis[0] == tgt:
            hypothesis = hypothesis[1:]
        out = self.sp.decode(hypothesis).strip()
        self.last_latency_s = elapsed
        return out

    def translate_many(self, texts: list[str], source: str, target: str) -> list[str]:
        """Batched translation for file mode (one decoder call instead of N)."""
        assert self.translator is not None and self.sp is not None, "call load() first"
        src, tgt = nllb_code(source), nllb_code(target)
        batch = [[src] + list(self.sp.encode(t, out_type=str)) for t in texts if t.strip()]
        if not batch:
            return ["" for _ in texts]
        results = self.translator.translate_batch(
            batch, target_prefix=[[tgt]] * len(batch), beam_size=self.beam_size,
            max_batch_size=len(batch), max_decoding_length=512, replace_unknowns=True,
        )
        out: list[str] = []
        for result in results:
            hypothesis = list(result.hypotheses[0])
            if hypothesis and hypothesis[0] == tgt:
                hypothesis = hypothesis[1:]
            out.append(self.sp.decode(hypothesis).strip())
        return out


class IdentityMt:
    """Same-language passthrough — used for caption-only mode and for tests."""

    estimated_mb = 0
    load_s = 0.0

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def translate(self, text: str, source: str, target: str) -> str:
        return text

    def translate_many(self, texts: list[str], source: str, target: str) -> list[str]:
        return list(texts)

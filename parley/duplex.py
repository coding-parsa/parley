"""Bidirectional interpretation for a real meeting.

Two cascades, one resident pair:

    my microphone ──► STT ──► NLLB ──► german voice ──► virtual cable ──► meeting
    meeting audio ──► STT ──► NLLB ──► english voice ─► your headphones

The recogniser and the translator are shared (``Engines``), so the VRAM cost is a
pair, not a pair per direction. The two voices run on the CPU.

What this cannot do, stated up front: it inserts latency into a conversation, so
the other side hears you ~1 s behind and you hear them ~1 s behind. That is the
same order as a human interpreter, and it is why the commit policy keeps
utterances short.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .audio import AudioSource, Speaker, find_loopback_device, open_source
from .config import Config, language_name
from .pipeline import Engines, Interpreter
from .vram import Budget


@dataclass
class DuplexConfig:
    a: str = "en"                       # the language you speak
    b: str = "de"                       # the language the room speaks
    profile: str = "balanced"
    mic_device: str | int | None = None
    loopback_device: str | int | None = None
    cable_device: str | int | None = None      # virtual cable the meeting listens to
    headphone_device: str | int | None = None
    captions_only_inbound: bool = False
    glossary_path: str | None = None
    vram_ceiling_mb: int = 5600
    block_ms: int = 30

    def to_config(self, source: str, target: str, input_device, output_device, speak: bool) -> Config:
        return Config(
            source=source, target=target, profile=self.profile,
            input_device=input_device, output_device=output_device,
            speak=speak, glossary_path=self.glossary_path,
            vram_ceiling_mb=self.vram_ceiling_mb, block_ms=self.block_ms,
        )


class Duplex:
    def __init__(self, cfg: DuplexConfig, on_event=None):
        self.cfg = cfg
        self.on_event = on_event or (lambda event: None)
        budget = Budget(cfg.vram_ceiling_mb, log=self._log)
        self.engines = Engines(budget=budget)
        self.outbound = Interpreter(
            cfg.to_config(cfg.a, cfg.b, cfg.mic_device, cfg.cable_device, speak=True),
            on_event=self._tag("outbound"), engines=self.engines)
        self.inbound = Interpreter(
            cfg.to_config(cfg.b, cfg.a, cfg.loopback_device, cfg.headphone_device,
                          speak=not cfg.captions_only_inbound),
            on_event=self._tag("inbound"), engines=self.engines)

    def _tag(self, direction: str):
        def emit(event: dict) -> None:
            if event.get("type") in ("segment", "partial_commit", "rejected", "caption"):
                event = {**event, "direction": direction}
            self.on_event(event)
        return emit

    def warm(self) -> None:
        self.outbound.warm()          # loads STT + MT + first voice
        self.inbound.warm()           # reuses STT + MT, loads the second voice

    def run(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        self._log(f"duplex {language_name(self.cfg.a)} ↔ {language_name(self.cfg.b)}")
        if self.cfg.cable_device is None:
            self._log("warning: no --cable device given — the meeting will not hear the outbound "
                      "translation. Install VB-Cable and pass its playback endpoint.")

        out_src = open_source(self.cfg.mic_device, loopback=False, block_ms=self.cfg.block_ms)
        in_src = open_source(self.cfg.loopback_device or find_loopback_device(),
                             loopback=True, block_ms=self.cfg.block_ms)
        out_spk = Speaker(self.cfg.cable_device) if self.cfg.cable_device is not None else None
        in_spk = None
        if not self.cfg.captions_only_inbound:
            in_spk = Speaker(self.cfg.headphone_device)

        threads = []
        with out_src, in_src:
            if out_spk is not None:
                out_spk.open()
            if in_spk is not None:
                in_spk.open()
            try:
                threads = [
                    threading.Thread(target=self.outbound.run_live, args=(out_src, out_spk, stop),
                                     name="outbound", daemon=True),
                    threading.Thread(target=self.inbound.run_live, args=(in_src, in_spk, stop),
                                     name="inbound", daemon=True),
                ]
                for thread in threads:
                    thread.start()
                while not stop.wait(0.3):
                    if not any(t.is_alive() for t in threads):
                        break
            except KeyboardInterrupt:
                stop.set()
            finally:
                stop.set()
                for thread in threads:
                    thread.join(timeout=20)
                if out_spk is not None:
                    out_spk.close()
                if in_spk is not None:
                    in_spk.close()
        self._log(f"duplex stopped · outbound lag p50 "
                  f"{self.outbound.stats()['lag_p50']:.2f}s · inbound "
                  f"{self.inbound.stats()['lag_p50']:.2f}s")

    def _log(self, message: str) -> None:
        self.on_event({"type": "log", "message": message})

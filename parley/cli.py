"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .audio import find_loopback_device, list_devices, open_source, read_wav, Speaker
from .config import PROFILES, Config, language_name
from .pipeline import Interpreter
from .vram import Budget, GpuSampler, gpu_name, gpu_used_mb
from .vad import SR

console = Console()


# --- helpers ----------------------------------------------------------------------
def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config(
        source=getattr(args, "source", "en"),
        target=getattr(args, "target", "de"),
        profile=getattr(args, "profile", "balanced"),
        device=getattr(args, "device", "cuda"),
        input_device=getattr(args, "input_device", None),
        output_device=getattr(args, "output_device", None),
        speak=not getattr(args, "no_speak", False),
        glossary_path=getattr(args, "glossary", None),
    )
    ceiling = getattr(args, "vram_ceiling", 0)
    if ceiling:
        cfg.vram_ceiling_mb = ceiling
    return cfg


def _pretty_event(event: dict) -> None:
    kind = event.get("type")
    if kind == "model":
        console.print(f"[dim]model {event['name']}: ~{event['vram_mb']} MB VRAM, loaded in {event['load_s']}s[/dim]")
    elif kind == "ready":
        console.print(f"[green]ready[/green] {event['config']}")
    elif kind == "listening":
        console.print("[bold]listening[/bold] — Ctrl+C to stop")
    elif kind == "segment":
        if event.get("dropped"):
            console.print(f"[yellow]skip[/yellow] {event['dropped']}")
            return
        console.print(f"[cyan]{event['src_lang'].upper()}[/cyan] {event['source_text']}")
        console.print(f"  [bold]{event['translated_text']}[/bold] "
                      f"[dim](stt {event['stt_s']:.2f}s · mt {event['mt_s']:.2f}s · "
                      f"lag {event['lag_s']:.2f}s)[/dim]")
    elif kind == "partial_commit":
        console.print(f"  [dim]… {event['translated']}[/dim]")
    elif kind == "partial":
        console.print(f"    [dim]{event.get('audio_s')}s heard → “{event.get('text', '')[:70]}”[/dim]")
    elif kind == "rejected":
        console.print(f"[yellow]language gate[/yellow] {event['reason']}")
    elif kind == "error":
        console.print(f"[red]error[/red] {event['where']}: {event['message']}")
    elif kind == "log":
        console.print(f"[dim]{event['message']}[/dim]")


# --- commands ---------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    table = Table(title="parley doctor", show_header=False, box=None)
    table.add_row("version", __version__)
    table.add_row("python", sys.version.split()[0])
    gpu = gpu_name()
    table.add_row("gpu", gpu or "none detected")
    used = gpu_used_mb()
    table.add_row("gpu in use", f"{used} MB" if used is not None else "unknown")
    cfg = _config_from_args(args)
    table.add_row("profile", f"{cfg.profile} — {cfg.prof.note}")
    table.add_row("vram ceiling", f"{cfg.vram_ceiling_mb} MB")
    table.add_row("stt model", f"{cfg.stt_path()} {'OK' if (cfg.stt_path() / 'model.bin').exists() else 'MISSING'}")
    table.add_row("mt model", f"{cfg.mt_path()} {'OK' if (cfg.mt_path() / 'model.bin').exists() else 'MISSING'}")
    voice = cfg.voice_path(cfg.target)
    table.add_row("tts voice", f"{voice} {'OK' if voice.exists() else 'MISSING (CPU, 0 MB VRAM)'}")
    loop = find_loopback_device()
    table.add_row("loopback", f"device {loop}" if loop is not None else "none — install VB-Cable for the inbound direction")
    console.print(table)

    for check in ("faster_whisper", "ctranslate2", "piper", "sounddevice", "sentencepiece"):
        try:
            __import__(check)
            console.print(f"  [green]✓[/green] {check}")
        except Exception as exc:                      # noqa: BLE001
            console.print(f"  [red]✗[/red] {check}: {exc}")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    for dev in list_devices():
        marker = ""
        if any(tag in dev.name.lower() for tag in ("loopback", "stereo mix", "cable")):
            marker = "  ← loopback/virtual"
        console.print(f"{dev}{marker}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    table = Table(title="profiles")
    table.add_column("profile")
    table.add_column("STT")
    table.add_column("MT")
    table.add_column("~VRAM")
    table.add_column("note")
    for name, prof in PROFILES.items():
        est = (900 if "turbo" in prof.stt_model else 1600) + (700 if prof.mt_size == "600m" else 1500)
        table.add_row(name, prof.stt_model, f"nllb-{prof.mt_size}", f"{est} MB", prof.note)
    console.print(table)
    store = Config().models_dir
    console.print(f"model store: [bold]{store}[/bold]")
    return 0


def cmd_file(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    cfg.speak = True
    out_dir = Path(args.out) if args.out else None
    interpreter = Interpreter(cfg, on_event=_pretty_event)
    interpreter.warm()
    started = time.perf_counter()
    results = interpreter.translate_file(args.path, out_dir=out_dir)
    elapsed = time.perf_counter() - started
    stats = interpreter.stats()
    audio, sr = read_wav(args.path)

    table = Table(title=f"{Path(args.path).name} · {cfg.source} → {cfg.target}")
    table.add_column("#", justify="right")
    table.add_column("source")
    table.add_column("translation")
    table.add_column("lag", justify="right")
    for r in results:
        table.add_row(str(r.index), r.source_text or f"[{r.dropped}]", r.translated_text,
                      f"{r.lag_s:.2f}s" if r.translated_text else "-")
    console.print(table)
    console.print(f"[dim]{len(audio)/SR:.1f}s audio processed in {elapsed:.1f}s "
                  f"({len(audio)/SR/max(elapsed,1e-6):.1f}× real time) · "
                  f"peak VRAM {stats['vram']['gpu_peak_mb']} MB of {stats['vram']['ceiling_mb']} MB budget[/dim]")
    if out_dir:
        console.print(f"audio written to [bold]{out_dir}[/bold]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    interpreter = Interpreter(cfg, on_event=_pretty_event)
    interpreter.warm()
    source = open_source(cfg.input_device, loopback=args.loopback, block_ms=cfg.block_ms)
    speaker = Speaker(cfg.output_device) if cfg.speak else None
    stop = threading.Event()
    console.print(f"[bold]interpreting[/bold] {language_name(cfg.source)} → {language_name(cfg.target)} "
                  f"from device {source.device}")
    try:
        with source:
            if speaker is not None:
                speaker.open()
                try:
                    interpreter.run_live(source, speaker, stop, partials=not args.no_partials)
                finally:
                    speaker.close()
            else:
                interpreter.run_live(source, None, stop, partials=not args.no_partials)
    except KeyboardInterrupt:
        stop.set()
    stats = interpreter.stats()
    console.print(json.dumps({k: v for k, v in stats.items() if k != "gate_rejections"}, indent=2, default=str))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Run every wav in a folder; optionally score against reference translations."""
    cfg = _config_from_args(args)
    cfg.speak = False
    folder = Path(args.path)
    files = sorted(folder.glob("*.wav"))
    if not files:
        console.print(f"[red]no .wav files in {folder}[/red]")
        return 1
    interpreter = Interpreter(cfg, on_event=lambda e: None)
    interpreter.warm()
    rows = []
    hyps: list[str] = []
    refs: list[str] = []
    for wav in files:
        started = time.perf_counter()
        results = interpreter.translate_file(wav)
        wall = time.perf_counter() - started
        audio, _ = read_wav(wav)
        hyp = " ".join(r.translated_text for r in results if r.translated_text).strip()
        ref_path = wav.with_suffix(f".{cfg.target}.txt")
        ref = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else ""
        if ref:
            hyps.append(hyp)
            refs.append(ref)
        rows.append({
            "file": wav.name,
            "audio_s": round(len(audio) / SR, 1),
            "wall_s": round(wall, 1),
            "rtf": round(wall / max(len(audio) / SR, 1e-6), 2),
            "segments": len([r for r in results if r.translated_text]),
            "hypothesis": hyp,
        })
    table = Table(title=f"bench · {cfg.profile} · {cfg.source}→{cfg.target}")
    for column in ("file", "audio_s", "wall_s", "rtf", "segments", "hypothesis"):
        table.add_column(column)
    for row in rows:
        table.add_row(str(row["file"]), str(row["audio_s"]), str(row["wall_s"]), str(row["rtf"]),
                      str(row["segments"]), row["hypothesis"][:70])
    console.print(table)
    if hyps:
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        console.print(f"BLEU [bold]{bleu.score:.1f}[/bold] · chrF [bold]{chrf.score:.1f}[/bold] "
                      f"over {len(hyps)} reference file(s)")
    stats = interpreter.stats()
    console.print(f"peak GPU {stats['vram']['gpu_peak_mb']} MB · own models {stats['vram']['own_peak_mb']} MB")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "stats": stats}, indent=2, default=str), encoding="utf-8")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Push a wav through the *live* pipeline, pacing it to wall clock.

    This is the number that matters: end-of-speech → first translated audio, with
    VAD, partial decoding and TTS in the loop. File mode is for content; this is for
    latency.
    """
    from .audio import FileSource

    cfg = _config_from_args(args)
    cfg.speak = not args.no_speak
    interpreter = Interpreter(cfg, on_event=_pretty_event)
    interpreter.warm()
    source = FileSource(args.path, block_ms=cfg.block_ms, realtime=not args.fast)
    started = time.perf_counter()
    with GpuSampler(interpreter.budget):
        results = interpreter.run_live(source, None, threading.Event(), partials=not args.no_partials)
    wall = time.perf_counter() - started
    stats = interpreter.stats()

    table = Table(title=f"simulate · {Path(args.path).name} · {cfg.profile}")
    for column in ("#", "source", "translation", "speech", "stt", "mt", "lag"):
        table.add_column(column, justify="right" if column != "source" and column != "translation" else "left")
    for r in results:
        table.add_row(str(r.index), r.source_text or f"[{r.dropped}]", r.translated_text,
                      f"{r.audio_s:.1f}s", f"{r.stt_s:.2f}s", f"{r.mt_s:.2f}s",
                      f"{r.lag_s:.2f}s" if r.lag_s else "-")
    console.print(table)
    console.print(f"lag p50 [bold]{stats['lag_p50']:.2f}s[/bold] · p90 {stats['lag_p90']:.2f}s · "
                  f"stt p50 {stats['stt_p50']:.2f}s · mt p50 {stats['mt_p50']:.2f}s · "
                  f"wall {wall:.1f}s · peak GPU {stats['vram']['gpu_peak_mb']} MB")
    if args.json:
        Path(args.json).write_text(json.dumps({"results": [r.row() for r in results], "stats": stats},
                                              indent=2, default=str), encoding="utf-8")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import main as gui_main

    return gui_main()


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    cfg = _config_from_args(args)
    return serve(cfg, host=args.host, port=args.port, loopback=args.loopback)


def cmd_meet(args: argparse.Namespace) -> int:
    from .duplex import Duplex, DuplexConfig

    duplex_cfg = DuplexConfig(
        a=args.a, b=args.b, profile=args.profile,
        mic_device=args.mic, loopback_device=args.loopback_device,
        cable_device=args.cable, headphone_device=args.headphones,
        captions_only_inbound=args.captions_only,
        glossary_path=args.glossary, vram_ceiling_mb=args.vram_ceiling or 5600,
    )
    duplex = Duplex(duplex_cfg, on_event=_pretty_event)
    duplex.warm()
    console.print(f"[bold]meeting mode[/bold] {language_name(args.a)} ↔ {language_name(args.b)} · "
                  f"speak into {args.mic or 'default mic'} · meeting hears {args.cable or 'NOTHING (pass --cable)'}")
    if args.dry_run:
        report = duplex.engines.budget.report()
        console.print(f"[green]dry run[/green] resident={report['resident']} "
                      f"own={report['resident_mb']} MB of {report['ceiling_mb']} MB ceiling")
        console.print("both directions share one recogniser and one translator; only the voices differ (CPU)")
        return 0
    try:
        duplex.run()
    except KeyboardInterrupt:
        pass
    return 0


# --- parser -----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parley", description="Offline live interpretation under 6 GB of VRAM")
    parser.add_argument("--version", action="version", version=f"parley {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, pair: bool = True) -> None:
        if pair:
            p.add_argument("--from", dest="source", default="en", help="source language (whisper code)")
            p.add_argument("--to", dest="target", default="de", help="target language (whisper code)")
        p.add_argument("--profile", default="balanced", choices=sorted(PROFILES))
        p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
        p.add_argument("--vram-ceiling", type=int, default=0, help="MiB; default from PARLEY_VRAM_CEILING_MB or 5600")
        p.add_argument("--glossary", default=None, help="TSV: source_term<TAB>target_term<TAB>bad variants")

    doctor = sub.add_parser("doctor", help="check GPU, models, devices")
    common(doctor)
    doctor.set_defaults(func=cmd_doctor)

    devices = sub.add_parser("devices", help="list audio endpoints")
    devices.set_defaults(func=cmd_devices)

    gui = sub.add_parser("gui", help="open the desktop app")
    gui.set_defaults(func=cmd_gui)

    models = sub.add_parser("models", help="show profiles and their VRAM cost")
    models.set_defaults(func=cmd_models)

    file_cmd = sub.add_parser("file", help="translate a wav file and report timings")
    common(file_cmd)
    file_cmd.add_argument("path")
    file_cmd.add_argument("--out", default=None, help="write translated speech here")
    file_cmd.add_argument("--no-speak", action="store_true")
    file_cmd.set_defaults(func=cmd_file)

    run = sub.add_parser("run", help="live interpretation from a microphone or loopback")
    common(run)
    run.add_argument("--input-device", default=None)
    run.add_argument("--output-device", default=None)
    run.add_argument("--loopback", action="store_true", help="capture system audio instead of the mic")
    run.add_argument("--no-speak", action="store_true", help="captions only")
    run.add_argument("--no-partials", action="store_true", help="wait for full utterances")
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("bench", help="score a folder of wavs (+ optional .<lang>.txt references)")
    common(bench)
    bench.add_argument("path")
    bench.add_argument("--json", default=None)
    bench.set_defaults(func=cmd_bench)

    simulate = sub.add_parser("simulate", help="push a wav through the live pipeline and measure lag")
    common(simulate)
    simulate.add_argument("path")
    simulate.add_argument("--no-speak", action="store_true")
    simulate.add_argument("--no-partials", action="store_true")
    simulate.add_argument("--fast", action="store_true", help="do not pace to wall clock (throughput only)")
    simulate.add_argument("--json", default=None)
    simulate.set_defaults(func=cmd_simulate)

    serve_cmd = sub.add_parser("serve", help="local web UI with live captions")
    common(serve_cmd)
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8770)
    serve_cmd.add_argument("--loopback", action="store_true")
    serve_cmd.set_defaults(func=cmd_serve)

    meet = sub.add_parser("meet", help="bidirectional meeting mode: mic ↔ system audio")
    meet.add_argument("--a", default="en", help="the language you speak")
    meet.add_argument("--b", default="de", help="the language the room speaks")
    meet.add_argument("--profile", default="balanced", choices=sorted(PROFILES))
    meet.add_argument("--mic", default=None, help="your microphone (name or index)")
    meet.add_argument("--loopback-device", default=None, help="system-audio capture endpoint")
    meet.add_argument("--cable", default=None, help="virtual cable the meeting listens to (VB-Cable)")
    meet.add_argument("--headphones", default=None, help="where you hear the inbound translation")
    meet.add_argument("--captions-only", action="store_true", help="do not speak the inbound direction")
    meet.add_argument("--dry-run", action="store_true", help="load both directions, print the VRAM plan, exit")
    meet.add_argument("--glossary", default=None)
    meet.add_argument("--vram-ceiling", type=int, default=0)
    meet.set_defaults(func=cmd_meet)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

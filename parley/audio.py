"""Audio I/O: capture (microphone or system loopback), playback, files.

The interesting part is that a live interpreter needs **two different inputs**:
your microphone for the outbound direction, and the meeting's audio — a *render*
endpoint, not a capture device — for the inbound one. On Windows that means WASAPI
loopback; :func:`open_source` handles both behind one interface.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import sounddevice as sd

TARGET_SR = 16000

# Names Windows drivers use for "whatever is playing right now".
LOOPBACK_HINTS = ("loopback", "stereo mix", "what u hear", "what you hear", "cable output", "wave out mix")


# --- device enumeration -----------------------------------------------------------
@dataclass
class Device:
    index: int
    name: str
    hostapi: str
    max_input: int
    max_output: int
    default_sr: int

    def __str__(self) -> str:
        kind = []
        if self.max_input:
            kind.append(f"in:{self.max_input}")
        if self.max_output:
            kind.append(f"out:{self.max_output}")
        return f"[{self.index}] {self.name} ({self.hostapi}, {', '.join(kind)})"


def list_devices() -> list[Device]:
    out: list[Device] = []
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(sd.query_devices()):
        out.append(Device(
            index=idx,
            name=dev["name"],
            hostapi=hostapis[dev["hostapi"]]["name"],
            max_input=int(dev["max_input_channels"]),
            max_output=int(dev["max_output_channels"]),
            default_sr=int(dev["default_samplerate"]),
        ))
    return out


def find_device(match: str | int | None, want_input: bool) -> int | None:
    """Resolve a user-supplied device by index or case-insensitive substring."""
    if match is None or match == "":
        return None
    if isinstance(match, int) or str(match).isdigit():
        return int(match)
    needle = str(match).lower()
    for dev in list_devices():
        if needle in dev.name.lower() and ((dev.max_input and want_input) or (dev.max_output and not want_input)):
            return dev.index
    raise SystemExit(f"no {'input' if want_input else 'output'} device matching {match!r}; run `parley devices`")


def resolve_default_input() -> int | None:
    """Prefer the OS default mic, but never a loopback or virtual device."""
    try:
        idx, _ = sd.default.device  # (input, output)
        return idx
    except Exception:  # noqa: BLE001
        return None


def resample(audio: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = np.gcd(int(src_sr), int(dst_sr))
    return resample_poly(audio.astype(np.float32), dst_sr // g, src_sr // g).astype(np.float32)


# --- capture ----------------------------------------------------------------------
class AudioSource:
    """Blocking iterator of 16 kHz mono float32 frames."""

    def __init__(self, device: int | None, block_ms: int = 30, loopback: bool = False,
                 samplerate: int | None = None):
        self.device = device
        self.block_ms = block_ms
        self.loopback = loopback
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._stream: sd.InputStream | None = None
        self.samplerate = samplerate
        self.overflow = 0

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001 - PortAudio signature
        if status:
            self.overflow += 1
        mono = indata[:, 0] if indata.ndim > 1 else indata
        try:
            self._q.put_nowait(mono.copy())
        except queue.Full:
            self.overflow += 1

    def open(self) -> "AudioSource":
        info = sd.query_devices(self.device, "input") if self.device is not None else sd.query_devices(kind="input")
        sr = self.samplerate or int(info["default_samplerate"])
        self.samplerate = sr
        if self.loopback:
            self._check_loopback_device(info["name"])
        self._stream = sd.InputStream(
            device=self.device, channels=1, samplerate=sr, dtype="float32",
            blocksize=int(sr * self.block_ms / 1000), callback=self._callback,
        )
        self._stream.start()
        return self

    @staticmethod
    def _check_loopback_device(name: str) -> None:
        lowered = name.lower()
        if not any(tag in lowered for tag in LOOPBACK_HINTS):
            raise SystemExit(
                f"{name!r} is not a loopback device. For the inbound direction you need either "
                f"'Stereo Mix' / a 'Loopback' endpoint, or a virtual cable (VB-Cable / VoiceMeeter) "
                f"that the meeting's audio is routed into. Run `parley devices` to list what this "
                f"machine exposes.")

    def frames(self) -> Iterator[np.ndarray]:
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            yield resample(chunk, self.samplerate or TARGET_SR)

    def close(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def __enter__(self) -> "AudioSource":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_source(device: str | int | None = None, loopback: bool = False, block_ms: int = 30) -> AudioSource:
    idx = find_device(device, want_input=True) if device is not None else None
    if loopback and idx is None:
        idx = find_loopback_device()
        if idx is None:
            raise SystemExit(
                "no loopback capture device found. Either enable 'Stereo Mix' in the sound control "
                "panel, or install VB-Cable and route the meeting audio into it. `parley devices` "
                "shows every endpoint this machine exposes.")
    return AudioSource(idx, block_ms=block_ms, loopback=loopback)


def find_loopback_device() -> int | None:
    """First input endpoint that carries system audio (Stereo Mix, VB-Cable output, ...)."""
    for dev in list_devices():
        if dev.max_input and any(tag in dev.name.lower() for tag in LOOPBACK_HINTS):
            return dev.index
    return None


# --- playback ---------------------------------------------------------------------
class Speaker:
    """Queued playback with a callback so the pipeline can timestamp first audio."""

    def __init__(self, device: str | int | None = None, gain: float = 1.0):
        idx = find_device(device, want_input=False) if device is not None else None
        info = sd.query_devices(idx, "output") if idx is not None else sd.query_devices(kind="output")
        self.device = idx
        self.samplerate = int(info["default_samplerate"])
        self.gain = gain
        self._q: "queue.Queue[tuple[np.ndarray, object]]" = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending = np.zeros(0, dtype=np.float32)

    def open(self) -> "Speaker":
        self._stream = sd.OutputStream(
            device=self.device, channels=1, samplerate=self.samplerate, dtype="float32",
            blocksize=0, callback=self._callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._worker, name="speaker", daemon=True)
        self._thread.start()
        return self

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        need = frames
        while need > 0:
            if self._pending.size == 0:
                try:
                    audio, done = self._q.get_nowait()
                except queue.Empty:
                    outdata[:need] = 0
                    return
                if audio is not None and audio.size:
                    self._pending = resample(audio, self.samplerate, TARGET_SR) * self.gain
                if done is not None:
                    done.set()
            take = min(need, self._pending.size)
            outdata[:take, 0] = self._pending[:take]
            self._pending = self._pending[take:]
            need -= take

    def _worker(self) -> None:
        """Kept for symmetry with future streaming writers; playback happens in the callback."""
        while not self._stop.wait(0.5):
            continue

    def enqueue(self, audio: np.ndarray, sr: int = TARGET_SR) -> threading.Event:
        done = threading.Event()
        if sr != TARGET_SR:
            audio = resample(audio, sr, TARGET_SR)
        self._q.put((np.ascontiguousarray(audio, dtype=np.float32), done))
        return done

    def busy(self) -> bool:
        return not self._q.empty() or self._pending.size > 0

    def flush(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while self.busy() and time.time() < deadline:
            time.sleep(0.02)

    def close(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def __enter__(self) -> "Speaker":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- files ------------------------------------------------------------------------
def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path

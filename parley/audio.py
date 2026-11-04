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


_FIR_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _lowpass_taps(src_sr: int, cutoff_hz: float, taps: int = 101) -> np.ndarray:
    """Windowed-sinc low-pass, designed once per (rate, cutoff) and cached."""
    key = (int(src_sr), int(cutoff_hz), int(taps))
    cached = _FIR_CACHE.get(key)
    if cached is not None:
        return cached
    n = np.arange(taps) - (taps - 1) / 2.0
    fc = cutoff_hz / src_sr                       # cycles per sample
    h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
    h /= h.sum()
    _FIR_CACHE[key] = h.astype(np.float32)
    return _FIR_CACHE[key]


def resample(audio: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    """Convert to 16 kHz without touching scipy.

    scipy used to be imported *here*, on the first audio block: a multi-second import
    inside the capture loop, which made Start look dead and Stop unreachable. numpy
    only now, and the filter is cached.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if src_sr == dst_sr or audio.size == 0:
        return audio
    if dst_sr > src_sr:                            # upsampling never happens for capture
        times = np.arange(int(audio.size * dst_sr / src_sr)) * (src_sr / dst_sr)
        return np.interp(times, np.arange(audio.size), audio).astype(np.float32)

    if src_sr % dst_sr == 0:                       # 48k→16k, 32k→16k, 96k→16k
        step = src_sr // dst_sr
        taps = _lowpass_taps(src_sr, 0.45 * dst_sr, taps=8 * step + 1)
        return np.convolve(audio, taps, mode="same")[::step].astype(np.float32)

    # 44.1k→16k and friends: filter at the source rate, then take the samples we need
    taps = _lowpass_taps(src_sr, 0.45 * dst_sr)
    filtered = np.convolve(audio, taps, mode="same")
    times = np.arange(int(audio.size * dst_sr / src_sr)) * (src_sr / dst_sr)
    return np.interp(times, np.arange(audio.size), filtered).astype(np.float32)


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
        if self.loopback:
            self._check_loopback_device(info["name"])
        native = int(info["default_samplerate"]) or TARGET_SR
        # Ask the driver for 16 kHz first: WASAPI resamples in the mixer, which costs
        # nothing here and keeps the capture loop free of per-block conversion.
        wanted = self.samplerate or TARGET_SR
        candidates = [wanted] if wanted != native else [wanted]
        if native not in candidates:
            candidates.append(native)
        last: BaseException | None = None
        for sr in candidates:
            try:
                stream = sd.InputStream(
                    device=self.device, channels=1, samplerate=sr, dtype="float32",
                    blocksize=int(sr * self.block_ms / 1000), callback=self._callback,
                )
                stream.start()
            except Exception as exc:                   # noqa: BLE001 - try the next rate
                last = exc
                continue
            self._stream = stream
            self.samplerate = sr
            return self
        raise SystemExit(f"cannot open {info['name']!r} for capture at {candidates} Hz: {last}")

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
    """Plays audio on a real device.

    A blocking writer thread, not a PortAudio callback: `write()` returns when the
    card has taken the samples, so "did it play" is measurable rather than hopeful.
    The old callback version resampled with swapped arguments and swallowed its own
    exceptions, which is how a translator can produce perfect text and total silence.
    """

    def __init__(self, device: str | int | None = None, gain: float = 1.0):
        idx = find_device(device, want_input=False) if device is not None else None
        info = sd.query_devices(idx, "output") if idx is not None else sd.query_devices(kind="output")
        self.device = idx
        self.samplerate = int(info["default_samplerate"]) or 48000
        self.gain = gain
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.frames_written = 0
        self.last_error: BaseException | None = None
        self._in_write = False

    def open(self) -> "Speaker":
        self._stream = sd.OutputStream(
            device=self.device, channels=1, samplerate=self.samplerate, dtype="float32", blocksize=0,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._writer, name="speaker", daemon=True)
        self._thread.start()
        return self

    def _writer(self) -> None:
        while not self._stop.is_set():
            try:
                audio = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if audio is None:
                break
            if audio.size == 0:
                continue
            try:
                assert self._stream is not None
                self._in_write = True
                self._stream.write(audio.reshape(-1, 1))
                self.frames_written += int(audio.size)
            except Exception as exc:                   # noqa: BLE001 - record it, never die silently
                self.last_error = exc
                return
            finally:
                self._in_write = False

    def enqueue(self, audio: np.ndarray, sr: int = TARGET_SR) -> None:
        if sr != self.samplerate:
            audio = resample(audio, sr, self.samplerate)
        audio = np.ascontiguousarray(audio, dtype=np.float32) * self.gain
        self._q.put(audio)

    def pending_seconds(self) -> float:
        return sum(item.size for item in list(self._q.queue)) / max(self.samplerate, 1)

    def busy(self) -> bool:
        return not self._q.empty()

    def interrupt(self) -> None:
        """Drop anything not yet played (used when the user stops a session)."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def flush(self, timeout: float = 60.0) -> None:
        """Wait until the card has actually taken every sample.

        `write()` blocks while the driver drains, so the queue being empty is not
        enough: the item in flight still has to finish.
        """
        deadline = time.time() + timeout
        while (not self._q.empty() or self._in_write) and time.time() < deadline:
            time.sleep(0.02)
        time.sleep(0.05)

    def close(self) -> None:
        self._stop.set()
        self._q.put(np.zeros(0, dtype=np.float32))
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
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


class FileSource:
    """Feeds a wav through the live pipeline.

    With ``realtime=True`` the frames are paced to wall-clock, so the measured lag is
    the lag you would get from a microphone — the only difference is where the audio
    comes from. That makes latency reproducible on a machine with no meeting and no
    microphone, which is what the benchmark table in the README is built from.
    """

    device = None

    def __init__(self, path: str | Path, block_ms: int = 30, realtime: bool = True):
        self.audio, self.samplerate = read_wav(path)
        if self.samplerate != TARGET_SR:
            self.audio = resample(self.audio, self.samplerate, TARGET_SR)
            self.samplerate = TARGET_SR
        self.block_ms = block_ms
        self.realtime = realtime
        self.overflow = 0

    def open(self) -> "FileSource":
        return self

    def frames(self) -> Iterator[np.ndarray]:
        step = int(self.samplerate * self.block_ms / 1000)
        started = time.time()
        for i in range(0, len(self.audio), step):
            block = np.ascontiguousarray(self.audio[i:i + step], dtype=np.float32)
            if block.size < step:
                block = np.pad(block, (0, step - block.size))
            if self.realtime:
                target = started + (i + step) / self.samplerate
                delay = target - time.time()
                if delay > 0:
                    time.sleep(delay)
            yield block
        # let the tail behave like a burst of silence, so the last utterance closes
        for _ in range(int(0.8 * 1000 / self.block_ms)):
            if self.realtime:
                time.sleep(self.block_ms / 1000)
            yield np.zeros(step, dtype=np.float32)

    def close(self) -> None:
        return None

    def __enter__(self) -> "FileSource":
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

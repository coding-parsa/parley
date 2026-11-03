"""VRAM budget: one language pair resident, never more than the ceiling.

Two jobs:

* **Admission control.** A model may only be loaded if the resident set plus its
  estimated footprint fits under ``ceiling_mb``. When it does not, least-recently
  used models are unloaded — that is how "load one pair at a time" is enforced
  rather than promised.
* **Measurement.** After a load, the real GPU delta is read back from `nvidia-smi`
  and replaces the estimate, so the numbers in the README come from the card, not
  from a table someone typed.

`nvidia-smi` is polled instead of NVML so the package needs no extra dependency
and behaves the same on Windows and Linux.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

_NVIDIA_SMI = shutil.which("nvidia-smi")


class VramError(RuntimeError):
    """Raised when the resident set cannot be made to fit the ceiling."""


@dataclass
class Resident:
    key: str
    mb: int
    unload: Callable[[], None] | None
    loaded_at: float = field(default_factory=time.time)
    touched_at: float = field(default_factory=time.time)
    pinned: bool = False
    measured: bool = False


def gpu_used_mb() -> int | None:
    """Whole-card usage in MiB, or None when the card cannot be queried."""
    if not _NVIDIA_SMI:
        return None
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def gpu_total_mb() -> int | None:
    if not _NVIDIA_SMI:
        return None
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def gpu_free_mb() -> int | None:
    """Free memory on the card, or None when it cannot be queried.

    Other applications are not parley's to manage, but a model that does not fit
    in what is *actually* free will fail inside CUDA with an unhelpful error, so
    this is checked before every load.
    """
    used, total = gpu_used_mb(), gpu_total_mb()
    if used is None or total is None:
        return None
    return max(0, total - used)


def gpu_name() -> str | None:
    if not _NVIDIA_SMI:
        return None
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class Budget:
    """Tracks what parley itself has loaded, and evicts to stay under a ceiling."""

    def __init__(self, ceiling_mb: int, log: Callable[[str], None] | None = None):
        self.ceiling_mb = int(ceiling_mb)
        self._resident: "OrderedDict[str, Resident]" = OrderedDict()
        self._lock = threading.RLock()
        self._log = log or (lambda msg: None)
        self.peak_own_mb = 0
        self.peak_gpu_mb = gpu_used_mb() or 0
        self._baseline_gpu_mb = self.peak_gpu_mb

    # -- bookkeeping ---------------------------------------------------------
    @property
    def resident_mb(self) -> int:
        with self._lock:
            return sum(r.mb for r in self._resident.values())

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._resident)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {k: r.mb for k, r in self._resident.items()}

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._resident

    def touch(self, key: str) -> None:
        with self._lock:
            if key in self._resident:
                self._resident[key].touched_at = time.time()
                self._resident.move_to_end(key)

    def pin(self, key: str, pinned: bool = True) -> None:
        with self._lock:
            if key in self._resident:
                self._resident[key].pinned = pinned

    # -- admission -----------------------------------------------------------
    def admit(self, key: str, estimate_mb: int, unload: Callable[[], None] | None = None) -> None:
        """Make room for ``key`` and register it as resident."""
        with self._lock:
            if key in self._resident:
                self.touch(key)
                return
            if estimate_mb > self.ceiling_mb:
                raise VramError(
                    f"{key} needs ~{estimate_mb} MB but the ceiling is {self.ceiling_mb} MB; "
                    f"pick a smaller profile (parley models list)")
            while self.resident_mb + estimate_mb > self.ceiling_mb:
                victim = self._pick_victim()
                if victim is None:
                    raise VramError(
                        f"cannot fit {key} (~{estimate_mb} MB) under {self.ceiling_mb} MB "
                        f"with pinned models {[k for k, r in self._resident.items() if r.pinned]}")
                self._evict(victim)
            self._resident[key] = Resident(key=key, mb=estimate_mb, unload=unload)
            self.peak_own_mb = max(self.peak_own_mb, self.resident_mb)

    def _pick_victim(self) -> str | None:
        for k, r in self._resident.items():          # OrderedDict is LRU-ordered
            if not r.pinned:
                return k
        return None

    def _evict(self, key: str) -> None:
        entry = self._resident.pop(key, None)
        if entry is None:
            return
        self._log(f"vram: evict {key} (~{entry.mb} MB) to stay under {self.ceiling_mb} MB")
        if entry.unload is not None:
            try:
                entry.unload()
            except Exception as exc:                  # noqa: BLE001 - eviction must never crash the loop
                self._log(f"vram: unload of {key} failed: {type(exc).__name__}: {exc}")

    def release(self, key: str) -> None:
        with self._lock:
            self._evict(key)

    # -- measurement ---------------------------------------------------------
    def ensure_free(self, key: str, estimate_mb: int, margin_mb: int = 250) -> None:
        """Fail early, with advice, when the card itself has no room left."""
        free = gpu_free_mb()
        if free is None or estimate_mb == 0:
            return
        if free < estimate_mb + margin_mb:
            raise VramError(
                f"only {free} MB free on the GPU, but {key} needs ~{estimate_mb} MB (+{margin_mb} MB "
                f"headroom). Close the other GPU apps, or use --profile balanced / --profile fast.")

    def measure(self, key: str, load: Callable[[], None]) -> None:
        """Run ``load`` and correct ``key``'s footprint with the observed GPU delta.

        Windows WDDM reports ``[N/A]`` for per-process memory, so the only figure
        available is whole-card usage before and after. That is noisy when other
        applications load models at the same moment, so the delta is only trusted
        when it is plausible (between a fifth and three times the estimate);
        otherwise the estimate stands and the run still reports the *peak card*
        usage measured by :class:`GpuSampler`.
        """
        before = gpu_used_mb() or 0
        load()
        after = gpu_used_mb() or 0
        delta = after - before
        with self._lock:
            entry = self._resident.get(key)
            if entry is not None and entry.mb > 0:
                if 0.2 * entry.mb <= delta <= 3 * entry.mb:
                    entry.mb = max(1, delta)
                    entry.measured = True
                self.peak_own_mb = max(self.peak_own_mb, self.resident_mb)
            self.peak_gpu_mb = max(self.peak_gpu_mb, after)

    def sample_gpu(self) -> int:
        used = gpu_used_mb() or 0
        with self._lock:
            self.peak_gpu_mb = max(self.peak_gpu_mb, used)
        return used

    def report(self) -> dict[str, object]:
        with self._lock:
            used = gpu_used_mb()
            return {
                "ceiling_mb": self.ceiling_mb,
                "resident": self.snapshot(),
                "resident_mb": self.resident_mb,
                "own_peak_mb": self.peak_own_mb,
                "gpu_peak_mb": self.peak_gpu_mb,
                "gpu_used_mb": used,
                "gpu_baseline_mb": self._baseline_gpu_mb,
                "headroom_mb": (self.ceiling_mb - self.resident_mb),
            }


class GpuSampler:
    """Background thread that keeps the peak GPU figure honest while a run is live."""

    def __init__(self, budget: Budget, interval_s: float = 0.5):
        self.budget = budget
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, name="vram-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.budget.sample_gpu()

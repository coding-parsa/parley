"""Does opening the output kill the microphone?

Symptom: status says listening, the user speaks, nothing appears. Count the frames
that actually arrive before and after the output stream is opened — a Bluetooth
headset switching from A2DP to hands-free can starve the capture stream silently.
"""

import sys
import threading
import time

sys.path.insert(0, r"C:\Users\parsa\parley")

import numpy as np

from parley.audio import Speaker, open_source

spoken = []


def meter(label: str, seconds: float, speaker: Speaker | None = None) -> None:
    source = open_source("", loopback=False, block_ms=30)
    frames = 0
    peak = 0.0
    stop = threading.Event()

    def watch() -> None:
        nonlocal frames, peak
        for chunk in source.frames():
            frames += 1
            peak = max(peak, float(np.abs(chunk).max()))
            if stop.is_set():
                break

    with source:
        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        if speaker is not None:
            speaker.open()
        time.sleep(seconds)
        stop.set()
        source.close()
        thread.join(timeout=3)
        if speaker is not None:
            speaker.close()
    spoken.append(f"{label}: device={source.samplerate} Hz frames={frames} "
                  f"({'DEAD — no audio arrived' if frames == 0 else 'alive'}) peak={peak:.3f} "
                  f"overflows={source.overflow}")


def main() -> int:
    print("1) microphone alone", flush=True)
    meter("mic only", 4.0)

    print("2) microphone while the default output (Bluetooth?) is open", flush=True)
    meter("mic + output", 4.0, speaker=Speaker())

    print("3) microphone with speak disabled path", flush=True)
    meter("mic only again", 4.0)

    print("\n".join(spoken))
    return 0


if __name__ == "__main__":
    sys.exit(main())

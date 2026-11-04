"""Where does a live Start actually block? Dump every thread's stack on a timer."""

import faulthandler
import sys
import threading
import time

sys.path.insert(0, r"C:\Users\parsa\parley")

from parley.audio import Speaker
from parley.config import Config
from parley.pipeline import Interpreter

faulthandler.dump_traceback_later(20, repeat=True)

cfg = Config(source="en", target="de", profile="fast", speak=True)
interpreter = Interpreter(cfg, on_event=lambda e: print("EVENT", e, flush=True))
print("warming…", flush=True)
t0 = time.perf_counter()
interpreter.warm()
print(f"warm done in {time.perf_counter() - t0:.1f}s", flush=True)

from parley.audio import open_source

print("opening source…", flush=True)
source = open_source("", loopback=False, block_ms=cfg.block_ms)
print("source:", type(source).__name__, flush=True)
speaker = Speaker("")
stop = threading.Event()

def stopper():
    time.sleep(12)
    print("REQUESTING STOP", flush=True)
    stop.set()

threading.Thread(target=stopper, daemon=True).start()

with source:
    speaker.open()
    try:
        interpreter.run_live(source, speaker, stop, partials=cfg.speak)
    finally:
        print("run_live returned", flush=True)
        speaker.close()

print("stopped cleanly", flush=True)
faulthandler.cancel_dump_traceback_later()

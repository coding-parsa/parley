"""Second Start must work.

Reproduces the reported failure: run a session, stop, run another one on the same
loaded models, and check that the second session translates instead of erroring.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, r"C:\Users\parsa\parley")

from parley.audio import FileSource
from parley.config import Config
from parley.pipeline import Interpreter

AUDIO = Path(r"C:\Users\parsa\parley\tests\audio")


def session(interpreter: Interpreter, clip: Path) -> dict:
    events: list[dict] = []
    interpreter.on_event = events.append
    interpreter.reset_session()
    source = FileSource(clip, realtime=False)
    stop = threading.Event()
    interpreter.run_live(source, None, stop, partials=False)
    stats = interpreter.stats()
    errors = [e["message"] for e in events if e.get("type") == "error"]
    return {"segments": stats["segments"], "translated": stats["translated"], "errors": errors,
            "vram": stats["vram"]["resident_mb"]}


def main() -> int:
    cfg = Config(source="en", target="de", profile="balanced", speak=False)
    interpreter = Interpreter(cfg, on_event=lambda _e: None)
    interpreter.warm()

    first = session(interpreter, AUDIO / "en_meeting_01.wav")
    print("session 1:", first)
    second = session(interpreter, AUDIO / "en_meeting_02.wav")
    print("session 2:", second)
    third = session(interpreter, AUDIO / "en_meeting_03.wav")
    print("session 3:", third)

    ok = all(s["translated"] > 0 and not s["errors"] for s in (first, second, third))
    ok = ok and second["vram"] == first["vram"]          # no model duplication between sessions
    print("RESULT:", "SESSIONS OK" if ok else "SECOND SESSION BROKEN")
    interpreter.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

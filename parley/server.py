"""Local web console: live captions, lag and VRAM while a meeting runs.

Serves on 127.0.0.1 only — this is a cockpit, not a service. The page is a plain
HTML file; the pipeline pushes every event over one WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from .audio import Speaker, open_source
from .config import Config
from .pipeline import Interpreter

WEB_DIR = Path(__file__).parent / "web"


class Runtime:
    """Holds the live pipeline and fans its events out to connected browsers."""

    def __init__(self, cfg: Config, loopback: bool = False):
        self.cfg = cfg
        self.loopback = loopback
        self.interpreter: Interpreter | None = None
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.history: list[dict] = []
        self.state = "idle"

    # -- event fan-out -------------------------------------------------------
    def emit(self, event: dict) -> None:
        if event.get("type") in ("segment", "caption", "rejected"):
            self.history.append(event)
            del self.history[:-200]
        if self.loop is None:
            return
        for queue in list(self.clients):
            try:
                self.loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                pass

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self.state == "running":
            return
        self.state = "running"
        self.stop.clear()
        self.thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.interpreter = Interpreter(self.cfg, on_event=self.emit)
            self.interpreter.warm()
            source = open_source(self.cfg.input_device, loopback=self.loopback)
            speaker = Speaker(self.cfg.output_device) if self.cfg.speak else None
            with source:
                if speaker is not None:
                    speaker.open()
                try:
                    self.interpreter.run_live(source, speaker, self.stop)
                finally:
                    if speaker is not None:
                        speaker.close()
        except SystemExit as exc:
            self.emit({"type": "error", "where": "start", "message": str(exc)})
        except Exception as exc:                       # noqa: BLE001
            self.emit({"type": "error", "where": "runtime", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.state = "idle"
            self.emit({"type": "stopped", "stats": self.stats()})

    def stop_now(self) -> None:
        self.stop.set()
        self.state = "stopping"

    def stats(self) -> dict:
        if self.interpreter is None:
            return {"state": self.state, "config": self.cfg.describe()}
        return {"state": self.state, "config": self.cfg.describe(), **self.interpreter.stats()}


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8770, loopback: bool = False) -> int:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="parley", docs_url=None, redoc_url=None)
    runtime = Runtime(cfg, loopback=loopback)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    async def status() -> dict:
        return runtime.stats()

    @app.post("/api/start")
    async def start() -> dict:
        runtime.start()
        return {"state": runtime.state}

    @app.post("/api/stop")
    async def stop() -> dict:
        runtime.stop_now()
        return {"state": runtime.state}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        runtime.loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        runtime.clients.add(queue)
        try:
            await socket.send_text(json.dumps({"type": "hello", "config": cfg.describe(),
                                               "history": runtime.history[-40:]}))
            while True:
                event = await queue.get()
                await socket.send_text(json.dumps(event, default=str))
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            runtime.clients.discard(queue)

    print(f"parley console on http://{host}:{port}  ({cfg.describe()})", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0

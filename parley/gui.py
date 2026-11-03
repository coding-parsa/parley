"""parley desktop app — a window with one big button, not a command line.

Design rule: someone who has never opened a terminal must be able to start a meeting
translation in two clicks. Three pages and nothing else:

    Live   — Start / demo, two columns of text, four numbers at the bottom
    Setup  — languages, microphone, quality, glossary. Saved automatically.
    Help   — how to plug it into Zoom/Teams, and what to do when something is red.

Everything long-running (model load, download, the pipeline) runs on a worker thread;
the UI only ever reads from a queue.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, ttk

from .audio import Speaker, list_devices, open_source
from .config import MODELS_DIR, WHISPER_TO_NLLB, Config, language_name
from .pipeline import Interpreter
from .vram import gpu_free_mb, gpu_name

# --- palette ----------------------------------------------------------------------
BG = "#070a10"
PANEL = "#0e1520"
PANEL_2 = "#131c2a"
LINE = "#26344a"
TEXT = "#eaf0f9"
MUTED = "#8b9db5"
DIM = "#5d6d84"
ACCENT = "#2f6fe4"
ACCENT_HOVER = "#3b7ef0"
OK = "#34d399"
WARN = "#fbbf24"
ERR = "#f87171"

UI = "Segoe UI"
F_BODY = (UI, 10)
F_SMALL = (UI, 9)
F_TINY = (UI, 8)
F_MONO = ("Consolas", 10)
F_SRC = (UI, 13)
F_TGT = (UI, 17)

PAD = 24                                          # one grid, used everywhere

SETTINGS_PATH = Path(os.environ.get("APPDATA", str(Path.home()))) / "parley" / "settings.json"


@dataclass
class Settings:
    source: str = "en"
    target: str = "de"
    profile: str = "balanced"
    input_device: str = ""
    use_system_audio: bool = False
    output_device: str = ""
    speak: bool = True
    glossary: str = ""
    duplex: bool = False

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            known = set(cls().__dict__)
            return cls(**{k: v for k, v in data.items() if k in known})
        except Exception:                       # noqa: BLE001 - settings must never block startup
            return cls()

    def save(self) -> None:
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except Exception:                       # noqa: BLE001
            pass


LANG_CHOICES = sorted(((code, language_name(code)) for code in WHISPER_TO_NLLB), key=lambda p: p[1])


def lang_choices() -> list[str]:
    return [f"{name}  ({code})" for code, name in LANG_CHOICES]


def lang_from_label(label: str) -> str:
    return label.rsplit("(", 1)[-1].rstrip(")").strip() if "(" in label else label


def lang_label(code: str) -> str:
    return f"{language_name(code)}  ({code})"


# --- widgets ----------------------------------------------------------------------
class FlatButton(tk.Label):
    """A clickable label that honours colours on Windows (tk.Button ignores bg)."""

    KINDS = {
        "primary": (ACCENT, "#ffffff", ACCENT_HOVER),
        "secondary": (PANEL_2, TEXT, "#1b2738"),
        "ghost": (BG, MUTED, PANEL),
        "danger": ("#3a1620", ERR, "#4a1c28"),
    }

    def __init__(self, master, text, command, kind="secondary", padx=18, pady=9, **kwargs):
        bg, fg, hover = self.KINDS[kind]
        super().__init__(master, text=text, bg=bg, fg=fg, font=(UI, 10, "bold"), padx=padx, pady=pady,
                         cursor="hand2", highlightthickness=1,
                         highlightbackground=LINE if kind == "secondary" else bg, **kwargs)
        self._normal, self._hover, self._fg = bg, hover, fg
        self._command = command
        self._enabled = True
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, _event=None):
        if self._enabled:
            self.configure(bg=self._hover)

    def _on_leave(self, _event=None):
        if self._enabled:
            self.configure(bg=self._normal)

    def _on_click(self, _event=None):
        if self._enabled:
            self._command()

    def set_text(self, text: str) -> None:
        self.configure(text=text)

    def set_kind(self, kind: str) -> None:
        bg, fg, hover = self.KINDS[kind]
        self._normal, self._hover, self._fg = bg, hover, fg
        self.configure(bg=bg, fg=fg)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.configure(bg=self._normal if enabled else BG,
                       fg=self._fg if enabled else "#3d4b5e",
                       cursor="hand2" if enabled else "arrow")


class Card(tk.Frame):
    def __init__(self, master, title: str | None = None, **kwargs):
        super().__init__(master, bg=PANEL, highlightbackground=LINE, highlightthickness=1, **kwargs)
        if title:
            tk.Label(self, text=title.upper(), bg=PANEL, fg=DIM, font=(UI, 8, "bold")).pack(
                anchor="w", padx=20, pady=(16, 0))


def combo(master, values: list[str], width: int = 34) -> ttk.Combobox:
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Parley.TCombobox", fieldbackground=PANEL_2, background=PANEL_2, foreground=TEXT,
                    arrowcolor=MUTED, bordercolor=LINE, lightcolor=LINE, darkcolor=LINE,
                    selectbackground=PANEL_2, selectforeground=TEXT, padding=7)
    style.map("Parley.TCombobox", fieldbackground=[("readonly", PANEL_2)], foreground=[("readonly", TEXT)])
    style.configure("Parley.Vertical.TScrollbar", background=PANEL_2, troughcolor=PANEL,
                    bordercolor=PANEL, arrowcolor=MUTED)
    return ttk.Combobox(master, values=values, state="readonly", style="Parley.TCombobox", width=width)


# --- the app ------------------------------------------------------------------------
class ParleyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:                                # noqa: BLE001
            pass

        self.title("parley — live interpretation")
        self.configure(bg=BG)
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.settings = Settings.load()
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.interpreter: Interpreter | None = None
        self.running = False
        self.page = "live"
        self._placeholder = True

        self._build_header()
        self._build_footer()
        self._build_content()
        self._show_page("live")
        self._refresh_ready_state()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<F5>", lambda _e: self.toggle())
        self.bind("<Escape>", lambda _e: self.stop())
        self.bind("<Control-d>", lambda _e: self.run_demo())
        self.after(60, self._darken_titlebar)
        self.after(90, self._pump)

    # -- chrome --------------------------------------------------------------
    def _darken_titlebar(self) -> None:
        """Windows draws a white title bar by default, which ruins a dark app."""
        try:
            import ctypes

            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            enabled = ctypes.c_int(1)
            for attribute in (20, 19):           # DWMWA_USE_IMMERSIVE_DARK_MODE (+ pre-20H1 variant)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(enabled),
                                                           ctypes.sizeof(enabled))
        except Exception:                        # noqa: BLE001
            pass

    def _build_header(self) -> None:
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=PAD, pady=(18, 8))

        brand = tk.Frame(bar, bg=BG)
        brand.pack(side="left")
        tk.Label(brand, text="parley", bg=BG, fg=TEXT, font=(UI, 21, "bold")).pack(side="left")
        tk.Label(brand, text="offline live interpretation", bg=BG, fg=DIM, font=F_SMALL).pack(
            side="left", padx=(12, 0), pady=(10, 0))

        right = tk.Frame(bar, bg=BG)
        right.pack(side="right")
        self.state_pill = tk.Label(right, text="●  ready", bg=PANEL_2, fg=OK, font=(UI, 9), padx=12, pady=5,
                                   highlightthickness=1, highlightbackground=LINE)
        self.state_pill.pack(side="right", padx=(16, 0))

        self.nav_buttons: dict[str, tk.Label] = {}
        self.nav_underlines: dict[str, tk.Frame] = {}
        for key, label in (("help", "Help"), ("setup", "Setup"), ("live", "Live")):
            holder = tk.Frame(right, bg=BG)
            holder.pack(side="right", padx=6)
            button = tk.Label(holder, text=label, bg=BG, fg=MUTED, font=(UI, 10), padx=8, pady=4, cursor="hand2")
            button.pack()
            underline = tk.Frame(holder, bg=BG, height=2)
            underline.pack(fill="x")
            button.bind("<Button-1>", lambda _e, k=key: self._show_page(k))
            self.nav_buttons[key] = button
            self.nav_underlines[key] = underline

        tk.Frame(self, bg=LINE, height=1).pack(fill="x", padx=PAD)

    def _build_footer(self) -> None:
        strip = tk.Frame(self, bg=BG)
        strip.pack(fill="x", side="bottom")
        tk.Frame(strip, bg=LINE, height=1).pack(fill="x")
        bar = tk.Frame(strip, bg=BG)
        bar.pack(fill="x", padx=PAD, pady=14)
        self.stat_labels: dict[str, tk.Label] = {}
        for index, (key, label) in enumerate((("lag", "lag p50"), ("stt", "speech"), ("mt", "translate"),
                                              ("vram", "vram"), ("models", "models"))):
            cell = tk.Frame(bar, bg=BG)
            cell.pack(side="left", padx=(0 if index == 0 else 34, 0))
            tk.Label(cell, text=label.upper(), bg=BG, fg="#556781", font=(UI, 7, "bold")).pack(anchor="w")
            value = tk.Label(cell, text="–", bg=BG, fg="#9aabc2", font=F_MONO)
            value.pack(anchor="w")
            self.stat_labels[key] = value
        self.hint_label = tk.Label(bar, text="", bg=BG, fg=MUTED, font=F_SMALL)
        self.hint_label.pack(side="right")

    def _build_content(self) -> None:
        self.content = tk.Frame(self, bg=BG)
        self.content.pack(fill="both", expand=True, padx=PAD, pady=(14, 8))
        self.pages: dict[str, tk.Frame] = {}
        self._build_live()
        self._build_setup()
        self._build_help()

    # -- live page -----------------------------------------------------------
    def _build_live(self) -> None:
        page = tk.Frame(self.content, bg=BG)
        self.pages["live"] = page

        toolbar = tk.Frame(page, bg=BG)
        toolbar.pack(fill="x", pady=(0, 12))
        self.start_button = FlatButton(toolbar, "Start", self.toggle, kind="primary", padx=30, pady=11)
        self.start_button.pack(side="left")
        self.demo_button = FlatButton(toolbar, "Try a demo", self.run_demo, kind="secondary")
        self.demo_button.pack(side="left", padx=(10, 0))
        self.clear_button = FlatButton(toolbar, "Clear", self.clear_captions, kind="ghost", padx=12)
        self.clear_button.pack(side="left", padx=(10, 0))
        self.clear_button.set_enabled(False)
        self.toolbar_hint = tk.Label(toolbar, text="", bg=BG, fg=DIM, font=F_SMALL)
        self.toolbar_hint.pack(side="right")

        self.notice = tk.Frame(page, bg="#221a0c", highlightbackground="#443410", highlightthickness=1)
        self.notice_label = tk.Label(self.notice, text="", bg="#221a0c", fg=WARN, font=F_BODY,
                                     justify="left", anchor="w", padx=16, pady=12)
        self.notice_label.pack(side="left")
        self.notice_action = FlatButton(self.notice, "Download models", self.download_models, kind="primary",
                                        padx=14, pady=7)
        self.notice_action.pack(side="right", padx=12, pady=8)

        board = tk.Frame(page, bg=BG)
        board.pack(fill="both", expand=True)
        board.grid_columnconfigure(0, weight=1, uniform="col")
        board.grid_columnconfigure(1, weight=1, uniform="col")
        board.grid_rowconfigure(1, weight=1)

        panels = []
        for column, (title, colour, font) in enumerate((
                (f"YOU  ·  {language_name(self.settings.source).upper()}", MUTED, F_SRC),
                (f"THEY HEAR  ·  {language_name(self.settings.target).upper()}", "#7fb0ff", F_TGT))):
            head = tk.Label(board, text=title, bg=BG, fg=colour, font=(UI, 8, "bold"), anchor="w")
            head.grid(row=0, column=column, sticky="ew", padx=(0, 6) if column == 0 else (6, 0), pady=(0, 8))
            holder = tk.Frame(board, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
            holder.grid(row=1, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0))
            text = tk.Text(holder, bg=PANEL, fg=TEXT, font=font, wrap="word", bd=0, padx=20, pady=18,
                           insertwidth=0, highlightthickness=0, spacing1=3, spacing3=12, cursor="arrow")
            scroll = ttk.Scrollbar(holder, orient="vertical", command=text.yview,
                                   style="Parley.Vertical.TScrollbar")
            text.configure(yscrollcommand=scroll.set)
            text.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            text.tag_configure("meta", foreground="#5a6b83", font=F_TINY)
            text.tag_configure("rejected", foreground=WARN, font=(UI, 9, "italic"))
            text.tag_configure("placeholder", foreground="#556781", font=(UI, 12), spacing3=10)
            text.configure(state="disabled")
            panels.append(text)

        self.src_text, self.tgt_text = panels
        self.live_headers = [board.grid_slaves(row=0, column=0)[0], board.grid_slaves(row=0, column=1)[0]]

    # -- setup page ----------------------------------------------------------
    def _build_setup(self) -> None:
        page = tk.Frame(self.content, bg=BG)
        self.pages["setup"] = page
        page.grid_columnconfigure(0, weight=1, uniform="c")
        page.grid_columnconfigure(1, weight=1, uniform="c")
        page.grid_rowconfigure(2, weight=1)

        languages = Card(page, "Languages")
        languages.grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=(0, 14))
        row = tk.Frame(languages, bg=PANEL)
        row.pack(fill="x", padx=20, pady=(12, 20))
        block = tk.Frame(row, bg=PANEL)
        block.pack(side="left", fill="x", expand=True)
        tk.Label(block, text="I speak", bg=PANEL, fg=MUTED, font=F_SMALL).pack(anchor="w")
        self.src_combo = combo(block, lang_choices(), 24)
        self.src_combo.pack(fill="x", pady=(5, 12))
        tk.Label(block, text="They hear", bg=PANEL, fg=MUTED, font=F_SMALL).pack(anchor="w")
        self.tgt_combo = combo(block, lang_choices(), 24)
        self.tgt_combo.pack(fill="x", pady=(5, 0))
        self.src_combo.set(lang_label(self.settings.source))
        self.tgt_combo.set(lang_label(self.settings.target))
        FlatButton(row, "⇄", self._swap_languages, kind="secondary", padx=14, pady=9).pack(
            side="left", padx=(12, 0), pady=(20, 0))

        audio = Card(page, "Audio")
        audio.grid(row=1, column=0, sticky="ew", padx=(0, 10))
        body = tk.Frame(audio, bg=PANEL)
        body.pack(fill="x", padx=20, pady=(12, 20))
        inputs = ["Default microphone"] + [f"{d.index} · {d.name}" for d in list_devices() if d.max_input]
        outputs = ["Default output (speakers / headphones)"] + [f"{d.index} · {d.name}"
                                                               for d in list_devices() if d.max_output]
        tk.Label(body, text="Microphone", bg=PANEL, fg=MUTED, font=F_SMALL).pack(anchor="w")
        self.in_combo = combo(body, inputs, 40)
        self.in_combo.set(inputs[0])
        self.in_combo.pack(fill="x", pady=(5, 12))
        tk.Label(body, text="Play the translation to", bg=PANEL, fg=MUTED, font=F_SMALL).pack(anchor="w")
        self.out_combo = combo(body, outputs, 40)
        self.out_combo.set(outputs[0])
        self.out_combo.pack(fill="x", pady=(5, 12))
        if self.settings.input_device:
            match = next((v for v in inputs if v.startswith(f"{self.settings.input_device} · ")), None)
            if match:
                self.in_combo.set(match)
        if self.settings.output_device:
            match = next((v for v in outputs if v.startswith(f"{self.settings.output_device} · ")), None)
            if match:
                self.out_combo.set(match)

        self.speak_var = tk.BooleanVar(value=self.settings.speak)
        self._check(body, "Speak the translation out loud", self.speak_var)
        self.duplex_var = tk.BooleanVar(value=self.settings.duplex)
        self._check(body, "Two-way: also translate the meeting audio coming back", self.duplex_var,
                    hint="needs Stereo Mix or VB-Cable — see Help")

        quality = Card(page, "Quality")
        quality.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=(0, 14))
        qbody = tk.Frame(quality, bg=PANEL)
        qbody.pack(fill="x", padx=20, pady=(12, 20))
        self.profile_var = tk.StringVar(value=self.settings.profile)
        hints = {"fast": "fastest · 1.0 GB · small quality cost",
                 "balanced": "default · 1.8 GB · 1.2 s behind you",
                 "quality": "best translation · 3.0 GB · 2.0 s behind you"}
        for name in ("fast", "balanced", "quality"):
            line = tk.Frame(qbody, bg=PANEL)
            line.pack(fill="x", pady=4)
            tk.Radiobutton(line, text=name.capitalize(), variable=self.profile_var, value=name, bg=PANEL,
                           fg=TEXT, activebackground=PANEL, activeforeground=TEXT, selectcolor=PANEL_2,
                           font=F_BODY, highlightthickness=0, bd=0).pack(side="left")
            tk.Label(line, text=hints[name], bg=PANEL, fg=DIM, font=F_SMALL).pack(side="left", padx=12)

        glossary = Card(page, "Glossary — optional")
        glossary.grid(row=1, column=1, sticky="ew", padx=(10, 0))
        gbody = tk.Frame(glossary, bg=PANEL)
        gbody.pack(fill="x", padx=20, pady=(12, 20))
        tk.Label(gbody, text="Names and terms the translator must get right.", bg=PANEL, fg=MUTED,
                 font=F_SMALL).pack(anchor="w")
        row2 = tk.Frame(gbody, bg=PANEL)
        row2.pack(fill="x", pady=(8, 0))
        self.glossary_label = tk.Label(row2, text=self._glossary_display(), bg=PANEL_2, fg=TEXT,
                                       font=F_SMALL, anchor="w", padx=12, pady=8,
                                       highlightthickness=1, highlightbackground=LINE)
        self.glossary_label.pack(side="left", fill="x", expand=True)
        FlatButton(row2, "Choose…", self._choose_glossary, kind="secondary", padx=14, pady=7).pack(
            side="left", padx=(8, 0))

        footer = tk.Frame(page, bg=BG)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        FlatButton(footer, "Save settings", self._save_settings, kind="primary", padx=22, pady=10).pack(side="left")
        self.saved_label = tk.Label(footer, text="Saved automatically when you press Start.", bg=BG, fg=DIM,
                                    font=F_SMALL)
        self.saved_label.pack(side="left", padx=14)

    def _check(self, master, text: str, var: tk.BooleanVar, hint: str = "") -> None:
        line = tk.Frame(master, bg=PANEL)
        line.pack(fill="x", pady=(0, 2))
        tk.Checkbutton(line, text=text, variable=var, bg=PANEL, fg=TEXT, activebackground=PANEL,
                       activeforeground=TEXT, selectcolor=PANEL_2, font=F_BODY, highlightthickness=0,
                       bd=0).pack(anchor="w")
        if hint:
            tk.Label(line, text=hint, bg=PANEL, fg=DIM, font=F_TINY).pack(anchor="w", padx=28)

    # -- help page -----------------------------------------------------------
    def _build_help(self) -> None:
        page = tk.Frame(self.content, bg=BG)
        self.pages["help"] = page
        card = Card(page, "Using it in a real meeting")
        card.pack(fill="both", expand=True)
        text = tk.Text(card, bg=PANEL, fg=TEXT, font=F_BODY, wrap="word", bd=0, padx=22, pady=18,
                       highlightthickness=0, spacing1=3, spacing3=9)
        text.pack(fill="both", expand=True)
        text.tag_configure("h", font=(UI, 11, "bold"), foreground=TEXT, spacing1=16)
        text.tag_configure("m", foreground=DIM, font=F_SMALL)
        text.insert("end", "One-way — you speak, the room hears\n", "h")
        text.insert("end", "1.   Install VB-Cable once: vb-audio.com/Cable. It adds a playback device "
                           "called CABLE Input.\n"
                           "2.   In Setup, choose your microphone, and set “Play the translation to” "
                           "to CABLE Input.\n"
                           "3.   In Zoom / Teams / Meet, choose CABLE Output as your microphone.\n"
                           "4.   Press Start and talk normally. They hear German; you see both "
                           "languages.\n\n")
        text.insert("end", "Two-way — you also hear them in your language\n", "h")
        text.insert("end", "Tick “Two-way” in Setup and pick Stereo Mix (or VB-Cable) as the meeting input. "
                           "Your headphones get the English voice, the room gets German.\n"
                           "Both directions share one recogniser and one translator, so this costs the "
                           "same VRAM as one direction.\n\n")
        text.insert("end", "The numbers at the bottom\n", "h")
        text.insert("end", "lag p50 — how long after you stop talking the translation starts playing.\n"
                           "speech / translate — time spent recognising and translating one utterance.\n"
                           "vram — what parley itself has resident. It never exceeds the ceiling; if a "
                           "profile cannot fit, it says so instead of crashing.\n\n")
        text.insert("end", "If something is red\n", "h")
        text.insert("end", "Missing models → the Live page offers a download (about 3 GB, once).\n"
                           "“language gate” lines mean the room was speaking outside your pair — parley "
                           "refuses rather than translating the wrong language into the meeting.\n"
                           "Shortcuts: F5 start/stop · Esc stop · Ctrl+D demo.\n\n")
        text.insert("end", f"Models: {MODELS_DIR}\nSettings: {SETTINGS_PATH}\n"
                           f"GPU: {gpu_name() or 'not detected'}\n"
                           f"Source: github.com/coding-parsa/parley\n", "m")
        text.configure(state="disabled")

    # -- page switching ------------------------------------------------------
    def _show_page(self, key: str) -> None:
        self.page = key
        for name, page in self.pages.items():
            page.pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        for name, button in self.nav_buttons.items():
            active = name == key
            button.configure(fg=TEXT if active else MUTED, font=(UI, 10, "bold" if active else "normal"))
            self.nav_underlines[name].configure(bg=ACCENT if active else BG)

    # -- settings ------------------------------------------------------------
    def _glossary_display(self) -> str:
        return Path(self.settings.glossary).name if self.settings.glossary else "none"

    def _choose_glossary(self) -> None:
        path = filedialog.askopenfilename(title="Choose a glossary",
                                          filetypes=[("Glossary", "*.tsv *.txt"), ("All files", "*.*")])
        if path:
            self.settings.glossary = path
            self.glossary_label.configure(text=self._glossary_display())
            self.settings.save()

    def _swap_languages(self) -> None:
        source, target = self.src_combo.get(), self.tgt_combo.get()
        self.src_combo.set(target)
        self.tgt_combo.set(source)

    def _save_settings(self) -> None:
        self._collect_settings()
        self.settings.save()
        self.saved_label.configure(text="Saved.", fg=OK)
        self._refresh_ready_state()
        self.after(1600, lambda: self.saved_label.configure(text="Saved automatically when you press Start.",
                                                            fg=DIM))

    def _collect_settings(self) -> None:
        self.settings.source = lang_from_label(self.src_combo.get())
        self.settings.target = lang_from_label(self.tgt_combo.get())
        self.settings.profile = self.profile_var.get()
        in_text, out_text = self.in_combo.get(), self.out_combo.get()
        self.settings.input_device = in_text.split(" · ")[0] if " · " in in_text else ""
        self.settings.output_device = out_text.split(" · ")[0] if " · " in out_text else ""
        self.settings.speak = bool(self.speak_var.get())
        self.settings.duplex = bool(self.duplex_var.get())

    def build_config(self) -> Config:
        self._collect_settings()
        return Config(source=self.settings.source, target=self.settings.target,
                      profile=self.settings.profile, speak=self.settings.speak,
                      glossary_path=self.settings.glossary or None,
                      input_device=self.settings.input_device or None,
                      output_device=self.settings.output_device or None)

    def _missing(self) -> list[str]:
        cfg = self.build_config()
        missing = []
        if not (cfg.stt_path() / "model.bin").exists():
            missing.append("speech recognition")
        if not (cfg.mt_path() / "model.bin").exists():
            missing.append("translator")
        if cfg.speak and not cfg.voice_path(cfg.target).exists():
            missing.append(f"{language_name(cfg.target)} voice")
        return missing

    # -- ready state ---------------------------------------------------------
    def _refresh_ready_state(self) -> None:
        missing = self._missing()
        free = gpu_free_mb()
        self.stat_labels["vram"].configure(text=f"{free} MB free" if free else "GPU not detected")
        self.stat_labels["models"].configure(text="missing" if missing else f"{self.settings.profile} · ok",
                                             fg=WARN if missing else MUTED)
        for header, language in zip(self.live_headers,
                                    (language_name(self.settings.source), language_name(self.settings.target))):
            prefix = "YOU  ·  " if header.cget("text").startswith("YOU") else "THEY HEAR  ·  "
            header.configure(text=f"{prefix}{language.upper()}")

        if missing:
            self._set_state("setup needed", WARN)
            self._show_notice(f"{' and '.join(missing).capitalize()} not downloaded — "
                              f"one download of about 3 GB, once.")
            self.start_button.set_enabled(False)
        else:
            self._set_state("ready", OK)
            self._hide_notice()
            self.start_button.set_enabled(True)
        self._render_placeholder()

    def _show_notice(self, text: str, action: bool = True) -> None:
        self.notice_label.configure(text=text)
        if action:
            self.notice_action.pack(side="right", padx=12, pady=8)
        else:
            self.notice_action.pack_forget()
        if not self.notice.winfo_ismapped():
            self.notice.pack(fill="x", pady=(0, 12), after=self.pages["live"].winfo_children()[0])

    def _hide_notice(self) -> None:
        self.notice.pack_forget()

    def _set_state(self, text: str, colour: str) -> None:
        self.state_pill.configure(text=f"●  {text}", fg=colour)

    # -- captions ------------------------------------------------------------
    def _render_placeholder(self) -> None:
        if not self._placeholder:
            return
        for widget, message in ((self.src_text, "Press Start and speak normally.\n"
                                               "Your words appear here as you talk."),
                                (self.tgt_text, "The translation appears here —\n"
                                                "and is spoken out loud.")):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("end", message + "\n", "placeholder")
            widget.configure(state="disabled")

    def _clear_placeholder(self) -> None:
        if not self._placeholder:
            return
        self._placeholder = False
        for widget in (self.src_text, self.tgt_text):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.configure(state="disabled")
        self.clear_button.set_enabled(True)

    def _append(self, widget: tk.Text, text: str, tag: str | None = None) -> None:
        widget.configure(state="normal")
        if tag:
            widget.insert("end", text, tag)
        else:
            widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def add_segment(self, event: dict) -> None:
        self._clear_placeholder()
        stamp = time.strftime("%H:%M:%S")
        if event.get("dropped"):
            self._append(self.src_text, f"{event['dropped']}\n", "rejected")
            return
        if event.get("source_text"):
            self._append(self.src_text, event["source_text"] + "\n")
            self._append(self.src_text, f"{stamp}  ·  {event.get('audio_s', 0):.1f}s speech  ·  "
                                        f"recognised in {event.get('stt_s', 0):.2f}s\n", "meta")
        if event.get("translated_text"):
            self._append(self.tgt_text, event["translated_text"] + "\n")
            self._append(self.tgt_text, f"lag {(event.get('lag_s') or 0):.2f}s  ·  "
                                        f"translated in {event.get('mt_s', 0):.2f}s\n", "meta")

    def clear_captions(self) -> None:
        self._placeholder = True
        self._render_placeholder()
        self.clear_button.set_enabled(False)

    # -- running -------------------------------------------------------------
    def toggle(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if self.running or self._missing():
            return
        self._collect_settings()
        self.settings.save()
        self._clear_placeholder()
        self._hide_notice()
        self.running = True
        self.start_button.set_text("Stop")
        self.start_button.set_kind("danger")
        self._set_state("starting…", WARN)
        self.toolbar_hint.configure(text="loading models — a few seconds")
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run_pipeline, daemon=True)
        self.worker.start()

    def _run_pipeline(self) -> None:
        cfg = self.build_config()
        loopback = self.settings.use_system_audio
        try:
            self.events.put({"type": "status", "text": "loading models…"})
            interpreter = Interpreter(cfg, on_event=self.events.put)
            self.interpreter = interpreter
            interpreter.warm()
            source = open_source(cfg.input_device, loopback=loopback, block_ms=cfg.block_ms)
            speaker = Speaker(cfg.output_device) if cfg.speak else None
            with source:
                if speaker is not None:
                    speaker.open()
                try:
                    interpreter.run_live(source, speaker, self.stop_event, partials=cfg.speak)
                finally:
                    if speaker is not None:
                        speaker.close()
        except SystemExit as exc:
            self.events.put({"type": "error", "message": str(exc)})
        except Exception as exc:                       # noqa: BLE001 - surface, never crash the window
            self.events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.events.put({"type": "stopped"})

    def stop(self) -> None:
        if self.running:
            self.stop_event.set()
            self._set_state("stopping…", WARN)

    def _stopped(self) -> None:
        self.running = False
        self.start_button.set_text("Start")
        self.start_button.set_kind("primary")
        self.toolbar_hint.configure(text="")
        if self.interpreter is not None:
            self._apply_stats(self.interpreter.stats())
        self._set_state("ready", OK)

    # -- demo ----------------------------------------------------------------
    def run_demo(self) -> None:
        if self.running or self._missing():
            return
        self._clear_placeholder()
        self._set_state("demo…", WARN)
        self.demo_button.set_text("Demo running…")
        self.demo_button.set_enabled(False)
        threading.Thread(target=self._demo_worker, daemon=True).start()

    def _demo_worker(self) -> None:
        cfg = self.build_config()
        cfg.speak = True
        out_dir = Path(os.environ.get("TEMP", ".")) / "parley-demo"
        try:
            audio_path = self._demo_audio()
            interpreter = Interpreter(cfg, on_event=self.events.put)
            interpreter.warm()
            results = interpreter.translate_file(audio_path, out_dir=out_dir)
            spoken = [r.audio_path for r in results if r.audio_path]
            if spoken:
                self.events.put({"type": "demo_audio", "path": spoken})
            self.events.put({"type": "demo_stats", "stats": interpreter.stats()})
        except SystemExit as exc:
            self.events.put({"type": "error", "message": str(exc)})
        except Exception as exc:                       # noqa: BLE001
            self.events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.events.put({"type": "demo_done"})

    def _demo_audio(self) -> str:
        """Synthesise the demo clip locally, so the app ships no audio assets."""
        from .audio import write_wav
        from .tts import PiperTts

        voices = {"en": "en_US-hfc_male-medium", "de": "de_DE-thorsten-high"}
        voice = voices.get(self.settings.source, voices["en"])
        tts = PiperTts(MODELS_DIR / "piper" / f"{voice}.onnx")
        tts.load()
        sentence = {
            "en": "Thanks everyone for joining. Today we need to review the deployment schedule "
                  "and the budget for the next quarter.",
            "de": "Die Besprechung findet morgen um zehn Uhr statt. Bitte schicken Sie mir den "
                  "Finanzbericht vorher.",
        }.get(self.settings.source, "This is a short demo sentence for the interpreter.")
        audio, rate = tts.synthesize(sentence)
        out = Path(os.environ.get("TEMP", ".")) / "parley-demo" / "demo_source.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_wav(out, audio, rate)
        return str(out)

    # -- downloads -----------------------------------------------------------
    def download_models(self) -> None:
        self._show_notice("Downloading models — about 3 GB. You can leave this window open.", action=False)
        self._set_state("downloading…", WARN)
        self.start_button.set_enabled(False)
        threading.Thread(target=self._download_worker, daemon=True).start()

    def _download_worker(self) -> None:
        from .fetch import fetch_all

        def progress(message: str, fraction: float) -> None:
            self.events.put({"type": "download", "message": message, "fraction": fraction})

        try:
            fetch_all(self.settings.source, self.settings.target, self.settings.profile, MODELS_DIR, progress)
            self.events.put({"type": "download_done", "ok": True})
        except Exception as exc:                       # noqa: BLE001
            self.events.put({"type": "download_done", "ok": False, "message": f"{type(exc).__name__}: {exc}"})

    # -- event pump ----------------------------------------------------------
    def _apply_stats(self, stats: dict) -> None:
        def fmt(value: float) -> str:
            return f"{value:.2f}s" if value else "–"

        self.stat_labels["lag"].configure(text=fmt(stats.get("lag_p50", 0)))
        self.stat_labels["stt"].configure(text=fmt(stats.get("stt_p50", 0)))
        self.stat_labels["mt"].configure(text=fmt(stats.get("mt_p50", 0)))
        vram = stats.get("vram") or {}
        if vram.get("resident_mb"):
            self.stat_labels["vram"].configure(text=f"{vram['resident_mb']} MB")
        if stats.get("dropped"):
            self.hint_label.configure(text=f"{stats['dropped']} utterance(s) refused by the language gate")

    def _pump(self) -> None:
        try:
            while True:
                self._handle_event(self.events.get_nowait())
        except queue.Empty:
            pass
        self.after(90, self._pump)

    def _handle_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "model":
            self.toolbar_hint.configure(text=f"{event['name']} · {event.get('vram_mb', 0)} MB")
        elif kind == "ready":
            self._set_state("listening", OK)
            self.toolbar_hint.configure(text="")
            vram = event.get("vram") or {}
            if vram.get("resident_mb"):
                self.stat_labels["vram"].configure(text=f"{vram['resident_mb']} MB")
        elif kind == "listening":
            self._set_state("listening", OK)
        elif kind == "fallback":
            self._show_notice(f"{event.get('from')} did not fit in the card's free memory — "
                              f"running {event.get('to')} instead.", action=False)
        elif kind == "log":
            self.toolbar_hint.configure(text=event.get("message", "")[:120])
        elif kind == "segment":
            self.add_segment(event)
        elif kind == "partial_commit":
            self._clear_placeholder()
            self._append(self.src_text, event.get("source", "") + "\n")
            self._append(self.tgt_text, event.get("translated", "") + "\n", "meta")
        elif kind == "rejected":
            self._clear_placeholder()
            self._append(self.src_text, f"{event.get('reason')}\n", "rejected")
        elif kind == "error":
            self._set_state("error", ERR)
            self._show_notice(event.get("message", "Something went wrong."), action=False)
            self.running = False
            self.start_button.set_text("Start")
            self.start_button.set_kind("primary")
            self.start_button.set_enabled(True)
        elif kind == "stopped":
            self._stopped()
        elif kind == "demo_audio":
            self._play(list(event.get("path") or []))
        elif kind == "demo_stats":
            self._apply_stats(event.get("stats") or {})
        elif kind == "demo_done":
            self.demo_button.set_text("Try a demo")
            self.demo_button.set_enabled(True)
            if not self.running:
                self._set_state("ready", OK)
        elif kind == "download":
            self._show_notice(f"{event.get('message')}  ·  {event.get('fraction', 0) * 100:.0f}%",
                              action=False)
        elif kind == "download_done":
            if event.get("ok"):
                self._refresh_ready_state()
            else:
                self._show_notice(f"Download failed: {event.get('message', '')}")

    def _play(self, paths: list[str]) -> None:
        """Play the demo audio with the stdlib — no extra audio dependency for this."""
        if not paths:
            return

        def run() -> None:
            try:
                import winsound

                for path in paths:
                    winsound.PlaySound(path, winsound.SND_FILENAME)
            except Exception:                      # noqa: BLE001
                pass

        threading.Thread(target=run, daemon=True).start()

    # -- lifecycle -----------------------------------------------------------
    def _on_close(self) -> None:
        try:
            self.stop_event.set()
            self._collect_settings()
            self.settings.save()
        finally:
            self.destroy()


def _crash_log_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home()))
    return base / "parley" / "crash.log"


def _write_crash(exc: BaseException) -> str:
    """A windowed .exe has nowhere to print, so crashes go to a file the user can send."""
    import traceback

    path = _crash_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(time.strftime("\n===== %Y-%m-%d %H:%M:%S =====\n"))
            handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:                                  # noqa: BLE001
        pass
    return str(path)


def main() -> int:
    try:
        app = ParleyApp()

        def report(exc_type, exc, tb):                 # Tk callback errors
            _write_crash(exc if isinstance(exc, BaseException) else RuntimeError(str(exc)))

        app.report_callback_exception = report
        app.mainloop()
        return 0
    except Exception as exc:                           # noqa: BLE001
        log = _write_crash(exc)
        try:
            import tkinter.messagebox as box

            box.showerror("parley could not start",
                          f"{type(exc).__name__}: {exc}\n\nDetails written to:\n{log}")
        except Exception:                              # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())

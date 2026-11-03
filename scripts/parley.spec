# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the parley desktop app.

    pyinstaller --noconfirm scripts/parley.spec

Builds a one-folder app (dist/parley/parley.exe): one-folder starts fast and keeps
the native CUDA/ONNX libraries next to the exe, where they can be inspected when
something goes wrong. Models are never bundled — the app downloads them, or uses
%LOCALAPPDATA%\\parley\\models (PARLEY_MODELS overrides).
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent      # noqa: F821 - SPECPATH is injected by PyInstaller

datas, binaries, hiddenimports = [], [], []
for package in ("ctranslate2", "faster_whisper", "piper", "espeakng_loader", "onnxruntime",
                "sounddevice", "sentencepiece", "huggingface_hub", "soundfile", "tkinter"):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
    except Exception:                       # noqa: BLE001 - a missing optional package must not stop the build
        continue
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# the caption console's page is small and lets `parley serve` work from the frozen build too
datas += [(str(ROOT / "parley" / "web"), "parley/web")]

a = Analysis(                                    # noqa: F821
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "transformers", "matplotlib", "pandas", "PyQt5", "PySide6", "PySide2",
              "IPython", "notebook", "tensorflow"],
    noarchive=False,
)
pyz = PYZ(a.pure)                                # noqa: F821

exe = EXE(                                       # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="parley",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(                                  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="parley",
)

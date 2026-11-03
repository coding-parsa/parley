# Contributing

Small project, sharp constraints. The two hard rules:

1. **Nothing may push the resident set past the VRAM ceiling.** If your change adds a
   model, it goes through `Budget.admit()` with an estimate, and the estimate is
   corrected by `Budget.measure()` after loading.
2. **Numbers over adjectives.** A PR that claims "faster" or "better" needs a row in
   `parley bench` output or a measurement note in `docs/`.

## Where help is worth the most

- A real meeting corpus (with consent) for `parley bench` — the current test audio is
  synthetic, which flatters every model.
- German-specific post-editing: verb-final handling, `Sie`/`du`, compound splitting.
- A loopback path that does not need Stereo Mix on Windows (PortAudio WASAPI loopback,
  or a pipewire virtual source on Linux).
- Alternative MT backends behind `NllbMt`'s interface, measured on the same audio.

## Dev loop

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e . pytest
pytest tests/test_units.py -q          # no GPU, no downloads, ~1 s
parley file tests/audio/en_meeting_01.wav --from en --to de
```

Keep the unit tests model-free: they are the CI signal, and CI has no GPU.

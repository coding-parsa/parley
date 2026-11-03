@echo off
rem parley launcher — runs the CLI from this checkout's own virtualenv.
rem   parley.cmd doctor
rem   parley.cmd file tests\audio\en_meeting_01.wav --from en --to de --out out
rem   parley.cmd run --from en --to de
setlocal
set HERE=%~dp0
if not exist "%HERE%.venv\Scripts\python.exe" (
  echo no .venv here. Create it once:
  echo   python -m venv .venv ^&^& .venv\Scripts\pip install -e .
  exit /b 1
)
set PYTHONPATH=%HERE%
"%HERE%.venv\Scripts\python.exe" -m parley.cli %*

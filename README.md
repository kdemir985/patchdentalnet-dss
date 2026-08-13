# PatchDentalNet DSS (English Desktop App)

This is the English version of the PatchDentalNet Decision Support System,
packaged to launch like a regular desktop app from an icon instead of a
browser tab.

## How to launch

Double-click the **"PatchDentalNet DSS"** icon on the Desktop. A loading
screen appears immediately; the full app window replaces it once the AI
models finish loading (up to ~30 seconds on first launch).

If something goes wrong and you want to see the error output, run
`START_DEBUG.bat` instead (opens a console window with logs).

## How this relates to the Turkish app (`kds/`)

This folder only contains translated frontend/backend **code**. It does
**not** duplicate the trained models or the Python virtual environment —
both are shared with the Turkish `kds/` app:

- Models: `predictor.py` points `MODELS_DIR` at `../../kds/backend/models`.
- Python venv: the desktop shortcut and `START_DEBUG.bat` both call
  `kds/backend/.venv/Scripts/python(w).exe` directly.

This means: if you retrain or replace a model file under `kds/backend/models/`,
both apps pick it up automatically. If you `pip install` something new, do
it once in `kds/backend/.venv` and both apps get it.

The English app runs on **port 8001** (the Turkish app uses 8000), so both
can run at the same time without conflicting.

## Files

- `icon.ico` — the desktop shortcut's icon (generated to match the in-app logo)
- `backend/app_window.py` — desktop app launcher (runs the API server in a
  background thread + opens it in a native window via `pywebview`, no
  browser address bar)
- `backend/main.py` — same FastAPI app as the Turkish version, English titles, port 8001
- `backend/predictor.py`, `svm_pipeline.py`, `cnn_arch.py` — English-translated
  copies of the Turkish backend logic
- `frontend/` — English-translated copy of the Turkish frontend (same structure)

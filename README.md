# PatchDentalNet DSS

A clinical decision support desktop application for PatchDentalNet, packaged
to launch like a regular desktop app from an icon instead of a browser tab.

## How to launch

Double-click the **"PatchDentalNet DSS"** icon on the Desktop. A loading
screen appears immediately; the full app window replaces it once the AI
models finish loading (up to ~30 seconds on first launch).

If something goes wrong and you want to see the error output, run
`START_DEBUG.bat` instead (opens a console window with logs).

## Project layout and requirements

This repository contains only the application **code**. It expects a sibling
`kds/backend/` directory (one level above this project's parent folder) that
provides:

- `kds/backend/models/` — the trained model files (`seed_42/`, `seed_123/`,
  `seed_456/`, each containing `patchdentalnet.pth`) and the cached
  `svm_pipeline.joblib` + `features/` directory. Not included in this
  repository due to size (~1GB); available on request.
- `kds/backend/.venv/` — the Python virtual environment used to run the app
  (see `backend/requirements.txt` for the package list).

If you're setting this up from scratch, create `kds/backend/.venv`, install
the requirements there, and place the trained models under
`kds/backend/models/` before launching.

The app runs on **port 8001** by default (see `backend/main.py`), so it can
run alongside another instance on port 8000 without conflicting.

## Files

- `icon.ico` — the desktop shortcut's icon (generated to match the in-app logo)
- `backend/app_window.py` — desktop app launcher (runs the API server in a
  background thread + opens it in a native window via `pywebview`, no
  browser address bar)
- `backend/main.py` — FastAPI app, serves the frontend and the prediction/SHAP endpoints
- `backend/predictor.py`, `svm_pipeline.py`, `cnn_arch.py` — inference pipeline
  (preprocessing, CNN feature extraction, SVM ensemble, SHAP explainability)
- `frontend/` — HTML/CSS/JS interface (no build step)

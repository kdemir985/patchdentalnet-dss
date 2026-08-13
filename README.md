# PatchDentalNet DSS

A clinical decision support interface for **PatchDentalNet**, a hybrid
VGG19 + ResNet152 + PatchAttention deep learning model for lesion
classification (No Lesion / Radiolucent / Radiopaque / Mixed) on dental
panoramic radiographs. The interface exposes the trained model's
predictions, confidence, triage level, and SHAP-based explainability
through a lightweight FastAPI backend and a dependency-free HTML/CSS/JS
frontend.

## Demo

<video src="https://github.com/kdemir985/patchdentalnet-dss/raw/master/assets/demo.mp4" controls width="720"></video>

## Features

- **Case intake** — case ID, patient age/gender, chief complaint, and
  clinical notes alongside the radiograph upload
- **Preprocessing visualization** — CLAHE + Unsharp Masking output and the
  2×8 anatomical patch grid used by the model, shown before classification
- **Prediction panel** — predicted class, confidence score, triage level,
  class probability distribution, and structured clinical findings
- **Model focus map** — attention-weighted patch grid highlighting the
  regions that most influenced the prediction
- **SHAP explainability report** — per-class feature-group attribution
  (attention context, patch statistics, softmax scores) computed live for
  the submitted case
- **Report export** — downloadable HTML and JSON case reports

## Architecture

- **Backend** — FastAPI (`backend/main.py`), serving both the REST API
  (`/api/predict`, `/api/shap-report`, `/api/health`) and the static
  frontend from a single process
- **Inference pipeline** — `cnn_arch.py` (CNN feature extraction) +
  `svm_pipeline.py` (PCA + BorderlineSMOTE + SVM ensemble classifier, with
  live SHAP explanation) — see `predictor.py` for the orchestration
- **Frontend** — plain HTML/CSS/JavaScript, no build step or framework
  dependency

## Requirements

- Python 3.10+, plus the packages listed in `backend/requirements.txt`
- PyTorch/torchvision (CUDA build recommended), installed separately
- Trained model weights (not included in this repository, see below)

## Setup

```bash
cd backend
pip install -r requirements.txt
# install torch/torchvision separately for your platform/CUDA version
python main.py
```

Then open `http://localhost:8001` in a browser.

## Model weights

Trained model weights are not included in this repository due to their
size (~1GB) and are not required to review the source code. By default,
`predictor.py` resolves `MODELS_DIR` two directories up from `backend/`
(see the `SHARED_BACKEND_DIR` constant) and expects: three CNN checkpoints
(`seed_42/`, `seed_123/`, `seed_456/`, each containing
`patchdentalnet.pth`), the cached SVM ensemble (`svm_pipeline.joblib`),
and the `features/` directory used to rebuild the SHAP background set.
Adjust `MODELS_DIR` to point at wherever you place these files. Weights
are available on request.

## Project layout

```
backend/
  main.py           FastAPI application and API routes
  predictor.py       Inference orchestration (preprocessing -> CNN -> SVM -> SHAP)
  cnn_arch.py         CNN architecture and feature extraction
  svm_pipeline.py     SVM ensemble classifier and SHAP explainability
  requirements.txt
frontend/
  index.html
  css/style.css
  js/app.js
```

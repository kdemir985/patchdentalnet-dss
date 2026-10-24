"""
Dental DSS - Backend API
=========================
Run:
    cd backend
    pip install -r requirements.txt
    python main.py

Then open in a browser:  http://localhost:8001
(The frontend is served from this same backend, no separate server needed.)

Runs on port 8001 by default so it can coexist with another instance of
this app running on port 8000.
"""

import io
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from predictor import INVALID_IMAGE_MESSAGE, DentalPredictor, InvalidImageError

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="PatchDentalNet DSS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = DentalPredictor()


def _load_image(contents: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail=INVALID_IMAGE_MESSAGE)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_mode": predictor.mode,
        "model_version": predictor.model_version,
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    case_id: str = Form(""),
    age: str = Form(""),
    gender: str = Form(""),
    complaint: str = Form(""),
    notes: str = Form(""),
):
    img = _load_image(await image.read())

    case_info = {
        "case_id": case_id,
        "age": age,
        "gender": gender,
        "complaint": complaint,
        "notes": notes,
    }

    try:
        return predictor.predict(img, case_info)
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/shap-report")
async def shap_report(image: UploadFile = File(...)):
    img = _load_image(await image.read())
    try:
        return predictor.build_shap_report(img)
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))


# Serve frontend static files at the root path (must be mounted last)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    # reload=False: the CNN models load ~900MB at every startup, so auto-reload
    # is both slow and unreliable at detecting file changes in the OneDrive
    # sync folder. After changing code: Ctrl+C, then re-run "python main.py".
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)

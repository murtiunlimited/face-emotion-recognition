# api/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import time
from pydantic import BaseModel
from src.llm.groq_client import explain_emotion

from src.inference.predict import predict_emotion

# =========================
# FastAPI app
# =========================
app = FastAPI(title="FER Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later if needed
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================
# Health check
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read file
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # =limit file size (5MB)
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        # Measure latency (nice touch, no logging needed)
        start = time.time()

        label = predict_emotion(img)

        duration = time.time() - start

        return {
            "emotion": label,
            "filename": file.filename,
            "latency_ms": round(duration * 1000, 2)
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")

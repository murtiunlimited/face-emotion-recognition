# tests/test_api.py
import io
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


# =========================
# Helpers
# =========================
def create_test_image():
    """Create a simple valid grayscale image"""
    img = np.zeros((48, 48), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer.tobytes())


# =========================
# Health endpoint
# =========================
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# =========================
# Predict - success
# =========================
def test_predict_success(monkeypatch):
    def mock_predict(_):
        return "happy"

    # Mock the model
    monkeypatch.setattr("api.app.predict_emotion", mock_predict)

    img_file = create_test_image()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_file, "image/jpeg")}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["emotion"] == "happy"
    assert data["filename"] == "test.jpg"
    assert "latency_ms" in data


# =========================
# Invalid file type
# =========================
def test_predict_invalid_file_type():
    file = io.BytesIO(b"not an image")

    response = client.post(
        "/predict",
        files={"file": ("test.txt", file, "text/plain")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be an image"


# =========================
# Empty file
# =========================
def test_predict_empty_file():
    file = io.BytesIO(b"")

    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", file, "image/jpeg")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Empty file"


# =========================
# File too large
# =========================
def test_predict_file_too_large(monkeypatch):
    def mock_predict(_):
        return "happy"

    monkeypatch.setattr("api.app.predict_emotion", mock_predict)

    big_file = io.BytesIO(b"a" * (5 * 1024 * 1024 + 1))

    response = client.post(
        "/predict",
        files={"file": ("big.jpg", big_file, "image/jpeg")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File too large"


# =========================
# Invalid image data
# =========================
def test_predict_invalid_image(monkeypatch):
    def mock_predict(_):
        return "happy"

    monkeypatch.setattr("api.app.predict_emotion", mock_predict)

    fake_img = io.BytesIO(b"this is not a real image")

    response = client.post(
        "/predict",
        files={"file": ("fake.jpg", fake_img, "image/jpeg")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image format"


# =========================
# Model failure (500)
# =========================
def test_predict_model_failure(monkeypatch):
    def mock_predict(_):
        raise Exception("model crashed")

    monkeypatch.setattr("api.app.predict_emotion", mock_predict)

    img_file = create_test_image()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_file, "image/jpeg")}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed"

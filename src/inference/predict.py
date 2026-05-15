# src/inference/predict.py
import os
import cv2
import numpy as np
import tensorflow as tf

from src.llm.groq_client import explain_emotion

# =========================
# Config
# =========================
MODEL_PATH = os.path.join("models_artifacts/final_emotion_model.keras")
IMG_SIZE = (48, 48)
CLASS_NAMES = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# =========================
# Load model
# =========================
#model = tf.keras.models.load_model(MODEL_PATH)

model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model
# =========================
# Preprocess face
# =========================
def preprocess_face(face_img):
    face = cv2.resize(face_img, IMG_SIZE)
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)  # (48,48,1)
    face = np.expand_dims(face, axis=0)   # (1,48,48,1)
    return face

# =========================
# Predict emotion
# =========================
def predict_emotion(face_img):
    model = get_model()
    processed = preprocess_face(face_img)
    preds = model.predict(processed, verbose=0)
    class_idx = np.argmax(preds)
    return CLASS_NAMES[class_idx]

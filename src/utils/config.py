# src/utils/config.py

import os

# =========================
# Dataset paths
# =========================
BASE_DIR = "data/processed"
RAW_DIR = "data/raw"                  # original raw images
RAW_SPLIT_DIR = "data/raw_split"
MODEL_DIR = "models_artifacts" 


TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validation")

# =========================
# Model paths
# =========================
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_emotion_model.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_emotion_model.keras")

# =========================
# Image and training config
# =========================
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 7

CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# Data augmentation
# =========================
import tensorflow as tf

DATA_AUGMENTATION = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

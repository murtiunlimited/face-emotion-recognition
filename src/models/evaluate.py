# src/models/evaluate.py
import os
import tensorflow as tf
import mlflow
from sklearn.metrics import classification_report
from src.utils.mlflow_config import setup_mlflow
import mlflow

# -------------------------
# Config
# -------------------------
VAL_DIR = "data/processed/validation"
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def evaluate_model(model_path="models/final_emotion_model.keras"):
        
    # -------------------------
    # MLflow Setup
    # -------------------------
    setup_mlflow("FER-Emotion-Recognition")

    with mlflow.start_run(run_name="evaluation"):
    
    # -------------------------
    # Preprocessing
    # -------------------------
    def preprocess_image(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)
        x = tf.image.per_image_standardization(x)
        return x, y

    # -------------------------
    # Load dataset
    # -------------------------
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    ).map(preprocess_image).prefetch(tf.data.AUTOTUNE)

    # -------------------------
    # Load model
    # -------------------------
    model = tf.keras.models.load_model(model_path)

    # -------------------------
    # Predictions
    # -------------------------
    y_true, y_pred = [], []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(preds, axis=1).numpy())
  
    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    # -------------------------
    # Report
    # -------------------------
    print("✅ Evaluation complete. Classification report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    return classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

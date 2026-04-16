# src/models/train.py
import os
import tensorflow as tf
import mlflow
import mlflow.keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.model import build_light_model
from src.utils.mlflow_config import setup_mlflow

def train_model():
    
    # -------------------------
    # MLflow Setup
    # -------------------------
    setup_mlflow("FER-Emotion-Recognition")

    with mlflow.start_run():
        
    # -------------------------
    # Config
    # -------------------------
    BASE_DIR = "data/processed"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    VAL_DIR = os.path.join(BASE_DIR, "validation")

    IMG_SIZE = (48, 48)
    BATCH_SIZE = 32
    EPOCHS = 50
    CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Log params
    mlflow.log_params({
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "model": "light_cnn",
        "classes": len(CLASS_NAMES)
    })

    # -------------------------
    # Preprocessing
    # -------------------------
    def preprocess_image(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)
        x = tf.image.per_image_standardization(x)
        return x, y

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    # -------------------------
    # Datasets
    # -------------------------
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    ).map(preprocess_image).map(lambda x, y: (data_augmentation(x), y)).prefetch(tf.data.AUTOTUNE)

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
    # Build Model
    # -------------------------
    model = build_light_model()
    model.compile(
        optimizer="adam", 
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )

    # -------------------------
    # Callbacks
    # -------------------------
    os.makedirs("models_artifacts", exist_ok=True)  # ensure folder exists
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss", 
            patience=10, 
            restore_best_weights=True
        ),
        ModelCheckpoint(
            "models/best_emotion_model.keras", 
            monitor="val_accuracy", 
            save_best_only=True
        )
    ]

    # -------------------------
    # Training
    # -------------------------
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        callbacks=callbacks
    )

    # -------------------------
    # Log metrics (final epoch)
    # -------------------------
    mlflow.log_metrics({
        "train_acc": history.history["accuracy"][-1],
        "val_acc": history.history["val_accuracy"][-1],
        "train_loss": history.history["loss"][-1],
        "val_loss": history.history["val_loss"][-1]
    })

    # -------------------------
    # Save models
    # -------------------------
    model.save("models_artifacts/final_emotion_model.keras")

    # Log model to MLflow
    mlflow.keras.log_model(model, "emotion_model")
    
    print("✅ Training complete + MLflow logging done")


if __name__ == "__main__":
    train_model()

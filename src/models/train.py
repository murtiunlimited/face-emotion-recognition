# src/models/train.py
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.model import build_light_model

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
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # -------------------------
    # Callbacks
    # -------------------------
    os.makedirs("models", exist_ok=True)  # ensure folder exists
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint("models/best_emotion_model.keras", monitor="val_accuracy", save_best_only=True)
    ]

    # -------------------------
    # Training
    # -------------------------
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # -------------------------
    # Save final model
    # -------------------------
    model.save("models/final_emotion_model.keras")
    print("✅ Training complete. Model saved as models/final_emotion_model.keras")

if __name__ == "__main__":
    train_model()  

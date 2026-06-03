import os
import json
from pathlib import Path
from string import ascii_uppercase

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models


# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "dataSet_kaggle_processed"
TRAIN_DIR = DATA_DIR / "trainingData"
VAL_DIR = DATA_DIR / "validationData"
TEST_DIR = DATA_DIR / "testingData"

MODELS_DIR = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "asl_cnn_kaggle.keras"
CLASS_PATH = MODELS_DIR / "asl_cnn_kaggle_classes.json"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

CLASS_NAMES = ["blank"] + list(ascii_uppercase)
NUM_CLASSES = len(CLASS_NAMES)


# =====================================================
# LOAD DATA
# =====================================================

def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


# =====================================================
# BUILD CNN MODEL
# =====================================================

def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =====================================================
# TRAIN
# =====================================================

def main():
    if not TRAIN_DIR.exists():
        print("Chưa tìm thấy trainingData.")
        print("Hãy chạy prepare_kaggle_asl.py trước.")
        return

    train_ds, val_ds, test_ds = load_datasets()

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\nĐánh giá trên tập test:")
    test_loss, test_acc = model.evaluate(test_ds)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    model.save(MODEL_PATH)

    with open(CLASS_PATH, "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=4)

    print("\n==============================")
    print("TRAIN XONG MODEL CNN KAGGLE")
    print("==============================")
    print(f"Model đã lưu tại: {MODEL_PATH}")
    print(f"File class đã lưu tại: {CLASS_PATH}")


if __name__ == "__main__":
    main()
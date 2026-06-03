from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "landmark_dataset" / "asl_landmarks.csv"

MODELS_DIR = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "asl_landmark_model.joblib"


def main():
    if not DATA_PATH.exists():
        print("Không tìm thấy:")
        print(DATA_PATH)
        print("Hãy chạy prepare_kaggle_landmarks_standard.py trước.")
        return

    df = pd.read_csv(DATA_PATH)

    print("Số mẫu từng lớp:")
    print(df["label"].value_counts().sort_index())

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(
            n_estimators=1000,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    print("\nĐang train model Kaggle landmark chuẩn...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("KẾT QUẢ MODEL KAGGLE LANDMARK")
    print("==============================")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    classes = list(model.named_steps["clf"].classes_)

    package = {
        "model": model,
        "classes": classes
    }

    joblib.dump(package, MODEL_PATH)

    print("\nĐã lưu model:")
    print(MODEL_PATH)
    print("Class order:", classes)


if __name__ == "__main__":
    main()
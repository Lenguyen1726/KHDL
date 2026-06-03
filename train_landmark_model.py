import json
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "landmark_dataset" / "asl_landmarks.csv"

MODELS_DIR = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "asl_landmark_model.joblib"
CLASS_PATH = MODELS_DIR / "asl_landmark_classes.json"


def main():
    if not DATA_PATH.exists():
        print("Chưa có file asl_landmarks.csv.")
        print("Hãy chạy prepare_kaggle_landmarks.py trước.")
        return

    df = pd.read_csv(DATA_PATH)

    custom_path = BASE_DIR / "landmark_dataset" / "custom_weak_landmarks.csv"

    if custom_path.exists():
        custom_df = pd.read_csv(custom_path)

        # Giới hạn dữ liệu custom mỗi lớp để tránh làm lệch model
        custom_df = (
            custom_df
            .groupby("label", group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), 500), random_state=42))
        )

        df = pd.concat([df, custom_df], ignore_index=True)

        print(f"Đã thêm dữ liệu custom sau khi giới hạn: {len(custom_df)} mẫu")

    X = df.drop(columns=["label"])
    y = df["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])
    print("Đang train model landmark...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("KẾT QUẢ MODEL LANDMARK")
    print("==============================")
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    ))

    package = {
        "model": model,
        "classes": list(encoder.classes_)
    }

    joblib.dump(package, MODEL_PATH)

    with open(CLASS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(encoder.classes_), f, ensure_ascii=False, indent=4)

    print("\nĐã lưu model:")
    print(MODEL_PATH)
    print(CLASS_PATH)


if __name__ == "__main__":
    main()
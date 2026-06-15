from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "landmark_dataset" / "asl_landmarks.csv"

MODELS_DIR = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "model_evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)

REPORT_TXT_PATH = RESULTS_DIR / "classification_report_asl_landmark.txt"
REPORT_CSV_PATH = RESULTS_DIR / "classification_report_asl_landmark.csv"
CONFUSION_MATRIX_IMG_PATH = RESULTS_DIR / "confusion_matrix_asl_landmark.png"
CONFUSION_MATRIX_CSV_PATH = RESULTS_DIR / "confusion_matrix_asl_landmark.csv"
SUMMARY_PATH = RESULTS_DIR / "training_summary.txt"
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
    classes = list(model.named_steps["clf"].classes_)

    report_text = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=classes
    )

    print("\nClassification Report:")
    print(report_text)

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT - ASL LANDMARK MODEL\n")
        f.write("==========================================\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report_text)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=classes,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(REPORT_CSV_PATH, encoding="utf-8-sig")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.to_csv(CONFUSION_MATRIX_CSV_PATH, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 14))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=True,
        values_format="d"
    )

    plt.title("Confusion Matrix - ASL Landmark Model")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMG_PATH, dpi=300)
    plt.close()

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("TRAINING SUMMARY - ASL LANDMARK MODEL\n")
        f.write("=====================================\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Number of classes: {len(classes)}\n")
        f.write(f"Number of features: {X.shape[1]}\n")
        f.write(f"Accuracy: {acc:.4f}\n")

    print("\nĐã lưu kết quả đánh giá:")
    print(REPORT_TXT_PATH)
    print(REPORT_CSV_PATH)
    print(CONFUSION_MATRIX_IMG_PATH)
    print(CONFUSION_MATRIX_CSV_PATH)
    print(SUMMARY_PATH)

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
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
CUSTOM_PATH = BASE_DIR / "landmark_dataset" / "custom_weak_landmarks.csv"
CUSTOM_LABELS = ["U", "V"]
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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Dùng để đánh giá riêng dữ liệu webcam U/V
    X_custom_test = None
    y_custom_test = None

    if CUSTOM_PATH.exists():
        custom_df = pd.read_csv(CUSTOM_PATH)

        # Chuẩn hóa nhãn
        custom_df["label"] = (
            custom_df["label"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        feature_cols = list(X.columns)
        required_cols = ["label"] + feature_cols

        missing_cols = [
            col for col in required_cols
            if col not in custom_df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Custom dataset thiếu các cột: {missing_cols}"
            )

        # Chỉ lấy dữ liệu U và V
        custom_selected = custom_df[
            custom_df["label"].isin(CUSTOM_LABELS)
        ].copy()

        # Chuyển feature sang dạng số
        for col in feature_cols:
            custom_selected[col] = pd.to_numeric(
                custom_selected[col],
                errors="coerce"
            )

        # Loại bỏ các dòng lỗi hoặc thiếu dữ liệu
        custom_selected = custom_selected.dropna(
            subset=feature_cols
        )

        # Loại bỏ các frame gần như giống hệt nhau.
        # Camera thường lưu nhiều frame liên tiếp rất giống nhau.
        rounded_features = custom_selected[feature_cols].round(4)
        duplicate_check = pd.concat(
            [
                custom_selected[["label"]].reset_index(drop=True),
                rounded_features.reset_index(drop=True)
            ],
            axis=1
        )

        keep_indices = duplicate_check.drop_duplicates(
            subset=["label"] + feature_cols
        ).index

        custom_selected = custom_selected.iloc[
            keep_indices
        ].reset_index(drop=True)

        print("\nSố mẫu custom sau khi loại gần trùng:")
        print(custom_selected["label"].value_counts())

        counts = custom_selected["label"].value_counts()

        missing_labels = [
            label for label in CUSTOM_LABELS
            if label not in counts.index
        ]

        if missing_labels:
            raise ValueError(
                f"Thiếu dữ liệu custom cho lớp: {missing_labels}"
            )

        # Cân bằng số mẫu U và V.
        # Không để U nhiều hơn V hoặc ngược lại.
        samples_per_class = min(
            int(counts["U"]),
            int(counts["V"]),
            1000
        )

        balanced_parts = []

        for label in CUSTOM_LABELS:
            label_data = custom_selected[
                custom_selected["label"] == label
                ]

            label_data = label_data.sample(
                n=samples_per_class,
                random_state=42
            )

            balanced_parts.append(label_data)

        custom_balanced = pd.concat(
            balanced_parts,
            ignore_index=True
        )

        print("\nDữ liệu custom cân bằng:")
        print(custom_balanced["label"].value_counts())

        X_custom = custom_balanced[feature_cols].astype("float32")
        y_custom = custom_balanced["label"]

        # Giữ lại 20% dữ liệu webcam để kiểm tra riêng.
        # Phần này không được đưa vào train.
        (
            X_custom_train,
            X_custom_test,
            y_custom_train,
            y_custom_test
        ) = train_test_split(
            X_custom,
            y_custom,
            test_size=0.2,
            random_state=42,
            stratify=y_custom
        )

        # Chỉ thêm 80% dữ liệu custom vào train
        X_train = pd.concat(
            [
                X_train.reset_index(drop=True),
                X_custom_train.reset_index(drop=True)
            ],
            ignore_index=True
        )

        y_train = pd.concat(
            [
                y_train.reset_index(drop=True),
                y_custom_train.reset_index(drop=True)
            ],
            ignore_index=True
        )

        print(
            f"\nĐã thêm {len(X_custom_train)} "
            "mẫu custom U/V vào tập train."
        )
        print(
            f"Giữ lại {len(X_custom_test)} "
            "mẫu custom U/V để kiểm tra."
        )
        print(f"Tổng số mẫu train mới: {len(X_train)}")

    else:
        print(
            "Không có custom_weak_landmarks.csv, "
            "tiếp tục train bằng dữ liệu Kaggle."
        )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(
            n_estimators=1000,
            max_features=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=1
        ))
    ])

    print("\nĐang train model Kaggle landmark chuẩn...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if X_custom_test is not None:
        y_custom_pred = model.predict(X_custom_test)

        print("\n==============================")
        print("ĐÁNH GIÁ RIÊNG DỮ LIỆU U/V WEBCAM")
        print("==============================")

        print(
            classification_report(
                y_custom_test,
                y_custom_pred,
                labels=["U", "V"],
                zero_division=0
            )
        )

        print("Bảng nhãn thật và nhãn dự đoán:")

        print(
            pd.crosstab(
                y_custom_test,
                y_custom_pred,
                rownames=["Nhãn thật"],
                colnames=["Nhãn dự đoán"],
                margins=True
            )
        )

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
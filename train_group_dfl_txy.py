import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent

MAIN_CSV = BASE_DIR / "landmark_dataset" / "asl_landmarks.csv"
CUSTOM_CSV = BASE_DIR / "landmark_dataset" / "custom_weak_landmarks.csv"

MODELS_DIR = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_LABELS = ["D", "F", "L", "T", "X", "Y"]

OUTPUT_MODEL = MODELS_DIR / "asl_group_dfl_txy.joblib"


def main():
    if not MAIN_CSV.exists():
        print("Không tìm thấy file:")
        print(MAIN_CSV)
        print("Hãy chạy prepare_kaggle_landmarks.py trước.")
        return

    if not CUSTOM_CSV.exists():
        print("Không tìm thấy file custom:")
        print(CUSTOM_CSV)
        print("Hãy thu dữ liệu custom trước.")
        return

    main_df = pd.read_csv(MAIN_CSV)
    custom_df = pd.read_csv(CUSTOM_CSV)

    # Chỉ lấy các chữ cần sửa
    custom_df = custom_df[custom_df["label"].isin(TARGET_LABELS)].copy()

    if len(custom_df) == 0:
        print("File custom không có dữ liệu D/F/L/T/X/Y.")
        return

    # Giới hạn mỗi chữ tối đa 400 mẫu custom
    custom_df = (
        custom_df
        .groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), 400), random_state=42))
    )

    # Lấy thêm dữ liệu Kaggle sạch của đúng nhóm này
    main_group_df = main_df[main_df["label"].isin(TARGET_LABELS)].copy()

    df = pd.concat([main_group_df, custom_df], ignore_index=True)

    print("Số mẫu từng lớp trước cân bằng:")
    print(df["label"].value_counts().sort_index())

    # Cân bằng số mẫu giữa các lớp
    min_count = df["label"].value_counts().min()

    df = (
        df
        .groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=42))
    )

    print("\nSố mẫu từng lớp sau cân bằng:")
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

    print("\nĐang train model phụ D/F/L/T/X/Y...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("KẾT QUẢ MODEL PHỤ D/F/L/T/X/Y")
    print("==============================")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    actual_classes = list(model.named_steps["clf"].classes_)

    package = {
        "model": model,
        "classes": actual_classes
    }

    joblib.dump(package, OUTPUT_MODEL)

    print("\nĐã lưu model phụ:")
    print(OUTPUT_MODEL)
    print("Class order:", actual_classes)


if __name__ == "__main__":
    main()
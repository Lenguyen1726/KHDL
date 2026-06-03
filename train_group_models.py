import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score


BASE_DIR = Path(__file__).resolve().parent

MAIN_CSV = BASE_DIR / "landmark_dataset" / "asl_landmarks.csv"
CUSTOM_CSV = BASE_DIR / "landmark_dataset" / "custom_weak_landmarks.csv"

MODELS_DIR = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(MAIN_CSV)

    if CUSTOM_CSV.exists():
        custom_df = pd.read_csv(CUSTOM_CSV)
        df = pd.concat([df, custom_df], ignore_index=True)

    return df


def train_group(df, group_labels, output_name):
    group_df = df[df["label"].isin(group_labels)].copy()

    X = group_df.drop(columns=["label"])
    y = group_df["label"]

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
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    print(f"\nĐang train group model: {group_labels}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    package = {
        "model": model,
        "classes": group_labels
    }

    output_path = MODELS_DIR / output_name
    joblib.dump(package, output_path)

    print(f"Đã lưu: {output_path}")


def main():
    df = load_data()

    train_group(df, ["M", "N", "S"], "asl_group_mns.joblib")
    train_group(df, ["X", "R", "Z"], "asl_group_xrz.joblib")


if __name__ == "__main__":
    main()
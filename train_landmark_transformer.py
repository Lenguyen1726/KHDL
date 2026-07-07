from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from landmark_transformer_model import LandmarkTransformer


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "landmark_dataset" / "asl_landmarks.csv"
CUSTOM_PATH = BASE_DIR / "landmark_dataset" / "custom_weak_landmarks.csv"

MODELS_DIR = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)

RESULTS_DIR = BASE_DIR / "model_evaluation_results_transformer"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "asl_landmark_transformer.pth"

REPORT_TXT_PATH = RESULTS_DIR / "classification_report_transformer.txt"
REPORT_CSV_PATH = RESULTS_DIR / "classification_report_transformer.csv"
CONFUSION_MATRIX_IMG_PATH = RESULTS_DIR / "confusion_matrix_transformer.png"
CONFUSION_MATRIX_CSV_PATH = RESULTS_DIR / "confusion_matrix_transformer.csv"
SUMMARY_PATH = RESULTS_DIR / "training_summary_transformer.txt"

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(
            X,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def load_base_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_PATH}. "
            "Hãy chạy prepare_kaggle_landmarks_standard.py trước."
        )

    df = pd.read_csv(DATA_PATH)

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[df["label"].isin(CLASS_NAMES)].copy()

    feature_cols = [
        col for col in df.columns
        if col != "label"
    ]

    X = df[feature_cols].astype("float32").values
    y_text = df["label"].values

    label_to_id = {
        label: index
        for index, label in enumerate(CLASS_NAMES)
    }

    y = np.array(
        [label_to_id[label] for label in y_text],
        dtype=np.int64,
    )

    return X, y, feature_cols, df


def load_custom_dataset(feature_cols):
    if not CUSTOM_PATH.exists():
        return None, None

    custom_df = pd.read_csv(CUSTOM_PATH)

    custom_df["label"] = (
        custom_df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    custom_df = custom_df[
        custom_df["label"].isin(CLASS_NAMES)
    ].copy()

    if len(custom_df) == 0:
        return None, None

    missing_cols = [
        col for col in feature_cols
        if col not in custom_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Custom dataset thiếu cột: {missing_cols}"
        )

    for col in feature_cols:
        custom_df[col] = pd.to_numeric(
            custom_df[col],
            errors="coerce",
        )

    custom_df = custom_df.dropna(
        subset=feature_cols,
    )

    custom_df = custom_df.drop_duplicates(
        subset=["label"] + feature_cols,
    )

    label_to_id = {
        label: index
        for index, label in enumerate(CLASS_NAMES)
    }

    X_custom = custom_df[feature_cols].astype("float32").values
    y_custom = np.array(
        [label_to_id[label] for label in custom_df["label"].values],
        dtype=np.int64,
    )

    print("\nDữ liệu custom được thêm vào train:")
    print(custom_df["label"].value_counts().sort_index())

    return X_custom, y_custom


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim=1)

        all_true.extend(
            y_batch.detach().cpu().numpy().tolist()
        )

        all_pred.extend(
            preds.detach().cpu().numpy().tolist()
        )

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_pred)

    return avg_loss, acc


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(logits, dim=1)

            all_true.extend(
                y_batch.detach().cpu().numpy().tolist()
            )

            all_pred.extend(
                preds.detach().cpu().numpy().tolist()
            )

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_pred)

    return avg_loss, acc, all_true, all_pred


def main():
    set_seed(RANDOM_STATE)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Thiết bị train:", device)

    X, y, feature_cols, df = load_base_dataset()

    print("\nSố mẫu từng lớp trong dataset gốc:")
    print(df["label"].value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_custom, y_custom = load_custom_dataset(feature_cols)

    if X_custom is not None:
        X_train = np.concatenate(
            [X_train, X_custom],
            axis=0,
        )

        y_train = np.concatenate(
            [y_train, y_custom],
            axis=0,
        )

    # Chuẩn hóa theo tập train.
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)

    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_dataset = LandmarkDataset(
        X_train,
        y_train,
    )

    test_dataset = LandmarkDataset(
        X_test,
        y_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = LandmarkTransformer(
        num_classes=len(CLASS_NAMES),
        input_dim=3,
        num_landmarks=21,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    best_acc = 0.0
    best_state = None

    print("\nĐang train Landmark Transformer...")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    _, final_acc, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report_text)

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as file:
        file.write("CLASSIFICATION REPORT - LANDMARK TRANSFORMER\n")
        file.write("============================================\n\n")
        file.write(f"Accuracy: {final_acc:.4f}\n\n")
        file.write(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_df.to_csv(
        REPORT_CSV_PATH,
        encoding="utf-8-sig",
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    cm_df.to_csv(
        CONFUSION_MATRIX_CSV_PATH,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(14, 14))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=True,
        values_format="d",
    )

    plt.title("Confusion Matrix - Landmark Transformer")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMG_PATH, dpi=300)
    plt.close()

    config = {
        "num_classes": len(CLASS_NAMES),
        "input_dim": 3,
        "num_landmarks": 21,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 128,
        "dropout": 0.1,
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": CLASS_NAMES,
        "config": config,
        "mean": mean.astype("float32"),
        "std": std.astype("float32"),
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write("TRAINING SUMMARY - LANDMARK TRANSFORMER\n")
        file.write("=======================================\n\n")
        file.write(f"Total base samples: {len(df)}\n")
        file.write(f"Train samples: {len(X_train)}\n")
        file.write(f"Test samples: {len(X_test)}\n")
        file.write(f"Number of classes: {len(CLASS_NAMES)}\n")
        file.write(f"Number of features: {X.shape[1]}\n")
        file.write(f"Best accuracy: {best_acc:.4f}\n")
        file.write(f"Final accuracy: {final_acc:.4f}\n")

    print("\nĐã lưu model Transformer:")
    print(MODEL_PATH)

    print("\nĐã lưu kết quả:")
    print(REPORT_TXT_PATH)
    print(REPORT_CSV_PATH)
    print(CONFUSION_MATRIX_IMG_PATH)
    print(CONFUSION_MATRIX_CSV_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
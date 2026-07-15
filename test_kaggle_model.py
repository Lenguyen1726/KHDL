from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from landmark_transformer_model import LandmarkTransformer

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "asl_landmark_transformer.pth"
)
TEST_DIR = BASE_DIR / "kaggle_data" / "asl_alphabet_test" / "asl_alphabet_test"

mp_hands = mp.solutions.hands


def normalize_landmarks(hand_landmarks):
    points = []

    for lm in hand_landmarks.landmark:
        points.append([lm.x, lm.y, lm.z])

    points = np.array(points, dtype=np.float32)

    wrist = points[0].copy()
    points = points - wrist

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))

    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    return points.flatten().reshape(1, -1)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Thiết bị test:",
        device
    )

    (
        model,
        classes,
        mean,
        std,
    ) = load_transformer(device)

    print(
        "Loaded Transformer model"
    )

    print(
        "Classes:",
        classes
    )

    if not TEST_DIR.exists():
        print("Không tìm thấy thư mục test:")
        print(TEST_DIR)
        return

    correct = 0
    total = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as hands:

        for img_path in sorted(TEST_DIR.glob("*_test.jpg")):
            label = img_path.name.split("_")[0].upper()

            if label not in classes:
                continue

            img = cv2.imread(str(img_path))

            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                print(f"{img_path.name}: không detect được tay")
                continue

            features = normalize_landmarks(results.multi_hand_landmarks[0])

            pred, conf = predict_transformer(
                model=model,
                features=features,
                classes=classes,
                mean=mean,
                std=std,
                device=device,
            )

            total += 1

            if pred == label:
                correct += 1
                status = "OK"
            else:
                status = "SAI"

            print(f"{img_path.name:12s} | thật: {label:2s} | đoán: {pred:2s} | {conf:.2f} | {status}")

    print("\n==============================")
    print(f"Đúng: {correct}/{total}")
    if total > 0:
        print(f"Accuracy test Kaggle: {correct / total:.4f}")

def load_checkpoint(
    model_path,
    device
):
    try:
        return torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            model_path,
            map_location=device,
        )
def load_transformer(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: "
            f"{MODEL_PATH}\n"
            "Hãy chạy "
            "train_landmark_transformer.py "
            "trước."
        )

    checkpoint = load_checkpoint(
        MODEL_PATH,
        device
    )

    classes = list(
        checkpoint["classes"]
    )

    config = dict(
        checkpoint["config"]
    )

    mean = np.asarray(
        checkpoint["mean"],
        dtype=np.float32,
    ).reshape(1, -1)

    std = np.asarray(
        checkpoint["std"],
        dtype=np.float32,
    ).reshape(1, -1)

    std[
        std < 1e-6
    ] = 1.0

    model = LandmarkTransformer(
        **config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return (
        model,
        classes,
        mean,
        std
    )
def predict_transformer(
    model,
    features,
    classes,
    mean,
    std,
    device,
):
    features = (
        features - mean
    ) / std

    input_tensor = torch.from_numpy(
        features.astype(
            np.float32
        )
    ).to(device)

    with torch.no_grad():
        logits = model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    index = int(
        torch.argmax(
            probabilities
        ).item()
    )

    pred = classes[index]

    confidence = float(
        probabilities[index].item()
    )

    return pred, confidence
if __name__ == "__main__":
    main()
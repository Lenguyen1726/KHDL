from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Models" / "asl_landmark_model.joblib"

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
    package = joblib.load(MODEL_PATH)

    model = package["model"]
    classes = package["classes"]

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

            probs = model.predict_proba(features)[0]
            idx = int(np.argmax(probs))

            pred = classes[idx]
            conf = float(probs[idx])

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


if __name__ == "__main__":
    main()
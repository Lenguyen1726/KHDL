import json
from pathlib import Path
from string import ascii_uppercase

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "kaggle_data" / "asl_alphabet_train" / "asl_alphabet_train"

OUTPUT_DIR = BASE_DIR / "landmark_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "asl_landmarks.csv"

CLASS_NAMES = list(ascii_uppercase)

# Lần đầu để 800 cho nhanh.
# Sau ổn có thể tăng 1500 hoặc None.
MAX_IMAGES_PER_CLASS = 800

mp_hands = mp.solutions.hands


def normalize_landmarks(hand_landmarks):
    """
    Chuyển 21 landmark thành vector 63 đặc trưng.
    Chuẩn hóa theo cổ tay để giảm ảnh hưởng vị trí tay trong ảnh.
    """
    points = []

    for lm in hand_landmarks.landmark:
        points.append([lm.x, lm.y, lm.z])

    points = np.array(points, dtype=np.float32)

    # Lấy cổ tay làm gốc tọa độ
    wrist = points[0].copy()
    points = points - wrist

    # Chuẩn hóa theo kích thước bàn tay
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))

    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    return points.flatten()


def mirror_features(features):
    """
    Tạo dữ liệu mirror để model học được cả tay trái/tay phải hoặc camera bị lật.
    """
    points = features.reshape(21, 3).copy()
    points[:, 0] = -points[:, 0]
    return points.flatten()


def extract_landmark_from_image(image_path, hands):
    img = cv2.imread(str(image_path))

    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    features = normalize_landmarks(hand_landmarks)

    return features


def main():
    if not RAW_DIR.exists():
        print(f"Không tìm thấy thư mục Kaggle: {RAW_DIR}")
        return

    rows = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as hands:

        for class_name in CLASS_NAMES:
            class_dir = RAW_DIR / class_name

            if not class_dir.exists():
                print(f"Không tìm thấy lớp {class_name}: {class_dir}")
                continue

            image_paths = [
                p for p in class_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ]

            if MAX_IMAGES_PER_CLASS is not None:
                image_paths = image_paths[:MAX_IMAGES_PER_CLASS]

            success = 0
            fail = 0

            print(f"\nĐang xử lý lớp {class_name} - {len(image_paths)} ảnh")

            for image_path in image_paths:
                features = extract_landmark_from_image(image_path, hands)

                if features is None:
                    fail += 1
                    continue

                row = {"label": class_name}

                for i, value in enumerate(features):
                    row[f"f{i}"] = value

                rows.append(row)

                # Thêm bản mirror để tăng dữ liệu
                mirrored = mirror_features(features)

                row_mirror = {"label": class_name}

                for i, value in enumerate(mirrored):
                    row_mirror[f"f{i}"] = value

                rows.append(row_mirror)

                success += 1

            print(f"Thành công: {success}")
            print(f"Bỏ qua: {fail}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    class_path = OUTPUT_DIR / "classes.json"

    with open(class_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=4)

    print("\n==============================")
    print("HOÀN THÀNH TRÍCH XUẤT LANDMARK")
    print("==============================")
    print(f"File CSV: {OUTPUT_CSV}")
    print(f"Tổng mẫu: {len(df)}")


if __name__ == "__main__":
    main()
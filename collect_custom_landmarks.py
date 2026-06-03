import sys
from pathlib import Path
from string import ascii_uppercase

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "landmark_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

CUSTOM_CSV = OUTPUT_DIR / "custom_weak_landmarks.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


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

    return points.flatten()


def main():
    if len(sys.argv) < 2:
        print("Cách dùng:")
        print("python collect_custom_landmarks.py M")
        print("python collect_custom_landmarks.py N")
        return

    label = sys.argv[1].upper()

    if label not in list(ascii_uppercase):
        print("Label phải là A-Z.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không mở được camera.")
        return

    rows = []
    count = 0
    frame_skip = 0

    print(f"Đang thu dữ liệu cho chữ: {label}")
    print("Nhấn Q để dừng.")
    print("Cố gắng thay đổi nhẹ góc tay, khoảng cách, ánh sáng.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                frame_skip += 1

                # Cứ 3 frame lưu 1 mẫu để tránh dữ liệu quá giống nhau
                if frame_skip >= 3:
                    features = normalize_landmarks(hand_landmarks)

                    row = {"label": label}

                    for i, value in enumerate(features):
                        row[f"f{i}"] = value

                    rows.append(row)
                    count += 1
                    frame_skip = 0

                cv2.putText(
                    frame,
                    f"Collecting {label}: {count}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:
                cv2.putText(
                    frame,
                    "No hand",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Collect Custom Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(rows) == 0:
        print("Không có dữ liệu được lưu.")
        return

    new_df = pd.DataFrame(rows)

    if CUSTOM_CSV.exists():
        old_df = pd.read_csv(CUSTOM_CSV)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(CUSTOM_CSV, index=False, encoding="utf-8-sig")

    print("\nĐã lưu dữ liệu:")
    print(CUSTOM_CSV)
    print(f"Số mẫu mới: {len(new_df)}")
    print(f"Tổng số mẫu custom: {len(df)}")


if __name__ == "__main__":
    main()
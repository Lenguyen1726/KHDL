import sys
import time
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


# Mỗi chữ thu bao nhiêu mẫu
TARGET_SAMPLES = 400

# Cứ bao nhiêu frame thì lưu 1 mẫu
SAVE_EVERY_N_FRAMES = 3

# Đợi vài giây đầu để bạn chuẩn bị tay
WARMUP_SECONDS = 3


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


def is_hand_quality_ok(hand_landmarks):
    """
    Lọc bớt frame xấu:
    - tay quá nhỏ
    - tay quá sát camera
    - landmark bị co cụm bất thường
    """
    points = []

    for lm in hand_landmarks.landmark:
        points.append([lm.x, lm.y])

    points = np.array(points, dtype=np.float32)

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    box_w = x_max - x_min
    box_h = y_max - y_min

    # Tay quá nhỏ, model khó học
    if box_w < 0.12 or box_h < 0.12:
        return False

    # Tay quá to, dễ bị cắt mất ngón
    if box_w > 0.85 or box_h > 0.85:
        return False

    return True


def save_to_csv(rows):
    if len(rows) == 0:
        print("Không có dữ liệu mới để lưu.")
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


def main():
    if len(sys.argv) < 2:
        print("Cách dùng:")
        print("python collect_custom_landmarks_auto_safe.py D")
        print("python collect_custom_landmarks_auto_safe.py F")
        print("python collect_custom_landmarks_auto_safe.py L")
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
    frame_counter = 0
    paused = False

    start_time = time.time()

    print(f"Đang thu tự động chữ: {label}")
    print(f"Mục tiêu: {TARGET_SAMPLES} mẫu")
    print("P: tạm dừng / tiếp tục")
    print("Q: dừng và lưu")
    print("R: xóa mẫu đang thu của chữ hiện tại")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    ) as hands:

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            elapsed = time.time() - start_time
            in_warmup = elapsed < WARMUP_SECONDS

            status_text = ""

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                quality_ok = is_hand_quality_ok(hand_landmarks)

                if in_warmup:
                    remain = int(WARMUP_SECONDS - elapsed) + 1
                    status_text = f"Ready for {label} in {remain}s"

                elif paused:
                    status_text = f"PAUSED {label}: {count}/{TARGET_SAMPLES}"

                elif not quality_ok:
                    status_text = "Move hand closer/farther"

                else:
                    frame_counter += 1

                    if frame_counter >= SAVE_EVERY_N_FRAMES:
                        features = normalize_landmarks(hand_landmarks)

                        row = {"label": label}

                        for i, value in enumerate(features):
                            row[f"f{i}"] = value

                        rows.append(row)
                        count += 1
                        frame_counter = 0

                    status_text = f"Collecting {label}: {count}/{TARGET_SAMPLES}"

            else:
                status_text = "No hand"

            color = (0, 255, 0)

            if "No hand" in status_text or "Move" in status_text:
                color = (0, 0, 255)

            if paused:
                color = (0, 255, 255)

            cv2.putText(
                frame,
                status_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

            cv2.putText(
                frame,
                "P: Pause | Q: Save & Quit | R: Reset current",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("Auto Safe Landmark Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("p"):
                paused = not paused

            elif key == ord("r"):
                rows = []
                count = 0
                frame_counter = 0
                print(f"Đã reset mẫu đang thu cho chữ {label}")

            elif key == ord("q"):
                break

            if count >= TARGET_SAMPLES:
                print(f"Đã đủ {TARGET_SAMPLES} mẫu cho chữ {label}")
                break

    cap.release()
    cv2.destroyAllWindows()

    save_to_csv(rows)


if __name__ == "__main__":
    main()
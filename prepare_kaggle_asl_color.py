import random
from pathlib import Path
from string import ascii_uppercase

import cv2


BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "kaggle_data" / "asl_alphabet_train" / "asl_alphabet_train"

OUTPUT_DIR = BASE_DIR / "dataSet_kaggle_color"
TRAIN_DIR = OUTPUT_DIR / "trainingData"
VAL_DIR = OUTPUT_DIR / "validationData"
TEST_DIR = OUTPUT_DIR / "testingData"

IMG_SIZE = 128
RANDOM_STATE = 42

# Lần đầu để 500 cho nhanh, sau ổn có thể tăng 1500 hoặc None
MAX_IMAGES_PER_CLASS = 500

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

CLASS_NAMES = ["blank"] + list(ascii_uppercase)

SOURCE_MAP = {"blank": "nothing"}

for letter in ascii_uppercase:
    SOURCE_MAP[letter] = letter


def preprocess_image_color(image_path):
    """
    Đọc ảnh màu, resize về 128x128.
    Không grayscale, không threshold.
    """
    img = cv2.imread(str(image_path))

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return img


def save_images(image_paths, target_folder):
    target_folder.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(image_paths):
        processed = preprocess_image_color(image_path)

        if processed is None:
            continue

        output_path = target_folder / f"{idx:05d}.png"
        cv2.imwrite(str(output_path), processed)


def main():
    if not RAW_DIR.exists():
        print(f"Không tìm thấy thư mục: {RAW_DIR}")
        return

    random.seed(RANDOM_STATE)

    for class_name in CLASS_NAMES:
        source_folder = RAW_DIR / SOURCE_MAP[class_name]

        if not source_folder.exists():
            print(f"Bỏ qua lớp {class_name}, không tìm thấy {source_folder}")
            continue

        image_paths = [
            p for p in source_folder.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]

        random.shuffle(image_paths)

        if MAX_IMAGES_PER_CLASS is not None:
            image_paths = image_paths[:MAX_IMAGES_PER_CLASS]

        total = len(image_paths)

        train_end = int(total * TRAIN_RATIO)
        val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

        train_paths = image_paths[:train_end]
        val_paths = image_paths[train_end:val_end]
        test_paths = image_paths[val_end:]

        print(f"\nLớp {class_name}")
        print(f"Tổng ảnh: {total}")
        print(f"Train: {len(train_paths)}")
        print(f"Validation: {len(val_paths)}")
        print(f"Test: {len(test_paths)}")

        save_images(train_paths, TRAIN_DIR / class_name)
        save_images(val_paths, VAL_DIR / class_name)
        save_images(test_paths, TEST_DIR / class_name)

    print("\n==============================")
    print("HOÀN THÀNH XỬ LÝ DATASET ẢNH MÀU")
    print("==============================")
    print(f"Dataset màu nằm tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
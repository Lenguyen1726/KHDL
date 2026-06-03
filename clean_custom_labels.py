import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CUSTOM_CSV = BASE_DIR / "landmark_dataset" / "custom_weak_landmarks.csv"
BACKUP_CSV = BASE_DIR / "landmark_dataset" / "custom_weak_landmarks_backup.csv"


def main():
    if not CUSTOM_CSV.exists():
        print("Không tìm thấy custom_weak_landmarks.csv")
        return

    labels_to_remove = [x.upper() for x in sys.argv[1:]]

    if len(labels_to_remove) == 0:
        print("Cách dùng:")
        print("python clean_custom_labels.py X R Z")
        return

    df = pd.read_csv(CUSTOM_CSV)

    df.to_csv(BACKUP_CSV, index=False, encoding="utf-8-sig")

    before = len(df)

    df_clean = df[~df["label"].isin(labels_to_remove)].copy()

    after = len(df_clean)

    df_clean.to_csv(CUSTOM_CSV, index=False, encoding="utf-8-sig")

    print("Đã backup file cũ tại:")
    print(BACKUP_CSV)

    print(f"Số mẫu trước khi xóa: {before}")
    print(f"Số mẫu sau khi xóa: {after}")
    print(f"Đã xóa các nhãn: {labels_to_remove}")


if __name__ == "__main__":
    main()
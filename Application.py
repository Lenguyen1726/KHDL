# Importing Libraries
import joblib
import os
import json
import operator
from pathlib import Path
from string import ascii_uppercase

import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk

from spellchecker import SpellChecker

# Giảm log TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


# =====================================================
# PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"


# =====================================================
# BUILD MODEL TỪ FILE JSON CŨ
# Không dùng model_from_json vì Keras mới dễ lỗi Python 3.13
# =====================================================

def build_model_from_old_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    model_name = model_config.get("config", {}).get("name", "sequential")
    layers_config = model_config["config"]["layers"]

    model = Sequential(name=model_name)

    input_added = False

    for layer_info in layers_config:
        class_name = layer_info["class_name"]
        config = layer_info["config"]

        if class_name == "InputLayer":
            batch_shape = config.get("batch_input_shape") or config.get("batch_shape")

            if batch_shape is not None:
                input_shape = tuple(batch_shape[1:])
            else:
                input_shape = tuple(config.get("shape"))

            if not input_added:
                model.add(InputLayer(
                    input_shape=input_shape,
                    name=config.get("name"),
                    dtype=config.get("dtype", "float32")
                ))
                input_added = True

        elif class_name == "Conv2D":
            # Nếu JSON không có InputLayer riêng thì lấy input_shape từ Conv2D đầu tiên
            if not input_added and "batch_input_shape" in config:
                input_shape = tuple(config["batch_input_shape"][1:])
                model.add(InputLayer(input_shape=input_shape))
                input_added = True

            model.add(Conv2D(
                filters=config["filters"],
                kernel_size=tuple(config["kernel_size"]),
                strides=tuple(config.get("strides", (1, 1))),
                padding=config.get("padding", "valid"),
                activation=config.get("activation", None),
                use_bias=config.get("use_bias", True),
                dilation_rate=tuple(config.get("dilation_rate", (1, 1))),
                name=config.get("name")
            ))

        elif class_name == "MaxPooling2D":
            pool_size = tuple(config.get("pool_size", (2, 2)))
            strides = config.get("strides")

            if strides is not None:
                strides = tuple(strides)
            else:
                strides = pool_size

            model.add(MaxPooling2D(
                pool_size=pool_size,
                strides=strides,
                padding=config.get("padding", "valid"),
                name=config.get("name")
            ))

        elif class_name == "Flatten":
            model.add(Flatten(
                name=config.get("name")
            ))

        elif class_name == "Dense":
            model.add(Dense(
                units=config["units"],
                activation=config.get("activation", None),
                use_bias=config.get("use_bias", True),
                name=config.get("name")
            ))

        elif class_name == "Dropout":
            model.add(Dropout(
                rate=config["rate"],
                name=config.get("name")
            ))

        else:
            raise ValueError(f"Layer chưa hỗ trợ: {class_name}")

    return model

def load_old_model(json_file, h5_file, model_name):
    json_path = MODELS_DIR / json_file
    h5_path = MODELS_DIR / h5_file

    model = build_model_from_old_json(json_path)
    model.load_weights(str(h5_path))

    print(f"Loaded {model_name}")

    return model


# =====================================================
# APPLICATION
# =====================================================

class Application:

    def __init__(self):
        # Suggestions mới, thay cho Hunspell
        self.spell = SpellChecker(language="en")

        # Camera
        self.vs = cv2.VideoCapture(0)

        if not self.vs.isOpened():
            print("Không mở được camera 0. Đang thử camera 1...")
            self.vs = cv2.VideoCapture(1)

        if not self.vs.isOpened():
            print("Không mở được camera. Kiểm tra quyền camera Windows.")

        # =====================================================
        # MEDIAPIPE HAND DETECTION
        # =====================================================

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.current_image = None
        self.current_image2 = None

        # =====================================================
        # LOAD MODELS
        # =====================================================

        print("Loading landmark model...")

        landmark_package = joblib.load(MODELS_DIR / "asl_landmark_model.joblib")

        self.landmark_model = landmark_package["model"]
        self.class_names = landmark_package["classes"]

        print("Loaded landmark model")

        # Tắt toàn bộ group model để test chuẩn Kaggle trước
        self.use_group_mns = False
        self.group_mns = None
        self.group_mns_labels = ["M", "N", "S"]

        self.use_group_dfl_txy = False
        self.group_dfl_txy = None
        self.group_dfl_txy_labels = ["D", "F", "L", "T", "X", "Y"]

        # =====================================================
        # LOAD GROUP MODEL M/N/S
        # =====================================================

        mns_path = MODELS_DIR / "asl_group_mns.joblib"

        if mns_path.exists():
            self.group_mns = joblib.load(mns_path)
            self.group_mns_labels = ["M", "N", "S"]
            self.use_group_mns = True
            print("Loaded group model M/N/S")
        else:
            self.group_mns = None
            self.group_mns_labels = ["M", "N", "S"]
            self.use_group_mns = False
            print("Group model M/N/S not found")

        # =====================================================
        # TẠM TẮT GROUP D/F/L/T/X/Y
        # =====================================================

        self.group_dfl_txy = None
        self.group_dfl_txy_labels = ["D", "F", "L", "T", "X", "Y"]
        self.use_group_dfl_txy = False

        print("Group D/F/L/T/X/Y is disabled for stability")

        group_path = MODELS_DIR / "asl_group_dfl_txy.joblib"

        if group_path.exists():
            self.group_dfl_txy = joblib.load(group_path)
            self.use_group_dfl_txy = True
            print("Loaded group model D/F/L/T/X/Y")
        else:
            self.group_dfl_txy = None
            self.use_group_dfl_txy = False
            print("Group model D/F/L/T/X/Y not found, running main model only")

        # =====================================================
        # COUNTER
        # =====================================================

        self.ct = {}
        self.ct["blank"] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
            self.ct[i] = 0

        # =====================================================
        # GUI
        # =====================================================

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("1000x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)
        # self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(
            text="Sign Language To Text Conversion",
            font=("Courier", 30, "bold")
        )

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word :", font=("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=690)
        self.T4.config(
            text="Suggestions :",
            fg="red",
            font=("Courier", 30, "bold")
        )

        self.bt1 = tk.Button(self.root, command=self.action1, height=1, width=12)
        self.bt1.place(x=26, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2, height=1, width=12)
        self.bt2.place(x=220, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3, height=1, width=12)
        self.bt3.place(x=414, y=745)

        self.bt4 = tk.Button(self.root, command=self.action4, height=1, width=12)
        self.bt4.place(x=608, y=745)

        self.bt5 = tk.Button(self.root, command=self.action5, height=1, width=12)
        self.bt5.place(x=802, y=745)

        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.current_confidence = 0.0
        self.min_confidence = 0.70

        self.last_symbol = None
        self.stable_count = 0
        self.stable_required = 10

        self.add_cooldown = 0
        self.no_hand_count = 0
        self.no_hand_required = 20

        self.photo = "Empty"

        self.video_loop()

    # =====================================================
    # GET SUGGESTIONS
    # =====================================================

    def get_suggestions(self):
        word = self.word.strip().lower()

        if len(word) == 0:
            return []

        try:
            candidates = self.spell.candidates(word)

            if candidates is None:
                return []

            candidates = list(candidates)

            # Ưu tiên từ gần độ dài với từ hiện tại
            candidates = sorted(
                candidates,
                key=lambda x: (abs(len(x) - len(word)), x)
            )

            candidates = [w.upper() for w in candidates]

            return candidates[:5]

        except Exception:
            return []

    def preprocess_hand_mediapipe(self, frame):
        """
        Dùng MediaPipe phát hiện bàn tay.
        Cắt sát vùng bàn tay, xóa nền và giữ ảnh màu RGB.
        Phù hợp với model CNN ảnh màu.
        """

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            cv2.putText(
                frame,
                "No hand",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            return None, frame

        hand_landmarks = results.multi_hand_landmarks[0]

        points = []

        for lm in hand_landmarks.landmark:
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            points.append([x, y])

        points = np.array(points)

        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])

        # Cắt theo hình vuông để không làm méo tay
        box_w = x_max - x_min
        box_h = y_max - y_min
        box_size = max(box_w, box_h)

        # Padding vừa phải, không quá lớn để tránh lấy nhiều nền
        padding = int(box_size * 0.35)

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        half = box_size // 2 + padding

        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, frame_w)
        y2 = min(cy + half, frame_h)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None, frame

        # Vẽ landmark và khung tay lên camera
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Hand detected",
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Tạo mask bàn tay từ 21 landmark
        points_roi = points - np.array([x1, y1])

        mask = np.zeros(roi.shape[:2], dtype=np.uint8)

        hull = cv2.convexHull(points_roi)
        cv2.fillConvexPoly(mask, hull, 255)

        # Nới mask nhẹ để giữ đủ vùng ngón tay
        kernel = np.ones((25, 25), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Xóa nền, nền chuyển thành đen
        hand_only = cv2.bitwise_and(roi, roi, mask=mask)

        # Resize ảnh màu
        processed = cv2.resize(hand_only, (128, 128))

        # OpenCV là BGR, model train bằng RGB
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        return processed, frame

    def reset_stability(self):
        self.last_symbol = None
        self.stable_count = 0

    def commit_word(self):
        if len(self.word) == 0:
            return

        if len(self.str) > 0:
            self.str += " "

        self.str += self.word
        self.word = ""

    def extract_landmark_features(self, hand_landmarks):
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

    def get_landmark_from_frame(self, frame):
        frame = cv2.flip(frame, 1)

        frame_h, frame_w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            cv2.putText(
                frame,
                "No hand",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            return None, frame

        hand_landmarks = results.multi_hand_landmarks[0]

        # Vẽ landmark
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

        # Vẽ bounding box
        points = []

        for lm in hand_landmarks.landmark:
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            points.append([x, y])

        points = np.array(points)

        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])

        padding = 40

        x1 = max(x_min - padding, 0)
        y1 = max(y_min - padding, 0)
        x2 = min(x_max + padding, frame_w)
        y2 = min(y_max + padding, frame_h)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Hand detected",
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        features = self.extract_landmark_features(hand_landmarks)

        return features, frame

    def predict_landmark(self, features):
        probabilities = self.landmark_model.predict_proba(features)[0]

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

        self.current_symbol = self.class_names[index]
        self.current_confidence = confidence

        if self.current_confidence < self.min_confidence:
            self.current_symbol = "Uncertain"
            self.reset_stability()
            return

        symbol = self.current_symbol

        if symbol == self.last_symbol:
            self.stable_count += 1
        else:
            self.last_symbol = symbol
            self.stable_count = 1

        if self.add_cooldown > 0:
            self.add_cooldown -= 1
            return

        if self.stable_count >= self.stable_required:
            self.word += symbol
            self.add_cooldown = 20
            self.stable_count = 0
    # =====================================================
    # VIDEO LOOP
    # =====================================================

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            features, display_frame = self.get_landmark_from_frame(frame)

            if features is not None:
                self.no_hand_count = 0
                self.predict_landmark(features)

            else:
                self.current_symbol = "No hand"
                self.current_confidence = 0.0
                self.reset_stability()

                self.no_hand_count += 1

                if self.no_hand_count >= self.no_hand_required:
                    self.commit_word()
                    self.no_hand_count = 0

            # Hiển thị camera chính
            display_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(display_image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # Cập nhật chữ
            self.panel3.config(
                text=f"{self.current_symbol} ({self.current_confidence:.2f})",
                font=("Courier", 30)
            )

            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            # Suggestions
            predicts = self.get_suggestions()

            buttons = [self.bt1, self.bt2, self.bt3, self.bt4, self.bt5]

            for i, button in enumerate(buttons):
                if len(predicts) > i:
                    button.config(text=predicts[i], font=("Courier", 18))
                else:
                    button.config(text="")

        self.root.after(5, self.video_loop)

    def reset_counters(self):
        self.ct["blank"] = 0

        for i in ascii_uppercase:
            self.ct[i] = 0

    def reset_counters(self):
        self.ct["blank"] = 0

        for i in ascii_uppercase:
            self.ct[i] = 0
    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))

        # Nếu lỡ ảnh là grayscale thì chuyển sang RGB
        if len(test_image.shape) == 2:
            test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)

        test_image = test_image.astype("float32")
        test_image = test_image.reshape(1, 128, 128, 3)

        result = self.loaded_model.predict(test_image, verbose=0)

        index = int(np.argmax(result[0]))
        confidence = float(result[0][index])

        self.current_symbol = self.class_names[index]
        self.current_confidence = confidence

        # Ngưỡng tin cậy, nếu thấp thì không ghi chữ
        if confidence < self.min_confidence:
            self.current_symbol = "Uncertain"
            self.reset_counters()
            return

        if self.current_symbol == "blank":
            for i in ascii_uppercase:
                self.ct[i] = 0

        if self.current_symbol not in self.ct:
            return

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 30:

            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = abs(self.ct[self.current_symbol] - self.ct[i])

                if tmp <= 10:
                    self.reset_counters()
                    return

            self.reset_counters()

            if self.current_symbol == "blank":

                if self.blank_flag == 0:
                    self.blank_flag = 1

                    if len(self.str) > 0:
                        self.str += " "

                    self.str += self.word
                    self.word = ""

            else:
                if len(self.str) > 16:
                    self.str = ""

                self.blank_flag = 0
                self.word += self.current_symbol
    # =====================================================
    # ACTION SUGGESTION
    # =====================================================

    def use_suggestion(self, index):
        predicts = self.get_suggestions()

        if len(predicts) > index:
            selected_word = predicts[index]

            self.word = ""

            if len(self.str) > 0:
                self.str += " "

            self.str += selected_word

    def action1(self):
        self.use_suggestion(0)

    def action2(self):
        self.use_suggestion(1)

    def action3(self):
        self.use_suggestion(2)

    def action4(self):
        self.use_suggestion(3)

    def action5(self):
        self.use_suggestion(4)

    # =====================================================
    # CLOSE APP
    # =====================================================

    def destructor(self):
        print("Closing Application...")

        try:
            self.root.destroy()
        except Exception:
            pass

        try:
            self.vs.release()
        except Exception:
            pass

        try:
            self.hands.close()
        except Exception:
            pass

        cv2.destroyAllWindows()


# =====================================================
# RUN APP
# =====================================================

print("Starting Application...")

app = Application()
app.root.mainloop()
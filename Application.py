
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
import torch

from PIL import Image, ImageTk
from spellchecker import SpellChecker
from landmark_transformer_model import LandmarkTransformer


# =====================================================
# PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
MODEL_PATH = (
    MODELS_DIR
    / "asl_landmark_transformer.pth"
)

# =====================================================
# APPLICATION
# =====================================================

class Application:

    def __init__(self):
        self.spell = SpellChecker(language="en")
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Transformer device:",
            self.device
        )

        # =====================================================
        # CAMERA
        # =====================================================

        self.vs = cv2.VideoCapture(0)

        if not self.vs.isOpened():
            print("Không mở được camera 0. Đang thử camera 1...")
            self.vs = cv2.VideoCapture(1)

        if not self.vs.isOpened():
            print("Không mở được camera. Kiểm tra quyền camera Windows.")

        # =====================================================
        # MEDIAPIPE
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

        # =====================================================
        # LOAD LANDMARK TRANSFORMER
        # =====================================================
        print("Loading landmark Transformer...")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {MODEL_PATH}\n"
                "Hãy chạy train_landmark_transformer.py trước."
            )

        checkpoint = self.load_checkpoint(
            MODEL_PATH
        )

        self.class_names = list(
            checkpoint["classes"]
        )

        self.transformer_config = dict(
            checkpoint["config"]
        )

        self.landmark_mean = np.asarray(
            checkpoint["mean"],
            dtype=np.float32,
        ).reshape(1, -1)

        self.landmark_std = np.asarray(
            checkpoint["std"],
            dtype=np.float32,
        ).reshape(1, -1)

        self.landmark_std[
            self.landmark_std < 1e-6
            ] = 1.0

        self.landmark_model = LandmarkTransformer(
            **self.transformer_config
        ).to(self.device)

        self.landmark_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.landmark_model.eval()

        print(
            "Loaded landmark Transformer"
        )

        print(
            "Classes:",
            self.class_names
        )

        print(
            "Transformer config:",
            self.transformer_config
        )

        # Không dùng group model nữa
        self.use_group_mns = False
        self.use_group_dfl_txy = False
        self.use_group_xrz = False

        # =====================================================
        # TEXT STATE
        # =====================================================

        self.str = ""
        self.word = ""

        self.current_symbol = "Empty"
        self.current_confidence = 0.0

        self.min_confidence = 0.70

        self.last_symbol = None
        self.stable_count = 0
        self.stable_required = 10

        self.add_cooldown = 20
        self.no_hand_count = 0
        self.no_hand_required = 20

        # =====================================================
        # GUI
        # =====================================================

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("1000x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

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

        self.video_loop()

    def load_checkpoint(
            self,
            model_path
    ):
        try:
            return torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )

        except TypeError:
            return torch.load(
                model_path,
                map_location=self.device,
            )

    # =====================================================
    # SUGGESTIONS
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

            candidates = sorted(
                candidates,
                key=lambda x: (abs(len(x) - len(word)), x)
            )

            candidates = [w.upper() for w in candidates]

            return candidates[:5]

        except Exception:
            return []

    # =====================================================
    # TEXT CONTROL
    # =====================================================

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

    # =====================================================
    # LANDMARK FEATURE
    # =====================================================

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

        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

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

    # =====================================================
    # PREDICT LANDMARK
    # =====================================================

    def predict_landmark(
            self,
            features
    ):
        normalized_features = (
                                      features - self.landmark_mean
                              ) / self.landmark_std

        input_tensor = torch.from_numpy(
            normalized_features.astype(
                np.float32
            )
        ).to(self.device)

        with torch.no_grad():
            logits = self.landmark_model(
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

        confidence = float(
            probabilities[index].item()
        )

        self.current_symbol = (
            self.class_names[index]
        )

        self.current_confidence = confidence

        if (
                self.current_confidence
                < self.min_confidence
        ):
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

        if (
                self.stable_count
                >= self.stable_required
        ):
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

            display_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(display_image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            self.panel3.config(
                text=f"{self.current_symbol} ({self.current_confidence:.2f})",
                font=("Courier", 30)
            )

            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            predicts = self.get_suggestions()

            buttons = [self.bt1, self.bt2, self.bt3, self.bt4, self.bt5]

            for i, button in enumerate(buttons):
                if len(predicts) > i:
                    button.config(text=predicts[i], font=("Courier", 18))
                else:
                    button.config(text="")

        self.root.after(5, self.video_loop)

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
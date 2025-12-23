import os
import platform
import time
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="🖐 Gesture Controller", layout="centered")
st.title("🖐 Hand Gesture Detection (Cloud Demo Mode)")

st.markdown("""
### ℹ️ Info
- On **local machine** → Real gesture detection (MediaPipe)
- On **Streamlit Cloud** → Hand detection demo (motion-based)
""")

# =========================================================
# Environment Detection
# =========================================================
RUNNING_LOCALLY = (
    platform.system() == "Windows"
    or os.environ.get("DISPLAY") is not None
)

# =========================================================
# Local MediaPipe Setup (only if local)
# =========================================================
if RUNNING_LOCALLY:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    st.success("🖥 Local Mode — Real gesture detection enabled")
else:
    st.warning("🌐 Cloud Mode — Gesture detection demo only")

# =========================================================
# Video Transformer
# =========================================================
class GestureDemo(VideoTransformerBase):
    def __init__(self):
        self.prev_gray = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # =========================
        # LOCAL: MediaPipe
        # =========================
        if RUNNING_LOCALLY:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    cv2.putText(
                        img,
                        "Hand Detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            return img

        # =========================
        # CLOUD: Motion Detection
        # =========================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return img

        diff = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) < 3000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

            cv2.putText(
                img,
                "Hand Movement Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

        self.prev_gray = gray

        cv2.putText(
            img,
            "Gesture Detection: DEMO MODE",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        return img


# =========================================================
# WebRTC Stream
# =========================================================
webrtc_streamer(
    key="gesture-demo",
    video_transformer_factory=GestureDemo,
    media_stream_constraints={"video": True, "audio": False},
)

st.success("✅ Webcam active")

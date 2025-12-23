import os
import platform
import time
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# =========================================================
# Streamlit Setup
# =========================================================
st.set_page_config(
    page_title="🖐 Gesture Controller (Demo)",
    layout="centered"
)

st.title("🖐 Hand Gesture Detection (Demo Mode)")

st.markdown("""
### ℹ️ Note
- Streamlit Cloud **cannot run MediaPipe**
- This demo shows **camera-based hand movement detection**
- Run locally to enable **real gesture recognition**
""")

# =========================================================
# Video Transformer (Motion-based detection)
# =========================================================
class GestureDemoTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_gray = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

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

        detected = False

        for cnt in contours:
            if cv2.contourArea(cnt) < 3000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(
                img, (x, y), (x + w, y + h),
                (0, 255, 255), 2
            )
            detected = True

        self.prev_gray = gray

        # UI Overlay
        cv2.rectangle(img, (10, 10), (450, 130), (0, 0, 0), -1)

        if detected:
            cv2.putText(
                img,
                "Gesture Detected (Demo)",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                img,
                "No Gesture Detected",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.putText(
            img,
            "Cloud-safe demo (motion based)",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return img


# =========================================================
# Run WebRTC Stream
# =========================================================
webrtc_streamer(
    key="gesture-demo",
    video_transformer_factory=GestureDemoTransformer,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)

st.success("✅ Webcam active — move your hand in front of the camera")

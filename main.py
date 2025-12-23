import os
import platform
import subprocess
import time
from collections import deque
from statistics import mode

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# =========================================================
# Streamlit Setup
# =========================================================
st.set_page_config(page_title="🖐 Gesture Controller (WebRTC)", layout="centered")
st.title("🖐 Real-Time Hand Gesture Control (WebRTC Version)")

st.markdown("""
### 🎥 How to Use
- Allow webcam access when prompted.
- Move your hand into the camera view.
- Gestures:
  - ✋ Open Palm → Mute/Unmute  
  - ✊ Fist → Lock Screen  
  - 👍 Thumbs Up / 👎 Thumbs Down → Volume Up/Down  
  - 👉 / 👈 Point Right/Left → Switch Apps  
  - ☝ One Finger Up → Open File Explorer  
  - ✌ Two Fingers Up → Screenshot  
""")

# =========================================================
# Environment Detection
# =========================================================
RUNNING_LOCALLY = (
    platform.system() == "Windows"
    or os.environ.get("DISPLAY") is not None
)

if RUNNING_LOCALLY:
    import mediapipe as mp
    import pyautogui

    pyautogui.FAILSAFE = False

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )
    mp_draw = mp.solutions.drawing_utils

    st.success("🖥 Running locally — full gesture control enabled.")
else:
    st.warning("🌐 Running on Streamlit Cloud")
    st.info("Gesture detection is disabled. Run locally to enable hand tracking.")

    mp_hands = None
    hands = None
    mp_draw = None

# =========================================================
# State
# =========================================================
landmark_history = deque(maxlen=5)
gesture_history = deque(maxlen=8)
COOLDOWN_TIME = 1.0


# =========================================================
# Gesture Logic
# =========================================================
def perform_action(gesture):
    if not RUNNING_LOCALLY:
        return

    system = platform.system()

    if gesture == "OPEN_PALM":
        pyautogui.press("volumemute")

    elif gesture == "FIST":
        if system == "Windows":
            os.system("rundll32.exe user32.dll,LockWorkStation")
        elif system == "Darwin":
            subprocess.call([
                "/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession",
                "-suspend"
            ])
        elif system == "Linux":
            os.system("gnome-screensaver-command -l")

    elif gesture == "THUMBS_UP":
        pyautogui.press("volumeup")

    elif gesture == "THUMBS_DOWN":
        pyautogui.press("volumedown")

    elif gesture == "POINT_RIGHT":
        pyautogui.hotkey("alt", "tab")

    elif gesture == "POINT_LEFT":
        pyautogui.hotkey("alt", "shift", "tab")

    elif gesture == "ONE_FINGER_UP":
        if system == "Windows":
            os.system("explorer")
        elif system == "Darwin":
            subprocess.call(["open", "."])
        elif system == "Linux":
            subprocess.call(["xdg-open", "."])

    elif gesture == "TWO_FINGERS_UP":
        pyautogui.hotkey("win", "prtsc")


def smooth_landmarks(hand_landmarks):
    current = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    landmark_history.append(current)
    return np.mean(landmark_history, axis=0)


def fingers_up(landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    fingers = []
    fingers.append(landmarks[4][0] > landmarks[3][0])  # Thumb

    for tip, pip in zip(tips[1:], pips[1:]):
        fingers.append(landmarks[tip][1] < landmarks[pip][1])

    return fingers


def detect_gesture(landmarks, img_w, img_h):
    f = fingers_up(landmarks)
    count = sum(f)

    wrist = landmarks[0][:2] * [img_w, img_h]
    index_tip = landmarks[8][:2] * [img_w, img_h]
    index_mcp = landmarks[5][:2] * [img_w, img_h]
    pinky_mcp = landmarks[17][:2] * [img_w, img_h]

    hand_dir = index_mcp[0] - pinky_mcp[0]
    finger_dir = index_tip[0] - wrist[0]

    if count == 5:
        return "OPEN_PALM"
    if count == 0:
        return "FIST"
    if f[0] and not any(f[1:]):
        return "THUMBS_UP" if landmarks[4][1] < landmarks[3][1] else "THUMBS_DOWN"
    if f[1] and not any(f[2:]) and not f[0]:
        if finger_dir * hand_dir > 50:
            return "POINT_RIGHT"
        if finger_dir * hand_dir < -50:
            return "POINT_LEFT"
        return "ONE_FINGER_UP"
    if f[1] and f[2] and not any(f[3:]):
        return "TWO_FINGERS_UP"

    return "UNKNOWN"


def stable_gesture(gesture):
    gesture_history.append(gesture)
    try:
        dominant = mode(gesture_history)
    except:
        dominant = gesture

    confidence = gesture_history.count(dominant) / len(gesture_history)
    return dominant if confidence > 0.6 else "UNKNOWN", confidence


# =========================================================
# WebRTC Video Processing
# =========================================================
class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_action_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if not RUNNING_LOCALLY:
            cv2.putText(
                img,
                "Gesture detection disabled on Cloud",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            return img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        h, w, _ = img.shape
        gesture = "NONE"
        confidence = 0

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                smoothed = smooth_landmarks(hand_landmarks)
                detected = detect_gesture(smoothed, w, h)
                gesture, confidence = stable_gesture(detected)
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        if gesture not in ["UNKNOWN", "NONE"]:
            now = time.time()
            if now - self.last_action_time > COOLDOWN_TIME:
                perform_action(gesture)
                self.last_action_time = now

        color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255)
        cv2.rectangle(img, (10, 10), (420, 130), (0, 0, 0), -1)
        cv2.putText(img, f"Gesture: {gesture}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Confidence: {confidence:.2f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img


# =========================================================
# Run WebRTC Stream
# =========================================================
webrtc_streamer(
    key="gesture-controller",
    video_transformer_factory=HandGestureTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

st.success("✅ Webcam active")

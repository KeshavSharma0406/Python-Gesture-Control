import streamlit as st
import mediapipe as mp
import numpy as np
import time
from collections import deque
from statistics import mode
import os
import platform
import subprocess

# =========================================================
# Handle pyautogui safely (for local use)
# =========================================================
HEADLESS = os.environ.get("DISPLAY") is None
if not HEADLESS:
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
    except ImportError:
        pyautogui = None
else:
    class MockPyAutoGUI:
        def press(self, *args, **kwargs): print(f"[Mock] press {args}")
        def hotkey(self, *args, **kwargs): print(f"[Mock] hotkey {args}")
    pyautogui = MockPyAutoGUI()
    st.warning("⚠ Running in headless mode — system actions disabled.")

# =========================================================
# Mediapipe Hands
# =========================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="🖐 Gesture Controller (Cloud Mode)", layout="wide")
st.title("🖐 Hand Gesture Control — Streamlit Cloud Safe Mode")

run = st.checkbox("Start Demo", value=False)
frame_window = st.image([])

landmark_history = deque(maxlen=5)
gesture_history = deque(maxlen=8)
COOLDOWN_TIME = 1.0
last_action_time = 0

# =========================================================
# Helper functions
# =========================================================
def perform_action(gesture):
    print(f"🖐 Gesture Detected: {gesture}")
    if pyautogui is None:
        return
    if gesture == "OPEN_PALM":
        pyautogui.press("volumemute")
    elif gesture == "THUMBS_UP":
        pyautogui.press("volumeup")
    elif gesture == "THUMBS_DOWN":
        pyautogui.press("volumedown")

def stable_gesture(current_gesture):
    gesture_history.append(current_gesture)
    try:
        dominant = mode(gesture_history)
    except:
        dominant = current_gesture
    confidence = gesture_history.count(dominant) / len(gesture_history)
    if confidence > 0.6:
        return dominant, confidence
    return "UNKNOWN", confidence

# =========================================================
# Fake "demo" hand feed using a placeholder image
# =========================================================
if run:
    st.info("🖐 Running demo mode — no webcam available on Streamlit Cloud.")
    placeholder = np.zeros((360, 480, 3), dtype=np.uint8)
    color = (255, 255, 255)

    demo_gestures = ["OPEN_PALM", "FIST", "THUMBS_UP", "THUMBS_DOWN", "UNKNOWN"]
    idx = 0
    while run:
        # Simulate rotating gestures every few seconds
        gesture = demo_gestures[idx % len(demo_gestures)]
        idx += 1

        placeholder[:] = (0, 0, 0)
        text = f"Gesture: {gesture}"
        # Draw text manually
        import cv2
        cv2.putText(placeholder, text, (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
        cv2.putText(placeholder, "Demo Mode (no webcam)", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        frame_window.image(placeholder)
        time.sleep(1.5)
else:
    st.write("👋 Click 'Start Demo' to see simulated gesture control.")

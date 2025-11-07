import os
import sys
import time
import platform
import subprocess
import numpy as np
import streamlit as st
from collections import deque
from statistics import mode

# =========================================================
# Safe import for OpenCV
# =========================================================
try:
    import cv2
except ImportError:
    st.warning("⚠ Falling back to headless OpenCV.")
    os.system("pip install opencv-python-headless -q")
    import cv2

# =========================================================
# Safe import for mediapipe
# =========================================================
try:
    import mediapipe as mp
except ImportError:
    os.system("pip install mediapipe -q")
    import mediapipe as mp

# =========================================================
# Handle headless (no DISPLAY) environments safely
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
# Initialize MediaPipe Hands
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
st.set_page_config(page_title="🖐 Gesture Controller", layout="wide")
st.title("🖐 Real-Time Hand Gesture Control")

run = st.checkbox("Start Gesture Control", value=False)
frame_window = st.image([])

system = platform.system()
landmark_history = deque(maxlen=5)
gesture_history = deque(maxlen=8)
last_action_time = 0
COOLDOWN_TIME = 1.0

# =========================================================
# Helper functions
# =========================================================
def perform_action(gesture):
    print(f"\n🖐 Gesture Detected: {gesture}")
    if pyautogui is None:
        print("⚠ PyAutoGUI unavailable — skipping.")
        return
    if gesture == "OPEN_PALM":
        pyautogui.press("volumemute")
    elif gesture == "THUMBS_UP":
        pyautogui.press("volumeup")
    elif gesture == "THUMBS_DOWN":
        pyautogui.press("volumedown")

def smooth_landmarks(hand_landmarks):
    current = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    landmark_history.append(current)
    return np.mean(landmark_history, axis=0)

def fingers_up(landmarks):
    tips, pips = [4, 8, 12, 16, 20], [3, 6, 10, 14, 18]
    fingers = [landmarks[4][0] > landmarks[3][0]]
    for tip, pip in zip(tips[1:], pips[1:]):
        fingers.append(landmarks[tip][1] < landmarks[pip][1])
    return fingers

def detect_gesture(landmarks, img_w, img_h):
    f = fingers_up(landmarks)
    count = sum(f)
    if count == 5:
        return "OPEN_PALM", f
    elif count == 0:
        return "FIST", f
    elif f[0] and not any(f[1:]):
        if landmarks[4][1] < landmarks[3][1]:
            return "THUMBS_UP", f
        else:
            return "THUMBS_DOWN", f
    else:
        return "UNKNOWN", f

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
# Main Loop
# =========================================================
if run:
    st.info("🖐 Starting gesture detection...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.warning("⚠ Webcam not found. Displaying test image instead.")
        test_img = np.zeros((300, 400, 3), np.uint8)
        cv2.putText(test_img, "No Webcam", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        frame_window.image(test_img)
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Could not read frame.")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            h, w, _ = frame.shape
            gesture, confidence = "NONE", 0

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    smoothed = smooth_landmarks(hand_landmarks)
                    detected, _ = detect_gesture(smoothed, w, h)
                    gesture, confidence = stable_gesture(detected)
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
else:
    st.write("👋 Click the checkbox to start gesture detection.")

import cv2
import mediapipe as mp
import pyautogui
import os
import time
import platform
import subprocess
import numpy as np
from collections import deque
from statistics import mode

# Disable PyAutoGUI failsafe (so corners of screen won't trigger exceptions)
pyautogui.FAILSAFE = False

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils
import mediapipe as mp
import pyautogui
import os
import time
import platform
import subprocess
import numpy as np
from collections import deque
from statistics import mode

# Disable PyAutoGUI failsafe (prevents exceptions when moving cursor to corners)
pyautogui.FAILSAFE = False

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# === Webcam ===
cap = cv2.VideoCapture(0)

# === Memory for smoothing and stability ===
landmark_history = deque(maxlen=5)
gesture_history = deque(maxlen=8)
last_gesture = None
last_action_time = 0

# System detection
system = platform.system()

# Cooldown between actions (seconds)
COOLDOWN_TIME = 1.0


# -------------------------------------------------------------
#  Perform System Action
# -------------------------------------------------------------
def perform_action(gesture):
    print(f"\n🖐 Gesture Detected: {gesture}")

    if gesture == "OPEN_PALM":
        pyautogui.press("volumemute")
        print("🔇 Mute/Unmute Audio")

    elif gesture == "FIST":
        print("🔒 Locking Screen")
        if system == "Windows":
            os.system("rundll32.exe user32.dll,LockWorkStation")
        elif system == "Darwin":
            subprocess.call([
                '/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession', '-suspend'
            ])
        elif system == "Linux":
            os.system("gnome-screensaver-command -l")

    elif gesture == "THUMBS_UP":
        pyautogui.press("volumeup")
        print("🔊 Volume Up")

    elif gesture == "THUMBS_DOWN":
        pyautogui.press("volumedown")
        print("🔉 Volume Down")

    elif gesture == "POINT_RIGHT":
        pyautogui.hotkey("alt", "tab")
        print("➡ Switch to Next App")

    elif gesture == "POINT_LEFT":
        pyautogui.hotkey("alt", "shift", "tab")
        print("⬅ Switch to Previous App")

    elif gesture == "ONE_FINGER_UP":
        print("🗂 Opening File Explorer")
        if system == "Windows":
            os.system("explorer")
        elif system == "Darwin":
            subprocess.call(["open", "."])
        elif system == "Linux":
            subprocess.call(["xdg-open", "."])

    elif gesture == "TWO_FINGERS_UP":
        pyautogui.hotkey("win", "prtsc")
        print("📸 Screenshot Taken")


# -------------------------------------------------------------
#  Landmark Smoothing
# -------------------------------------------------------------
def smooth_landmarks(hand_landmarks):
    current = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    landmark_history.append(current)
    avg = np.mean(landmark_history, axis=0)
    return avg


# -------------------------------------------------------------
#  Finger Detection
# -------------------------------------------------------------
def fingers_up(landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    fingers = []

    # Thumb (horizontal movement)
    fingers.append(landmarks[4][0] > landmarks[3][0])

    # Other fingers (vertical)
    for tip, pip in zip(tips[1:], pips[1:]):
        fingers.append(landmarks[tip][1] < landmarks[pip][1])
    return fingers


# -------------------------------------------------------------
#  Gesture Recognition (orientation-aware)
# -------------------------------------------------------------
def detect_gesture(landmarks, img_w, img_h):
    f = fingers_up(landmarks)
    count = sum(f)

    wrist = landmarks[0][:2] * [img_w, img_h]
    index_tip = landmarks[8][:2] * [img_w, img_h]
    index_mcp = landmarks[5][:2] * [img_w, img_h]
    pinky_mcp = landmarks[17][:2] * [img_w, img_h]

    # Hand and finger orientation
    hand_dir = index_mcp[0] - pinky_mcp[0]
    finger_dir = index_tip[0] - wrist[0]

    # Gesture mapping
    if count == 5:
        return "OPEN_PALM", f
    elif count == 0:
        return "FIST", f
    elif f[0] and not any(f[1:]):
        if landmarks[4][1] < landmarks[3][1]:
            return "THUMBS_UP", f
        else:
            return "THUMBS_DOWN", f
    elif f[1] and not any(f[2:]) and not f[0]:
        if finger_dir * hand_dir > 50:
            return "POINT_RIGHT", f
        elif finger_dir * hand_dir < -50:
            return "POINT_LEFT", f
        else:
            return "ONE_FINGER_UP", f
    elif f[1] and f[2] and not any(f[3:]):
        return "TWO_FINGERS_UP", f
    else:
        return "UNKNOWN", f


# -------------------------------------------------------------
#  Gesture Stabilization
# -------------------------------------------------------------
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


# -------------------------------------------------------------
#  Main Loop
# -------------------------------------------------------------
print("🖐 Gesture Controller Running — Press 'Q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    h, w, _ = frame.shape
    gesture = "NONE"
    confidence = 0
    fingers = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            smoothed = smooth_landmarks(hand_landmarks)
            detected, fingers = detect_gesture(smoothed, w, h)
            gesture, confidence = stable_gesture(detected)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw debug arrows
            wrist = smoothed[0][:2] * [w, h]
            index_tip = smoothed[8][:2] * [w, h]
            index_mcp = smoothed[5][:2] * [w, h]
            pinky_mcp = smoothed[17][:2] * [w, h]
            cv2.arrowedLine(frame, tuple(pinky_mcp.astype(int)), tuple(index_mcp.astype(int)), (255, 0, 0), 3)
            cv2.arrowedLine(frame, tuple(wrist.astype(int)), tuple(index_tip.astype(int)), (0, 255, 0), 3)

    current_time = time.time()

    # Allow same gesture again after cooldown
    if gesture not in ["UNKNOWN", "NONE"]:
        if current_time - last_action_time > COOLDOWN_TIME:
            perform_action(gesture)
            last_gesture = gesture
            last_action_time = current_time

    # Draw overlay info
    color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255)
    cv2.rectangle(frame, (10, 10), (420, 140), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture: {gesture}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if fingers:
        cv2.putText(frame, f"Fingers Up: {sum(fingers)}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("🖐 Gesture-Based System Controller (Final Version)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exited Gesture Control.")


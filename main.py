import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp

st.set_page_config(page_title="🖐 Gesture Recognition Demo", layout="centered")
st.title("🖐 Real-Time Hand Gesture Recognition (WebRTC)")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# Define a custom video transformer class
class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def transform(self, frame):
        # Convert the incoming video frame to a numpy array
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)

        # Draw landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Flip horizontally for better UX
        img = cv2.flip(img, 1)
        return img


# Stream the webcam feed via browser
webrtc_streamer(
    key="gesture-demo",
    video_transformer_factory=HandGestureTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("""
### ✋ Instructions:
- Allow webcam access when prompted.
- Move your hand in front of the camera — landmarks will appear in real time.
- This works fully in your browser — no installation needed.
""")

import cv2
from deepface import DeepFace
import streamlit as st

def run_face_detection(st):
    st.subheader("📸 Facial Emotion Detection")
    start = st.button("Start Face Detection")

    webcam_placeholder = st.empty()
    emotion_placeholder = st.empty()

    if start:
        cap = cv2.VideoCapture(0)
        st.success("Detecting facial emotion...")

        stop = st.button("Stop Detection")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                cv2.putText(frame, emotion, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                webcam_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                emotion_placeholder.markdown(f"### 📸 The detected facial emotion is: **{emotion.capitalize()}**")
            except:
                emotion_placeholder.markdown("😐 Detecting...")

            # Optional: Add a break condition if stop is pressed
            if stop:
                cap.release()
                break

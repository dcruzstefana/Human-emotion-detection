import streamlit as st
from voice_emotion import run_voice_detection
from face_emotion import run_face_detection

st.set_page_config(page_title="Real-Time Emotion Detector")
st.title("🧠 Real-Time Emotion Detector")
st.write("Choose input mode to start detecting emotions in real-time.")

mode = st.radio("Select Mode", ["🎤 Voice Only", "📸 Face Only", "🧠 Combined"])

if mode == "🎤 Voice Only":
    run_voice_detection()
elif mode == "📸 Face Only":
    run_face_detection(st)
elif mode == "🧠 Combined":
    run_voice_detection()
    run_face_detection(st)

import sounddevice as sd
import numpy as np
import joblib
import librosa
import streamlit as st
import pandas as pd
import queue
import datetime
import plotly.express as px

model = joblib.load("voice_model.pkl")
scaler = joblib.load("scaler.pkl")

emotion_q = queue.Queue()

# Track session data
if 'voice_emotions' not in st.session_state:
    st.session_state.voice_emotions = []

def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    return np.hstack((
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0)
    ))

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio = indata[:, 0]
    features = extract_features(audio, 44100)
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    emotion_q.put((timestamp, pred))

def run_voice_detection():
    st.subheader("🎤 Real-Time Voice Emotion Detection")

    if 'listening' not in st.session_state:
        st.session_state.listening = False

    if not st.session_state.listening:
        if st.button("Start Voice Detection"):
            st.session_state.listening = True
    else:
        if st.button("Stop Voice Detection"):
            st.session_state.listening = False

    placeholder = st.empty()
    graph_placeholder = st.empty()
    download_placeholder = st.empty()

    if st.session_state.listening:
        st.success("Listening... Speak into the mic 🎙️")
        try:
            with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100, blocksize=22050):
                while st.session_state.listening:
                    try:
                        timestamp, pred = emotion_q.get(timeout=1)
                        placeholder.markdown(f"**Voice Emotion:** _{pred}_")
                        st.session_state.voice_emotions.append({"Time": timestamp, "Emotion": pred})

                        # Update graph
                        df = pd.DataFrame(st.session_state.voice_emotions)
                        fig = px.line(df, x="Time", y="Emotion", title="Emotional Trend Over Time")
                        graph_placeholder.plotly_chart(fig, use_container_width=True)
                    except queue.Empty:
                        pass
        except Exception as e:
            st.error(f"Microphone error: {e}")

    # Downloadable session data
    if st.session_state.voice_emotions:
        df = pd.DataFrame(st.session_state.voice_emotions)
        csv = df.to_csv(index=False).encode('utf-8')
        download_placeholder.download_button("Download Session Data", data=csv, file_name="voice_emotion_session.csv", mime="text/csv")

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Path to your dataset folder (update this!)
DATA_PATH = "Audio_Speech_Actors_01-24"  # Replace with the path to your dataset

# Emotions (modify this according to your dataset)
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    return np.hstack((np.mean(mfccs.T, axis=0),
                      np.mean(chroma.T, axis=0),
                      np.mean(mel.T, axis=0)))

# Load and extract features
X, y = [], []
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotion_map.get(emotion_code)
            if emotion:
                features = extract_features(os.path.join(root, file))
                X.append(features)
                y.append(emotion)

# Prepare data
X = np.array(X)
y = np.array(y)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "voice_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")

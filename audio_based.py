import os
import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

# 📍 Dataset directory
dataset_path = "D:\Emotion Detection\Emotion Detection\Data\Tess"

# Clear stale recording
if os.path.exists("temp.wav") and "audio_uploaded" not in st.session_state:
    os.remove("temp.wav")

# 🎯 Feature Extraction
def extract_features(file_path, duration=5, sample_rate=22050):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs
    except:
        return None

# 🧠 Train Model
def train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, "label_encoder.pkl")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    joblib.dump(model, "xgb_emotion_model.pkl")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)



    

    
# 🚀 Streamlit App

# 🚦 Start Training
if st.button("🚀 Start Training", key="train_button"):
    if not os.path.exists(dataset_path):
        st.error("❌ Dataset not found!")
    else:
        st.info("🔍 Extracting Features from WAV Files...")
        features = []
        labels = []

        class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        progress_bar = st.progress(0)
        status = st.empty()
        total_files = sum(len(files) for _, _, files in os.walk(dataset_path))
        processed = 0

        for label_dir in class_dirs:
            sub_path = os.path.join(dataset_path, label_dir)
            for file in os.listdir(sub_path):
                if file.endswith(".wav"):
                    path = os.path.join(sub_path, file)
                    feature = extract_features(path, duration=5)
                    if feature is not None:
                        features.append(feature)
                        emotion_label = label_dir.split("_")[-1].lower().replace("pleasant", "surprise").replace("surprised", "surprise")
                        labels.append(emotion_label)
                    processed += 1
                    progress_bar.progress(min(processed / total_files, 1.0))
                    status.text(f"Processed {processed}/{total_files} files...")

        st.success("✅ Feature Extraction Completed.")
        model = train_model(features, labels)

        st.session_state.model_trained = True
        st.session_state.audio_uploaded = False

# ✅ Prediction Section
if "model_trained" in st.session_state and st.session_state.model_trained:
    sample_rate = 22050
    file_added = False

    # 📂 Upload Button (only this part is kept)
    uploaded = st.file_uploader("📂 Upload a WAV File", type=["wav"])
    if uploaded is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded.read())
        st.success("✅ File uploaded successfully.")
        file_added = True
        st.session_state.audio_uploaded = True

    if file_added and os.path.exists("temp.wav"):
        feature = extract_features("temp.wav", duration=duration, sample_rate=sample_rate)
        if feature is not None:
            try:
                model = joblib.load("xgb_emotion_model.pkl")
                encoder = joblib.load("label_encoderr.pkl")
                scaler = joblib.load("scalerr.pkl")
                feature_scaled = scaler.transform([feature])
                pred = model.predict(feature_scaled)
                emotion = encoder.inverse_transform(pred)[0]
                st.success(f"🎭 Detected Emotion: **{emotion.upper()}**")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
        else:
            st.error("❌ Could not extract features from audio.")
    elif not file_added:
        st.warning("⚠️ Please upload an audio file before prediction.")

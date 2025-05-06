import streamlit as st
import joblib
import librosa
import numpy as np
import speech_recognition as sr
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
import os
from PIL import Image                                        
import pickle

# Load trained models and encoders
@st.cache_resource
def load_models():
    log_model = joblib.load(r"D:\SER\logistic_text_emotion.pkl")
    xgb_model = joblib.load(r"D:\SER\xgb_speech_emotion.pkl")
    vectorizer = joblib.load(r"D:\SER\text_emotion\tfidf_vectorizer.pkl")
    scaler = joblib.load(r"D:\SER\text_emotion\scaler.pkl")
    label_encoder_text = joblib.load(r"D:\SER\label_encoder_logistic.pkl")
    label_encoder_speech = joblib.load(r"D:\SER\text_emotion\label_encoder.pkl")
    return log_model, xgb_model, vectorizer, scaler, label_encoder_text, label_encoder_speech

log_model, xgb_model, vectorizer, scaler, label_encoder_text, label_encoder_speech = load_models()

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Predict emotion from text with keyword correction
def predict_text_emotion(text):
    text_cleaned = clean_text(text)
    text_vectorized = vectorizer.transform([text_cleaned]).toarray()  # Remove scaler.transform here
    emotion_index = log_model.predict(text_vectorized)[0]
    predicted_emotion = label_encoder_text.inverse_transform([emotion_index])[0]

    # Debug print
    print(f"Text: {text} | Cleaned: {text_cleaned} | log Prediction: {predicted_emotion}")

    # Keyword-based override
    keywords = {
        "happy": ["happy", "glad", "joy", "excited", "pleased"],
        "sad": ["sad", "unhappy", "depressed", "down"],
        "angry": ["angry", "mad", "furious", "annoyed"],
        "neutral": ["okay", "fine", "normal"],
        "surprise": ["surprised", "shocked", "amazed"]
    }

    for emotion, words in keywords.items():
        if any(word in text_cleaned for word in words):
            print(f"Keyword match found. Overriding with: {emotion}")
            return emotion

    return predicted_emotion


# Extract features from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Predict emotion from speech
def predict_speech_emotion(audio_data):
    speech_features = extract_features(audio_data).reshape(1, -1)
    emotion_index = xgb_model.predict(speech_features)[0]
    emotion = label_encoder_speech.inverse_transform([emotion_index])[0]
    return emotion

# Convert speech to text
def speech_to_text(audio_data):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_data) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

# Combine emotions and calculate percentages
def get_emotion_probabilities(text_emotion, speech_emotion):
    emotions = [text_emotion, speech_emotion]
    counts = Counter(emotions)
    total = sum(counts.values())
    prob_text = (counts[text_emotion] / total) * 100
    prob_speech = (counts[speech_emotion] / total) * 100
    return f"{text_emotion}: {prob_text:.1f}%, {speech_emotion}: {prob_speech:.1f}%"

# Decide final emotion
def get_final_emotion(text_emotion, speech_emotion):
    if text_emotion.lower() == speech_emotion.lower():
        return text_emotion
    elif "happy" in [text_emotion.lower(), speech_emotion.lower()]:
        return "happy"
    else:
        return speech_emotion  # fallback to speech tone

# Custom styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #4CAF50;
            font-size: 40px;
            font-family: 'Arial', sans-serif;
        }
        .subheader {
            font-size: 25px;
            color: #2196F3;
            font-weight: bold;
            padding-top: 20px;
        }
        .container {
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .button:hover {
            background-color: #45a049;
        }
        .success {
            color: #28a745;
        }
        .warning {
            color: #ff9800;
        }
        .info {
            color: #2196F3;
        }
        .slider {
            width: 100%;
        }
        .stApp{
            background-color:white;
            }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="title">🎭 Real-Time Emotion Recognition from Text & Speech</div>', unsafe_allow_html=True)

# Instructions
st.markdown("""
    
""", unsafe_allow_html=True)

# Real-time capture and prediction
def capture_and_predict(duration):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        st.info(f"🎙 Listening for {duration} seconds...")

        try:
            audio_data = recognizer.listen(source, timeout=duration)
            st.audio(audio_data.get_wav_data(), format="audio/wav")

            with open("temp_audio.wav", "wb") as f:
                f.write(audio_data.get_wav_data())

            text_emotion = None
            speech_emotion = None

            # Step 1: Speech-to-text
            speech_text = speech_to_text("temp_audio.wav")
            if speech_text:
                st.write(f"**Transcribed Text**: {speech_text}")
                text_emotion = predict_text_emotion(speech_text)
                st.success(f"**Predicted Emotion from Text**: {text_emotion}")
            else:
                st.warning("🛑 Unable to transcribe speech. Try speaking clearly.")

            # Step 2: Speech-based emotion
            try:
                speech_emotion = predict_speech_emotion("temp_audio.wav")
                st.success(f"**Predicted Emotion from Speech**: {speech_emotion}")
            except Exception as e:
                st.warning(f"❌ Error in speech emotion prediction: {str(e)}")

            # Step 3: Combined emotion result
            if text_emotion and speech_emotion:
                result = get_emotion_probabilities(text_emotion, speech_emotion)
                final = get_final_emotion(text_emotion, speech_emotion)
                st.success(f"🎯 **Combined Emotion Analysis**: {result} → Final: **{final.upper()}**")
            elif text_emotion:
                st.info(f"📘 Only Text Emotion Detected: **{text_emotion}**")
            elif speech_emotion:
                st.info(f"🔊 Only Speech Emotion Detected: **{speech_emotion}**")
            else:
                st.warning("⚠️ No emotion detected from speech or text.")

        except sr.WaitTimeoutError:
            st.warning(f"⌛ No speech detected in {duration} seconds.")
        except sr.UnknownValueError:
            st.warning("🤷 Could not understand the audio.")
        except sr.RequestError:
            st.warning("🚫 Speech recognition service error.")
        finally:
            # Clean-up
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
            st.info("✅ Ready for the next recording. Click the button again to start!")

# Slider to set duration

# White label above the slider
st.markdown('<p style="color: black; font-size: 18px;">Select recording duration (seconds)</p>', unsafe_allow_html=True)

# Custom style for white slider numbers (3, 10, 5)
st.markdown("""
    <style>
    .stSlider > div[data-baseweb="slider"] span {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# Slider itself (with no default label)
duration = st.slider("", min_value=3, max_value=10, value=5)


# Start button
if st.button("Start Recording", key="start_button"):
    capture_and_predict(duration)
    


st.subheader("Confusion Matrix of Speech Emotion")# Subheading
st.image("speech.png")
st.subheader("Classification Report of Speech Emotion")# Subheading
st.image("xclass.png")
st.subheader("Confusion Matrix of Text Emotion")# Subheading
st.image("textclass.png")
st.subheader("Classification Report of Text Emotion")# Subheading
st.image("textc.png")



import streamlit as st

# 🧠 Run the full audio_based.py file
with open("audio_based.py", "r", encoding="utf-8") as f:
    exec(f.read(), globals())

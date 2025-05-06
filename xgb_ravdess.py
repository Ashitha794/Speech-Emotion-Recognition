import os
import librosa
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score


# Function to extract MFCC features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=5)  # Load 5s of audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCCs
    return np.mean(mfcc.T, axis=0)  # Compute mean across time steps


# Function for parallel extraction using multiprocessing
def extract_features_parallel(file_path):
    return extract_features(file_path)


# Main code execution wrapped in a function
def main():
    # Dataset Path (Update with actual path)
    DATASET_PATH = r"D:\SER\text_emotion\archive (1)"

    # Emotion labels based on RAVDESS filenames
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    # Collecting audio files and corresponding labels
    audio_files = []
    labels = []

    for root, dirs, files in os.walk(DATASET_PATH):
        for file in tqdm(files):
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion_code = file.split("-")[2]  # Emotion is the 3rd part in filename
                if emotion_code in emotion_map:
                    audio_files.append(file_path)
                    labels.append(emotion_map[emotion_code])

    # Using ProcessPoolExecutor for parallel feature extraction
    with ProcessPoolExecutor() as executor:
        X_audio = np.array(list(executor.map(extract_features_parallel, audio_files)))
    
    y_audio = np.array(labels)

    # Encoding labels
    label_encoder = LabelEncoder()
    y_audio_encoded = label_encoder.fit_transform(y_audio)

    # Save the label encoder
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # Scaling features
    scaler = StandardScaler()
    X_audio_scaled = scaler.fit_transform(X_audio)

    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')

    # Balancing dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_audio_resampled, y_audio_resampled = smote.fit_resample(X_audio_scaled, y_audio_encoded)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_audio_resampled, y_audio_resampled, test_size=0.2, random_state=42)

    # XGBoost Model
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    # Training the model
    xgb_model.fit(X_train, y_train)

    # Saving the trained model
    joblib.dump(xgb_model, 'xgb_speech_emotion.pkl')

    # Model evaluation
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"XGBoost Speech Emotion Accuracy: {accuracy:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Speech Emotion Recognition")
    plt.show()

# Entry point for script execution
if __name__ == '__main__':
    main()
    # Optional: Test a known angry file from RAVDESS
    test_file = r"D:\SER\text_emotion\archive (1)\Actor_11\03-01-02-02-02-01-11.wav"

    # Load label encoder and scaler (already saved earlier)
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    # Extract features and scale
    test_features = extract_features(test_file).reshape(1, -1)
    test_features_scaled = scaler.transform(test_features)

    # Load model and predict
    model = joblib.load('xgb_speech_emotion.pkl')
    predicted_label = model.predict(test_features_scaled)[0]
    predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

    print(f"\n🔊 Testing File: {os.path.basename(test_file)}")
    print(f"🎯 Predicted Emotion: {predicted_emotion}")

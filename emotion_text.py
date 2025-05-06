from sklearn.svm import LinearSVC  # Use LinearSVC for faster training
from sklearn.decomposition import TruncatedSVD  # For dimensionality reduction
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report


# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load datasets
train_df = pd.read_csv(r"D:\text_emotion\text_emotion\archive\training.csv")
test_df = pd.read_csv(r"D:\text_emotion\text_emotion\archive\test.csv")
val_df = pd.read_csv(r"D:\text_emotion\text_emotion\archive\validation.csv")

# Label mapping
label_mapping = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Neutral", 4: "Excited", 5: "Fear"}
train_df['label'] = train_df['label'].map(label_mapping)
test_df['label'] = test_df['label'].map(label_mapping)
val_df['label'] = val_df['label'].map(label_mapping)

# Clean the text data
train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)
val_df['text'] = val_df['text'].apply(clean_text)

# Label encoding
label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])
test_df['encoded_label'] = label_encoder.transform(test_df['label'])
val_df['encoded_label'] = label_encoder.transform(val_df['label'])
joblib.dump(label_encoder, 'label_encoder_text.pkl')

# Vectorization with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words=stopwords.words('english'))
X_train = vectorizer.fit_transform(train_df['text']).toarray()
X_test = vectorizer.transform(test_df['text']).toarray()
X_val = vectorizer.transform(val_df['text']).toarray()
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

y_train = train_df['encoded_label']
y_test = test_df['encoded_label']
y_val = val_df['encoded_label']

# Apply SMOTE for balancing the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize the features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
joblib.dump(scaler, 'scaler.pkl')

# Dimensionality reduction with TruncatedSVD
svd = TruncatedSVD(n_components=500)  # Reduce to 500 components
X_train_svd = svd.fit_transform(X_train_resampled)
X_test_svd = svd.transform(X_test)

# Train the SVM model
svm_model = LinearSVC(class_weight='balanced')
svm_model.fit(X_train_svd, y_train_resampled)
y_pred = svm_model.predict(X_test_svd)

# Evaluate the model
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(svm_model, 'svm_text_emotion.pkl')
joblib.dump(svd, 'svd_model.pkl')  # Save the SVD model

# Predict text emotion function
def predict_text_emotion(text):
    try:
        # Load necessary files
        svm_model = joblib.load('svm_text_emotion.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder_text.pkl')
        svd = joblib.load('svd_model.pkl')  # Load the SVD model

        # Clean and vectorize the input text
        text_cleaned = clean_text(text)
        text_vectorized = vectorizer.transform([text_cleaned]).toarray()
        text_vectorized = scaler.transform(text_vectorized)
        text_svd = svd.transform(text_vectorized)

        # Predict emotion
        emotion_index = svm_model.predict(text_svd)[0]
        emotion = label_encoder.inverse_transform([emotion_index])[0]

        print("\nPredicted Emotion:", emotion)
        return emotion
    
    except Exception as e:
        print("Error in prediction:", e)
        return None

# Test the prediction function
predict_text_emotion("I am very proud")

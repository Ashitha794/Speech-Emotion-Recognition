import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
import logging

# Setting up logging to track the process
logging.basicConfig(filename='app.log', level=logging.INFO)
logging.info("Starting model training process...")

# Start measuring time
start_time = time.time()

# Download NLTK resources
logging.info("Downloading NLTK resources...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
logging.info("NLTK resources downloaded successfully!")

# Load the datasets
logging.info("Loading datasets...")
file_paths = [
    r"D:\text_emotion - Copy - Copy\text_emotion\archive\goemotions_1.csv",
    r"D:\text_emotion - Copy - Copy\text_emotion\archive\goemotions_2.csv",
    r"D:\text_emotion - Copy - Copy\text_emotion\archive\goemotions_3.csv"
]

try:
    dfs = [pd.read_csv(fp) for fp in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    logging.info("Datasets loaded successfully!")
except Exception as e:
    logging.error(f"Error loading datasets: {e}")
    raise

# Filter to 5 selected emotions
selected_emotions = ['joy', 'sadness', 'anger', 'fear', 'neutral']
df['label'] = df[selected_emotions].idxmax(axis=1)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single one
    return text

# Lemmatization function
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

logging.info("Cleaning and lemmatizing text data...")
try:
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['lemmatized_text'] = df['cleaned_text'].apply(lemmatize_text)
    logging.info("Text cleaning and lemmatization completed!")
except Exception as e:
    logging.error(f"Error during text cleaning and lemmatization: {e}")
    raise

# Encode labels
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])

# TF-IDF vectorization (with fewer features)
logging.info("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=500, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(df['lemmatized_text']).toarray()
y = df['encoded_label']

# Train-test split
logging.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for oversampling the minority class
logging.info("Applying SMOTE for oversampling...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning for Logistic Regression with class weighting
param_grid = {
    'C': [0.1, 1],  # Reduced grid
    'solver': ['liblinear'],  # Use only one solver for faster computation
    'penalty': ['l2']  # Only L2 penalty for simplicity
}
logging.info("Starting grid search for hyperparameter tuning...")
try:
    grid_search = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=500), param_grid, cv=5, verbose=1, n_jobs=-1)  # Parallel processing
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_lr_model = grid_search.best_estimator_
    logging.info("Grid search completed successfully!")
except Exception as e:
    logging.error(f"Error during grid search: {e}")
    raise

# Train Logistic Regression
logging.info("Training Logistic Regression model...")
try:
    best_lr_model.fit(X_train_resampled, y_train_resampled)
    logging.info("Model trained successfully!")
except Exception as e:
    logging.error(f"Error during model training: {e}")
    raise

# Predict with Logistic Regression
logging.info("Predicting with Logistic Regression...")
y_pred_lr = best_lr_model.predict(X_test)
logging.info(f"Tuned Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
logging.info("\nClassification Report (Logistic Regression):\n" + classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))

# ----------------------------------
# Predicting emotion from new input
# ----------------------------------
def predict_text_emotion(text):
    logging.info(f"Predicting emotion for input: {text}")
    text = clean_text(text)
    text = lemmatize_text(text)
    vector = vectorizer.transform([text]).toarray()
    prediction = best_lr_model.predict(vector)
    emotion = label_encoder.inverse_transform(prediction)
    print(f"Predicted Emotion: {emotion[0]}")

# ----------------------------------
# Test the prediction function
# ----------------------------------
try:
    print("Testing prediction function:")
    predict_text_emotion("I am very proud")   # Should ideally predict 'joy' or 'neutral'
    predict_text_emotion("I feel so sad and hopeless")   # Should predict 'sadness'
    predict_text_emotion("That’s scary and makes me anxious")   # Should predict 'fear'
    predict_text_emotion("I am happy today")   # Should predict 'joy'
    predict_text_emotion("I don't care anymore")   # Should predict 'neutral'
except Exception as e:
    logging.error(f"Error during prediction: {e}")
    print(f"Error: {e}")

# End measuring time
end_time = time.time()
total_time = end_time - start_time
logging.info(f"Total Runtime: {total_time:.2f} seconds")

# Print total runtime
print(f"\n⏳ Total Runtime: {total_time:.2f} seconds")
import joblib
import os

# Save models to D:\text_emotion - Copy - Copy
save_dir = r"D:\text_emotion - Copy - Copy"
os.makedirs(save_dir, exist_ok=True)

# Save Logistic Regression model
joblib.dump(best_lr_model, os.path.join(save_dir, "logistic_text_emotion.pkl"))
print("✅ Logistic Regression model saved as 'logistic_text_emotion.pkl'")

# Save TF-IDF Vectorizer
joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer.pkl"))
print("✅ TF-IDF Vectorizer saved as 'tfidf_vectorizer.pkl'")

# Save Label Encoder
joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder_logistic.pkl"))
print("✅ Label Encoder saved as 'label_encoder_logistic.pkl'")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Print classification report to console
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_lr)
labels = label_encoder.classes_

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()

# Save the confusion matrix
conf_matrix_path = os.path.join(save_dir, "confusion_matrix_logistic_text_emotion.png")
plt.savefig(conf_matrix_path)
plt.show()

print(f"✅ Confusion matrix saved to: {conf_matrix_path}")

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize labels
y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
n_classes = len(label_encoder.classes_)
y_score = best_lr_model.predict_proba(X_test)

# Compute ROC curve and AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression (Text Emotion)')
plt.legend(loc='lower right')
plt.grid(True)

# Save ROC Curve
roc_path = os.path.join(save_dir, "roc_curve_logistic_text_emotion.png")
plt.savefig(roc_path)
plt.show()
print(f"✅ ROC Curve saved to: {roc_path}")

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize labels
y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
n_classes = len(label_encoder.classes_)
y_score = best_lr_model.predict_proba(X_test)

# Compute ROC curve and AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression (Text Emotion)')
plt.legend(loc='lower right')
plt.grid(True)

# Save ROC Curve
roc_path = os.path.join(save_dir, "roc_curve_logistic_text_emotion.png")
plt.savefig(roc_path)
plt.show()
print(f"✅ ROC Curve saved to: {roc_path}")


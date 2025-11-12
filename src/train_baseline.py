# src/train_baseline.py
# Baseline: TF-IDF + Logistic Regression for Emotion Detection

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset

# Import preprocessing
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import preprocess_tweet

# --------------- CONFIG ---------------
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------- LOAD DATA ---------------
print("Loading TweetEval (emotion)...")
dataset = load_dataset("tweet_eval", "emotion")
train_df = pd.DataFrame(dataset["train"])
val_df   = pd.DataFrame(dataset["validation"])
test_df  = pd.DataFrame(dataset["test"])
label_names = dataset["train"].features["label"].names

print("Classes:", label_names)

# --------------- PREPROCESS ---------------
def preprocess_series(series):
    return series.apply(preprocess_tweet)

print("Preprocessing text...")
train_df["text_clean"] = preprocess_series(train_df["text"])
val_df["text_clean"]   = preprocess_series(val_df["text"])
test_df["text_clean"]  = preprocess_series(test_df["text"])

# --------------- TF-IDF FEATURES ---------------
print("Building TF-IDF features...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    max_df=0.95,
)
X_train = tfidf.fit_transform(train_df["text_clean"])
X_val   = tfidf.transform(val_df["text_clean"])
X_test  = tfidf.transform(test_df["text_clean"])

y_train = train_df["label"]
y_val   = val_df["label"]
y_test  = test_df["label"]

# --------------- TRAIN MODEL ---------------
print("Training Logistic Regression...")
model = LogisticRegression(
    max_iter=200,
    C=2.0,
    class_weight="balanced",
    solver="liblinear"
)
model.fit(X_train, y_train)

# --------------- EVALUATE ---------------
print("\nValidation performance:")
y_pred_val = model.predict(X_val)
print(classification_report(y_val, y_pred_val, target_names=label_names))

print("Confusion matrix (validation):")
print(confusion_matrix(y_val, y_pred_val))

print("\nTest performance:")
y_pred_test = model.predict(X_test)
print(classification_report(y_test, y_pred_test, target_names=label_names))

# --------------- SAVE MODEL ---------------
model_path = os.path.join(SAVE_DIR, "logreg_model.joblib")
tfidf_path = os.path.join(SAVE_DIR, "tfidf_vectorizer.joblib")
joblib.dump(model, model_path)
joblib.dump(tfidf, tfidf_path)
print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Vectorizer saved to: {tfidf_path}")

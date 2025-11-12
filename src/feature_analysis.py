import joblib, numpy as np, pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/logreg_model.joblib")
VEC_PATH = Path("models/tfidf_vectorizer.joblib")

clf = joblib.load(MODEL_PATH)
vec = joblib.load(VEC_PATH)
feature_names = np.array(vec.get_feature_names_out())

for i, label in enumerate(clf.classes_):
    topn = np.argsort(clf.coef_[i])[-15:][::-1]
    print(f"\nTop features for '{label}':")
    print(", ".join(feature_names[topn]))

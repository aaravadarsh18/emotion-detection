# src/streamlit_app.py
# Streamlit UI with Predict tab and Model Insights tab (reads outputs and top features).
# Run: streamlit run src/streamlit_app.py

import sys
import os
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st

# ensure root on path
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Preprocessing
try:
    from src.preprocessing import preprocess_tweet
except Exception as e:
    preprocess_tweet = None
    preprocess_error = e
else:
    preprocess_error = None

# Model files and outputs
MODEL_PIPELINE_PATH = PROJECT_ROOT / "models" / "pipeline_full.joblib"
MODEL_CLF_PATH = PROJECT_ROOT / "models" / "logreg_model.joblib"
VEC_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
OUT_DIR = PROJECT_ROOT / "outputs"
F1_PNG = OUT_DIR / "f1_per_emotion.png"
TOP_FEATS_CSV = OUT_DIR / "top_features.csv"
REPORT_JSON = OUT_DIR / "classification_report_test.json"

# Load pipeline/model
pipeline = None
clf = None
vec = None
try:
    if MODEL_PIPELINE_PATH.exists():
        pipeline = joblib.load(MODEL_PIPELINE_PATH)
        if hasattr(pipeline, "named_steps"):
            clf = pipeline.named_steps.get("clf", None)
            vec = pipeline.named_steps.get("tfidf", None)
        else:
            clf = pipeline
except Exception as e:
    pipeline = None
if clf is None:
    try:
        if MODEL_CLF_PATH.exists():
            clf = joblib.load(MODEL_CLF_PATH)
        if VEC_PATH.exists():
            vec = joblib.load(VEC_PATH)
    except Exception:
        clf, vec = None, None

# labels from report or classifier
def load_labels():
    if REPORT_JSON.exists():
        with open(REPORT_JSON, "r") as f:
            report = json.load(f)
        labels = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
        return labels
    if clf is not None and hasattr(clf, "classes_"):
        classes = clf.classes_
        return [str(x) for x in classes]
    return []

labels = load_labels()

# Streamlit layout
st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("Emotion Detection")

tab1, tab2 = st.tabs(["Predict", "Model Insights"])

with tab1:
    st.subheader("Input")
    text = st.text_area("Enter text", value="I am so happy today! 😄", height=140)
    c1, c2 = st.columns([1,1])
    with c1:
        do_preprocess = st.button("Preprocess")
    with c2:
        do_predict = st.button("Predict")

    st.subheader("Result")
    result = st.container()
    chart_area = st.container()

    if do_preprocess:
        if preprocess_tweet is None:
            result.error("Preprocessing not available.")
            if preprocess_error:
                with result.expander("Import error"):
                    result.text(str(preprocess_error))
        else:
            result.markdown("**Preprocessed**")
            result.code(preprocess_tweet(text))

    if do_predict:
        if preprocess_tweet is None:
            result.error("Preprocessing not available.")
        elif clf is None:
            result.error("Model not loaded. Train and save models to models/first.")
        else:
            processed = preprocess_tweet(text)
            try:
                if vec is not None:
                    X = vec.transform([processed])
                    probs = clf.predict_proba(X)[0]
                else:
                    probs = clf.predict_proba([processed])[0]
            except Exception:
                probs = None

            if probs is not None:
                pred_idx = int(np.argmax(probs))
                pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
                result.success(f"Predicted: **{pred_label}**")
                # show probabilities as bar chart
                prob_df = pd.DataFrame({"label": labels if labels else list(range(len(probs))), "prob": probs})
                prob_df = prob_df.sort_values("prob", ascending=True)
                chart_area.markdown("**Probability distribution**")
                st.bar_chart(prob_df.set_index("label")["prob"])
            else:
                pred = clf.predict([processed])[0]
                result.success(f"Predicted: **{pred}** (no probabilities)")

with tab2:
    st.subheader("Model Insights")
    st.write("This page displays saved evaluation plots and top features per emotion (if available).")

    if F1_PNG.exists():
        st.image(str(F1_PNG), caption="F1 per emotion (test)", use_column_width=False)
    else:
        st.info("F1 plot not found. Run src/evaluate_model.py to generate outputs/f1_per_emotion.png")

    if TOP_FEATS_CSV.exists():
        df_top = pd.read_csv(TOP_FEATS_CSV)
        # Show top 5 features per label as a pivoted small table
        grouped = df_top.groupby("label").head(5).reset_index(drop=True)
        st.write("Top features (sample 5 per label):")
        st.dataframe(grouped[["label", "feature", "weight"]].sort_values(["label","weight"], ascending=[True, False]).reset_index(drop=True))
        with st.expander("Open full top features CSV"):
            st.download_button("Download top_features.csv", data=TOP_FEATS_CSV.read_bytes(), file_name="top_features.csv")
    else:
        st.info("Top features file not found. Run src/feature_analysis.py to generate outputs/top_features.csv")

    # Show classification report summary
    if REPORT_JSON.exists():
        with open(REPORT_JSON, "r") as f:
            rep = json.load(f)
        if "macro avg" in rep:
            st.metric("Macro F1 (test)", f"{rep['macro avg']['f1-score']:.3f}")
        st.write("Classification report (test):")
        st.json(rep)
    else:
        st.info("Classification report not found. Run the training script to generate outputs.")

tab1, tab2, tab3 = st.tabs(["Predict", "Model Insights", "About & Ethics"])

with tab3:
    st.subheader("About This Project")
    st.markdown("""
    **Emotion Detection in Social Media Posts**  
    - Baseline: TF-IDF + Logistic Regression  
    - Contextual Model: TweetEval RoBERTa-base  
    - Dataset: TweetEval Emotion (CardiffNLP)
    """)

    st.subheader("Ethics & Bias")
    ethics_path = PROJECT_ROOT / "docs" / "ethics.md"
    if ethics_path.exists():
        st.markdown(ethics_path.read_text())
    else:
        st.info("Ethics document not found (docs/ethics.md).")

# src/streamlit_app.py
# Clean Streamlit UI — shows predicted label + probability distribution bar chart
# Run from project root: streamlit run src/streamlit_app.py

import sys
import os
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd

# Ensure project root on sys.path so "src" imports work
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except Exception as e:
    print("Streamlit not installed. Install it: python3 -m pip install streamlit")
    raise

# Try import preprocessing
try:
    from src.preprocessing import preprocess_tweet
except Exception as e:
    preprocess_tweet = None
    preprocess_import_error = e
else:
    preprocess_import_error = None

# Paths
MODEL_PIPELINE_PATH = PROJECT_ROOT / "models" / "pipeline_full.joblib"
MODEL_CLF_PATH = PROJECT_ROOT / "models" / "logreg_model.joblib"
VEC_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
REPORT_JSON = PROJECT_ROOT / "outputs" / "classification_report_test.json"

# Load pipeline or fallback
pipeline, clf, vec = None, None, None
try:
    if MODEL_PIPELINE_PATH.exists():
        pipeline = joblib.load(MODEL_PIPELINE_PATH)
        if hasattr(pipeline, "named_steps"):
            clf = pipeline.named_steps.get("clf", None)
            vec = pipeline.named_steps.get("tfidf", None)
        else:
            clf = pipeline
            vec = None
except Exception as e:
    print("Could not load pipeline:", e)

if clf is None:
    try:
        if MODEL_CLF_PATH.exists():
            clf = joblib.load(MODEL_CLF_PATH)
        if VEC_PATH.exists():
            vec = joblib.load(VEC_PATH)
    except Exception as e:
        print("Could not load model/vectorizer fallback:", e)
        clf, vec = None, None

# Determine label names
def load_label_names_from_report(report_path: Path):
    if not report_path.exists():
        return None
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
        return [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
    except Exception:
        return None

label_names = None
if clf is not None and hasattr(clf, "classes_"):
    classes = clf.classes_
    if all(isinstance(c, str) for c in classes):
        label_names = list(classes)
    elif all(isinstance(c, (bytes, bytearray)) for c in classes):
        label_names = [c.decode("utf-8") for c in classes]
    else:
        label_names = load_label_names_from_report(REPORT_JSON) or [str(c) for c in classes]
else:
    label_names = load_label_names_from_report(REPORT_JSON) or []

# -------------------------
# Streamlit UI layout
# -------------------------
st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("Emotion Detection")

# Top-level layout: left main column, right info column
left_col, right_col = st.columns([3, 1])

with left_col:
    st.subheader("Input")
    text = st.text_area("Enter text to classify", value="I am so happy today! 😄", height=150)

    # Action buttons
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        do_preprocess = st.button("Preprocess")
    with btn_col2:
        do_predict = st.button("Predict")

    # Output area
    st.subheader("Result")
    result_container = st.container()
    chart_container = st.container()

with right_col:
    st.subheader("Model status")
    if pipeline is not None:
        st.write("Loaded: pipeline_full.joblib")
    elif clf is not None and vec is not None:
        st.write("Loaded: model + vectorizer")
    else:
        st.write("No model loaded")
    if label_names:
        st.write("Labels:", ", ".join(label_names))
    else:
        st.write("Label names: unknown")
    st.markdown("---")
    st.write("Files checked:")
    st.write(f"- pipeline_full.joblib: {'yes' if MODEL_PIPELINE_PATH.exists() else 'no'}")
    st.write(f"- logreg_model.joblib: {'yes' if MODEL_CLF_PATH.exists() else 'no'}")
    st.write(f"- tfidf_vectorizer.joblib: {'yes' if VEC_PATH.exists() else 'no'}")
    st.write(f"- classification_report_test.json: {'yes' if REPORT_JSON.exists() else 'no'}")
    if preprocess_import_error:
        with st.expander("Preprocessing import error"):
            st.text(str(preprocess_import_error))

# -------------------------
# Actions
# -------------------------
if do_preprocess:
    if preprocess_tweet is None:
        result_container.error("Preprocessing function not available. Check src/preprocessing.py.")
    else:
        processed = preprocess_tweet(text)
        result_container.markdown("**Preprocessed text:**")
        result_container.code(processed)

if do_predict:
    if preprocess_tweet is None:
        result_container.error("Cannot run prediction because preprocessing is unavailable.")
    elif clf is None:
        result_container.error("Model not loaded. Train model first and ensure models/ contains model files.")
    else:
        try:
            processed = preprocess_tweet(text)

            # If vectorizer present
            if vec is not None:
                X = vec.transform([processed])
                probs = clf.predict_proba(X)[0]
            else:
                # Try with pipeline or clf directly
                probs = clf.predict_proba([processed])[0]

            pred_idx = int(np.argmax(probs))
            pred_label = label_names[pred_idx] if pred_idx < len(label_names) else str(pred_idx)

            # Show prediction
            result_container.success(f"Predicted emotion: **{pred_label}**")

            # Display bar chart for full probability distribution
            labels = label_names if len(label_names) == len(probs) else [str(i) for i in range(len(probs))]
            prob_df = pd.DataFrame({"Emotion": labels, "Probability": probs})
            prob_df = prob_df.sort_values("Probability", ascending=True)

            chart_container.markdown("**Probability distribution across emotions**")
            st.bar_chart(prob_df.set_index("Emotion"))

        except Exception as e:
            result_container.error("Error during prediction. See console for details.")
            print("Prediction error:", e)

tab1, tab2 = st.tabs(["Predict", "Model Insights"])

# src/streamlit_app.py
# Simple, normal-looking Streamlit UI for Emotion Detection
# - Keeps sys.path fix so "src" imports work
# - Minimal, no custom CSS or retro styling

import sys
import os

# Ensure project root on sys.path so package imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import streamlit as st
except Exception:
    print("Streamlit is not installed. Install it with: pip install streamlit")
    raise

# Try to import preprocessing; handle missing module gracefully
try:
    from src.preprocessing import preprocess_tweet
except Exception as e:
    preprocess_tweet = None
    import_error = e
else:
    import_error = None

st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("Emotion Detection")

st.write("Enter a short text (tweet) below. Use **Preprocess** to see cleaned text, or **Predict** once a model is available.")

# Input area
text = st.text_area("Text input", value="I am so happy today! 😄", height=140)

col1, col2 = st.columns(2)
with col1:
    do_preprocess = st.button("Preprocess")
with col2:
    do_predict = st.button("Predict")

# Output area
st.subheader("Output")
if do_preprocess:
    if preprocess_tweet is None:
        st.error("Preprocessing function not available. Check src/preprocessing.py and imports.")
        if import_error:
            st.write("Import error details:", import_error)
    else:
        processed = preprocess_tweet(text)
        st.code(processed)

if do_predict:
    st.info("Prediction functionality not yet implemented. In later steps we'll load the saved TF-IDF vectorizer and LogisticRegression model and display predictions here.")
    # Example for future wiring:
    # processed = preprocess_tweet(text)
    # vec = joblib.load("models/tfidf.joblib")
    # model = joblib.load("models/logreg.joblib")
    # X = vec.transform([processed])
    # pred = model.predict(X)
    # proba = model.predict_proba(X)
    # st.write("Predicted label:", pred[0])
    # st.write("Probabilities:", proba[0])

# Sidebar info
with st.sidebar:
    st.header("Info")
    st.write("- Project: Emotion Detection")
    st.write("- Preprocessing: configurable in src/preprocessing.py")
    if import_error:
        st.write("Import error: see console for details.")

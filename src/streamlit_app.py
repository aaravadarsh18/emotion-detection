# streamlit_app.py - simple UI scaffold for inference
import streamlit as st
from src.preprocessing import preprocess_tweet

st.title("Emotion Detection - Demo (TF-IDF + LR)")
st.write("Enter a tweet to classify emotion (Week 1: demo scaffold)")

tweet = st.text_area("Tweet", value="I am so happy today! 😄")
preproc = preprocess_tweet(tweet)
st.write("Preprocessed text:", preproc)

if st.button("Predict (placeholder)"):
    # placeholder: later load model and vectorizer
    st.info("Model not yet hooked. In Week 2 we'll load the trained TF-IDF + LR model and show predictions.")

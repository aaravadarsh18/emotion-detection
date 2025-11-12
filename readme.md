# Emotion Detection in Social Media Posts

Project: TF-IDF + Logistic Regression baseline, TweetEval benchmark, and advanced analyses.

## Week 1 - Setup & EDA

### Install
1. Create virtualenv (recommended)
   python -m venv .venv
   source .venv/bin/activate
2. Install requirements
   pip install -r requirements.txt

### SpaCy model
   python -m spacy download en_core_web_sm

### Run notebook
Open `notebooks/Week1_EDA.ipynb` in Jupyter and run cells.

### Streamlit UI (scaffold)
Run:
   streamlit run src/streamlit_app.py

## Git
Initialize repo and push to GitHub (see git setup below)

## Notes
- Data (TweetEval) is downloaded within the notebook using `datasets`. If you prefer to use local files, put them in `data/raw`.

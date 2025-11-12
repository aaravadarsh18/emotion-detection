# src/streamlit_app.py
"""
Enhanced Streamlit App for Emotion Detection

Features:
- Modern, polished UI with improved visuals
- Real-time prediction with confidence visualization
- Comprehensive model insights and analytics
- Bias analysis dashboard
- Export capabilities
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Sequence
import time

# Project paths
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

PIPELINE_PATH = MODELS_DIR / "pipeline_full.joblib"
CLF_PATH = MODELS_DIR / "logreg_model.joblib"
VEC_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
F1_PNG = OUTPUTS_DIR / "f1_per_emotion.png"
TOP_FEATS_CSV = OUTPUTS_DIR / "top_features.csv"
REPORT_JSON = OUTPUTS_DIR / "classification_report_test.json"
BIAS_CSV = OUTPUTS_DIR / "bias_analysis.csv"

# Page config
st.set_page_config(
    page_title="Emotion Detection Platform",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .prediction-box {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .emoji-display {
        font-size: 4rem;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Emotion emoji mapping
EMOTION_EMOJIS = {
    "anger": "😠",
    "joy": "😄",
    "sadness": "😢",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️",
    "optimism": "🌟",
    "pessimism": "😔"
}

EMOTION_COLORS = {
    "anger": "#e74c3c",
    "joy": "#f39c12",
    "sadness": "#3498db",
    "fear": "#9b59b6",
    "surprise": "#1abc9c",
    "love": "#e91e63",
    "optimism": "#2ecc71",
    "pessimism": "#95a5a6"
}

# --- Loaders with caching --- #
@st.cache_resource
def load_pipeline():
    if PIPELINE_PATH.exists():
        try:
            return joblib.load(PIPELINE_PATH)
        except Exception:
            return None
    return None

@st.cache_resource
def load_clf_and_vectorizer() -> Tuple[Optional[object], Optional[object]]:
    clf = joblib.load(CLF_PATH) if CLF_PATH.exists() else None
    vec = joblib.load(VEC_PATH) if VEC_PATH.exists() else None
    return clf, vec

@st.cache_data
def load_label_names() -> Sequence[str]:
    if REPORT_JSON.exists():
        try:
            rep = json.loads(REPORT_JSON.read_text())
            labels = [k for k in rep.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
            if labels:
                return labels
        except Exception:
            pass
    
    pipeline = load_pipeline()
    if pipeline is not None:
        clf = pipeline.named_steps.get("clf", pipeline)
        if hasattr(clf, "classes_"):
            return [str(x) for x in clf.classes_]
    
    clf, _ = load_clf_and_vectorizer()
    if clf is not None and hasattr(clf, "classes_"):
        return [str(x) for x in clf.classes_]
    
    return []

def safe_preprocess(text: str):
    try:
        from src.preprocessing import preprocess_tweet
        return preprocess_tweet(text)
    except Exception:
        import re
        t = str(text)
        t = re.sub(r"http\S+", "<URL>", t)
        t = re.sub(r"@\w+", "<USER>", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t.lower()

def predict_text(text: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    if not text:
        return None, None
    
    labels = load_label_names()
    pipeline = load_pipeline()
    clf, vec = load_clf_and_vectorizer()
    preprocessed = safe_preprocess(text)
    
    if pipeline is not None:
        try:
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba([preprocessed])[0]
                label = labels[np.argmax(probs)] if labels else str(np.argmax(probs))
                return label, probs
            else:
                label = pipeline.predict([preprocessed])[0]
                return str(label), None
        except Exception:
            pass
    
    if clf is not None and vec is not None:
        try:
            X = vec.transform([preprocessed])
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)[0]
                label = labels[np.argmax(probs)] if labels else str(np.argmax(probs))
                return label, probs
            else:
                label = clf.predict(X)[0]
                return str(label), None
        except Exception:
            return None, None
    
    return None, None

# --- UI Components --- #
def render_header():
    st.markdown('<h1 class="main-header">🎭 Emotion Detection Platform</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Analyze emotions in text using state-of-the-art NLP models</p>", unsafe_allow_html=True)
    st.markdown("---")

def render_prediction_tab():
    st.subheader("🔮 Predict Emotion")
    
    # Create two columns (we instantiate the quick-example buttons first so they can
    # update session state before the text_area widget is created on the same run).
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Quick Examples")
        examples = {
            "Joy": "I'm so happy today! Everything is wonderful! 😊",
            "Anger": "This is completely unacceptable and frustrating!",
            "Sadness": "I feel so lonely and disappointed...",
            "Surprise": "Wow! I can't believe this just happened!"
        }

        for emotion, example in examples.items():
            if st.button(f"{EMOTION_EMOJIS.get(emotion.lower(), '😐')} {emotion}", use_container_width=True):
                # Set the session-state value before the text_area is instantiated.
                # No explicit rerun call — Streamlit will re-run the script on interaction
                # and the text_area (instantiated later) will read from session_state.
                st.session_state["input_text"] = example

    with col1:
        st.markdown("### Enter your text")

        # Ensure there's a default in session_state so the widget can be safely created
        if "input_text" not in st.session_state:
            st.session_state["input_text"] = "I am so excited about this amazing opportunity! Can't wait to get started! 🎉"

        text_input = st.text_area(
            "Type or paste text to analyze:",
            height=150,
            help="Enter any text and we'll predict the emotion",
            key="input_text",
        )

        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            predict_btn = st.button("🎯 Predict Emotion", type="primary", use_container_width=True)
        with col_b:
            preprocess_btn = st.button("🔧 Show Preprocessed", use_container_width=True)
    
    # Note: example buttons write directly to `st.session_state['input_text']` so
    # the text_area widget updates automatically on rerun.
    
    # Show preprocessed text
    if preprocess_btn:
        with st.spinner("Preprocessing..."):
            processed = safe_preprocess(text_input)
            st.success("✅ Preprocessing complete")
            st.code(processed, language=None)
            st.info(f"**Original length:** {len(text_input)} chars → **Processed:** {len(processed)} chars")
    
    # Make prediction
    if predict_btn:
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze")
        else:
            with st.spinner("Analyzing emotion..."):
                time.sleep(0.5)  # Brief delay for UX
                label, probs = predict_text(text_input)
            
            if label is None:
                st.error("❌ Could not load model. Please check models/ directory.")
            else:
                # Main prediction display
                emoji = EMOTION_EMOJIS.get(label.lower(), "😐")
                st.markdown(f'<div class="emoji-display">{emoji}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-box"><h2 style="text-align: center; color: #667eea;">Detected Emotion: {label.upper()}</h2></div>', unsafe_allow_html=True)
                
                if probs is not None:
                    labels = load_label_names()
                    if len(labels) != len(probs):
                        labels = [f"Class {i}" for i in range(len(probs))]
                    
                    # Create probability dataframe
                    prob_df = pd.DataFrame({
                        "Emotion": labels,
                        "Confidence": probs * 100
                    }).sort_values("Confidence", ascending=False)

                    st.markdown("### 📊 Confidence Scores")

                    # Top 3 metrics (summary cards)
                    top3 = prob_df.head(3)
                    cols = st.columns(3)
                    for idx, (_, row) in enumerate(top3.iterrows()):
                        with cols[idx]:
                            emoji_icon = EMOTION_EMOJIS.get(row['Emotion'].lower(), "😐")
                            st.markdown(f"""
                            <div class="info-card" style="border-left: 4px solid {EMOTION_COLORS.get(row['Emotion'].lower(), '#666')};">
                                <div style="font-size: 2rem;">{emoji_icon}</div>
                                <h3>{row['Emotion'].title()}</h3>
                                <h2 style="color: {EMOTION_COLORS.get(row['Emotion'].lower(), '#667eea')};">{row['Confidence']:.1f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)

                    # Show full distribution as a table inside an expander
                    with st.expander("Show full distribution"):
                        df_display = prob_df.copy()
                        df_display['Confidence'] = df_display['Confidence'].map(lambda x: f"{x:.1f}%")
                        st.dataframe(df_display.reset_index(drop=True), use_container_width=True)
                    
                    # Interactive bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=prob_df['Confidence'],
                            y=prob_df['Emotion'],
                            orientation='h',
                            marker=dict(
                                color=[EMOTION_COLORS.get(e.lower(), '#667eea') for e in prob_df['Emotion']],
                                line=dict(color='rgba(0,0,0,0.3)', width=1)
                            ),
                            text=[f"{c:.1f}%" for c in prob_df['Confidence']],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="All Emotion Probabilities",
                        xaxis_title="Confidence (%)",
                        yaxis_title="",
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export option
                    csv_data = prob_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv_data,
                        "emotion_prediction.csv",
                        "text/csv",
                        use_container_width=True
                    )

def render_insights_tab():
    st.subheader("📈 Model Performance Insights")
    
    # Load metrics
    if REPORT_JSON.exists():
        report = json.loads(REPORT_JSON.read_text())
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = report.get("macro avg", {})
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Macro F1</h3>
                <h1>{metrics.get('f1-score', 0):.3f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Precision</h3>
                <h1>{metrics.get('precision', 0):.3f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Recall</h3>
                <h1>{metrics.get('recall', 0):.3f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Accuracy</h3>
                <h1>{report.get('accuracy', 0):.3f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # F1 scores visualization
    if F1_PNG.exists():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(str(F1_PNG), caption="F1 Score per Emotion", use_container_width=True)
        
        with col2:
            if REPORT_JSON.exists():
                # Create interactive F1 chart
                labels = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
                f1_scores = [report[k]["f1-score"] for k in labels]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=labels,
                        y=f1_scores,
                        marker=dict(
                            color=[EMOTION_COLORS.get(l.lower(), '#667eea') for l in labels]
                        ),
                        text=[f"{f:.3f}" for f in f1_scores],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="F1 Scores by Emotion (Interactive)",
                    xaxis_title="Emotion",
                    yaxis_title="F1 Score",
                    yaxis_range=[0, 1],
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top features
    if TOP_FEATS_CSV.exists():
        st.markdown("### 🔍 Top Predictive Features")
        df_top = pd.read_csv(TOP_FEATS_CSV)
        
        # Select emotion to view
        emotions = df_top['label'].unique()
        selected_emotion = st.selectbox("Select emotion to view top features:", emotions)
        
        emotion_features = df_top[df_top['label'] == selected_emotion].head(15)
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=emotion_features['weight'],
                y=emotion_features['feature'],
                orientation='h',
                marker=dict(color=EMOTION_COLORS.get(selected_emotion.lower(), '#667eea')),
                text=[f"{w:.3f}" for w in emotion_features['weight']],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Top 15 Features for '{selected_emotion}'",
            xaxis_title="Weight",
            yaxis_title="",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download option
        st.download_button(
            "📥 Download Full Feature Analysis",
            TOP_FEATS_CSV.read_bytes(),
            "top_features.csv",
            use_container_width=True
        )

def render_bias_tab():
    st.subheader("⚖️ Bias Analysis")
    
    st.info("This section analyzes potential biases in emotion predictions based on demographic attributes.")
    
    if BIAS_CSV.exists():
        bias_df = pd.read_csv(BIAS_CSV)
        
        st.markdown("### Gender-based Analysis")
        st.dataframe(bias_df, use_container_width=True)
        
        # Visualize bias
        male_data = bias_df[bias_df['text'].str.contains('He|man', case=False)]
        female_data = bias_df[bias_df['text'].str.contains('She|woman', case=False)]
        
        if not male_data.empty and not female_data.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Male pronouns',
                x=male_data['pred_label'],
                y=male_data['score'],
                marker_color='#3498db'
            ))
            
            fig.add_trace(go.Bar(
                name='Female pronouns',
                x=female_data['pred_label'],
                y=female_data['score'],
                marker_color='#e91e63'
            ))
            
            fig.update_layout(
                title="Prediction Confidence by Gender Pronouns",
                xaxis_title="Predicted Emotion",
                yaxis_title="Confidence Score",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Bias analysis data not found. Run `python src/bias_analysis.py` to generate it.")
        
        st.markdown("""
        **Bias analysis helps identify:**
        - Systematic differences in predictions based on demographic indicators
        - Fairness issues in model behavior
        - Areas for model improvement
        
        Generate bias analysis by running: `python src/bias_analysis.py`
        """)

def render_about_tab():
    st.subheader("ℹ️ About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This emotion detection platform uses advanced NLP techniques to analyze and classify emotions in text data, specifically optimized for social media content like tweets.
        
        **Key Features:**
        - Real-time emotion prediction
        - Multi-class classification (anger, joy, sadness, fear, etc.)
        - Comprehensive model insights and analytics
        - Bias detection and fairness analysis
        
        ### 🛠️ Technical Stack
        
        - **Models:** TF-IDF + Logistic Regression, RoBERTa Transformer
        - **Dataset:** TweetEval emotion dataset (CardiffNLP)
        - **Framework:** scikit-learn, Transformers, Streamlit
        - **Preprocessing:** Custom tweet normalization pipeline
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Model Architecture
        
        **Baseline Model:**
        - TF-IDF vectorization (1-2 grams)
        - Logistic Regression (multinomial)
        - Optimized via GridSearchCV
        
        **Advanced Model:**
        - RoBERTa-base (twitter-optimized)
        - Fine-tuned on emotion dataset
        - Transfer learning approach
        
        ### 🚀 Getting Started
        
        1. **Train models:** `python src/train_baseline.py`
        2. **Generate insights:** `python src/evaluate_model.py`
        3. **Run bias analysis:** `python src/bias_analysis.py`
        4. **Launch app:** `streamlit run src/streamlit_app.py`
        
        ### 📧 Contact & Support
        
        For questions or issues, please refer to the project documentation or contact the development team.
        """)

# --- Main App --- #
def main():
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Settings")
        
        st.markdown("### Model Status")
        pipeline = load_pipeline()
        clf, vec = load_clf_and_vectorizer()
        
        if pipeline or (clf and vec):
            st.success("✅ Models loaded")
        else:
            st.error("❌ Models not found")
            st.info("Train models using:\n`python src/train_baseline.py`")
        
        st.markdown("---")
        st.markdown("### 📚 Resources")
        st.markdown("- [TweetEval Dataset](https://huggingface.co/datasets/cardiffnlp/tweet_eval)")
        st.markdown("- [Project Documentation](#)")
        st.markdown("- [GitHub Repository](#)")
        
        st.markdown("---")
        st.markdown("### ⚡ Quick Stats")
        labels = load_label_names()
        st.metric("Emotion Classes", len(labels))
        
        if REPORT_JSON.exists():
            report = json.loads(REPORT_JSON.read_text())
            st.metric("Model Accuracy", f"{report.get('accuracy', 0):.1%}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📈 Insights", "⚖️ Bias Analysis", "ℹ️ About"])
    
    with tab1:
        render_prediction_tab()
    
    with tab2:
        render_insights_tab()
    
    with tab3:
        render_bias_tab()
    
    with tab4:
        render_about_tab()
    
    # Footer
    

if __name__ == "__main__":
    main()
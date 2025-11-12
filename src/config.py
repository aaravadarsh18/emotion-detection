# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

# Model hyperparameters
TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 20000,
    "min_df": 2,
    "max_df": 0.95,
}

LOGREG_CONFIG = {
    "solver": "saga",
    "multi_class": "multinomial",
    "max_iter": 1000,
    "class_weight": "balanced",
    "n_jobs": -1,
}

# Emotion labels
EMOTION_LABELS = ["anger", "joy", "optimism", "sadness", "fear", "love"]
# src/train_baseline.py
"""
Baseline training script for Emotion Detection.

Features:
- Loads TweetEval emotion dataset (explicit repo id)
- Preprocesses using src.preprocessing.preprocess_tweet
- Builds TF-IDF features
- Trains LogisticRegression with solver='saga' (multinomial)
- Optional GridSearch (small or full)
- Evaluates on validation and test sets (classification report + confusion matrix)
- Saves best model, vectorizer, and metrics to outputs/
- Falls back to local sample CSV if dataset download fails
"""

import os
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# ensure src package import works
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import preprocess_tweet

# Constants / paths
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 20000,
    "min_df": 2,
    "max_df": 0.95,
}

LABEL_NAMES = None  # will be filled after loading dataset

def try_load_tweeteval():
    """Attempt to load the TweetEval emotion dataset from HuggingFace.
    Returns None if loading fails.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("cardiffnlp/tweet_eval", "emotion")
        return ds
    except Exception as e:
        print("Could not load TweetEval from HuggingFace:", e)
        return None

def load_local_sample(sample_path: Path):
    """Load a local processed CSV sample saved in Week1 (expects columns text_clean and label)"""
    if not sample_path.exists():
        raise FileNotFoundError(f"Local sample not found at {sample_path}")
    df = pd.read_csv(sample_path)
    # Ensure expected columns
    if "text_clean" not in df.columns:
        # if only 'text' present, preprocess it
        if "text" in df.columns:
            df["text_clean"] = df["text"].apply(preprocess_tweet)
        else:
            raise ValueError("Local sample must contain 'text' or 'text_clean' column.")
    return df

def prepare_data(grid_mode="small"):
    """Load dataset (HuggingFace) and return pandas train/val/test and label names.
    If HF fails, fall back to local sample (data/processed/train_sample_5k.csv).
    """
    ds = try_load_tweeteval()
    if ds is not None:
        train_df = pd.DataFrame(ds["train"])
        val_df = pd.DataFrame(ds["validation"])
        test_df = pd.DataFrame(ds["test"])
        global LABEL_NAMES
        LABEL_NAMES = ds["train"].features["label"].names
        print("Loaded TweetEval from HuggingFace. Classes:", LABEL_NAMES)
    else:
        # Fallback: load local processed sample (smaller)
        sample_path = PROJECT_ROOT / "data" / "processed" / "train_sample_5k.csv"
        print("Falling back to local sample:", sample_path)
        df = load_local_sample(sample_path)
        # For fallback, we will split into train/val/test roughly
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        train_df = df.iloc[: int(0.8 * n)].copy()
        val_df = df.iloc[int(0.8 * n): int(0.9 * n)].copy()
        test_df = df.iloc[int(0.9 * n):].copy()
        # Expect label column to be numeric or string; keep as-is
        if "label" in df.columns and LABEL_NAMES is None:
            # if label is numeric but no mapping known, create string labels
            unique_labels = sorted(train_df["label"].unique().tolist())
            LABEL_NAMES = [str(x) for x in unique_labels]
            print("Using fallback label names:", LABEL_NAMES)
        else:
            LABEL_NAMES = LABEL_NAMES or []

    # Preprocess text_clean column if not present
    for d in (train_df, val_df, test_df):
        if "text_clean" not in d.columns:
            print("Preprocessing texts (this may take a while)...")
            d["text_clean"] = d["text"].apply(preprocess_tweet)
    return train_df, val_df, test_df, LABEL_NAMES

def build_pipeline(tfidf_cfg=None, clf_kwargs=None):
    tfidf_cfg = tfidf_cfg or DEFAULT_TFIDF_CONFIG
    clf_kwargs = clf_kwargs or {}
    tfidf = TfidfVectorizer(
        ngram_range=tfidf_cfg.get("ngram_range", (1,2)),
        max_features=tfidf_cfg.get("max_features", None),
        min_df=tfidf_cfg.get("min_df", 2),
        max_df=tfidf_cfg.get("max_df", 0.95)
    )
    clf = LogisticRegression(
        solver="saga",
        multi_class="multinomial",
        max_iter=1000,
        n_jobs=-1,
        **clf_kwargs
    )
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf)
    ])
    return pipe

def run_grid_search(pipe, X_train, y_train, grid_mode="small"):
    """Run GridSearchCV. Two preset modes: 'small' and 'full'."""
    if grid_mode == "small":
        param_grid = {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__max_features": [10000, 20000],
            "clf__C": [0.5, 1.0, 2.0],
        }
    else:  # full mode (heavier)
        param_grid = {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__max_features": [5000, 10000, 20000],
            "tfidf__min_df": [1,2],
            "clf__C": [0.1, 0.5, 1.0, 2.0],
        }
    print("Running GridSearchCV with grid:", param_grid)
    gs = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=3, verbose=2, n_jobs=-1)
    gs.fit(X_train, y_train)
    print("GridSearch best params:", gs.best_params_)
    return gs.best_estimator_, gs

def evaluate_and_save(model, vectorizer, X_val, y_val, X_test, y_test, label_names, out_dir=OUT_DIR):
    # Validation
    print("Evaluating on validation set...")
    y_pred_val = model.predict(X_val)
    report_val = classification_report(y_val, y_pred_val, target_names=label_names, output_dict=True)
    cm_val = confusion_matrix(y_val, y_pred_val)

    # Test
    print("Evaluating on test set...")
    y_pred_test = model.predict(X_test)
    report_test = classification_report(y_test, y_pred_test, target_names=label_names, output_dict=True)
    cm_test = confusion_matrix(y_test, y_pred_test)

    # Save reports
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "classification_report_test.json"
    with open(report_path, "w") as f:
        json.dump(report_test, f, indent=2)
    print("Saved classification report (test) to", report_path)

    # Save confusion matrix figure
    cm_fig_path = out_dir / "confusion_matrix_test.png"
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens",
                xticklabels=label_names if label_names else None,
                yticklabels=label_names if label_names else None)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(cm_fig_path, dpi=200)
    plt.close()
    print("Saved confusion matrix to", cm_fig_path)

    # Save validation report too
    report_val_path = out_dir / "classification_report_val.json"
    with open(report_val_path, "w") as f:
        json.dump(report_val, f, indent=2)
    print("Saved classification report (val) to", report_val_path)

    return report_test, cm_test

def main(args):
    grid_mode = args.grid
    print("Grid mode:", grid_mode)

    train_df, val_df, test_df, label_names = prepare_data(grid_mode=grid_mode)
    X_train_text = train_df["text_clean"].astype(str).values
    X_val_text = val_df["text_clean"].astype(str).values
    X_test_text = test_df["text_clean"].astype(str).values
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # Build and optionally grid-search
    base_pipe = build_pipeline()
    if args.do_grid:
        print("Running grid search (this may take time). Use grid_mode='small' for faster runs.")
        best_pipe, gs = run_grid_search(base_pipe, X_train_text, y_train, grid_mode=grid_mode)
        model_pipe = best_pipe
    else:
        print("Fitting base pipeline (no grid search)...")
        model_pipe = base_pipe
        model_pipe.fit(X_train_text, y_train)

    # After fitting/predicting, extract components for saving
    # If pipeline, vectorizer is model_pipe.named_steps["tfidf"], model is model_pipe.named_steps["clf"]
    tfidf_vect = model_pipe.named_steps["tfidf"]
    clf = model_pipe.named_steps["clf"]

    # Evaluate
    X_val = tfidf_vect.transform(X_val_text)
    X_test = tfidf_vect.transform(X_test_text)
    report_test, cm_test = evaluate_and_save(clf, tfidf_vect, X_val, y_val, X_test, y_test, label_names)

    # Save model and vectorizer (joblib)
    model_path = MODELS_DIR / "logreg_model.joblib"
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(tfidf_vect, vec_path)
    print("Saved model to:", model_path)
    print("Saved vectorizer to:", vec_path)

    # Also save the pipeline if desired
    pipeline_path = MODELS_DIR / "pipeline_full.joblib"
    joblib.dump(model_pipe, pipeline_path)
    print("Saved full pipeline to:", pipeline_path)

    # Print a small summary
    print("\n=== SUMMARY ===")
    print("Test macro-F1 (approx):", report_test.get("macro avg", {}).get("f1-score"))
    print("Outputs saved under:", OUT_DIR)
    print("Models saved under:", MODELS_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do-grid", action="store_true", help="Run GridSearchCV (heavier).")
    parser.add_argument("--grid", choices=["small", "full"], default="small", help="Grid size for GridSearch.")
    args = parser.parse_args()
    main(args)

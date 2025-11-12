# Emotion Detection — Quick Start

Requirements

- Python 3.8 or newer
- A POSIX shell (macOS zsh tested)
- A virtual environment and the packages listed in `requirements.txt`

Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install project dependencies:

```bash
pip install -r requirements.txt
```

Core workflows

Train baseline model (TF-IDF + Logistic Regression):

```bash
python src/train_baseline.py
```

Evaluate and generate plots (requires `outputs/classification_report_test.json`):

```bash
python src/evaluate_model.py
```

Show top predictive features (requires saved model and vectorizer in `models/`):

```bash
python src/feature_analysis.py
```

Optional analyses

Bias analysis (requires `transformers` and network access):

```bash
python src/bias_analysis.py
```

Linguistic exploratory plots:

```bash
python src/linguistic_analysis.py
```

Transformer training (heavy; GPU recommended):

```bash
python src/train_transformer.py
```

Run interactive app

```bash
streamlit run src/streamlit_app.py
```

Open the URL shown in the terminal (default: http://localhost:8501).

Troubleshooting

- If a script fails because data or model files are missing, follow the script messages to create or generate the required files.
- If Hugging Face downloads fail, check network access and retry.
- For transformer training, install a PyTorch wheel that matches your system (CPU-only or CUDA-enabled).
- If your editor reports unresolved imports, ensure the editor uses the project virtual environment interpreter.

Files of interest

- `src/streamlit_app.py` — interactive web UI
- `src/train_baseline.py` — baseline training
- `src/train_transformer.py` — transformer training (Trainer)
- `src/preprocessing.py` — text preprocessing utilities
- `src/feature_analysis.py`, `src/evaluate_model.py`, `src/linguistic_analysis.py`, `src/bias_analysis.py` — analysis scripts

If you would like a single orchestration script to run selected steps in sequence or a short "quick mode" for transformer training, request it and I will add it.

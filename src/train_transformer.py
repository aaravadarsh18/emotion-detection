# src/train_transformer.py
"""
Robust transformer training script for TweetEval emotion task.

This version inspects TrainingArguments at runtime and only passes supported kwargs,
and avoids load_best_model_at_end when evaluation is not enabled to prevent mismatch errors.
Run:
    python3 src/train_transformer.py
"""

from pathlib import Path
import os
import json

def main():
    import numpy as np
    import inspect

    try:
        import transformers
    except Exception as e:
        raise RuntimeError("transformers not found. Install with: python3 -m pip install transformers") from e

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets not found. Install with: python3 -m pip install datasets") from e

    try:
        import evaluate as hf_evaluate
    except Exception:
        hf_evaluate = None

    print("transformers version:", transformers.__version__)

    OUT_DIR = Path("models") / "transformer_emotion"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TweetEval (emotion)...")
    dataset = load_dataset("cardiffnlp/tweet_eval", "emotion")

    if "label" in dataset["train"].features:
        dataset = dataset.rename_column("label", "labels")

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"
    print("Loading tokenizer and model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

    print("Tokenizing...")
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    metric = None
    if hf_evaluate is not None:
        metric = hf_evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1 = metric.compute(predictions=preds, references=labels, average="macro")["f1"] if metric is not None else None
        acc = float((preds == labels).mean())
        return {"macro_f1": f1, "accuracy": acc}

    ta_init_sig = inspect.signature(TrainingArguments.__init__)
    supported_params = set(ta_init_sig.parameters.keys())
    supported_params.discard('self')

    desired_args = {
        "output_dir": str(OUT_DIR),
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 2,
        "weight_decay": 0.01,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "save_total_limit": 2,
        "logging_strategy": "epoch",
    }

    ta_kwargs = {k: v for k, v in desired_args.items() if k in supported_params}

    if "evaluation_strategy" not in ta_kwargs:
        for key in ["load_best_model_at_end", "metric_for_best_model", "save_strategy", "save_total_limit"]:
            if key in ta_kwargs:
                ta_kwargs.pop(key, None)
        if "evaluate_during_training" in supported_params:
            ta_kwargs["evaluate_during_training"] = True
        if "do_eval" in supported_params:
            ta_kwargs["do_eval"] = True

    if "logging_strategy" not in supported_params and "logging_strategy" in ta_kwargs:
        ta_kwargs.pop("logging_strategy", None)

    if not ta_kwargs:
        print("No preferred TrainingArguments keys supported; using minimal args fallback.")
        ta_kwargs = {
            "output_dir": str(OUT_DIR),
            "per_device_train_batch_size": 16,
            "num_train_epochs": 2,
        }
    else:
        print("Using TrainingArguments kwargs:", ta_kwargs)

    training_args = TrainingArguments(**ta_kwargs)

    print("TrainingArguments constructed. Starting Trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(dataset["test"])
    print("Test metrics:", test_metrics)

    metrics_path = OUT_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Saved test metrics to", metrics_path)

    print("Done.")


if __name__ == "__main__":
    main()

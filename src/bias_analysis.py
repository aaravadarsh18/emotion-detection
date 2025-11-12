"""
Evaluate bias in emotion predictions using gendered word swaps and polarity groups.
"""
import re, pandas as pd
from transformers import pipeline
from datasets import load_dataset

model_name = "cardiffnlp/twitter-roberta-base-emotion"
classifier = pipeline("text-classification", model=model_name, top_k=None)

# Small synthetic test set
samples = [
    "He is so emotional about this!",
    "She is so emotional about this!",
    "The man is furious.",
    "The woman is furious.",
    "He loves it!",
    "She loves it!",
]

results = []
for text in samples:
    pred = classifier(text)[0]
    label = pred["label"]
    score = pred["score"]
    results.append({"text": text, "pred_label": label, "score": score})

df = pd.DataFrame(results)
df.to_csv("outputs/bias_analysis.csv", index=False)
print(df)

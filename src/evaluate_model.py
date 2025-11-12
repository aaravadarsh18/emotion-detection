import joblib, json, matplotlib.pyplot as plt, numpy as np

report = json.load(open("outputs/classification_report_test.json"))
labels = [k for k in report.keys() if k not in ("accuracy","macro avg","weighted avg")]
f1 = [report[k]["f1-score"] for k in labels]

plt.bar(labels, f1)
plt.title("F1-score per Emotion (Test)")
plt.ylabel("F1-score")
plt.ylim(0,1)
plt.savefig("outputs/f1_per_emotion.png", dpi=200)

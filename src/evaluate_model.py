import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
	report_path = Path("outputs") / "classification_report_test.json"
	if not report_path.exists():
		print(f"Report not found at {report_path}. Run evaluation first.")
		return

	with report_path.open() as fh:
		report = json.load(fh)

	labels = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
	f1 = [report[k]["f1-score"] for k in labels]

	out_dir = Path("outputs")
	out_dir.mkdir(parents=True, exist_ok=True)
	fig_path = out_dir / "f1_per_emotion.png"

	plt.figure(figsize=(8, 4))
	plt.bar(labels, f1)
	plt.title("F1-score per Emotion (Test)")
	plt.ylabel("F1-score")
	plt.ylim(0, 1)
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close()
	print("Saved F1 plot to", fig_path)


if __name__ == "__main__":
	main()

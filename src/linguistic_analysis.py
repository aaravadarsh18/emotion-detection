import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, re, emoji
from datasets import load_dataset
from pathlib import Path

def main():
	ds = load_dataset("cardiffnlp/tweet_eval", "emotion")
	df = pd.DataFrame(ds["train"])

	df["len"] = df["text"].str.len()
	df["emoji_count"] = df["text"].apply(lambda x: sum(ch in emoji.EMOJI_DATA for ch in x))

	sns.boxplot(data=df, x="label", y="len")
	plt.title("Tweet Length by Emotion")
	out_dir = Path("outputs")
	out_dir.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(out_dir / "tweet_length_by_emotion.png", dpi=200)


if __name__ == "__main__":
	main()

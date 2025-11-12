import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, re, emoji
from datasets import load_dataset

ds = load_dataset("cardiffnlp/tweet_eval", "emotion")
df = pd.DataFrame(ds["train"])

df["len"] = df["text"].str.len()
df["emoji_count"] = df["text"].apply(lambda x: sum(ch in emoji.EMOJI_DATA for ch in x))

sns.boxplot(data=df, x="label", y="len")
plt.title("Tweet Length by Emotion")
plt.savefig("outputs/tweet_length_by_emotion.png", dpi=200)

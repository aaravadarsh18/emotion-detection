# data_prep.py
import re, os, json
import pandas as pd
from datasets import load_dataset

CONTRACTIONS = {
    "ain't":"is not","can't":"can not","i'm":"i am","i've":"i have","you're":"you are",
    "it's":"it is","that's":"that is","don't":"do not","didn't":"did not"
}

def clean_text(t):
    t = t.lower()
    # replace contractions
    for k,v in CONTRACTIONS.items():
        t = re.sub(r"\b"+re.escape(k)+r"\b", v, t)
    # remove URLs, mentions, normalize hashtags
    t = re.sub(r"http\S+","",t)
    t = re.sub(r"@\w+","",t)
    t = re.sub(r"#"," #",t)  # separate hash
    # replace elongated characters e.g. soooo -> soo
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    t = t.strip()
    return t

def load_tweeteval_emotion():
    ds = load_dataset("tweet_eval", "emotion")
    # convert to pandas for convenience
    train = pd.DataFrame(ds['train'])
    valid = pd.DataFrame(ds['validation'])
    test = pd.DataFrame(ds['test'])
    for df in (train,valid,test):
        df['text'] = df['text'].apply(clean_text)
    return train, valid, test

def load_tweeteval_irony():
    ds = load_dataset("tweet_eval", "irony")
    train = pd.DataFrame(ds['train'])
    valid = pd.DataFrame(ds['validation'])
    test = pd.DataFrame(ds['test'])
    for df in (train,valid,test):
        df['text'] = df['text'].apply(clean_text)
    return train, valid, test

def load_jigsaw_toxicity(path=None):
    # If not available locally, user should download from Kaggle and set path
    # Simple placeholder: load sample from datasets if present
    try:
        ds = load_dataset("jigsaw_toxicity_predication")
    except Exception:
        return None
    return ds

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    train_e, val_e, test_e = load_tweeteval_emotion()
    train_e.to_csv("data/emotion_train.csv", index=False)
    val_e.to_csv("data/emotion_valid.csv", index=False)
    test_e.to_csv("data/emotion_test.csv", index=False)
    train_i, val_i, test_i = load_tweeteval_irony()
    train_i.to_csv("data/irony_train.csv", index=False)
    print("Saved datasets in ./data")

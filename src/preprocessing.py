"""
preprocessing.py
Basic, configurable preprocessing for tweets.
"""

import re
import string
from typing import List
import emoji
import spacy
from nltk.tokenize import TweetTokenizer

# Load spaCy inside functions to avoid import cost at module import time if needed
_tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)

URL_RE = re.compile(r"http\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
MULTI_WHITESPACE = re.compile(r"\s+")

def normalize_elongations(text: str) -> str:
    # Reduce repeated characters (loooove -> loove)
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def replace_urls_users(text: str, replace_url_token: str = "<URL>", replace_user_token: str = "<USER>"):
    text = URL_RE.sub(replace_url_token, text)
    text = USER_RE.sub(replace_user_token, text)
    return text

def map_emojis_to_text(text: str) -> str:
    # e.g. "I love it 😊" -> "I love it :smiling_face_with_smiling_eyes:"
    return emoji.demojize(text, language='en')

def basic_clean(text: str, remove_punct: bool = False) -> str:
    text = text.strip()
    text = MULTI_WHITESPACE.sub(" ", text)
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize(text: str) -> List[str]:
    return _tknzr.tokenize(text)

def preprocess_tweet(text: str,
                     lower: bool = True,
                     map_emoji: bool = True,
                     replace_urls: bool = True,
                     normalize_repeat: bool = True,
                     remove_punct: bool = False) -> str:
    if lower:
        text = text.lower()
    if replace_urls:
        text = replace_urls_users(text)
    if map_emoji:
        text = map_emojis_to_text(text)
    if normalize_repeat:
        text = normalize_elongations(text)
    text = basic_clean(text, remove_punct=remove_punct)
    return text

# POS tagging helper
def get_spacy_nlp(model="en_core_web_sm"):
    return spacy.load(model)

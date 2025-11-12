"""
features.py
TF-IDF vectorizer and basic feature stacking helpers.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import numpy as np

def build_tfidf(max_features=20000, ngram_range=(1,2), min_df=2, max_df=0.9):
    tfidf = TfidfVectorizer(max_features=max_features,
                            ngram_range=ngram_range,
                            min_df=min_df,
                            max_df=max_df)
    return tfidf

# Example: stacking custom numeric features with TF-IDF using FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array([x[self.key] for x in X]).reshape(-1,1)

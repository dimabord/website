from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Transformer(BaseEstimator, TransformerMixin):

    def __init__(self, vectorizer=TfidfVectorizer()):
        self.vectorizer = vectorizer

    def fit(self, X, y=None):
        return self.vectorizer.fit(X)

    def transform(self, X):
        return self.vectorizer.transform(X)
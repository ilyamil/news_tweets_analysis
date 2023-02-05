import pandas as pd
import numpy as np
from numbers import Number
from typing import Iterable
from gensim import downloader
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from news_tweets_analysis.preprocessing import TextPreprocessor


class AverageWordVectors(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = None, word_vectors: KeyedVectors = None):
        if (not model_name and not word_vectors) or (model_name and word_vectors):
            raise ValueError('Either model_name or model must be provided')

        self.model_name = model_name
        if self.model_name:
            self.word_vectors = None
        else:
            self.word_vectors = word_vectors

    def fit(self, X=None, y=None):
        if self.model_name:
            self.word_vectors = downloader.load(self.model_name)
        return self

    def tokenize(self, doc: str) -> Iterable[str]:
        return TextPreprocessor.tokenize(doc)

    def _transform(self, doc: str) -> Iterable[Number]:
        tokens = self.tokenize(doc)
        doc_vector = self.word_vectors.get_mean_vector(
            tokens, pre_normalize=False
        )
        return doc_vector

    def transform(self, X: Iterable[str]) -> pd.Series:
        if not isinstance(X, pd.Series):
            return pd.Series(X).apply(self._transform).apply(pd.Series)
        return X.apply(self._transform).apply(pd.Series)


class WeightedAverageWordVectors(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = None, word_vectors: KeyedVectors = None):
        if (not model_name and not word_vectors) or (model_name and word_vectors):
            raise ValueError('Either model_name or model must be provided')

        self.model_name = model_name
        if self.model_name:
            self.word_vectors = None
        else:
            self.word_vectors = word_vectors

    def fit(self, X=None, y=None):
        if self.model_name:
            self.word_vectors = downloader.load(self.model_name)
        return self    

    def tokenize(self, doc: str) -> Iterable[str]:
        return TextPreprocessor.tokenize(doc)

    def _transform(self, doc: str) -> Iterable[Number]:
        tokens = self.tokenize(doc)
        doc_vector = self.word_vectors.get_mean_vector(
            tokens, pre_normalize=False
        )
        return doc_vector

    def transform(self, X: Iterable[str]) -> np.ndarray:
        if not isinstance(X, pd.Series):
            X_preprocessed = pd.Series(X)
        else:
            X_preprocessed = X

        vectorizer = TfidfVectorizer(tokenizer=self.tokenize)
        tfidf_matrix = vectorizer.fit_transform(X_preprocessed)
        feature_names = vectorizer.get_feature_names_out()

        output_shape = (len(X_preprocessed), self.word_vectors.vector_size)
        wavg_word_vectors = np.zeros(output_shape)
        for i, row in enumerate(tfidf_matrix.todense()):
            _, nz_cols = row.nonzero()
            row = np.asarray(row).flatten()
            tokens = feature_names[nz_cols]
            weights = row[nz_cols]
            wavg_word_vectors[i] = self.word_vectors.get_mean_vector(
                tokens, weights=weights, pre_normalize=False
            )
        return wavg_word_vectors

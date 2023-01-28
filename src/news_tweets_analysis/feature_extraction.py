import pandas as pd
import numpy as np
from numbers import Number
from typing import Iterable, Text
from gensim import downloader
from gensim.models.keyedvectors import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin
from news_tweets_analysis.preprocessing import TextPreprocessor


class AverageWordVectors(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = None, model: KeyedVectors = None):
        if (not model_name and not model) or (model_name and model):
            raise ValueError('Either model_name or model must be provided')
        
        if model_name:
            self.word_vectors = downloader.load(model_name)
        else:
            self.word_vectors = model

    def _transform(self, doc: str) -> Iterable[Number]:
        doc_clean = TextPreprocessor.remove_useless(doc.lower())
        doc_clean = TextPreprocessor.fix_contractions(doc_clean)
        tokens = TextPreprocessor.tokenize(doc_clean)
        doc_vector = self.word_vectors.get_mean_vector(
            tokens, pre_normalize=False
        )
        return doc_vector

    def transform(self, X: Iterable[str]) -> pd.Series:
        if not isinstance(X, pd.Series):
            return pd.Series(X).apply(self._transform)
        return X.apply(self._transform)


class WightedAverageWordVectors(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = None, model: KeyedVectors = None):
        if (not model_name and not model) or (model_name and model):
            raise ValueError('Either model_name or model must be provided')
        
        if model_name:
            self.model = downloader.load(model_name)
        else:
            self.model = model

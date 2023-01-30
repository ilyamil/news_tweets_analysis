import pytest
import numpy as np
from gensim import downloader
from news_tweets_analysis.feature_extraction import (
    AverageWordVectors,
    WeightedAverageWordVectors
)

WORD_VECTORS_NAME = 'glove-wiki-gigaword-50'

@pytest.fixture
def word_vectors():
    return downloader.load(WORD_VECTORS_NAME)


def test_average_word_vectors_single_doc(word_vectors):
    averager = AverageWordVectors(model=word_vectors)
    doc = "I'm working"
    average_vector = averager._transform(doc)
    assert len(average_vector) == 50
    assert  np.isclose(
        average_vector,
        (word_vectors['i'] + word_vectors['am'] + word_vectors['working'])/3
    ).all()


def test_average_word_vectors_multiple_docs(word_vectors):
    averager = AverageWordVectors(model=word_vectors)
    docs = ["I'm working", "You're working"]
    average_vectors = averager.fit_transform(docs)
    assert average_vectors.shape == (2, 50)


def test_weighted_average_word_vectors_multiple_docs(word_vectors):
    averager = WeightedAverageWordVectors(model=word_vectors)
    docs = ["I am working", "You are working here"]
    average_vectors = averager.fit_transform(docs)
    true_vector = word_vectors.get_mean_vector(
        keys=['am', 'i', 'working'],
        weights = [0.6316672, 0.6316672, 0.44943642],
        pre_normalize=False
    )
    assert average_vectors.shape == (2, 50)
    assert np.isclose(average_vectors[0,:], true_vector).all()

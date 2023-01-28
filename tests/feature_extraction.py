import pytest
import numpy as np
from gensim import downloader
from news_tweets_analysis.feature_extraction import AverageWordVectors

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

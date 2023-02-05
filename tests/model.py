import pytest
from news_tweets_analysis.model import (
    TwitterRobertaBaseSentimentInference,
    predict_sentiment_textblob,
    predict_sentiment_vader,
    predict_sentiment_twitter_roberta
)


@pytest.fixture
def docs_and_labels():
    return {
        "You're beautiful today!": 'positive',
        "I failed an exam.": 'negative'
    }


def test_predict_sentiment_textblob(docs_and_labels):
    docs = list(docs_and_labels.keys())
    labels = list(docs_and_labels.values())
    sentiment = predict_sentiment_textblob(docs)
    assert sentiment == labels


def test_predict_sentiment_vader(docs_and_labels):
    docs = list(docs_and_labels.keys())
    labels = list(docs_and_labels.values())
    sentiment = predict_sentiment_vader(docs)
    assert sentiment == labels


def test_init_twitter_roberta_base_sentiment_inference():
    model = TwitterRobertaBaseSentimentInference()
    assert model


def test_predict_sentiment_twitter_roberta(docs_and_labels):
    docs = list(docs_and_labels.keys())
    labels = list(docs_and_labels.values())
    sentiment = predict_sentiment_twitter_roberta(docs)
    assert sentiment == labels    

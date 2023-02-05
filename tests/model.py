from news_tweets_analysis.model import predict_sentiment_textblob, predict_sentiment_vader


def test_predict_sentiment_textblob():
    docs = [
        "You're beautiful today!",
        "Sky is blue",
        "I failed an exam."
    ]
    sentiment = predict_sentiment_textblob(docs)
    assert sentiment == ['positive', 'neutral', 'negative']


def test_predict_sentiment_vader():
    docs = [
        "You're beautiful today!",
        "Sky is blue",
        "I failed an exam."
    ]
    sentiment = predict_sentiment_vader(docs)
    assert sentiment == ['positive', 'neutral', 'negative']

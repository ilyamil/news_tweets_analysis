import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import Iterable


nltk.download('vader_lexicon')


def predict_sentiment_textblob(docs: Iterable[str]) -> Iterable[str]:
    docs_sentiment = []
    for doc in docs:
        polarity = TextBlob(doc).sentiment.polarity
        if (polarity > -0.33) & (polarity < 0.33):
            sentiment = 'neutral'
        elif polarity >= 0.33:
            sentiment = 'positive'
        else:
            sentiment = 'negative'
        docs_sentiment.append(sentiment)
    return docs_sentiment


def predict_sentiment_vader(docs: Iterable[str]) -> Iterable[str]:
    vader = SentimentIntensityAnalyzer()
    docs_sentiment = []
    for doc in docs:
        polarity_scores = vader.polarity_scores(doc)
        polarity_scores.pop('compound')
        label = max(polarity_scores, key=polarity_scores.get)
        if label == 'neg':
            sentiment = 'negative'
        elif label == 'pos':
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
        docs_sentiment.append(sentiment)
    return docs_sentiment


def predict_sentiment_roberta(docs: Iterable[str]) -> Iterable[str]:
    pass

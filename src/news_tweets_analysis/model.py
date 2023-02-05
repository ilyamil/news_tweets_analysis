import os
import torch
import nltk
from scipy.special import softmax
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from pathlib import Path
from textblob import TextBlob
from typing import Iterable, Dict


ROOT = Path(__file__).resolve().parents[2].as_posix()


class TwitterRobertaBaseSentimentInference:
    """
    Roberta Base Model for sentiment classification.
    Ref: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    """
    name = 'twitter-roberta-base-sentiment-latest'
    labels = ['negative', 'neutral', 'positive']
    version = 1
    def __init__(self):
        model_name = os.path.join(
            ROOT,
            'models',
            'twitter-roberta-base-sentiment'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def _preprocess(self, doc: str) -> str:
        clean_doc = []
        for t in doc.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            clean_doc.append(t)
        return " ".join(clean_doc)
    
    def predict(self, docs: Iterable[str]) -> Iterable[Dict[str, float]]:
        preprocessed_docs = [self._preprocess(doc) for doc in docs]
        encoded_input = self.tokenizer(
            preprocessed_docs,
            padding=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            scores = self.model(**encoded_input)[0].detach().numpy()
        scores_normalized = softmax(scores, axis=1).tolist()

        output_dict = []
        for scores in scores_normalized:
            scores_dict = {k: v for k,v in zip(self.labels, scores)}
            output_dict.append(scores_dict)
        return output_dict


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
    nltk.download('vader_lexicon')
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


def predict_sentiment_twitter_roberta(docs: Iterable[str]) -> Iterable[str]:
    model = TwitterRobertaBaseSentimentInference()
    outputs = model.predict(docs)
    docs_sentiment = []
    for output in outputs:
        sentiment = max(output, key=output.get)
        docs_sentiment.append(sentiment)
    return docs_sentiment

import re
import nltk
import string
import contractions
import pandas as pd
from typing import List, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


nltk.download([
    'stopwords',
    'punkt',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger'],
    quiet=True
)

TAGS = {
    "J": nltk.corpus.wordnet.ADJ,
    "N": nltk.corpus.wordnet.NOUN,
    "V": nltk.corpus.wordnet.VERB,
    "R": nltk.corpus.wordnet.ADV
}
LEMMATIZER = nltk.stem.WordNetLemmatizer()
PUNCTUATION = set(string.punctuation)


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [
        token for token in tokens
        if token.lower() not in ENGLISH_STOP_WORDS
    ]
def remove_stopwords_and_punct(tokens: List[str]) -> List[str]:
    return [
        token for token in tokens
        if token.lower() not in ENGLISH_STOP_WORDS
        and token not in PUNCTUATION
    ]


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        remove_urls_flg: bool = False,
        remove_mentions_flg: bool = False,
        remove_hashtags_flg: bool = False,
        remove_numbers_flg: bool = False,
        remove_stopwords_flg: bool = False,
        fix_contractions_flg: bool = False,
        lemmatize_flg: bool = False,
        to_lowercase_flg: bool = False
    ):
        self.remove_urls_flg = remove_urls_flg
        self.remove_mentions_flg = remove_mentions_flg
        self.remove_hashtags_flg = remove_hashtags_flg
        self.remove_stopwords_flg = remove_stopwords_flg
        self.fix_contractions_flg = fix_contractions_flg
        self.lemmatize_flg = lemmatize_flg
        self.to_lowercase_flg = to_lowercase_flg
        self.remove_numbers_flg = remove_numbers_flg

    @staticmethod
    def remove_urls(doc: str) -> str:
        return re.sub(r"https?:[^\s]+", '', doc)

    @staticmethod
    def remove_mentions(doc: str) -> str:
        return re.sub('@[A-Za-z0-9_]+', '', doc)

    @staticmethod
    def remove_hashtags(doc: str) -> str:
        return re.sub('#[A-Za-z0-9_]+', '', doc)

    @staticmethod
    def remove_useless(doc: str) -> str:
        pattern = r"https?:[^\s]+|@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|[1-9]"
        return re.sub(pattern, '', doc)

    @staticmethod
    def remove_numbers(doc: str) -> str:
        return re.sub('[1-9]', '', doc)

    @staticmethod
    def fix_contractions(doc: str) -> str:
        return contractions.fix(doc)

    @staticmethod
    def to_lowercase(doc: str) -> str:
        return doc.lower()

    @staticmethod
    def tokenize(doc: str) -> List[str]:
        return nltk.word_tokenize(doc)

    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        return [
            token for token in tokens
            if token.lower() not in ENGLISH_STOP_WORDS
        ]

    @staticmethod
    def remove_stopwords_and_punct(tokens: List[str]) -> List[str]:
        return [
            token for token in tokens
            if token.lower() not in ENGLISH_STOP_WORDS
            and token not in PUNCTUATION
        ]

    @staticmethod
    def lemmatize(tokens: List[str]) -> List[str]:
        tags = nltk.pos_tag(tokens)
        return [
            LEMMATIZER.lemmatize(token, TAGS.get(tag[0], 'n'))
            for token, tag in tags
        ]

    def fit(self, X=None, y=None):
        return self

    def _transform(self, doc: str) -> List[str]:
        doc_clean = doc

        # use single function to speed up processing time
        if self.remove_hashtags_flg\
           and self.remove_mentions_flg\
           and self.remove_urls_flg\
           and self.remove_numbers_flg:
            doc_clean = TextPreprocessor.remove_useless(doc_clean)
        else:
            if self.remove_hashtags_flg:
                doc_clean = TextPreprocessor.remove_hashtags(doc_clean)

            if self.remove_mentions_flg:
                doc_clean = TextPreprocessor.remove_mentions(doc_clean)

            if self.remove_urls_flg:
                doc_clean = TextPreprocessor.remove_urls(doc_clean)

            if self.remove_numbers_flg:
                doc_clean = TextPreprocessor.remove_numbers(doc_clean)

        if self.fix_contractions_flg:
            doc_clean = TextPreprocessor.fix_contractions(doc_clean)

        if self.lemmatize_flg or self.remove_stopwords_flg:
            tokens = TextPreprocessor.tokenize(doc_clean)

            if self.lemmatize_flg:
                tokens = TextPreprocessor.lemmatize(tokens)

            if self.remove_stopwords_flg:
                tokens = TextPreprocessor.remove_stopwords_and_punct(tokens)

            doc_clean = ' '.join(tokens)

        if self.to_lowercase_flg:
            doc_clean = doc_clean.lower()

        return doc_clean

    def transform(self, X: Iterable[str]) -> pd.Series:
        if not isinstance(X, pd.Series):
            return pd.Series(X).apply(self._transform)
        return X.apply(self._transform)

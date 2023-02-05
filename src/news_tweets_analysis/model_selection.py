import os
import time
from tracemalloc import start
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import (
    GridSearchCV,
    PredefinedSplit,
    learning_curve
)
from news_tweets_analysis.preprocessing import TextPreprocessor
from news_tweets_analysis.feature_extraction import (
    AverageWordVectors,
    WeightedAverageWordVectors
)


SCORING = 'f1_macro'
WV_MODEL_NAME = 'glove-wiki-gigaword-300'
DEFAULT_PIPELINE = Pipeline([
    ('extractor', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
EXPERIMENT_SETTINGS = {
    'Classic Model + TFIDF': {
        'preprocessor': TextPreprocessor(
            remove_urls_flg=True,
            remove_hashtags_flg=True,
            remove_mentions_flg=True,
            remove_numbers_flg=True,
            fix_contractions_flg=True,
            remove_stopwords_flg=True,
            lemmatize_flg=True,
            to_lowercase_flg=True
        ),
        'param_grid': [
            {
                'extractor': [TfidfVectorizer()],
                'clf': [LogisticRegression()],
                'extractor__ngram_range': [(1, 1), (1, 2)],
                'extractor__max_features': [1000, 3000, 5000, 10000, None],
                'clf__C': [0.1, 1, 10],
                'clf__max_iter': [1000]
            },
            {
                'extractor': [CountVectorizer()],
                'clf': [MultinomialNB()],
                'extractor__ngram_range': [(1, 1), (1, 2)],
                'extractor__max_features': [1000, 3000, 5000, 10000, None],
                'clf__alpha': [0.1, 1, 10]
            },
            {
                'extractor': [TfidfVectorizer()],
                'clf': [LinearSVC()],
                'extractor__ngram_range': [(1, 1), (1, 2)],
                'extractor__max_features': [1000, 3000, 5000, 10000, None],
                'clf__C': [0.1, 1, 10]
            }
        ]
    },
    'Classic Model + Word Vectors': {
        'preprocessor': TextPreprocessor(
            remove_urls_flg=True,
            remove_hashtags_flg=True,
            remove_mentions_flg=True,
            remove_numbers_flg=True,
            fix_contractions_flg=True,
            to_lowercase_flg=True,
            remove_stopwords_flg=False,
            lemmatize_flg=False
        ),
        'param_grid': {
            'extractor': [
                AverageWordVectors(WV_MODEL_NAME),
                WeightedAverageWordVectors(WV_MODEL_NAME)
            ],
            'clf': [
                LogisticRegression(max_iter=1_000),
                MultinomialNB(),
                LinearSVC()
            ]         
        }
    }
}


def calc_learning_curve(estimator, X, y, scoring=None) -> pd.DataFrame:
    train_sizes, train_score, test_score = learning_curve(
        estimator, X, y,
        scoring=scoring,
        n_jobs=-1,
        verbose=2
    )
    lc = pd.DataFrame({
        'train_size': train_sizes,
        'train_score_mean': train_score.mean(axis=1),
        'train_score_std': train_score.std(axis=1),
        'test_score_mean': test_score.mean(axis=1),
        'test_score_std': test_score.std(axis=1)
    })
    return lc


def run_gridsearch(setting: str, dataset_path: str):
    dataset = pd.read_parquet(dataset_path)
    train = dataset[dataset['split'] == 'train']
    validation = dataset[dataset['split'] == 'validation']
    test = dataset[dataset['split'] == 'test']
    print('Reading datasets...')

    preprocessor = EXPERIMENT_SETTINGS[setting]['preprocessor']
    param_grid = EXPERIMENT_SETTINGS[setting]['param_grid']

    print('Preparing train, validation and tests datasets...')
    X_train, X_val, X_test = train['text'], validation['text'], test['text']
    y_train, y_val, y_test = train['label'], validation['label'], test['label']
    X = np.concatenate((X_train, X_val))
    y = np.concatenate((y_train, y_val))
    test_fold = np.concatenate(
        (np.zeros(len(X_train) - 1) - 1,
        np.ones(len(X_val)))
    )
    cv = PredefinedSplit(test_fold)

    with mlflow.start_run():
        mlflow.set_tag('experiment_setting', setting)

        print('Tuning hyperparameters...')
        gs = GridSearchCV(
            DEFAULT_PIPELINE,
            param_grid,
            scoring=SCORING,
            n_jobs=1,
            cv=cv,
            verbose=1
        )
        gs.fit(X, y)

        print('Training best estimator...')
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        best_model.steps.insert(0, ['preprocessor', preprocessor])
        best_model.fit(X, y)

        start_time = time.time()
        y_pred = best_model.predict(X_test)
        test_time = time.time() - start_time
        test_score = f1_score(y_test, y_pred, average='macro')

        print('Logging artifacts...')
        mlflow.log_params(best_params)
        mlflow.log_metric('val_score', gs.best_score_)
        mlflow.log_metric('test_score', test_score)
        mlflow.log_metric('test_time', test_time)

import os
import mlflow
import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import (
    GridSearchCV,
    PredefinedSplit,
    learning_curve,
    train_test_split
)
from news_tweets_analysis.data import load_tweet_eval
from news_tweets_analysis.preprocessing import TextPreprocessor


MODELS_SETTINGS = {
    'logreg_tfidf': {
        'description': 'Logreg + TFIDF',
        'pipeline': Pipeline([
            ('extractor', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000))
        ]),
        'param_grid': {
            'extractor__ngram_range': [(1, 1), (1, 2)],
            'extractor__max_features': [1000, 3000, 5000, 10000, None],
            'clf__C': [0.1, 1, 10]
        }
    },
    'nb_counts': {
        'description': 'Naive Bayes + counts',
        'pipeline': Pipeline([
            ('extractor', CountVectorizer()),
            ('clf', MultinomialNB())
        ]),
        'param_grid': {
            'extractor__ngram_range': [(1, 1), (1, 2)],
            'extractor__max_features': [1000, 3000, 5000, 10000, None],
            'clf__alpha': [0.1, 1, 10]
        }
    },
    'svm_tfidf': {
        'description': 'SVM + TFIDF',
        'pipeline': Pipeline([
            ('extractor', TfidfVectorizer()),
            ('clf', LinearSVC())
        ]),
        'param_grid': {
            'extractor__ngram_range': [(1, 1), (1, 2)],
            'extractor__max_features': [1000, 3000, 5000, 10000, None],
            'clf__C': [0.1, 1, 10]
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


def run_gridsearch(model: List[str], scoring: str, **preprocessor_params):
    print('Loading datasets...')
    train = load_tweet_eval('train')
    validation = load_tweet_eval('validation')
    print('Train and validation datasets have been downloaded')

    print('Preprocessing datasets...')
    # preprocess tweets first, because it is a time consuming task
    # then we add this as a first step in pipeline
    preprocessor = TextPreprocessor(**preprocessor_params)
    train['text'] = preprocessor.fit_transform(train['text'])
    validation['text'] = preprocessor.transform(validation['text'])
    print('Train and validation datasets have been preprocessed')

    X_train, X_val = train['text'], validation['text']
    y_train, y_val = train['label'], validation['label']
    X = np.concatenate((X_train, X_val))
    y = np.concatenate((y_train, y_val))
    test_fold = np.concatenate(
        (np.zeros(len(X_train) - 1) - 1,
        np.ones(len(X_val)))
    )
    cv = PredefinedSplit(test_fold)

    print('Tuning hyperparameters...')
    for m in model:
        settings = MODELS_SETTINGS[m]
        description = settings['description']
        model = settings['pipeline']
        params = settings['param_grid']
        with mlflow.start_run(description=description):
            print(f'Tuning hyperparameters of {description}...')
            gs = GridSearchCV(
                model, params,
                scoring=scoring,
                n_jobs=-1,
                cv=cv,
                verbose=1
            )
            gs.fit(X, y)

            print('Training best estimator...')
            best_model = gs.best_estimator_
            best_model.steps.insert(0, ['transformer', preprocessor])
            best_model.fit(X_train, y_train)
  
            print('Logging best estimator...')
            mlflow.sklearn.log_model(best_model)
            mlflow.log_metric('val_score', gs.best_score_)


def evaluate(estimator, X, y):
        # best_estimator.fit(X_train, y_train)
        # pred = best_estimator.predict(X_test)
        # test_score = f1_score(y_test, pred, average='macro')
        # mlflow.log_metric('f1_macro', test_score)

        # X = pd.concat([X_train, X_test]).reset_index()
        # y = pd.concat([X_test, y_test]).reset_index()
        # best_estimator.fit(X, y)
        # mlflow.sklearn.log_model(best_estimator, 'model')
    pass

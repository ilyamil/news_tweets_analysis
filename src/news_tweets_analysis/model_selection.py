import os
import mlflow
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
    train_test_split
)


CV = 3
MODELS = [
    'logreg_tfidf',
    'nb_counts',
    'svm_tfidf'
]
SVM_MAX_TRAIN_SIZE = 10_000


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


def tune_hyperparams(estimator, param_grid, X, y, desc, scoring):
    gs = GridSearchCV(
        estimator, param_grid,
        scoring=scoring,
        n_jobs=-1,
        cv=CV,
        return_train_score=True,
        verbose=1
    )
    with mlflow.start_run(description=desc):
        gs.fit(X, y)
        best_estimator = gs.best_estimator_
        cv_results = gs.cv_results_

        lc = calc_learning_curve(best_estimator, X, y, scoring)
        lc.to_csv('learning_curve.csv', index=False)

        mlflow.log_metrics({
            'mean_test_score': cv_results['mean_test_score'],
            'mean_train_score': cv_results['mean_train_score']
        })
        mlflow.log_artifact('learning_curve.csv')
        os.remove('learning_curve.csv')


def tune_and_log(estimator, params, X, y, run_desc, scoring):
    with mlflow.start_run(description=run_desc):
        gs = GridSearchCV(
            estimator, params,
            scoring=scoring,
            n_jobs=-1,
            cv=CV,
            verbose=1
        )
        gs.fit(X, y)
        best_estimator = gs.best_estimator_
        mlflow.sklearn.log_model(best_estimator, 'estimator')
        mlflow.log_metric('val_score', gs.best_score_)    


def tune_logreg(X, y, scoring):
    desc = 'Logreg + TFIDF'
    estimator = Pipeline([
        ('extractor', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    params = {
        'extractor__ngram_range': [(1, 1), (1, 2)],
        'exractor__max_features': [1000, 3000, 5000, 10000, None],
        'clf__penalty': ['l1', 'l2', 'elastic'],
        'clf__C': [0.1, 1, 10]
    }
    tune_and_log(estimator, params, X, y, desc, scoring)


def tune_naive_bayes(X, y, scoring):
    desc = 'Naive Bayes + counts'
    estimator = Pipeline([
        ('extractor', CountVectorizer()),
        ('clf', MultinomialNB())
    ])
    params = {
        'extractor__ngram_range': [(1, 1), (1, 2)],
        'exractor__max_features': [1000, 3000, 5000, 10000],
        'clf__alpha': [0.1, 1, 10]
    }
    tune_and_log(estimator, params, X, y, desc, scoring)


def tune_svm(X, y, scoring):
    desc = 'SVM + TFIDF'
    estimator = Pipeline([
        ('extractor', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', None)
    ])
    params = [
        {
            'extractor__max_features': [1000, 3000, 5000, 10000],
            'clf': [SVC(kernel='rbf')],
            'clf__C': [0.1, 1, 10]
        },
        {
            'extractor__max_features': [1000, 3000, 5000, 10000],
            'clf': [SVC(kernel='poly')],
            'clf__degree': [2, 3]
        },
        {
            'extractor__max_features': [1000, 3000, 5000, 10000],
            'clf': [LinearSVC()],
            'clf__C': [0.1, 1, 10]
        }
    ]
    tune_and_log(estimator, params, X, y, desc, scoring)


def tune_model(X, y, model_nm, scoring):
    if model_nm == 'logreg_tfidf':
        tune_logreg(X, y, scoring)
    elif model_nm == 'nb_counts':
        tune_naive_bayes(X, y, scoring)
    elif model_nm == 'svm_tfidf':
        if len(X) > SVM_MAX_TRAIN_SIZE:
            X_sample, _, y_sample, _ = train_test_split(
                X, y, train_size=SVM_MAX_TRAIN_SIZE
            )
            tune_svm(X_sample, y_sample, scoring)
        else:
            tune_svm(X, y, scoring)


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

from ast import arg
import os
import mlflow
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from news_tweets_analysis.data import load_tweet_eval
from news_tweets_analysis.preprocessing import TextPreprocessor
from news_tweets_analysis.model_selection import MODELS, tune_model


AWS_BUCKET = os.getenv('AWS_BUCKET')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY = os.getenv('AWS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_KEY')
# set hyperparams for cross-validation
RANDOM_STATE = 1
SCORING = 'f1_macro'
# set Mlflow paths
ROOT = Path(__file__).resolve().parent.as_posix()
TRACKING_URI = f'file://{ROOT}/mlruns'
ARTIFACT_LOCATION = f's3://{AWS_BUCKET}/news_tweets_analysis/mlflow_artifacts'



    # tweets = load_tweet_eval()
    # preprocessor = TextPreprocessor(
    #     remove_urls_flg=True,
    #     remove_hashtags_flg=True,
    #     remove_mentions_flg=True,
    #     fix_contractions_flg=True,
    #     remove_stopwords_flg=True,
    #     lemmatize_flg=True,
    #     to_lowercase=True
    # )
    # tweets['text_clean'] = preprocessor.fit_transform(tweets['text'])
    # tweets_train = tweets[tweets['split'].isin(['train', 'validation'])]
    # X_train, y_train = tweets_train['text'], tweets_train['label']
    # X_train_c, y_train_c = tweets_train['text_clean'], tweets_train['label']


argparser = ArgumentParser('Model hyperarameter tuning on grid')
argparser.add_argument(
    '--model',
    nargs='+',
    help='Name of model that require hyperparameter fune-tuning'
)
argparser.add_argument('--experiment', help='Name of MLflow experiment')
argparser.add_argument(
    '--remove_url',
    type=bool,
    default=True,
    help='Text preprocessing: remove urls'
)
argparser.add_argument(
    '--remove_hashtag',
    type=bool,
    default=True,
    help='Text preprocessing: remove hashtags'
)
argparser.add_argument(
    '--remove_mentions',
    type=bool,
    default=True,
    help='Text preprocessing: remove mentions'
)
argparser.add_argument(
    '--remove_numbers',
    type=bool,
    default=True,
    help='Text preprocessing: remove numbers'
)
argparser.add_argument(
    '--fix_contractions',
    type=bool,
    default=True,
    help='Text preprocessing: expand contractions'
)
argparser.add_argument(
    '--lemmatize',
    type=bool,
    default=True,
    help='Text preprocessing: lemmatize tokens'
)
argparser.add_argument(
    '--lowercase',
    type=bool,
    default=True,
    help='Text preprocessing: cast tokens to lowercase'
)


def run(model: List[str], experiment: str):
    print(model)
    # mlflow.set_tracking_uri(TRACKING_URI)
    # if not mlflow.get_experiment_by_name(experiment_name):
    #     mlflow.create_experiment(experiment_name, ARTIFACT_LOCATION)
    # mlflow.set_experiment(experiment_name)

    # print('Loading dataset...')
    # tweets = load_tweet_eval()
    # if preprocessed_flg:
    #     print('Preprocessing dataset...')
    #     preprocessor = TextPreprocessor(
    #         remove_urls_flg=True,
    #         remove_hashtags_flg=True,
    #         remove_mentions_flg=True,
    #         fix_contractions_flg=True,
    #         remove_stopwords_flg=True,
    #         lemmatize_flg=True,
    #         to_lowercase=True
    #     )
    #     tweets['text'] = preprocessor.fit_transform(tweets['text'])

    # tweets_train = tweets[tweets['split'].isin(['train', 'validation'])]
    # X_train, y_train = tweets_train['text'], tweets_train['label']

    # print('Tuning hyperparameters...')
    # tune_model(X_train, y_train, model_name, SCORING)


if __name__ == '__main__':
    args = argparser.parse_args()
    model = args.model
    experiment = args.experiment
    preprocessor_params = {
        'remove_urls': args.remove_url,
        'remove_hashtag': args.remove_hashtag,
        'remove_mentions': args.remove_mentions,
        'remove_numbers': args.remove_numbers,
        'fix_contractions': args.fix_contractions,
        'lemmatize': args.lemmatize,
        'lowercase': args.lowercase
    }
    print(model, experiment)
    print('Preprocessor params:', preprocessor_params)
    # mlflow.set_tracking_uri(TRACKING_URI)
    # if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    #     mlflow.create_experiment(EXPERIMENT_NAME, ARTIFACT_LOCATION)
    # mlflow.set_experiment(EXPERIMENT_NAME)
    # mlflow.sklearn.autolog()
    # np.random.seed(RANDOM_STATE)
    # run()
    pass
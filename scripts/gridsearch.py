import boto3
import os
import mlflow
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from dotenv import load_dotenv
from news_tweets_analysis.model_selection import MODELS, run_gridsearch

load_dotenv()

AWS_BUCKET = os.getenv('AWS_BUCKET')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY = os.getenv('AWS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_KEY')
# set hyperparams for cross-validation
RANDOM_STATE = 1
SCORING = 'f1_macro'
# set Mlflow paths
ROOT = Path(__file__).resolve().parent.parent.as_posix()
TRACKING_URI = f'file://{ROOT}/mlruns'
ARTIFACT_LOCATION = f's3://{AWS_BUCKET}/news_tweets_analysis/mlflow_artifacts'

argparser = ArgumentParser('Model hyperarameter tuning on grid')
argparser.add_argument(
    '--experiment',
    required=True,
    help='MLflow experiment name'
)
argparser.add_argument(
    '--model',
    nargs='+',
    choices=MODELS,
    required=True,
    help='Name of model that require hyperparameter fune-tuning'
)
argparser.add_argument(
    '--scoring',
    default=SCORING,
    help='Scoring being used in Grid Search'
)
argparser.add_argument(
    '--remove_urls',
    type=bool,
    default=True,
    help='Text preprocessing: remove urls'
)
argparser.add_argument(
    '--remove_hashtags',
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
    '--remove_stopwords',
    type=bool,
    default=True,
    help='Text preprocessing: remove stopwords'
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


def gridsearch():
    args = argparser.parse_args()
    model = args.model
    experiment = args.experiment
    scoring = args.scoring
    preprocessor_params = {
        'remove_urls_flg': args.remove_urls,
        'remove_hashtags_flg': args.remove_hashtags,
        'remove_mentions_flg': args.remove_mentions,
        'remove_numbers_flg': args.remove_numbers,
        'fix_contractions_flg': args.fix_contractions,
        'remove_stopwords_flg': args.remove_stopwords,
        'lemmatize_flg': args.lemmatize,
        'to_lowercase_flg': args.lowercase
    }

    mlflow.set_tracking_uri(TRACKING_URI)
    if not mlflow.get_experiment_by_name(experiment):
        mlflow.create_experiment(experiment, ARTIFACT_LOCATION)
    mlflow.set_experiment(experiment)
    np.random.seed(RANDOM_STATE)

    run_gridsearch(model, scoring, **preprocessor_params)


if __name__ == '__main__':
    gridsearch()

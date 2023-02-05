from subprocess import check_output
import boto3
import os
import mlflow
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from dotenv import load_dotenv
from news_tweets_analysis.model_selection import (
    EXPERIMENT_SETTINGS,
    run_gridsearch
)

RANDOM_STATE = 1
# set Mlflow paths
ROOT = Path(__file__).resolve().parent.parent.as_posix()
TRACKING_URI = f'file://{ROOT}/mlruns'
EXPERIMENT_NAME = 'Tuning classic ML models'


argparser = ArgumentParser('Model hyperarameter tuning on grid')
argparser.add_argument(
    '--experiment_setting',
    required=True,
    choices=list(EXPERIMENT_SETTINGS.keys()),
    help='Experiment setting'
)


def gridsearch():
    args = argparser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    np.random.seed(RANDOM_STATE)

    run_gridsearch(args.experiment_setting)


if __name__ == '__main__':
    gridsearch()

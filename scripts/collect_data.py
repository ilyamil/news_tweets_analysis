import os
import boto3
import pandas as pd
from dotenv import load_dotenv
from news_tweets_analysis.utils import load_yaml, write_df_to_s3
from news_tweets_analysis.data import load_new_tweets, update_log_file


load_dotenv()


CONFIG_FILE = os.path.join(
    os.path.dirname(__file__),
    '..',
    'scripts',
    'config.yaml'
)
SOURCES_FILE = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'sources.yaml'
)
TWITTER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
AWS_BUCKET = os.getenv('AWS_BUCKET')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY = os.getenv('AWS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_KEY')


def collect(*args):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
        config=boto3.session.Config(signature_version='s3v4')
    )

    # loading config files
    config = load_yaml(CONFIG_FILE, 'data_collection')
    sources = load_yaml(SOURCES_FILE, 'sources')

    # data collection
    freq = config['frequency']
    end_dt = pd.Timestamp.now('utc').floor(f'{freq}min')
    start_dt = end_dt - pd.offsets.Minute(freq)
    user_id = [s['twitter_id'] for s in sources.values()]
    tweets = load_new_tweets(
        token=TWITTER_TOKEN,
        user_id=user_id,
        max_results=config['max_tweets_per_user'],
        start_dt=start_dt.to_pydatetime(),
        end_dt=end_dt.to_pydatetime()
    )

    # write tweets to s3 and update logs
    data_key = config['path_template'].format(end_dt.strftime('%Y%m%d%H%M%S'))
    write_df_to_s3(s3_client, tweets, AWS_BUCKET, data_key, format='csv')
    update_log_file(s3_client, config['updates_path'], AWS_BUCKET,
                    start_dt, end_dt, len(tweets), data_key)


if __name__ == '__main__':
    collect()

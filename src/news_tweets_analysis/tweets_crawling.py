import tweepy
import pandas as pd
from typing import List
from datetime import datetime
from news_tweets_analysis.utils import (
    write_df_to_s3,
    read_df_from_s3,
    check_file
)


def load_new_tweets(
    token: str,
    user_id: List[int],
    max_results: int,
    start_dt: datetime,
    end_dt: datetime
) -> pd.DataFrame:
    """
    Load new tweets of some users.

    Args:
        token (str): Twitter API token. For more info
            go to https://developer.twitter.com/en/products/twitter-api
        user_id (List[int]): Twitter ID of users.
        max_results (int): Maximum number of tweets per one user.
        start_dt (datetime): Minimum date-time of posted tweet.
        end_dt (datetime): Maximum date-time of posted tweet.

    Returns:
        pd.DataFrame: Tweets from users with some meta and public metrics:
        likes and reposts.
    """
    client = tweepy.Client(token)
    users_tweets = []
    for uid in user_id:
        response = client.get_users_tweets(
            uid,
            max_results=max_results,
            start_time=start_dt,
            end_time=end_dt,
            tweet_fields=['public_metrics', 'created_at']
        )
        if not response.data:
            continue

        tweets_ = pd.DataFrame({
            'user_id': [uid]*len(response.data),
            'tweet_id': [tweet.id for tweet in response.data],
            'text': [tweet.text for tweet in response.data],
            'created_at': [tweet.created_at for tweet in response.data],
            'public_metrics': [tweet.public_metrics for tweet in response.data]
        })
        tweets = (
            tweets_
            .join(tweets_['public_metrics'].apply(pd.Series))
            .drop(['public_metrics'], axis=1)
        )
        users_tweets.append(tweets)
    return pd.concat(users_tweets)


def update_log_file(client, log_filename: str, bucket: str,
                    start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                    num_records: int, data_filename: str):
    updates_file_exist = check_file(client, bucket, log_filename)
    dtype = {
        'start_dt': 'datetime64[ns, UTC]',
        'end_dt': 'datetime64[ns, UTC]',
        'update_dt': 'datetime64[ns, UTC]',
        'num_records': 'int',
        'file_path': 'str'
    }
    if updates_file_exist:
        records = read_df_from_s3(client, bucket, log_filename, format='csv')
    else:
        records = pd.DataFrame(columns=dtype.keys())

    records = records.astype(dtype)

    new_record = pd.DataFrame({
        'start_dt': [start_dt],
        'end_dt': [end_dt],
        'update_dt': [pd.Timestamp.now('utc')],
        'num_records': [num_records],
        'file_path': [data_filename]
    }).astype(dtype)

    updates = pd.concat([records, new_record], ignore_index=True)
    write_df_to_s3(client, updates, bucket, log_filename, format='csv')
    print('Log file has been updated successfully!')

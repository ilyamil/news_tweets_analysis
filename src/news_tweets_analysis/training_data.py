import pandas as pd
import datasets


DATASETS = ['tweet_eval']


def load_tweet_eval():
    tweets = datasets.load_dataset('tweet_eval', 'sentiment')
    tweets_pdf = (
        pd.concat([
            pd.DataFrame(v).assign(split=k)
            for k, v in tweets.items()
        ])
        .reset_index(drop=True)
    )
    return tweets_pdf


def download_data(dataset: str, dst_path: str) -> pd.DataFrame:
    if dataset == 'tweet_eval':
        data = load_tweet_eval()
    else:
        raise ValueError(
            f'Unsupported dataset. Pick one from list: {DATASETS}'
        )
    data.to_parquet(dst_path)

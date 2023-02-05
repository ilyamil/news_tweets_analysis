from argparse import ArgumentParser
from news_tweets_analysis.training_data import download_data


parser = ArgumentParser(prog='Download dataset')
parser.add_argument(
    '--dataset',
    type=str,
    help='Dataset name'
)
parser.add_argument(
    '--dst_path',
    type=str,
    help='Destination path'
)


if __name__ == '__main__':
    args = parser.parse_args()
    download_data(args.dataset, args.dst_path)

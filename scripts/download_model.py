from argparse import ArgumentParser
from news_tweets_analysis.model_setup import download_model


parser = ArgumentParser(prog='Download pretrained huggingface model')
parser.add_argument(
    '--model',
    type=str,
    help='Huggingface model from Model Hub'
)
parser.add_argument(
    '--folder',
    type=str,
    help='Destination folder'
)


if __name__ == '__main__':
    args = parser.parse_args()
    download_model(args.model, args.folder)

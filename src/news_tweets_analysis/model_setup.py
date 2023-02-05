import os
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


def download_model(model_name: str, dst_folder: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer.save_pretrained(dst_folder)
    model.save_pretrained(dst_folder)

    if os.path.isdir(dst_folder) and os.listdir(dst_folder):
        print(f'{model_name} has been downloaded successfully!')
    else:
        print('Something went wrong, there is no model in destination folder')

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


def download_model(model_name: str, dst_folder: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=dst_folder)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=dst_folder
    )
    return tokenizer, model

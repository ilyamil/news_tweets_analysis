# Analysis of news tweets posted by well-known media 

# Results Reproducibility
1. Create Conda virtual environment

    ```conda create -n news_tweets_analysis_env python==3.9.12```

2. Activate this virtual environment

    ```conda activate news_tweets_analysis_env```

3. Install package with code related to analysis of tweets

    ```pip install -e .```

4. Download datasets to train, test and evaluate model quality (from root)
    1. For sentiment classification
        ```
        python scripts/download_data.py \
            --dataset tweet_eval \
            --dst_path ./data/tweet_eval.parquet
        ```

5. Run hyperparameter tuning script to find best (shallow) ML model
    ```
    python scripts/gridsearch.py \
        --experiment_setting "Classic Model + TFIDF" \
        --dataset data/tweet_eval.py
    ```

6. Download pretrained models
    1. For sentiment classification
        ```
        python scripts/download_model.py \
            --model cardiffnlp/twitter-roberta-base-sentiment-latest \
            --folder ./models
        ```
    2. 
from news_tweets_analysis.preprocessing import TextPreprocessor


def test_text_preprocessor_remove_url():
    doc = 'Link to google https://google.com'
    doc_without_link = TextPreprocessor.remove_urls(doc)
    assert doc_without_link == 'Link to google '


def test_text_preprocessor_fix_contractions():
    doc = "you're happy now"
    doc_without_contractions = TextPreprocessor.fix_contractions(doc)
    assert doc_without_contractions == 'you are happy now'


def test_text_preprocessor_remove_mentions():
    doc = 'What do you think @JohnDoe?'
    doc_without_mentions = TextPreprocessor.remove_mentions(doc)
    assert doc_without_mentions == 'What do you think ?'


def test_text_preprocessor_remove_hashtags():
    doc = 'Save Data Scientists from automl #DS #AutoML'
    doc_without_hashtags = TextPreprocessor.remove_hashtags(doc)
    assert doc_without_hashtags == 'Save Data Scientists from automl  '


def test_text_preprocessor_remove_numbers():
    doc = 'Sentence with numbers 123999'
    doc_without_numbers = TextPreprocessor.remove_numbers(doc)
    assert doc_without_numbers == 'Sentence with numbers '


def test_text_preprocessor_to_lowercase():
    doc = "FIRST HALF IN UPPERCASE, second half in lowercase"
    doc_lowercase = TextPreprocessor.to_lowercase(doc)
    assert doc_lowercase == 'first half in uppercase, second half in lowercase'


def test_text_preprocessor_lemmatize():
    tokens = ['These', 'cats', 'were', 'smaller', 'than', 'I', 'thought']
    tokens_true_lemmas = ['These', 'cat', 'be', 'small', 'than', 'I', 'think']
    assert TextPreprocessor.lemmatize(tokens) == tokens_true_lemmas

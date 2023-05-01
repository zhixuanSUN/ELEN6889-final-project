import re
import pandas as pd


def preprocess_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.lower()
    return text


file_name = 'tweeter.csv'
data = pd.read_excel(file_name)
data['cleaned_text'] = data['tweet_text'].apply(preprocess_tweet)
print(data['clean_text'])


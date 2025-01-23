import pandas as pd
import spacy
import re

# Load the data from the CSV file
data_path = '/Users/viktorkotuliak/Projects/unga_bunga/data/cyberbullying_tweets.csv'
data = pd.read_csv(data_path)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_tweet(tweet):
    # Remove URLs, mentions, and hashtags
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    
    # Tokenize and lemmatize
    doc = nlp(tweet)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    return " ".join(tokens)

# Apply preprocessing to all tweets
data['processed_tweet'] = data['tweet_text'].head(10).apply(preprocess_tweet)

for i in range(5):
    print(data['tweet_text'][i])
    print(data['processed_tweet'][i])
    print()
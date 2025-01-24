import pandas as pd
import spacy
import re
from tqdm import tqdm

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
# data['processed_tweet'] = data['tweet_text'].apply(preprocess_tweet)
tqdm.pandas()
data['processed_tweet'] = data['tweet_text'].progress_apply(preprocess_tweet)

# Save the processed tweets and their corresponding cyberbullying type to a new CSV file
output_path = '/Users/viktorkotuliak/Projects/unga_bunga/data/processed_cyberbullying_tweets.csv'
data[['processed_tweet', 'cyberbullying_type']].to_csv(output_path, index=False)
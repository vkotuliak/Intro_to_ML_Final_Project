from sklearn.model_selection import train_test_split
import pandas as pd

def split_data() -> tuple:
    # Load the data from the CSV file
    df = pd.read_csv('data/processed_cyberbullying_tweets.csv')

    # Split the data into training and testing sets
    X = df['processed_tweet']
    y = df['cyberbullying_type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test

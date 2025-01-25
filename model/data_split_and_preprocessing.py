from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

np.random.seed(103)

# Load the data from the CSV file
df = pd.read_csv("data/processed_cyberbullying_tweets.csv")

# Split the data into training and testing sets
X = df["processed_tweet"]
y = df["cyberbullying_type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2
)

# Save the training, validation, and testing sets to CSV files
train_data = pd.DataFrame(
    {"processed_tweet": X_train, "cyberbullying_type": y_train}
)
val_data = pd.DataFrame(
    {"processed_tweet": X_val, "cyberbullying_type": y_val}
)
test_data = pd.DataFrame(
    {"processed_tweet": X_test, "cyberbullying_type": y_test}
)

train_data.to_csv("data/train_data.csv", index=False)
val_data.to_csv("data/val_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)

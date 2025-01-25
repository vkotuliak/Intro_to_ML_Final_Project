import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from gensim.models import Word2Vec
import numpy as np


# import data
df_test = pd.read_csv("data/test_data.csv")
df_train = pd.read_csv("data/train_data.csv")
df_val = pd.read_csv("data/val_data.csv")

# Split data into x and y
x_test = df_test["processed_tweet"]
y_test = df_test["cyberbullying_type"]

x_train = df_train["processed_tweet"]
y_train = df_train["cyberbullying_type"]

x_val = df_val["processed_tweet"]
y_val = df_val["cyberbullying_type"]

# Label encode the target variable
Encoder = LabelEncoder()

y_test = Encoder.fit_transform(y_test)
y_train = Encoder.fit_transform(y_train)
y_val = Encoder.fit_transform(y_val)

# # Vectorize the word data
Vectorizer = TfidfVectorizer()

x_train_tfidf = Vectorizer.fit_transform(x_train.values.astype("U"))
x_test_tfidf = Vectorizer.transform(x_test.values.astype("U"))
x_val_tfidf = Vectorizer.transform(x_val.values.astype("U"))

# Train Word2Vec model
# x_train = x_train.astype(str)
# x_test = x_test.astype(str)
# x_val = x_val.astype(str)

# # Train Word2Vec model
# sentences = [tweet.split() for tweet in x_train.tolist()]
# word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# # Function to transform tweets to Word2Vec vectors
# def tweet_to_vec(tweet, model):
#     words = tweet.split()
#     word_vecs = [model.wv[word] for word in words if word in model.wv]
#     return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(model.vector_size)

# # Vectorize the word data using Word2Vec
# x_train_w2v = np.array([tweet_to_vec(tweet, word2vec_model) for tweet in x_train])
# x_test_w2v = np.array([tweet_to_vec(tweet, word2vec_model) for tweet in x_test])
# x_val_w2v = np.array([tweet_to_vec(tweet, word2vec_model) for tweet in x_val])

# The Naive Bayes model
model = LogisticRegression(max_iter=1000)

param_grid = {"C": [0.1, 1.0, 10.0], "penalty": ["l2"], "solver": ["lbfgs"]}

# Set up the grid search to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(x_train_tfidf, y_train)
# grid_search.fit(x_train_w2v, y_train)
best_model = grid_search.best_estimator_

predictions = best_model.predict(x_val_tfidf)

print(f"Best parameters found: {grid_search.best_params_}")


# Calculate accuracy
print(f"Accuracy: {accuracy_score(predictions, y_val):.4f}")
print(f"F1 Score: {f1_score(predictions, y_val, average='macro'):.4f}")
print(f"Precision: {precision_score(predictions, y_val, average='macro'):.4f}")
print(f"Recall: {recall_score(predictions, y_val, average='macro'):.4f}")

# Print all the parameter combinations with their accuracy
print("\nAccuracy for all alphas:")
for mean_score, params in zip(
    grid_search.cv_results_["mean_test_score"],
    grid_search.cv_results_["params"],
):
    print(
        f"Alpha: {params['var_smoothing']}, Fit Prior: {params['priors']} Accuracy: {mean_score:.4f}"
    )

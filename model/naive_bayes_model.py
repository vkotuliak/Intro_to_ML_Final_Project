import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


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

# Vectorize the word data
Vectorizer = TfidfVectorizer()

x_train_tfidf = Vectorizer.fit_transform(x_train.values.astype("U"))
x_test_tfidf = Vectorizer.transform(x_test.values.astype("U"))
x_val_tfidf = Vectorizer.transform(x_val.values.astype("U"))


# The Naive Bayes model
model = MultinomialNB()

param_grid = {"alpha": [0.1, 0.5, 1.0, 5.0, 10.0], "fit_prior": [True, False]}

# Set up the grid search to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(x_train_tfidf, y_train)
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
        f"Alpha: {params['alpha']}, Fit Prior: {params['fit_prior']} Accuracy: {mean_score:.4f}"
    )

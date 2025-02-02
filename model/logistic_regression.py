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


# Logistic Regression model

model2 = LogisticRegression(max_iter=1000)
model2.fit(x_train_tfidf, y_train)
predictions = model2.predict(x_val_tfidf)
print(f"Accuracy: {accuracy_score(predictions, y_val):.4f}")
print(f"F1 Score: {f1_score(predictions, y_val, average='macro'):.4f}")
print(f"Precision: {precision_score(predictions, y_val, average='macro'):.4f}")
print(f"Recall: {recall_score(predictions, y_val, average='macro'):.4f}")


model = LogisticRegression(max_iter=1000)

# Define the hyperparameters to search through
param_grid = [
    {
        "penalty": ["l2"],
        "C": [0.1, 1, 10, 100],
        "solver": ["lbfgs"],
        "class_weight": [None, "balanced"],
    },
    {
        "penalty": ["l1"],
        "C": [0.1, 1, 10],
        "solver": ["liblinear"],
        "class_weight": [None, "balanced"],
    },
]

# Set up the grid search to find the best hyperparameters
grid_search = GridSearchCV(
    model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2
)
grid_search.fit(x_train_tfidf, y_train)
best_model = grid_search.best_estimator_

predictions = best_model.predict(x_val_tfidf)

print(f"Best parameters found: {grid_search.best_params_}")

# Calculate accuracy
print(f"Accuracy: {accuracy_score(predictions, y_val):.4f}")
print(f"F1 Score: {f1_score(predictions, y_val, average='macro'):.4f}")
print(f"Precision: {precision_score(predictions, y_val, average='macro'):.4f}")
print(f"Recall: {recall_score(predictions, y_val, average='macro'):.4f}")

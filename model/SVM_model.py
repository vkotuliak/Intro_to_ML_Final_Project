import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, learning_curve
import matplotlib.pyplot as plt

# Import data
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

x_train_tfidf = Vectorizer.fit_transform(x_train.values.astype('U'))
x_test_tfidf = Vectorizer.transform(x_test.values.astype('U'))
x_val_tfidf = Vectorizer.transform(x_val.values.astype('U'))

# The SVM model
# param_grid = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf'],
#     'gamma': ['scale', 'auto']
# }

# # Initialize the GridSearchCV object to find the best hyperparameters
# grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
# grid_search.fit(x_train_tfidf, y_train)

# best_params = grid_search.best_params_
# print(f"Best parameters found: {best_params}")

# # Print all the parameter combinations with their accuracy
# means = grid_search.cv_results_['mean_test_score']
# stds = grid_search.cv_results_['std_test_score']
# params = grid_search.cv_results_['params']

# for mean, std, param in zip(means, stds, params):
#     print(f"Mean accuracy: {mean:.3f} (std: {std:.3f}) with parameters: {param}")

# Use the best parameters to initialize the SVM model
SVM = SVC(kernel="linear", C=1)

SVM.fit(x_train_tfidf, y_train)

predictions = SVM.predict(x_train_tfidf)

print(accuracy_score(predictions, y_train))
print(f1_score(predictions, y_train, average="macro"))
print(precision_score(predictions, y_train, average="macro"))
print(recall_score(predictions, y_train, average="macro"))

predictions = SVM.predict(x_test_tfidf)

print(accuracy_score(predictions, y_test))
print(f1_score(predictions, y_test, average="macro"))
print(precision_score(predictions, y_test, average="macro"))
print(recall_score(predictions, y_test, average="macro"))
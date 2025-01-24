import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
SVM = SVC()
SVM.fit(x_train_tfidf, y_train)

predictions = SVM.predict(x_val_tfidf)

print(accuracy_score(predictions, y_val))
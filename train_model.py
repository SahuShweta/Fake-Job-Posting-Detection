import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/fake_job_postings.csv")

# ==========================
# Fill Missing Values
# ==========================

df.fillna("", inplace=True)

# ==========================
# Combine Text Columns
# ==========================

df["text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

# ==========================
# Clean Text
# ==========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# ==========================
# Features and Target
# ==========================

X = df["text"]
y = df["fraudulent"]

# ==========================
# TF-IDF Vectorization
# ==========================

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(X)

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Logistic Regression
# ==========================

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n==============================")
print("Logistic Regression")
print("==============================")

print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall   :", recall_score(y_test, lr_pred))
print("F1 Score :", f1_score(y_test, lr_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

# ==========================
# Naive Bayes
# ==========================

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\n==============================")
print("Naive Bayes")
print("==============================")

print("Accuracy :", accuracy_score(y_test, nb_pred))
print("Precision:", precision_score(y_test, nb_pred))
print("Recall   :", recall_score(y_test, nb_pred))
print("F1 Score :", f1_score(y_test, nb_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, nb_pred))

print("\nClassification Report")
print(classification_report(y_test, nb_pred))

# ==========================
# Save Best Model
# ==========================

joblib.dump(lr_model, "model/fake_job_model.pkl")

joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel Saved Successfully!")

with open("model_results.txt", "w") as file:

    file.write("===== Logistic Regression =====\n")
    file.write(classification_report(y_test, lr_pred))

    file.write("\n\n===== Naive Bayes =====\n")
    file.write(classification_report(y_test, nb_pred))
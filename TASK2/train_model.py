import os
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/SMSSpamCollection"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "spam_classifier.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean an SMS message before vectorization.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Keep letters and numbers
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"]
)


print(f"Total messages: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# DATA CLEANING
# ============================================================

df = df.dropna()

df["message"] = df["message"].apply(
    clean_text
)

# Remove duplicate messages
df = df.drop_duplicates(
    subset=["message"]
)

# Convert labels
df["label"] = df["label"].map(
    {
        "ham": 0,
        "spam": 1
    }
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df["message"]
y = df["label"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

print("\nCreating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    "TF-IDF matrix shape:",
    X_train_tfidf.shape
)


# ============================================================
# MODEL 1 — MULTINOMIAL NAIVE BAYES
# ============================================================

print("\nTraining Multinomial Naive Bayes...")

nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_predictions = nb_model.predict(
    X_test_tfidf
)


# ============================================================
# MODEL 2 — LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_predictions = lr_model.predict(
    X_test_tfidf
)


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Ham", "Spam"],
            zero_division=0
        )
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


nb_results = evaluate_model(
    "Multinomial Naive Bayes",
    y_test,
    nb_predictions
)

lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

results = [
    nb_results,
    lr_results
]

results_df = pd.DataFrame(
    results
)

print("\nModel Comparison:")
print(results_df)


best_model_name = results_df.loc[
    results_df["f1"].idxmax(),
    "model"
]


if best_model_name == "Multinomial Naive Bayes":

    best_model = nb_model
    best_predictions = nb_predictions

else:

    best_model = lr_model
    best_predictions = lr_predictions


print(
    f"\nBest model selected: {best_model_name}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)


plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    best_model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)


print("\nModel saved to:")
print(MODEL_PATH)

print("\nVectorizer saved to:")
print(VECTORIZER_PATH)

print("\nTraining completed successfully! ✅")
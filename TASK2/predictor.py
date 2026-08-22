import os
import re

import joblib


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "spam_classifier.pkl"
)

VECTORIZER_PATH = os.path.join(
    "models",
    "tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

model = joblib.load(
    MODEL_PATH
)

vectorizer = joblib.load(
    VECTORIZER_PATH
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean an SMS message before prediction.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

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

    # Keep letters, numbers and spaces
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
# PREDICTION FUNCTION
# ============================================================

def predict_spam(message):
    """
    Predict whether a message is spam or ham.

    Returns:
        label
        confidence
    """

    cleaned_message = clean_text(
        message
    )

    if not cleaned_message:
        return "unknown", 0.0

    # Convert message into TF-IDF features
    message_vector = vectorizer.transform(
        [cleaned_message]
    )

    # Predict class
    prediction = model.predict(
        message_vector
    )[0]

    # Get probability
    probabilities = model.predict_proba(
        message_vector
    )[0]

    confidence = float(
        max(probabilities)
    )

    if prediction == 1:
        label = "spam"
    else:
        label = "ham"

    return label, confidence
# 🛡️ Spam Message Detector

A machine-learning powered SMS spam classifier with an interactive Streamlit web app. Enter any SMS/text message and the model predicts whether it's **spam** or **legitimate (ham)**, along with a confidence score and real model evaluation metrics.

---

## Features

- 🔍 **Real-time spam prediction** with confidence score
- 📊 **Model Performance dashboard** — accuracy, precision, recall, F1 score, and a confusion matrix computed on a held-out test set
- 🧠 Automatically selects the best of two trained models (Naive Bayes vs. Logistic Regression) based on F1 score
- 🎨 Clean, dark-themed UI built with Streamlit

---

## Project Structure

```
.
├── app.py                      # Streamlit web app
├── predictor.py                # Loads saved model/vectorizer and runs predictions
├── train.py                    # Trains, evaluates, and saves the model
├── data/
│   └── SMSSpamCollection        # Training dataset (tab-separated: label, message)
├── models/                     # Created after running train.py
│   ├── spam_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── eval_results.json
├── confusion_matrix.png        # Generated after running train.py
└── README.md
```

---

## How It Works

```
SMS Message → Text Cleaning → TF-IDF Vectorization →
Naive Bayes / Logistic Regression → Spam / Ham Prediction → Confidence Score
```

The model is trained on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset. During training, both a **Multinomial Naive Bayes** and a **Logistic Regression** model are trained on TF-IDF features (unigrams + bigrams), evaluated on a stratified 80/20 train/test split, and the better-performing model (by F1 score) is saved for use in the app.

---

## Setup

### 1. Clone / download the project

Make sure the following files are in place:
- `app.py`
- `predictor.py`
- `train.py`
- `data/SMSSpamCollection`

### 2. Install dependencies

```bash
pip install streamlit pandas scikit-learn joblib matplotlib seaborn
```

(Use a virtual environment if you'd like to keep dependencies isolated.)

### 3. Train the model

```bash
python train.py
```

This will:
- Load and clean the dataset
- Train both candidate models
- Print evaluation metrics for each to the terminal
- Save the best model to `models/spam_classifier.pkl` and `models/tfidf_vectorizer.pkl`
- Save evaluation metrics + confusion matrix to `models/eval_results.json`
- Save a confusion matrix plot to `confusion_matrix.png`

### 4. Run the app

```bash
streamlit run app.py
```

Or, for headless environments (servers, containers):

```bash
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
```

By default the app opens at `http://localhost:8501`.

---

## Model Performance

Once you've run `train.py`, the app's **Model Performance** section will automatically display:

- Which model was selected as best (Naive Bayes or Logistic Regression)
- Accuracy, Precision, Recall, and F1 Score on the held-out test set
- A color-coded confusion matrix (true positives/negatives in green, false positives/negatives in red)
- Dataset stats (training size, test size, total messages after cleaning)

These numbers come directly from `models/eval_results.json`, generated fresh each time you run `train.py` — not hardcoded.

---

## Tech Stack

- **Python**
- **Streamlit** — web app framework
- **scikit-learn** — TF-IDF vectorization, Naive Bayes, Logistic Regression, evaluation metrics
- **pandas** — data loading and cleaning
- **joblib** — model persistence
- **matplotlib / seaborn** — confusion matrix visualization (training-time only)

---

## Notes

- The dataset is cleaned (lowercased, URLs/emails stripped, duplicates removed) before training and before every prediction, using the same `clean_text()` logic in both `train.py` and `predictor.py`.
- If `models/eval_results.json` is missing, the app will still run — it just shows a prompt to run `train.py` first instead of the performance section.

---

## Author

Built by **Mehul Gupta**
import json
import os
from textwrap import dedent

import streamlit as st

from predictor import predict_spam


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    dedent("""
    <style>

    .stApp {
        background: #0f172a;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-header {
        text-align: center;
        padding: 1rem 0 1.5rem 0;
    }

    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.4rem;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    .info-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
    }

    .result-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 25px;
        margin-top: 20px;
        text-align: center;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .confidence-text {
        color: #94a3b8;
        font-size: 1rem;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sidebar-text {
        color: #94a3b8;
        line-height: 1.6;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 35px;
        padding: 15px;
    }

    .cm-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }

    .cm-table th {
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 8px;
        text-align: center;
    }

    .cm-table td {
        text-align: center;
        padding: 18px 10px;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .cm-correct {
        background: rgba(34, 197, 94, 0.18);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }

    .cm-wrong {
        background: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .cm-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 500;
        display: block;
        margin-top: 4px;
    }

    </style>
    """),
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🛡️ Spam Detector</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        dedent("""
        <div class="sidebar-text">
        An ML-powered SMS spam detection system
        built using Python and Scikit-learn.
        </div>
        """),
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🧠 Machine Learning")

    st.markdown(
        """
        - TF-IDF Vectorization
        - Multinomial Naive Bayes / Logistic Regression
        - Supervised Classification
        - Probability-based Prediction
        """
    )

    st.divider()

    st.markdown("### 👨‍💻 Developer")

    st.markdown(
        dedent("""
        <div style="
            text-align: center;
            color: #94a3b8;
            font-size: 0.85rem;
            padding: 10px;
        ">
            Made by <strong style="color: #e2e8f0;">
            Mehul Gupta
            </strong>
        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    dedent("""
    <div class="main-header">
        <h1>🛡️ Spam Message Detector</h1>
        <p>
            Machine-learning powered SMS spam classification
        </p>
    </div>
    """),
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    dedent("""
    <div class="info-card">
        <h3>📩 Check a Message</h3>
        <p style="color:#94a3b8;">
            Enter an SMS message below and the trained machine
            learning model will determine whether it is spam
            or a legitimate message.
        </p>
    </div>
    """),
    unsafe_allow_html=True
)


# ============================================================
# MESSAGE INPUT
# ============================================================

message = st.text_area(
    "Enter your message",
    placeholder=(
        "Example: Congratulations! "
        "You have won a free prize..."
    ),
    height=160
)


# ============================================================
# BUTTONS
# ============================================================

col1, col2 = st.columns(2)

with col1:
    check_message = st.button("🔍 Check Message", use_container_width=True)

with col2:
    clear_message = st.button("🧹 Clear", use_container_width=True)

if clear_message:
    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if check_message:

    if not message.strip():

        st.warning("Please enter a message before checking.")

    else:

        label, confidence = predict_spam(message)
        confidence_percentage = confidence * 100

        if label == "spam":

            st.markdown(
                dedent(f"""
                <div class="result-card">
                    <div class="result-title">🚨 SPAM MESSAGE</div>
                    <div class="confidence-text">
                        The model classified this message as spam.
                    </div>
                    <h2>{confidence_percentage:.2f}%</h2>
                    <div class="confidence-text">Prediction Confidence</div>
                </div>
                """),
                unsafe_allow_html=True
            )

            st.error(
                "⚠️ Be careful with links, requests for money, "
                "prizes, or personal information."
            )

        elif label == "ham":

            st.markdown(
                dedent(f"""
                <div class="result-card">
                    <div class="result-title">✅ NOT SPAM</div>
                    <div class="confidence-text">
                        The model classified this message as legitimate.
                    </div>
                    <h2>{confidence_percentage:.2f}%</h2>
                    <div class="confidence-text">Prediction Confidence</div>
                </div>
                """),
                unsafe_allow_html=True
            )

            st.success("✅ This message appears to be legitimate.")

        else:

            st.warning("Unable to classify the message.")


# ============================================================
# EXAMPLE MESSAGES
# ============================================================

st.divider()
st.markdown("### 💡 Try These Examples")

example_col1, example_col2 = st.columns(2)

with example_col1:
    st.markdown("**🚨 Spam Example**")
    st.code(
        "Congratulations! You have won a free "
        "iPhone. Click here to claim your prize!"
    )

with example_col2:
    st.markdown("**✅ Normal Example**")
    st.code("Hey, are we still meeting at 6 today?")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()
st.markdown("### 📊 Model Performance")

EVAL_RESULTS_PATH = os.path.join("models", "eval_results.json")

if not os.path.exists(EVAL_RESULTS_PATH):

    st.info(
        "Run `train.py` to generate `models/eval_results.json` "
        "and unlock this section with real evaluation metrics."
    )

else:

    with open(EVAL_RESULTS_PATH, "r") as f:
        eval_results = json.load(f)

    model_name = eval_results["model_name"]
    accuracy = eval_results["accuracy"] * 100
    precision = eval_results["precision"] * 100
    recall = eval_results["recall"] * 100
    f1 = eval_results["f1"] * 100

    cm = eval_results["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    dataset = eval_results["dataset"]

    st.markdown(
        dedent(f"""
        <div class="info-card">
            <p style="color:#94a3b8; margin-bottom: 6px;">
                Best performing model on the held-out test set:
            </p>
            <h3 style="margin-top: 0;">{model_name}</h3>
            <p style="color:#64748b; font-size: 0.85rem;">
                Trained on {dataset['train_size']} messages,
                evaluated on {dataset['test_size']} unseen
                test messages
                ({dataset['total_after_cleaning']} total after
                cleaning and de-duplication).
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Accuracy", f"{accuracy:.2f}%")

    with metric_col2:
        st.metric("Precision", f"{precision:.2f}%")

    with metric_col3:
        st.metric("Recall", f"{recall:.2f}%")

    with metric_col4:
        st.metric("F1 Score", f"{f1:.2f}%")

    st.markdown("#### Confusion Matrix")

    st.markdown(
        dedent(f"""
        <div class="info-card">
        <table class="cm-table">
            <tr>
                <th></th>
                <th>Predicted: Ham</th>
                <th>Predicted: Spam</th>
            </tr>
            <tr>
                <th>Actual: Ham</th>
                <td class="cm-correct">
                    {tn}
                    <span class="cm-label">True Negative</span>
                </td>
                <td class="cm-wrong">
                    {fp}
                    <span class="cm-label">False Positive</span>
                </td>
            </tr>
            <tr>
                <th>Actual: Spam</th>
                <td class="cm-wrong">
                    {fn}
                    <span class="cm-label">False Negative</span>
                </td>
                <td class="cm-correct">
                    {tp}
                    <span class="cm-label">True Positive</span>
                </td>
            </tr>
        </table>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.caption(
        "False Positive = a legitimate message wrongly flagged as spam. "
        "False Negative = a spam message that slipped through undetected."
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

with st.expander("🧠 How does the AI work?"):

    st.markdown(
        """
        ### Prediction Pipeline

        **SMS Message**

        ↓

        **Text Cleaning**

        ↓

        **TF-IDF Vectorization**

        ↓

        **Multinomial Naive Bayes / Logistic Regression**

        ↓

        **Spam / Ham Prediction**

        ↓

        **Confidence Score**

        The model was trained on the UCI SMS Spam Collection
        dataset using labeled spam and legitimate messages,
        with the best of two candidate models (by F1 score)
        selected automatically during training.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    dedent("""
    <div class="footer">
        Built with Python • Streamlit • Scikit-learn
    </div>
    """),
    unsafe_allow_html=True
)
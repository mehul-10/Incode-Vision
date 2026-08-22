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
    """
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

    </style>
    """,
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
        """
        <div class="sidebar-text">
        An ML-powered SMS spam detection system
        built using Python and Scikit-learn.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🧠 Machine Learning")

    st.markdown(
        """
        - TF-IDF Vectorization
        - Multinomial Naive Bayes
        - Supervised Classification
        - Probability-based Prediction
        """
    )

    st.divider()

    st.markdown("### 📊 Model Performance")

    st.metric(
        "Accuracy",
        "97.76%"
    )

    st.metric(
        "Precision",
        "100%"
    )

    st.metric(
        "F1 Score",
        "89.87%"
    )

    st.divider()

    st.markdown("### 👨‍💻 Developer")

    st.markdown(
        """
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
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ Spam Message Detector</h1>
        <p>
            Machine-learning powered SMS spam classification
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
    <div class="info-card">
        <h3>📩 Check a Message</h3>
        <p style="color:#94a3b8;">
            Enter an SMS message below and the trained machine
            learning model will determine whether it is spam
            or a legitimate message.
        </p>
    </div>
    """,
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

    check_message = st.button(
        "🔍 Check Message",
        use_container_width=True
    )


with col2:

    clear_message = st.button(
        "🧹 Clear",
        use_container_width=True
    )


if clear_message:

    st.rerun()

# ============================================================
# PREDICTION
# ============================================================

if check_message:

    if not message.strip():

        st.warning(
            "Please enter a message before checking."
        )

    else:

        label, confidence = predict_spam(message)

        confidence_percentage = confidence * 100

        if label == "spam":

            st.error("🚨 SPAM MESSAGE")

            st.markdown(
                "The model classified this message as **spam**."
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence_percentage:.2f}%"
            )

            st.warning(
                "⚠️ Be careful with links, requests for money, "
                "prizes, or personal information."
            )

        elif label == "ham":

            st.success("✅ NOT SPAM")

            st.markdown(
                "The model classified this message as **legitimate**."
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence_percentage:.2f}%"
            )

            st.info(
                "✅ This message appears to be legitimate."
            )

        else:

            st.warning(
                "Unable to classify the message."
            )
        # ----------------------------------------------------
        # SPAM RESULT
        # ----------------------------------------------------

        if label == "spam":

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">
                        🚨 SPAM MESSAGE
                    </div>

                    <div class="confidence-text">
                        The model classified this message
                        as spam.
                    </div>

                    <h2>
                        {confidence_percentage:.2f}%
                    </h2>

                    <div class="confidence-text">
                        Prediction Confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.error(
                "⚠️ Be careful with links, requests "
                "for money, prizes, or personal information."
            )


        # ----------------------------------------------------
        # HAM RESULT
        # ----------------------------------------------------

        elif label == "ham":

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">
                        ✅ NOT SPAM
                    </div>

                    <div class="confidence-text">
                        The model classified this message
                        as a legitimate message.
                    </div>

                    <h2>
                        {confidence_percentage:.2f}%
                    </h2>

                    <div class="confidence-text">
                        Prediction Confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "✅ This message appears to be legitimate."
            )

        else:

            st.warning(
                "Unable to classify the message."
            )


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

    st.code(
        "Hey, are we still meeting at 6 today?"
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
        
        **Multinomial Naive Bayes**
        
        ↓
        
        **Spam / Ham Prediction**
        
        ↓
        
        **Confidence Score**

        The model was trained on the UCI SMS Spam Collection
        dataset using labeled spam and legitimate messages.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with Python • Streamlit • Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)
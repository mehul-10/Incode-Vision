import streamlit as st
from chatbot import get_response


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

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

    .chat-header {
        text-align: center;
        padding: 1rem 0 1.5rem 0;
    }

    .chat-header h1 {
        font-size: 2.4rem;
        margin-bottom: 0.3rem;
    }

    .chat-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    .welcome-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        text-align: center;
    }

    .welcome-card h3 {
        margin-bottom: 8px;
    }

    .welcome-card p {
        color: #94a3b8;
    }

    .suggestion-title {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-bottom: 10px;
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
        margin-top: 30px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🤖 AI Chatbot</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
        A simple NLP-powered chatbot built using
        Python and Streamlit.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🧠 Technologies")

    st.markdown(
        """
        - Python
        - Streamlit
        - NLTK
        - Scikit-learn
        - TF-IDF
        - Cosine Similarity
        """
    )

    st.divider()

    st.markdown("### 💡 Try asking")

    st.markdown(
        """
        - What is AI?
        - What is NLP?
        - What is Python?
        - What can you do?
        - Who are you?
        """
    )

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="chat-header">
        <h1>🤖 AI Chatbot</h1>
        <p>
            A simple NLP-powered conversational assistant
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# WELCOME SCREEN
# --------------------------------------------------

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 Welcome!</h3>
            <p>
                I'm your AI chatbot. Ask me something
                about AI, NLP, Python, or my capabilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar="🤖" if message["role"] == "assistant" else "👤"
    ):
        st.markdown(message["content"])


# --------------------------------------------------
# SUGGESTED QUESTIONS
# --------------------------------------------------

if not st.session_state.messages:

    st.markdown(
        '<div class="suggestion-title">💡 Suggested questions</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "What is Artificial Intelligence?",
            use_container_width=True
        ):

            user_input = "What is Artificial Intelligence?"

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

            response, intent, confidence = get_response(
                user_input
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            st.rerun()

    with col2:

        if st.button(
            "What is NLP?",
            use_container_width=True
        ):

            user_input = "What is NLP?"

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

            response, intent, confidence = get_response(
                user_input
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            st.rerun()


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

user_input = st.chat_input(
    "Type your message here..."
)


if user_input:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Generate response
    response, intent, confidence = get_response(
        user_input
    )

    # Add chatbot response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Built with Python • Streamlit • NLTK • Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)
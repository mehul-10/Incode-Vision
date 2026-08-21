import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# INTENTS
# ============================================================

INTENTS = {
    "greeting": {
        "patterns": [
            "hello",
            "hi",
            "hey",
            "hey there",
            "hello there",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "how are you doing",
            "how is it going",
            "how are things",
            "how have you been"
        ],
        "responses": [
            "Hello! 👋 How can I help you today?",
            "Hi there! 😊 What can I do for you?",
            "Hey! 👋 Nice to chat with you!",
            "I'm doing great, thanks for asking! 😊 How can I help you?"
        ]
    },

    "goodbye": {
        "patterns": [
            "bye",
            "goodbye",
            "see you",
            "see you later",
            "talk to you later",
            "have a good day",
            "good night",
            "catch you later"
        ],
        "responses": [
            "Goodbye! 👋 Have a great day!",
            "See you later! 😊",
            "Bye! It was nice chatting with you!",
            "Take care! 👋"
        ]
    },

    "thanks": {
        "patterns": [
            "thanks",
            "thank you",
            "thank you so much",
            "thanks a lot",
            "thanks for helping",
            "thank you for helping",
            "thank you for your help",
            "thanks for your help",
            "i appreciate your help",
            "i appreciate it",
            "much appreciated"
        ],
        "responses": [
            "You're welcome! 😊",
            "Happy to help!",
            "Anytime! 👍",
            "You're very welcome!"
        ]
    },

    "about": {
        "patterns": [
            "who are you",
            "who are you exactly",
            "what are you",
            "what kind of chatbot are you",
            "tell me about yourself",
            "tell me about you",
            "about you",
            "introduce yourself",
            "can you introduce yourself",
            "what is your purpose"
        ],
        "responses": [
            "I'm a simple AI chatbot built using Python, NLP, and Streamlit.",
            "I'm an NLP-based chatbot designed to understand user intents and respond accordingly."
        ]
    },

    "capabilities": {
        "patterns": [
            "what can you do",
            "what do you do",
            "what can you help me with",
            "how can you help me",
            "what are your capabilities",
            "what are your features",
            "what features do you have",
            "what are you capable of",
            "tell me what you can do",
            "tell me your capabilities",
            "how can you assist me",
            "what kind of things can you do",
            "what services do you provide"
        ],
        "responses": [
            "I can understand common user intents, answer basic questions, and have simple conversations with you.",
            "I can respond to questions about AI, NLP, Python, and my own capabilities."
        ]
    },

    "ai": {
        "patterns": [
            "what is artificial intelligence",
            "what is ai",
            "define artificial intelligence",
            "explain artificial intelligence",
            "tell me about ai",
            "tell me about artificial intelligence",
            "meaning of ai",
            "explain ai",
            "what does ai mean",
            "artificial intelligence definition",
            "can you explain ai",
            "can you tell me about ai"
        ],
        "responses": [
            "Artificial Intelligence (AI) is the field of creating machines and software that can perform tasks that normally require human intelligence, such as learning, reasoning, problem-solving, and decision-making."
        ]
    },

    "nlp": {
        "patterns": [
            "what is natural language processing",
            "what is nlp",
            "define nlp",
            "explain natural language processing",
            "tell me about nlp",
            "meaning of nlp",
            "explain nlp",
            "what does nlp mean",
            "natural language processing definition",
            "can you explain nlp",
            "can you tell me about nlp",
            "what does natural language processing do"
        ],
        "responses": [
            "Natural Language Processing (NLP) is a branch of Artificial Intelligence that enables computers to understand, process, analyze, and generate human language."
        ]
    },

    "python": {
        "patterns": [
            "what is python",
            "tell me about python",
            "why use python",
            "python programming",
            "what is python used for",
            "explain python",
            "python language",
            "python programming language",
            "can you explain python",
            "can you tell me about python",
            "why is python popular"
        ],
        "responses": [
            "Python is a popular high-level programming language known for its simple syntax and wide use in web development, automation, data science, machine learning, and AI."
        ]
    },

    "help": {
        "patterns": [
            "help",
            "i need help",
            "can you help me",
            "help me",
            "i need assistance",
            "could you help me",
            "can you assist me",
            "i need some assistance"
        ],
        "responses": [
            "Of course! 😊 You can ask me about AI, NLP, Python, or my capabilities."
        ]
    }
}


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Normalize user input for intent detection.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Normalize common multi-word AI/NLP terms
    text = text.replace(
        "artificial intelligence",
        "artificial_intelligence"
    )

    text = text.replace(
        "natural language processing",
        "natural_language_processing"
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-zA-Z0-9_\s]",
        "",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# PREPARE TF-IDF TRAINING DATA
# ============================================================

patterns = []
intent_labels = []

for intent_name, intent_data in INTENTS.items():
    for pattern in intent_data["patterns"]:
        cleaned_pattern = preprocess_text(pattern)

        patterns.append(cleaned_pattern)
        intent_labels.append(intent_name)


# ============================================================
# TF-IDF MODEL
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

pattern_vectors = vectorizer.fit_transform(patterns)


# ============================================================
# EXACT / CONVERSATIONAL MATCHES
# ============================================================

EXACT_INTENTS = {
    # Greetings
    "hello": "greeting",
    "hi": "greeting",
    "hey": "greeting",
    "hey there": "greeting",
    "hello there": "greeting",
    "good morning": "greeting",
    "good afternoon": "greeting",
    "good evening": "greeting",
    "how are you": "greeting",
    "how are you doing": "greeting",
    "how is it going": "greeting",
    "how are things": "greeting",
    "how have you been": "greeting",

    # Goodbye
    "bye": "goodbye",
    "goodbye": "goodbye",
    "see you": "goodbye",
    "see you later": "goodbye",
    "talk to you later": "goodbye",
    "have a good day": "goodbye",
    "good night": "goodbye",
    "catch you later": "goodbye",

    # Thanks
    "thanks": "thanks",
    "thank you": "thanks",
    "thank you so much": "thanks",
    "thanks a lot": "thanks",
    "thanks for helping": "thanks",
    "thank you for helping": "thanks",
    "thank you for your help": "thanks",
    "thanks for your help": "thanks",
    "i appreciate your help": "thanks",
    "i appreciate it": "thanks",
    "much appreciated": "thanks",

    # About
    "who are you": "about",
    "who are you exactly": "about",
    "what are you": "about",
    "what kind of chatbot are you": "about",
    "tell me about yourself": "about",
    "tell me about you": "about",
    "about you": "about",
    "introduce yourself": "about",
    "can you introduce yourself": "about",
    "what is your purpose": "about",

    # Capabilities
    "what can you do": "capabilities",
    "what do you do": "capabilities",
    "what can you help me with": "capabilities",
    "how can you help me": "capabilities",
    "what are your capabilities": "capabilities",
    "what are your features": "capabilities",
    "what features do you have": "capabilities",
    "what are you capable of": "capabilities",
    "tell me what you can do": "capabilities",
    "tell me your capabilities": "capabilities",
    "how can you assist me": "capabilities",
    "what kind of things can you do": "capabilities",
    "what services do you provide": "capabilities"
}


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(user_input):
    """
    Detect the most appropriate intent and return:
    (intent, confidence_score)
    """

    cleaned_input = preprocess_text(user_input)

    if not cleaned_input:
        return "unknown", 0.0

    # --------------------------------------------------------
    # 1. Exact conversational matching
    # --------------------------------------------------------

    if cleaned_input in EXACT_INTENTS:
        intent = EXACT_INTENTS[cleaned_input]
        return intent, 1.0

    # --------------------------------------------------------
    # 2. Important concept matching
    # --------------------------------------------------------

    input_words = set(cleaned_input.split())

    if (
        "nlp" in input_words
        or "natural_language_processing" in input_words
    ):
        return "nlp", 0.95

    if (
        "ai" in input_words
        or "artificial_intelligence" in input_words
    ):
        return "ai", 0.95

    if "python" in input_words:
        return "python", 0.95

    # --------------------------------------------------------
    # 3. TF-IDF + Cosine Similarity
    # --------------------------------------------------------

    user_vector = vectorizer.transform(
        [cleaned_input]
    )

    similarities = cosine_similarity(
        user_vector,
        pattern_vectors
    )[0]

    best_index = similarities.argmax()

    best_score = min(
        float(similarities[best_index]),
        1.0
    )

    best_intent = intent_labels[best_index]

    # --------------------------------------------------------
    # 4. Confidence threshold
    # --------------------------------------------------------

    # Prevent unrelated questions from being assigned
    # to the nearest available intent.
    if best_score < 0.30:
        return "unknown", best_score

    return best_intent, best_score


# ============================================================
# RESPONSE GENERATION
# ============================================================

def get_response(user_input):
    """
    Return:
        response
        detected intent
        confidence score
    """

    intent, confidence = detect_intent(
        user_input
    )

    # --------------------------------------------------------
    # Unknown question
    # --------------------------------------------------------

    if intent == "unknown":
        return (
            "I'm sorry, I don't understand that yet. "
            "Try asking me about AI, NLP, Python, or my capabilities.",
            "unknown",
            confidence
        )

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    response = random.choice(
        INTENTS[intent]["responses"]
    )

    return (
        response,
        intent,
        confidence
    )
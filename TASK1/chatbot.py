import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# INTENTS
# --------------------------------------------------

INTENTS = {
    "greeting": {
        "patterns": [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "hey there"
        ],
        "responses": [
            "Hello! 👋 How can I help you today?",
            "Hi there! 😊 What can I do for you?",
            "Hey! 👋 Nice to chat with you!"
        ]
    },

    "goodbye": {
        "patterns": [
            "bye",
            "goodbye",
            "see you",
            "see you later",
            "talk to you later",
            "have a good day"
        ],
        "responses": [
            "Goodbye! 👋 Have a great day!",
            "See you later! 😊",
            "Bye! It was nice chatting with you!"
        ]
    },

    "thanks": {
        "patterns": [
            "thanks",
            "thank you",
            "thank you so much",
            "thanks a lot",
            "thx",
            "I appreciate your help"
        ],
        "responses": [
            "You're welcome! 😊",
            "Happy to help!",
            "Anytime! 👍"
        ]
    },

    "about": {
        "patterns": [
            "who are you",
            "what are you",
            "tell me about yourself",
            "about you",
            "introduce yourself"
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
            "your capabilities",
            "what are your features",
            "how can you help me",
            "what can you help me with",
            "what are you capable of",
            "what can you help with"
        ],
        "responses": [
            "I can understand common user intents, answer basic questions, and have simple conversations with you."
        ]
    },

    "ai": {
        "patterns": [
            "what is artificial intelligence",
            "what is AI",
            "define artificial intelligence",
            "explain artificial intelligence",
            "tell me about AI",
            "meaning of AI",
            "explain AI",
            "what does AI mean",
            "artificial intelligence definition"
        ],
        "responses": [
            "Artificial Intelligence (AI) is the field of creating machines and software that can perform tasks that normally require human intelligence, such as learning, reasoning, problem-solving, and decision-making."
        ]
    },

    "nlp": {
        "patterns": [
            "what is natural language processing",
            "what is NLP",
            "define NLP",
            "explain natural language processing",
            "tell me about NLP",
            "meaning of NLP",
            "explain NLP",
            "what does NLP mean",
            "natural language processing definition"
        ],
        "responses": [
            "Natural Language Processing (NLP) is a branch of Artificial Intelligence that enables computers to understand, process, analyze, and generate human language."
        ]
    },

    "python": {
        "patterns": [
            "what is Python",
            "tell me about Python",
            "why use Python",
            "Python programming",
            "what is Python used for",
            "explain Python",
            "Python language",
            "Python programming language"
        ],
        "responses": [
            "Python is a popular high-level programming language known for its simple syntax and wide use in web development, automation, data science, machine learning, and AI."
        ]
    },

    "help": {
        "patterns": [
            "help",
            "I need help",
            "can you help me",
            "help me",
            "I need assistance"
        ],
        "responses": [
            "Of course! 😊 You can ask me about AI, NLP, Python, or my capabilities."
        ]
    }
}


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

def preprocess_text(text):
    """
    Clean and normalize text.
    """

    text = text.lower()

    # Normalize common AI/NLP variations
    text = text.replace("artificial intelligence", "artificial_intelligence")
    text = text.replace("natural language processing", "natural_language_processing")

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9_\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# PREPARE TRAINING DATA
# --------------------------------------------------

patterns = []
intent_labels = []

for intent_name, intent_data in INTENTS.items():

    for pattern in intent_data["patterns"]:

        patterns.append(
            preprocess_text(pattern)
        )

        intent_labels.append(
            intent_name
        )


# --------------------------------------------------
# TF-IDF MODEL
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

pattern_vectors = vectorizer.fit_transform(patterns)


# --------------------------------------------------
# INTENT DETECTION
# --------------------------------------------------

def detect_intent(user_input):

    cleaned_input = preprocess_text(user_input)

    # Explicit keyword recognition for important concepts
    input_words = set(cleaned_input.split())

    if (
        "nlp" in input_words
        or "natural_language_processing" in input_words
    ):
        if any(
            word in input_words
            for word in [
                "nlp",
                "natural_language_processing"
            ]
        ):
            return "nlp", 0.95

    if (
        "ai" in input_words
        or "artificial_intelligence" in input_words
    ):
        return "ai", 0.95

    if "python" in input_words:
        return "python", 0.95

    # TF-IDF similarity
    user_vector = vectorizer.transform(
        [cleaned_input]
    )

    similarities = cosine_similarity(
        user_vector,
        pattern_vectors
    )[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]
    best_intent = intent_labels[best_index]

    return best_intent, best_score


# --------------------------------------------------
# RESPONSE GENERATION
# --------------------------------------------------

def get_response(user_input):

    intent, confidence = detect_intent(
        user_input
    )

    if confidence < 0.30:

        return (
            "I'm sorry, I don't understand that yet. "
            "Try asking me about AI, NLP, Python, or my capabilities."
        ), "unknown", confidence

    response = random.choice(
        INTENTS[intent]["responses"]
    )

    return response, intent, confidence
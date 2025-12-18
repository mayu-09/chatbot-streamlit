streamlit
scikit-learn

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

intents = {
    "hello": "Hi there! How can I help you today?",
    "bye": "Goodbye! Have a great day.",
    "thanks": "You're welcome!",
    "waste": "Try composting food scraps and recycling paper, plastic, and glass.",
    "methane": "Methane is a greenhouse gas. Capturing and converting it can reduce emissions."
}

training_phrases = [
    "hello", "hi", "hey", "good morning",
    "bye", "goodbye", "see you later",
    "thanks", "thank you", "appreciate it",
    "waste", "reduce waste", "how to recycle", "trash management",
    "methane", "methane gas", "tell me about methane", "greenhouse gas methane"
]

labels = [
    "hello", "hello", "hello", "hello",
    "bye", "bye", "bye",
    "thanks", "thanks", "thanks",
    "waste", "waste", "waste", "waste",
    "methane", "methane", "methane", "methane"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_phrases)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    best_match_index = similarities.argmax()
    best_intent = labels[best_match_index]
    return intents.get(best_intent, "Sorry, I don't understand that yet.")

st.title("Smart Chatbot 🌱")
user_input = st.text_input("You:", "")
if user_input:
    st.write("Bot:", chatbot_response(user_input))


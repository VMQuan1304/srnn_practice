import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = load_model('imdb_rnn_model.keras')

def decode_review(encoded_review):
    word_index = imdb.get_word_index()
    reversed_word_index = {value: key for key, value in word_index.items()}
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_review(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_review(review):
    padded_review = preprocess_review(review)
    prediction = model.predict(padded_review)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return  sentiment, prediction[0][0]


st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive/negative):")

user_input = st.text_area("Movie Review", "Type your review here...")
if st.button("Predict Sentiment"):
    if user_input:
        sentiment, confidence = predict_review(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction probability: {confidence:.2f}")
    else:
        st.write("Please enter a review to get a prediction.")
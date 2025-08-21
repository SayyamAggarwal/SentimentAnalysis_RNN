import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reversed_word_index = {value:key for key,value in word_index.items()}
model = load_model("SentimentalPrediction.keras")

def decode_review(encoded_text):
    return " ".join([reversed_word_index.get(i-3,'?') for i in word_index])



def add_padding(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word  in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review





def predict_sentiment(review):
    encoded_review = add_padding(review)
    prediction = model.predict(encoded_review)
    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]



st.title("IMDB Movie Review Sentimental Analysis")

st.write("Enter a movie review to classify it as positive and negative")

user_input = st.text_area("Movie Review")


if st.button("Predict"):
    encoded_review = add_padding(user_input)

    prediction = model.predict(encoded_review)

    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'


    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction score: {prediction[0][0]}")


else:
    st.write("Please write a movie review")










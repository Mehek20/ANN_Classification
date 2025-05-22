import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Load the IMDB dataset word index
word_index = imdb.get_word_index()
# Reverse the word index to get words from indices
reverse_word_index = dict((value, key) for (key, value) in word_index.items())

model = load_model('simple_rnn_imdb.h5')

## Step2 : Helper Functions
## Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

## Function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    # Pad the sequence to the same length as the training data
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Prediction function


## Streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive/negative):")
user_input = st.text_area("Movie Review", "Type your review here...")

if st.button("Predict Sentiment"):
    preprocess_input = preprocess_text(user_input)
    
    ## Predict sentiment
    predict = model.predict(preprocess_input)
    sentiment = 'positive' if predict[0][0] > 0.5 else 'negative'

    ## Display the result
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Prediction Score: {predict[0][0]}")
else:
    st.write('Please enter a review and click the button to predict sentiment.')
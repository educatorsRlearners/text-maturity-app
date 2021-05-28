import streamlit as st
import pandas as pd
import numpy as np

import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model



st.write("""
# Text Maturity - Beta

This app predicts the **MINIMUM** age required for reading a book!

Data was obtained from [Common Sense Media](https://www.commonsensemedia.org/book-reviews) using [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) by [Evan Simpson](https://github.com/educatorsRlearners/book-maturity).
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters

genre_list = ['Activity','Adventure', 'Advice', 'Alphabet', 'Animals', 'Anthropology', 'Arts','Autobiography', 'Biography','Board', 'Body Awareness',
        'Colors','Coming of Age', 'Contemporary Fiction','Counting', 'Emotions', 'Fairy Tale', 'Family Life', 'Fantasy', 'Folklore', 'For Beginning Readers',
        'Friendship', 'Graphic Novel','Historical Fiction', 'History', 'Holiday', 'Horror', 'Humor', 'Learning', 'Literary Fiction', 'Math', 'Media', 'Mystery', 'Picture Book',
        'Poetry', 'Romance', 'School', 'Science', 'Science Fiction', 'Short Stories', 'Sports', 'Technology', 'Transportation', 'Travel', 'Words']

book_type_list = ['Fiction', 'Non-fiction']

st.sidebar.header('Book Details')

def user_input_features():
    title = st.sidebar.text_input("Book Title", label="Enter the title")
    genre = st.sidebar.selectbox("Genre", options=genre_list)
    bookType = st.sidebar.selectbox('Book Type', options=book_type_list)
    data = {'title': title,
            'genre': genre,
            'book type': bookType}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model= load_model('lstm_title_no_year_h5.h5')


path = 'lstm_title_no_year_h5_tokenizer.pickle'

# load tokenizer
with open(path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Apply Model to Make Prediction
tester = df['title'] + " " + df["genre"] + " " + df["book type"]
test_entry = tokenizer.texts_to_sequences(list(tester))
test_entry_padded = pad_sequences(test_entry, maxlen=29)
prediction = (round(model.predict(test_entry_padded)[0][0], 0))


st.header('Minimum Recommended Age')
st.write(prediction)
#st.write(tester)
st.write('---')

import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import time

# set model and vectorizer
model_prediksi = joblib.load(open("model_prediksi.pkl","rb"))
vectorizer = joblib.load(open("vectorizer.pkl","rb"))

# make a text processor funtion
def olah_kata(text):
    # symbols and tabs removal
    text = re.sub(r'[^\w\s]', '', text)
    
    # lowercasing
    text = text.lower()
    
    # set stopwords and remove them from text
    stopword = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stopword]

    # stemming the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Rejoin words into a single string
    processed_txt = ' '.join(words)

    # make into list before vectorization
    processed_txt = [processed_txt]

    # and then vectorise the list
    vectorised_txt = vectorizer.transform(processed_txt)

    return vectorised_txt

def prediksi_txt(text):
    text = olah_kata(text)
    res_pred = model_prediksi.predict(text)
    return res_pred

# set tabs
tab1, tab2 = st.tabs(["Home", "About"])

with tab1:
    st.header("Text Based Misinformation and Disinformation Detector")
    st.write("Guide:")
    st.write("1. Put an article/text (in english) in the text box")
    st.write("2. Click the Predict button")
    st.write("Panduan:")
    st.write("1. Masukkan sebuah artikel/teks (dalam bahasa inggris) ke dalam kotak teks")
    st.write("2. Klik tombol Predict")

    Text = st.text_area(
        label="",
        height="content",
        placeholder="Insert text Input Here...")
    
    # if button is clicked or ctrl+enter pressed
    if st.button(type="primary",width="stretch",label="Predict") or Text:
        if len(Text.strip()) > 0:
            with st.spinner("Processing...", show_time=True):
                time.sleep(0.5)
                res = prediksi_txt(Text)
                if res == 1:
                    st.success("This article is true.", icon="✅")
                else:
                    st.error("This article is fake.", icon="❌")
        else:
            st.error("You didn't insert a text!", icon="🚨")

with tab2:
    st.header("About the App")
    
    st.write("This web app is made using streamlit, and the model used for classifying text input is the Support Vector Machine with linear kernel. " \
    "The model is trained with the ISOT news dataset which contained over 40,000 news article, labelled as True or Fake. " \
    "The trained model achieved a 99% in accuracy, precision, and recall")

    st.write("Aplikasi berbasis web ini dibuat menggunakan streamlit, dan model yang digunakan untuk mengklasifikasi masukkan dalam bentuk teks adalah Support Vector Machine dengan kernel Linear. " \
    "Model ini dilatih menggunakan data set berita ISOT yang memiliki lebih dari 40.000 artikel berita, yang sudah dilabeli sebagai True (nyata) atau Fake (Palsu). " \
    "Model yang telah dilatih mencapai 99% dalam akurasi, presisi, dan recall")
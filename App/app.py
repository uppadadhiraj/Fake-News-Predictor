import streamlit as st
import pandas as pd
import pickle as pkl
from newspaper import Article
import joblib
import re
import sklearn
from sklearn.preprocessing import LabelEncoder

model = pkl.load(open('model.pkl', 'rb'))
vectorizer = pkl.load(open('vectorizer.pkl', 'rb'))
le = pkl.load(open('le.pkl', 'rb'))

def scrape_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

# Text preprocessing (replicate training preprocessing steps)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    # Add other preprocessing as needed here
    return text

# Prediction function to classify the article text
def predict(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    confidence = max(model.predict_proba(vectorized_text)[0])
    return prediction, confidence

# Streamlit UI
st.title("Indian News Fake News Detector")
url = st.text_input("Enter the news article URL")

if st.button("Check News"):
    with st.spinner("Analyzing article..."):
        try:
            article_text = scrape_article(url)
            if len(article_text) < 20:
                st.warning("Couldn't extract sufficient text from the given URL.")
            else:
                label, confidence = predict(article_text)
                predicted_label = le.inverse_transform([label])[0]
                st.success(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")
        except Exception as e:
            st.error(f"Failed to process URL: {e}")
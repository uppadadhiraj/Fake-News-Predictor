# Indian News Fake News Detector

## Project Overview
This project aims to build an AI-powered fake news detector specialized for Indian news articles. Users can input a news article URL through a sleek Streamlit web interface. The app scrapes the full article text, preprocesses it, and predicts if the news is "Fake" or "Real" using a trained ML model built on the Indian Fake News dataset.

## Features
- URL-based news article text extraction with Newspaper3k
- Machine Learning classification model trained on Indian Fake News dataset
- Interactive and easy-to-use Streamlit interface
- Confidence score display for each prediction

## Technologies
- Python 3
- Streamlit for UI
- Newspaper3k for scraping
- Scikit-learn for ML and vectorization
- Pickle for saving/loading model artifacts

## Setup and Installation

### Prerequisites
Install dependencies:
pip install -r requirements.md

### Training and Saving Artifacts
In your Jupyter Notebook after training your model, save the following (last three lines):
import pickle

'pickle.dump(model, open('model.pkl', 'wb'))'

'pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))'

'pickle.dump(le,open('le.pkl','wb'))'

### Deployment Steps
1. Copy the files `model.pkl`, `vectorizer.pkl`, and `labelencoder.pkl` into the same folder as your Streamlit app file (`app.py`).
2. Run the app:
   
streamlit run app.py

3. Open the URL provided by Streamlit in your browser to start using the fake news detector.

## Usage
- Enter a valid Indian news article URL.
- Press "Check News".
- View prediction (Fake/Real) and confidence score.

## Troubleshooting
- Ensure `.pkl` files are correctly placed in app directory.
- If scraping fails, try a different news URL or check website access policies.
- Validate your saved model and vectorizer files correspond to your training setup.

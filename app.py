import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model & vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# UI
st.title(" TRUTHLENS-Fake News Detection App")
st.write("Paste a news title below and click Detect")

user_input = st.text_area("News Title")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter a news title")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

import pandas as pd
import nltk
import string
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

# Load dataset
data = pd.read_csv("FakeNewsNet.csv", encoding="utf-8", engine="python")
data = data[['title', 'real']]
data.columns = ['text', 'label']

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data['clean_text'] = data['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully")

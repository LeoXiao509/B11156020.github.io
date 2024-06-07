# preprocess.py

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    print(f"原始文本: {text}")
    tokens = nltk.word_tokenize(text)
    print(f"令牌: {tokens}")
    tokens = [word.lower() for word in tokens if word.isalnum()]
    print(f"過濾後的令牌: {tokens}")
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    print(f"去掉停用詞後: {tokens}")
    return ' '.join(tokens)

def extract_features(texts):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, 'vectorizer.pkl')
    return features

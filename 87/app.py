# app.py

from flask import Flask, request, jsonify
import joblib
from preprocess import preprocess_text

app = Flask(__name__)

# 加載模型和向量化器
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = preprocess_text(data['text'])
    features = vectorizer.transform([text])
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

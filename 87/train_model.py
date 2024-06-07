# train_model.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from preprocess import preprocess_text, extract_features

# 模型訓練函數
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    return model

if __name__ == '__main__':
    # 示例數據
    texts = [
        "This is a normal message.",
        "This is a fraudulent message.",
        "Another normal message.",
        "Suspicious activity detected.",
        "Normal communication.",
        "Fraudulent attempt detected.",
        "Please send your bank details.",
        "Your account has been compromised.",
        "Win a prize now!",
        "Click this link for a reward.",
        "Normal work-related message.",
        "Meeting scheduled for tomorrow."
    ]
    labels = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]  # 0代表正常，1代表詐騙

    # 預處理和特徵提取
    processed_texts = [preprocess_text(text) for text in texts]
    features = extract_features(processed_texts)

    # 訓練模型
    train_model(features, labels)

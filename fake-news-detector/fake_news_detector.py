# fake_news_detector.py
# Basic placeholder class

from sklearn.feature_extraction.text import TfidfVectorizer

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def load_and_prepare_data(self, fake_path="data/Fake.csv", true_path="data/True.csv"):
    import pandas as pd

    # Load both datasets
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Label them
    df_fake["label"] = 1  # fake = 1
    df_true["label"] = 0  # real = 0

    # Combine
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine title + text (optional but improves context)
    df['processed_text'] = (df['title'].astype(str) + " " + df['text'].astype(str)).str.lower()

    return df


    def train_models(self, X_train_vec, y_train, X_test_vec, y_test):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_train_vec, y_train)
        acc = model.score(X_test_vec, y_test)
        print(f"Logistic Regression Accuracy: {acc:.2f}")
        return {"Logistic Regression": acc}

    def predict_news(self, text):
        vec = self.vectorizer.transform([text.lower()])
        print("Prediction not implemented. Please extend this method.")

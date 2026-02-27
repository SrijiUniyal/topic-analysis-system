import pandas as pd
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class TopicClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", max_features=5000,min_df=1,sublinear_tf=True)
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced")

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text

    def train(self, X, y):
        X_processed = [self.preprocess_text(t) for t in X]
        X_features = self.vectorizer.fit_transform(X_processed)
        self.model.fit(X_features, y)
        return X_features

    def predict(self, text):
        processed = self.preprocess_text(text)
        features = self.vectorizer.transform([processed])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        return prediction, max(probability)

def main():
    print("📌 Loading dataset...")
    df = pd.read_csv("data/topic_dataset.csv")
    df = df.dropna(subset=["topic", "text", "label"])
    df["topic"] = df["topic"].astype(str)
    df["text"] = df["text"].astype(str)

    # Combine topic + text
    df["combined"] = df["topic"] + " " + df["text"]

    X = df["combined"].values
    y = df["label"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    classifier = TopicClassifier()
    classifier.train(X_train, y_train)

    # Evaluate
    X_test_processed = [classifier.preprocess_text(t) for t in X_test]
    X_test_features = classifier.vectorizer.transform(X_test_processed)
    y_pred = classifier.model.predict(X_test_features)

    print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Disadvantage", "Advantage"]))

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(classifier.model, "models/trained_model.pkl")
    joblib.dump(classifier.vectorizer, "models/vectorizer.pkl")
    joblib.dump(classifier, "models/full_classifier.pkl")
    print("\n💾 Model saved in /models")

if __name__ == "__main__":
    main()

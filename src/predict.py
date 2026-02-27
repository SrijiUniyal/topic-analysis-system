import os
import re
import joblib
import pandas as pd
from datetime import datetime
from textblob import TextBlob


class TopicPredictor:
    def __init__(self):
        # Load trained model and vectorizer
        self.model = joblib.load("models/trained_model.pkl")
        self.vectorizer = joblib.load("models/vectorizer.pkl")
        self.dataset = pd.read_csv("data/topic_dataset.csv")

        # Ensure outputs folder exists
        os.makedirs("outputs", exist_ok=True)

    # ✅ SAME preprocessing as train_model.py
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.strip()

    def predict_sentiment(self, topic, text):

        # Combine topic + user input
        combined_text = topic + " " + text

        # Apply preprocessing
        combined_text = self.preprocess_text(combined_text)

        # Convert to vector
        X = self.vectorizer.transform([combined_text])

        # If no valid words found
        if X.nnz == 0:
            return {
                "sentiment": "unknown",
                "confidence": 0.0,
                "polarity": 0.0,
                "subjectivity": 0.0,
                "top_features": [],
                "opposite_responses": ["No valid input for prediction"]
            }

        # Predict probabilities
        proba = self.model.predict_proba(X)[0]
        pred_label = int(proba.argmax())
        confidence = round(float(proba.max()) * 100, 2)

        sentiment = "advantage" if pred_label == 1 else "disadvantage"

        # TextBlob sentiment (only on user text)
        blob = TextBlob(text)
        polarity = round(blob.sentiment.polarity, 3)
        subjectivity = round(blob.sentiment.subjectivity, 3)

        # ✅ Top important words
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[0]

        nonzero_indices = X.nonzero()[1]

        top_features = sorted(
            [(coefs[i], feature_names[i]) for i in nonzero_indices],
            key=lambda x: abs(x[0]),
            reverse=True
        )[:5]

        top_features = [f"{word} ({round(coef,3)})" for coef, word in top_features]

        # ✅ Opposite responses (FIXED properly topic-wise)
        opposite_label = 0 if pred_label == 1 else 1

        filtered = self.dataset[
            (self.dataset["topic"].str.lower() == topic.lower()) &
            (self.dataset["label"] == opposite_label)
        ]

        if len(filtered) > 0:
            opposite_texts = filtered["text"].sample(
                min(3, len(filtered)),
                random_state=42
            ).tolist()
        else:
            opposite_texts = ["No opposite examples found for this topic"]

        # ✅ Save prediction log
        log_path = "outputs/predictions_log.csv"

        log_data = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topic": topic,
            "input_text": text,
            "predicted_label": sentiment,
            "confidence": confidence,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "top_features": "; ".join(top_features)
        }

        if os.path.exists(log_path):
            pd.DataFrame([log_data]).to_csv(
                log_path, mode="a", header=False, index=False
            )
        else:
            pd.DataFrame([log_data]).to_csv(
                log_path, mode="w", header=True, index=False
            )

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "top_features": top_features,
            "opposite_responses": opposite_texts
        }
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"
MODEL_PATH = BASE_DIR / "spam_model.joblib"


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    columns = {column.strip().lower(): column for column in data.columns}

    # Accept either label/text or the common v1/v2 dataset column names.
    label_column = columns.get("label") or columns.get("v1")
    text_column = columns.get("text") or columns.get("message") or columns.get("v2")

    if not label_column or not text_column:
        raise ValueError(
            "CSV needs label and text columns, or v1 and v2 columns."
        )

    data = data[[label_column, text_column]].dropna()
    labels = data[label_column].astype(str).str.strip().str.lower()
    texts = data[text_column].astype(str).str.strip()

    valid_labels = labels.isin(["spam", "ham"])
    labels = labels[valid_labels]
    texts = texts[valid_labels]

    if len(labels) < 2 or labels.nunique() < 2:
        raise ValueError("The CSV must contain both spam and ham examples.")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("classifier", MultinomialNB()),
    ])

    model.fit(texts, labels)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return model


def get_model():
    """Load the saved model, or train it if it doesn't exist yet."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_model()


if __name__ == "__main__":
    train_model()
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import urllib.request
import zipfile

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

REQUIRED_COLUMNS = ["label", "message"]
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
_stemmer = PorterStemmer()


def _tokenize_and_stem(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z']+", str(text).lower())
    return [_stemmer.stem(word) for word in words if word]


def _load_uci_dataset() -> pd.DataFrame:
    request = urllib.request.Request(
        DATASET_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        data_file = next(
            name for name in archive.namelist()
            if name.endswith("SMSSpamCollection")
        )
        with archive.open(data_file) as file:
            data = pd.read_csv(
                file, sep="\t", header=None, names=REQUIRED_COLUMNS,
                encoding="utf-8", on_bad_lines="skip"
            )
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data["message"] = data["message"].astype(str).str.strip()
    return data[data["label"].isin(["ham", "spam"])].dropna().reset_index(drop=True)


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, str]:
    """Load the UCI SMS Spam Collection, falling back to the bundled sample."""
    data_dir = Path(data_dir)
    fallback = data_dir / "sample_messages.csv"

    try:
        data = _load_uci_dataset()
        return data, "UCI SMS Spam Collection (online)"
    except Exception:
        if fallback.exists():
            data = pd.read_csv(fallback)
            data = data[REQUIRED_COLUMNS].dropna().copy()
            data["label"] = data["label"].astype(str).str.lower().str.strip()
            data["message"] = data["message"].astype(str).str.strip()
            data = data[data["label"].isin(["ham", "spam"])]
            return data.drop_duplicates(subset="message").reset_index(drop=True), \
                "Bundled offline sample"
        raise RuntimeError(
            "Could not download the UCI dataset and the offline sample is missing."
        )


def _make_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        tokenizer=_tokenize_and_stem,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", MultinomialNB(alpha=0.35)),
    ])


def train_classifier(data: pd.DataFrame) -> Pipeline:
    clean = data[REQUIRED_COLUMNS].dropna().copy()
    clean["label"] = clean["label"].astype(str).str.lower().str.strip()
    clean["message"] = clean["message"].astype(str).str.strip()
    clean = clean[clean["label"].isin(["ham", "spam"])]
    if len(clean) < 4 or clean["label"].nunique() != 2:
        raise ValueError("Training data must contain both ham and spam messages.")
    model = _make_pipeline()
    model.fit(clean["message"], clean["label"])
    return model


def predict_message(model_bundle: Pipeline, message: str) -> dict:
    probabilities = model_bundle.predict_proba([message])[0]
    classes = list(model_bundle.classes_)
    spam_probability = float(probabilities[classes.index("spam")])
    ham_probability = float(probabilities[classes.index("ham")])
    label = "spam" if spam_probability >= ham_probability else "ham"
    return {
        "label": label,
        "confidence": max(spam_probability, ham_probability),
        "spam_probability": spam_probability,
        "ham_probability": ham_probability,
    }


def evaluate_model(model_bundle: Pipeline, data: pd.DataFrame):
    clean = data[REQUIRED_COLUMNS].dropna().copy()
    X = clean["message"].astype(str)
    y = clean["label"].astype(str).str.lower()
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Each class needs at least two messages for evaluation.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    evaluation_model = _make_pipeline()
    evaluation_model.fit(X_train, y_train)
    y_pred = evaluation_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["ham", "spam"]),
    }
    return metrics, y_test, y_pred


def get_distinctive_terms(model_bundle: Pipeline, top_n: int = 10) -> dict:
    vectorizer = model_bundle.named_steps["tfidf"]
    classifier = model_bundle.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    if len(classifier.classes_) != 2:
        return {"spam": [], "ham": []}

    class_indices = {label: i for i, label in enumerate(classifier.classes_)}
    spam_scores = classifier.feature_log_prob_[class_indices["spam"]]
    ham_scores = classifier.feature_log_prob_[class_indices["ham"]]
    spam_order = (spam_scores - ham_scores).argsort()[::-1][:top_n]
    ham_order = (ham_scores - spam_scores).argsort()[::-1][:top_n]
    return {
        "spam": [str(feature_names[i]) for i in spam_order],
        "ham": [str(feature_names[i]) for i in ham_order],
    }

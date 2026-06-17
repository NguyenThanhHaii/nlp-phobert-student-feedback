"""Baseline model builders for clean text classification experiments."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_majority_classifier() -> DummyClassifier:
    """Build a majority-class baseline."""
    return DummyClassifier(strategy="most_frequent")


def build_tfidf_word_svm() -> Pipeline:
    """Build word-level TF-IDF + Linear SVM baseline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_tfidf_char_svm() -> Pipeline:
    """Build character-level TF-IDF + Linear SVM baseline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.95,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def get_baseline_builders() -> dict:
    """Return all baseline model builders used in Stage 2."""
    return {
        "majority_class": build_majority_classifier,
        "tfidf_word_svm": build_tfidf_word_svm,
        "tfidf_char_svm": build_tfidf_char_svm,
    }

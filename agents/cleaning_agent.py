from __future__ import annotations

import re
from typing import Any, List

import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:  # pragma: no cover - fallback in minimal environments
    nltk = None
    stopwords = None
    WordNetLemmatizer = None
    word_tokenize = None


class CleaningAgent:
    """Clean and normalize review data for downstream analysis."""

    def __init__(self, text_column: str = "Review") -> None:
        self.text_column = text_column
        self.lemmatizer = None
        self.stop_words: set[str] = set()
        self._initialize_nlp()

    def _initialize_nlp(self) -> None:
        """Attempt to initialize NLTK resources when available."""
        if nltk is None:
            return

        if WordNetLemmatizer is not None:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                self.lemmatizer = None

        if stopwords is not None:
            try:
                self.stop_words = set(stopwords.words("english"))
            except Exception:
                self.stop_words = set()

    def load_reviews(self, csv_path: str) -> pd.DataFrame:
        """Load a CSV file into a dataframe."""
        try:
            dataframe = pd.read_csv(csv_path)
            return dataframe
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"The file '{csv_path}' was not found.") from exc
        except Exception as exc:
            raise RuntimeError(f"Unable to read CSV file '{csv_path}'.") from exc

    def clean_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Perform cleaning, normalization, tokenization, and lemmatization."""
        cleaned = dataframe.copy()

        try:
            cleaned[self.text_column] = cleaned[self.text_column].fillna("").astype(str)
            cleaned["Product"] = cleaned.get("Product", pd.Series([""] * len(cleaned))).fillna("").astype(str)
            cleaned["Rating"] = pd.to_numeric(cleaned.get("Rating", pd.Series([0] * len(cleaned))), errors="coerce")
            cleaned["Date"] = cleaned.get("Date", pd.Series([""] * len(cleaned))).fillna("").astype(str)

            cleaned = cleaned.drop_duplicates(subset=[self.text_column], keep="first")
            cleaned["cleaned_review"] = cleaned[self.text_column].apply(self.clean_text)
            cleaned["tokens"] = cleaned["cleaned_review"].apply(self.tokenize)
            cleaned["lemmatized_tokens"] = cleaned["tokens"].apply(self.lemmatize_tokens)
            cleaned["normalized_text"] = cleaned["lemmatized_tokens"].apply(lambda tokens: " ".join(tokens))
            return cleaned
        except Exception as exc:
            raise RuntimeError(f"Review cleaning failed: {exc}") from exc

    def clean_text(self, text: Any) -> str:
        """Strip HTML, URLs, emojis, and punctuation from text."""
        if pd.isna(text):
            return ""

        text = str(text)
        try:
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"http\S+|www\.\S+", " ", text)
            text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
            text = re.sub(r"\s+", " ", text).strip()
            text = text.lower()
            return text
        except Exception:
            return str(text).lower()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLTK when available, otherwise a simple split."""
        if not text:
            return []

        try:
            if word_tokenize is not None:
                tokens = word_tokenize(text)
            else:
                tokens = text.split()

            return [token for token in tokens if token not in self.stop_words]
        except Exception:
            return text.split()

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens with a safe fallback."""
        if not tokens:
            return []

        if self.lemmatizer is None:
            return [token for token in tokens if token]

        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception:
            return [token for token in tokens if token]

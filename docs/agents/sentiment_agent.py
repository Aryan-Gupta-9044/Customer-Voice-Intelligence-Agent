from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None


class SentimentAgent:
    """Predict whether a review is positive, neutral, or negative."""

    def __init__(self) -> None:
        self.pipeline_model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Hugging Face sentiment pipeline when available."""
        if pipeline is None:
            return

        try:
            self.pipeline_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        except Exception:
            self.pipeline_model = None

    def predict(self, text: str) -> Dict[str, Any]:
        """Return sentiment and confidence for a single review."""
        cleaned_text = str(text or "").strip()
        if not cleaned_text:
            return {"Sentiment": "Neutral", "Confidence": 0.0, "Review": ""}

        if self.pipeline_model is not None:
            try:
                result = self.pipeline_model(cleaned_text[:512], truncation=True)[0]
                label = result.get("label", "Neutral").lower()
                if label.startswith("pos"):
                    sentiment = "Positive"
                elif label.startswith("neg"):
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                confidence = round(float(result.get("score", 0.0)), 3)
                return {"Sentiment": sentiment, "Confidence": confidence, "Review": cleaned_text}
            except Exception:
                pass

        positive_words = {"excellent", "great", "love", "good", "satisfied", "fast", "impressive", "strong", "durable", "easy"}
        negative_words = {"bad", "poor", "slow", "issue", "problem", "unreliable", "disappointing", "damage", "fail", "broken", "badly"}

        tokens = set(cleaned_text.lower().split())
        positive_hits = len(tokens & positive_words)
        negative_hits = len(tokens & negative_words)

        if positive_hits > negative_hits:
            sentiment = "Positive"
            confidence = min(0.95, 0.6 + (positive_hits * 0.05))
        elif negative_hits > positive_hits:
            sentiment = "Negative"
            confidence = min(0.95, 0.6 + (negative_hits * 0.05))
        else:
            sentiment = "Neutral"
            confidence = 0.55

        return {"Sentiment": sentiment, "Confidence": round(confidence, 3), "Review": cleaned_text}

    def analyze_dataframe(self, dataframe: pd.DataFrame, text_column: str = "normalized_text") -> pd.DataFrame:
        """Attach sentiment labels to a dataframe."""
        output = dataframe.copy()
        sentiments: List[Dict[str, Any]] = []
        for review in output[text_column].fillna(""):
            sentiments.append(self.predict(str(review)))

        output["Sentiment"] = [item["Sentiment"] for item in sentiments]
        output["Confidence"] = [item["Confidence"] for item in sentiments]
        output["Review"] = output.get("Review", output[text_column])
        return output

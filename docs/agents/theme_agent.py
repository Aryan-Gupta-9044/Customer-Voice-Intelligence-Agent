from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pandas as pd


class ThemeAgent:
    """Extract recurring complaint themes and representative reviews."""

    def __init__(self) -> None:
        self.theme_keywords = {
            "Service": ["service", "support", "customer", "response", "slow"],
            "Reliability": ["fail", "broken", "stopped", "work", "reliable", "issue"],
            "Performance": ["cooling", "heating", "performance", "strong", "fast", "power"],
            "App & Controls": ["app", "remote", "button", "control", "connect"],
            "Packaging": ["packaging", "damaged", "instructions", "delivery"],
            "Design": ["design", "build", "quality", "compact", "durable"],
        }

    def extract_themes(self, dataframe: pd.DataFrame, text_column: str = "normalized_text") -> List[Dict[str, Any]]:
        """Group reviews into themes based on keyword overlaps."""
        themes: Dict[str, List[dict[str, Any]]] = {}
        for _, row in dataframe.iterrows():
            review_text = str(row.get(text_column, "") or row.get("Review", "")).lower()
            matched_theme = "General"
            best_score = -1

            for theme_name, keywords in self.theme_keywords.items():
                score = sum(1 for keyword in keywords if keyword in review_text)
                if score > best_score:
                    best_score = score
                    matched_theme = theme_name

            if matched_theme not in themes:
                themes[matched_theme] = []
            themes[matched_theme].append(
                {
                    "Review": str(row.get("Review", "")),
                    "Product": str(row.get("Product", "")),
                    "Sentiment": str(row.get("Sentiment", "Neutral")),
                }
            )

        result: List[Dict[str, Any]] = []
        for theme_name, entries in sorted(themes.items(), key=lambda item: len(item[1]), reverse=True):
            keywords = self._keywords_for_theme(theme_name, dataframe, text_column)
            representative_reviews = [entry["Review"] for entry in entries[:3]]
            result.append(
                {
                    "Topic": theme_name,
                    "Keywords": keywords,
                    "Representative Reviews": representative_reviews,
                    "Frequency": len(entries),
                }
            )

        return result

    def _keywords_for_theme(self, theme_name: str, dataframe: pd.DataFrame, text_column: str) -> List[str]:
        """Pull the most frequent words for a theme from the matching reviews."""
        matching_rows = dataframe[dataframe["Review"].apply(lambda value: self._matches_theme(value, theme_name))]
        tokens = []
        for _, row in matching_rows.iterrows():
            text = str(row.get(text_column, "") or row.get("Review", "")).lower()
            tokens.extend([token for token in text.split() if len(token) > 3])

        counter = Counter(tokens)
        return [word for word, _ in counter.most_common(5)]

    def _matches_theme(self, review_text: Any, theme_name: str) -> bool:
        """Determine whether a review belongs to a theme."""
        text = str(review_text or "").lower()
        for keyword in self.theme_keywords.get(theme_name, []):
            if keyword in text:
                return True
        return False

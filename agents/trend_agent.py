from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


class TrendAgent:
    """Analyze review sentiment and theme trends over time."""

    def __init__(self) -> None:
        self.period_columns = ["Week", "Month", "Quarter"]

    def analyze_trends(self, dataframe: pd.DataFrame) -> dict:
        """Generate trend summaries and Plotly charts."""
        analysis_frame = dataframe.copy()
        analysis_frame["Date"] = pd.to_datetime(analysis_frame["Date"], errors="coerce")
        analysis_frame = analysis_frame.dropna(subset=["Date"])
        analysis_frame["Week"] = analysis_frame["Date"].dt.to_period("W").astype(str)
        analysis_frame["Month"] = analysis_frame["Date"].dt.to_period("M").astype(str)
        analysis_frame["Quarter"] = analysis_frame["Date"].dt.to_period("Q").astype(str)

        charts = {}
        for period_name in self.period_columns:
            counts = analysis_frame.groupby(period_name)["Sentiment"].value_counts().unstack(fill_value=0)
            counts = counts.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
            figure = go.Figure()
            figure.add_trace(go.Bar(x=counts.index, y=counts.get("Positive", 0), name="Positive"))
            figure.add_trace(go.Bar(x=counts.index, y=counts.get("Neutral", 0), name="Neutral"))
            figure.add_trace(go.Bar(x=counts.index, y=counts.get("Negative", 0), name="Negative"))
            figure.update_layout(title=f"Sentiment trend by {period_name}", barmode="group")
            charts[period_name.lower()] = figure

        complaint_keywords = self._extract_complaints(analysis_frame)
        positive_keywords = self._extract_keywords(analysis_frame, "Positive")
        negative_keywords = self._extract_keywords(analysis_frame, "Negative")

        return {
            "charts": charts,
            "complaint_trends": complaint_keywords,
            "positive_trends": positive_keywords,
            "negative_trends": negative_keywords,
            "theme_frequency": self._theme_frequency(analysis_frame),
        }

    def _extract_complaints(self, dataframe: pd.DataFrame) -> list[dict]:
        negative_reviews = dataframe[dataframe["Sentiment"] == "Negative"]
        tokens = []
        for review in negative_reviews["normalized_text"].fillna(""):
            tokens.extend(str(review).split())
        counts = pd.Series(tokens).value_counts().head(8)
        return [{"keyword": keyword, "count": int(count)} for keyword, count in counts.items()]

    def _extract_keywords(self, dataframe: pd.DataFrame, sentiment: str) -> list[dict]:
        subset = dataframe[dataframe["Sentiment"] == sentiment]
        tokens = []
        for review in subset["normalized_text"].fillna(""):
            tokens.extend(str(review).split())
        counts = pd.Series(tokens).value_counts().head(8)
        return [{"keyword": keyword, "count": int(count)} for keyword, count in counts.items()]

    def _theme_frequency(self, dataframe: pd.DataFrame) -> list[dict]:
        counts = dataframe["Sentiment"].value_counts()
        return [{"sentiment": sentiment, "count": int(count)} for sentiment, count in counts.items()]

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

# Below this similarity score a "retrieved" review is treated as noise rather than
# real evidence for the question. Reviews scored by the fallback (hashing) embedding
# in particular tend to land well under this for genuinely unrelated text.
MIN_RELEVANCE_SCORE = 0.18

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who", "whom",
    "about", "of", "for", "to", "in", "on", "with", "and", "or", "do", "does",
    "did", "how", "why", "main", "any", "there", "this", "that", "it", "be",
    "as", "at", "by", "from", "customers", "customer", "review", "reviews",
}


class QAAgent:
    """Answer user questions using only the retrieved reviews."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        self._initialize()

    def _initialize(self) -> None:
        """Configure the Gemini model when the API key is available."""
        if not self.api_key or genai is None:
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception:
            self.model = None

    def answer_question(self, question: str, retrieved_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a grounded answer and include supporting snippets.

        Only reviews that actually scored as relevant to the question are used - a raw
        top-k retrieval can include weakly-related filler (e.g. score 0.13) just to fill
        out 5 results, and treating those as equal evidence produces answers that don't
        actually address what was asked.
        """
        relevant_reviews = [
            review for review in retrieved_reviews if review.get("Similarity Score", 0.0) >= MIN_RELEVANCE_SCORE
        ]

        # If nothing clears the bar, fall back to the single best match rather than
        # declaring total failure outright - it's still worth showing the user what's
        # closest, with appropriately low confidence.
        if not relevant_reviews and retrieved_reviews:
            best = max(retrieved_reviews, key=lambda review: review.get("Similarity Score", 0.0))
            if best.get("Similarity Score", 0.0) > 0.0:
                relevant_reviews = [best]

        if not relevant_reviews:
            return {
                "Answer": "Not enough evidence.",
                "Supporting Reviews": [],
                "Confidence": 0.0,
                "Evidence": [],
            }

        if self.model is not None:
            prompt = self._build_prompt(question, relevant_reviews)
            try:
                response = self.model.generate_content(prompt)
                answer_text = getattr(response, "text", None) or str(response)
                avg_score = sum(r.get("Similarity Score", 0.0) for r in relevant_reviews) / len(relevant_reviews)
                return {
                    "Answer": answer_text.strip() or "Not enough evidence.",
                    "Supporting Reviews": [self._snippet(review) for review in relevant_reviews[:3]],
                    "Confidence": round(min(0.95, 0.5 + avg_score), 3),
                    "Evidence": relevant_reviews,
                }
            except Exception:
                pass

        return self._fallback_answer(question, relevant_reviews)

    def _build_prompt(self, question: str, retrieved_reviews: List[Dict[str, Any]]) -> str:
        """Create a prompt that forces evidence-only reasoning."""
        review_details = "\n".join(
            f"- Review: {item.get('Review', '')} | Product: {item.get('Product', '')} | Rating: {item.get('Rating', '')} | Date: {item.get('Date', '')}"
            for item in retrieved_reviews[:5]
        )
        return (
            "You are an AI Product Analyst. Answer ONLY using the retrieved reviews. "
            "Never invent facts. If the evidence is insufficient, reply exactly with 'Not enough evidence.'.\n"
            f"Question: {question}\n"
            f"Retrieved reviews:\n{review_details}"
        )

    def _fallback_answer(self, question: str, relevant_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a deterministic, question-aware answer when Gemini is not available."""
        avg_score = sum(r.get("Similarity Score", 0.0) for r in relevant_reviews) / len(relevant_reviews)

        positive_reviews = [review for review in relevant_reviews if review.get("Rating", 0) >= 4]
        negative_reviews = [review for review in relevant_reviews if review.get("Rating", 0) <= 2]

        summary: List[str] = []
        if negative_reviews:
            terms = self._top_terms(question, [r.get("Review", "") for r in negative_reviews])
            term_text = f" Common themes: {', '.join(terms)}." if terms else ""
            summary.append(f"{len(negative_reviews)} of {len(relevant_reviews)} matching review(s) report problems.{term_text}")
        if positive_reviews:
            terms = self._top_terms(question, [r.get("Review", "") for r in positive_reviews])
            term_text = f" Common themes: {', '.join(terms)}." if terms else ""
            summary.append(f"{len(positive_reviews)} matching review(s) are positive.{term_text}")
        if not summary:
            summary.append("Matching reviews show mixed feedback with no strongly positive or negative signal.")

        return {
            "Answer": " ".join(summary),
            "Supporting Reviews": [self._snippet(review) for review in relevant_reviews[:3]],
            "Confidence": round(min(0.9, 0.35 + avg_score), 3),
            "Evidence": relevant_reviews,
        }

    def _top_terms(self, question: str, texts: List[str], limit: int = 4) -> List[str]:
        """Pull the most frequent non-trivial words shared across a set of review texts."""
        question_terms = {word for word in re.findall(r"[a-z]+", question.lower()) if word not in STOPWORDS}
        tokens: List[str] = []
        for text in texts:
            tokens.extend(word for word in re.findall(r"[a-z]+", str(text).lower()) if len(word) > 3 and word not in STOPWORDS)

        counter = Counter(tokens)
        # Prefer words the question itself mentioned, then fill with the next most common terms.
        ordered = [word for word in question_terms if word in counter] + [
            word for word, _ in counter.most_common() if word not in question_terms
        ]
        seen: List[str] = []
        for word in ordered:
            if word not in seen:
                seen.append(word)
            if len(seen) >= limit:
                break
        return seen

    def _snippet(self, review: Dict[str, Any]) -> str:
        """Return a short evidence snippet."""
        text = str(review.get("Review", ""))
        return text[:140] + ("..." if len(text) > 140 else "")

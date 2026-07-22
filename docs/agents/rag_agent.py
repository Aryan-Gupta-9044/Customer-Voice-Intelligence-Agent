from __future__ import annotations

from typing import Any, List, Optional

from agents.embedding_agent import EmbeddingAgent


class RAGAgent:
    """Retrieve the most relevant reviews for a user query."""

    def __init__(self, embedding_agent: Optional[EmbeddingAgent] = None) -> None:
        self.embedding_agent = embedding_agent or EmbeddingAgent()

    def retrieve(self, query: str, top_k: int = 5) -> List[dict[str, Any]]:
        """Use embeddings to fetch the top-k relevant reviews."""
        if not query or not query.strip():
            return []

        try:
            results = self.embedding_agent.search(query=query, top_k=top_k)
            return results
        except Exception:
            return []

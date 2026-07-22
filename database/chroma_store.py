from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised in minimal environments
    chromadb = None


class ChromaStore:
    """A thin wrapper around ChromaDB with a safe fallback for local execution."""

    def __init__(self, persist_directory: Optional[str | Path] = None):
        self.persist_directory = Path(persist_directory or Path(__file__).resolve().parents[1] / "database" / "chroma_db")
        self.client = None
        self.collection = None
        self._fallback_documents: List[Dict[str, Any]] = []
        self._connect()

    def _connect(self) -> None:
        """Create a ChromaDB persistent client when the package is available."""
        if chromadb is None:
            return

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collection = self.client.get_or_create_collection(name="customer_reviews")
        except Exception:
            self.collection = None
            self.client = None

    def reset(self) -> None:
        """Clear all previously indexed documents before indexing a fresh dataset.

        Without this, re-analyzing (e.g. clicking "Analyze Reviews" again, or uploading a new
        CSV) would either raise a duplicate-ID error from ChromaDB or silently mix reviews from
        an old dataset into the new one's search results.
        """
        if self.client is not None:
            try:
                self.client.delete_collection("customer_reviews")
            except Exception:
                pass
            try:
                self.collection = self.client.get_or_create_collection(name="customer_reviews")
            except Exception:
                self.collection = None

        self._fallback_documents = []

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]], ids: List[str]) -> None:
        """Store documents and embeddings in ChromaDB or the in-memory fallback."""
        if not documents:
            return

        if self.collection is not None:
            try:
                # upsert (rather than add) is safe even if an id from a previous run survived.
                self.collection.upsert(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)
            except Exception:
                self.collection.add(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)
            return

        for doc, metadata, embedding, doc_id in zip(documents, metadatas, embeddings, ids):
            self._fallback_documents.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": metadata,
                    "embedding": embedding,
                }
            )

    def query(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """Query stored documents with cosine similarity semantics."""
        if self.collection is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            payload: List[Dict[str, Any]] = []
            for document, metadata, distance in zip(documents, metadatas, distances):
                similarity = max(0.0, 1.0 - float(distance))
                payload.append({"document": document, "metadata": metadata, "score": round(similarity, 4)})
            return payload

        scored: List[Dict[str, Any]] = []
        for item in self._fallback_documents:
            score = self._cosine_similarity(query_embedding, item["embedding"])
            if score > 0:
                scored.append({"document": item["document"], "metadata": item["metadata"], "score": round(score, 4)})

        scored.sort(key=lambda entry: entry["score"], reverse=True)
        return scored[:n_results]

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not left or not right or len(left) != len(right):
            return 0.0

        numerator = sum(a * b for a, b in zip(left, right))
        denom_left = sum(a * a for a in left) ** 0.5
        denom_right = sum(b * b for b in right) ** 0.5

        if denom_left == 0 or denom_right == 0:
            return 0.0

        return numerator / (denom_left * denom_right)

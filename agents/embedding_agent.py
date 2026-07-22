from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, List, Optional

import pandas as pd

from database.chroma_store import ChromaStore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

# Fixed dimensionality for the fallback (no sentence-transformers) embedding path.
# Every text is hashed into a vector of this size so that any two embeddings are
# always comparable with cosine similarity, regardless of vocabulary differences.
FALLBACK_EMBEDDING_DIM = 256


class EmbeddingAgent:
    """Generate embeddings and index reviews in ChromaDB."""

    def __init__(self, persist_directory: Optional[str] = None) -> None:
        self.store = ChromaStore(persist_directory)
        self.model = None
        # token -> inverse-document-frequency weight, learned from the indexed corpus and
        # reused when embedding queries. Only used by the fallback (hashing) embedding path.
        self._vocab_idf: dict[str, float] = {}
        self._default_idf: float = 1.0
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence-transformer model when the dependency is available."""
        if SentenceTransformer is None:
            return

        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.model = None

    def _build_vocab_idf(self, texts: List[str]) -> None:
        """Learn word importance from the corpus so common filler words (e.g. "product",
        "control") stop drowning out the distinctive words (e.g. "AC", "battery",
        "disconnecting") that actually distinguish one review from another.
        """
        doc_freq: Counter[str] = Counter()
        documents = [str(text) for text in texts if str(text).strip()]
        n_docs = len(documents) or 1

        for text in documents:
            for token in set(token.lower() for token in text.split() if token):
                doc_freq[token] += 1

        self._vocab_idf = {
            token: math.log((n_docs + 1) / (freq + 1)) + 1.0 for token, freq in doc_freq.items()
        }
        # Words that never appeared in the indexed corpus (e.g. a query-only word) are
        # treated as maximally distinctive rather than ignored.
        self._default_idf = max(self._vocab_idf.values(), default=1.0)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if self.model is not None:
            try:
                embeddings = self.model.encode(list(texts), convert_to_numpy=True)
                return embeddings.tolist()
            except Exception:
                pass

        return [self._fallback_embedding(text) for text in texts]

    def _fallback_embedding(self, text: str) -> List[float]:
        """Hashing-based, IDF-weighted bag-of-words embedding used when sentence-transformers
        is unavailable.

        Every text is mapped onto the SAME fixed-size vector (FALLBACK_EMBEDDING_DIM) using a
        stable hash of each token, so embeddings for different reviews/queries always share a
        dimension and can be compared with cosine similarity. Token contributions are weighted
        by corpus IDF (see `_build_vocab_idf`) so generic words that appear in almost every
        review contribute little, while specific/rare words dominate the similarity score.
        """
        vector = [0.0] * FALLBACK_EMBEDDING_DIM
        tokens = [token.lower() for token in str(text).split() if token]
        if not tokens:
            return vector

        for token in tokens:
            weight = self._vocab_idf.get(token, self._default_idf)
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % FALLBACK_EMBEDDING_DIM
            vector[bucket] += weight

        # L2 normalize so review length doesn't dominate the similarity score.
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    def index_reviews(self, dataframe: pd.DataFrame, text_column: str = "normalized_text") -> None:
        """Create embeddings and store them in ChromaDB."""
        if dataframe.empty:
            return

        self.store.reset()

        documents: List[str] = []
        metadatas: List[dict[str, Any]] = []
        ids: List[str] = []

        for _, row in dataframe.iterrows():
            review_text = str(row.get(text_column, "")) or str(row.get("Review", ""))
            metadata = {
                "Review": str(row.get("Review", "")),
                "Product": str(row.get("Product", "")),
                "Rating": int(row.get("Rating", 0)) if pd.notna(row.get("Rating")) else 0,
                "Date": str(row.get("Date", "")),
                "Sentiment": str(row.get("Sentiment", "")),
                "source": "uploaded",
            }
            documents.append(review_text)
            metadatas.append(metadata)
            ids.append(f"review-{len(ids) + 1}")

        # Learn corpus-level word importance BEFORE embedding, so both the indexed reviews
        # and any later query share the same IDF weighting.
        if self.model is None:
            self._build_vocab_idf(documents)

        # Batch-encode once instead of one API/model call per row - much faster and, for the
        # sentence-transformer path, avoids re-loading model state per row.
        embeddings = self.generate_embeddings(documents)

        self.store.add_documents(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)

    def search(self, query: str, top_k: int = 5) -> List[dict[str, Any]]:
        """Search indexed reviews with semantic similarity."""
        query_embedding = self.generate_embeddings([query])[0]
        results = self.store.query(query_embedding=query_embedding, n_results=top_k)
        return [
            {
                "Review": item.get("metadata", {}).get("Review", ""),
                "Product": item.get("metadata", {}).get("Product", ""),
                "Rating": item.get("metadata", {}).get("Rating", 0),
                "Date": item.get("metadata", {}).get("Date", ""),
                "Similarity Score": item.get("score", 0.0),
            }
            for item in results
        ]

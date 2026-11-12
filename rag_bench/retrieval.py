"""
Dense Semantic Vector Retriever for Enterprise Knowledge Bases
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rag_bench.models import Document, Query, RetrievalItem, RetrievalResult


class DenseVectorRetriever:
    """
    Vectorized dense semantic retriever combining TF-IDF lexical frequency
    with random orthogonal Gaussian projections onto the unit hypersphere.
    """

    def __init__(self, embedding_dim: int = 128, random_state: int = 42):
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=2048, stop_words="english")
        self.projection_matrix: Optional[np.ndarray] = None
        self.doc_embeddings: Optional[np.ndarray] = None
        self.documents: List[Document] = []
        self.is_indexed = False

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return v / norms

    def fit_and_index(self, documents: List[Document]) -> "DenseVectorRetriever":
        """Indexes corpus and computes dense unit-normalized embeddings."""
        if not documents:
            raise ValueError("Corpus cannot be empty.")

        self.documents = documents
        corpus_texts = [d.text for d in documents]

        tfidf_matrix = self.vectorizer.fit_transform(corpus_texts).toarray()
        vocab_size = tfidf_matrix.shape[1]

        rng = np.random.RandomState(self.random_state)
        # Johnson-Lindenstrauss Gaussian random projection matrix
        scale = 1.0 / np.sqrt(self.embedding_dim)
        self.projection_matrix = rng.normal(loc=0.0, scale=scale, size=(vocab_size, self.embedding_dim))

        dense_embeddings = np.dot(tfidf_matrix, self.projection_matrix)
        self.doc_embeddings = self._normalize(dense_embeddings)

        for i, doc in enumerate(self.documents):
            doc.embedding = self.doc_embeddings[i]

        self.is_indexed = True
        return self

    def embed_text(self, text: str) -> np.ndarray:
        """Projects arbitrary query string into the dense semantic embedding space."""
        if not self.is_indexed:
            raise RuntimeError("Retriever must be indexed before embedding queries.")

        vec = self.vectorizer.transform([text]).toarray()
        dense = np.dot(vec, self.projection_matrix)
        return self._normalize(dense)[0]

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieves top_k documents by cosine similarity."""
        if not self.is_indexed:
            raise RuntimeError("Corpus not indexed.")

        q_vec = self.embed_text(query)
        # Dot product with normalized document embeddings equals cosine similarity
        similarities = np.dot(self.doc_embeddings, q_vec)

        # Top-k indices descending
        k = min(top_k, len(self.documents))
        top_indices = np.argsort(-similarities)[:k]

        return [(self.documents[idx], float(similarities[idx])) for idx in top_indices]

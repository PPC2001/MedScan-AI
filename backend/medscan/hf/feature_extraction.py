"""
HF Task: Feature Extraction (Multi-modal Embeddings)
Used for: Generating dense vector embeddings for the vector store.
         Handles both text chunks and image descriptions.
Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast, accurate)
"""

import numpy as np
from typing import Any

from medscan.hf.base import BaseHFPipeline


class FeatureExtractionPipeline(BaseHFPipeline):
    """
    Generate dense embeddings for text and image descriptions.

    Output dimension: 384 (all-MiniLM-L6-v2)
    Used to populate the pgvector / Pinecone store for semantic retrieval.
    """

    task = "feature-extraction"
    default_model = "sentence-transformers/all-MiniLM-L6-v2"

    def run(self, text: str | list[str]) -> list[list[float]]:
        """
        Args:
            text: Single string or list of strings to embed.

        Returns:
            List of embeddings (each is a list of floats, length 384).
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # transformers feature-extraction returns [[[float]]] (batch x seq x dim)
        raw = self.pipe(texts, return_tensors=False)

        embeddings = []
        for item in raw:
            # Mean pool over sequence dimension
            arr = np.array(item)
            if arr.ndim == 2:
                pooled = arr.mean(axis=0)
            else:
                pooled = arr
            # L2 normalize
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
            embeddings.append(pooled.tolist())

        return embeddings

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.run(text=[text])[0]

    def embed_batch(
        self, texts: list[str], batch_size: int = 64
    ) -> list[list[float]]:
        """
        Embed a list of texts in batches for memory efficiency.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts per batch.

        Returns:
            All embeddings in order.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self.run(text=batch))
        return all_embeddings

    def cosine_similarity(
        self, embedding_a: list[float], embedding_b: list[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        a = np.array(embedding_a)
        b = np.array(embedding_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

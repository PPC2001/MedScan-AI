"""
Embedder — generates vector embeddings for document chunks.

Uses sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
Supports batched processing for efficiency.
"""

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import]

from medscan.config import get_settings
from medscan.pipeline.chunker import Chunk

logger = logging.getLogger(__name__)
settings = get_settings()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class DocumentEmbedder:
    """
    Generates dense vector embeddings for document chunks.

    Uses SentenceTransformers for efficient batch encoding.
    Embeddings are L2-normalized for cosine similarity.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or settings.hf_device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load and cache the SentenceTransformer model."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(
                self.model_name, device=self.device
            )
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return embedding.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string (alias for compatibility)."""
        return self.embed_text(text)

    def embed_texts(
        self, texts: list[str], batch_size: int = 64, show_progress: bool = False
    ) -> list[list[float]]:
        """
        Embed a list of texts in batches.

        Args:
            texts: Texts to embed.
            batch_size: Texts per batch.
            show_progress: Show tqdm progress bar.

        Returns:
            List of embeddings, each a list of 384 floats.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return [e.tolist() for e in embeddings]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed all chunks in a list and attach embeddings to metadata.

        Returns:
            Same list of chunks, with embeddings stored in chunk.metadata['embedding'].
        """
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]
        logger.info("Embedding %d chunks...", len(chunks))

        embeddings = self.embed_texts(texts=texts, show_progress=len(texts) > 20)

        for chunk, emb in zip(chunks, embeddings):
            chunk.metadata["embedding"] = emb

        return chunks

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two normalized embeddings."""
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb))  # Already normalized → dot product = cosine

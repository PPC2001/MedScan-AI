"""Abstract base for vector store implementations."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VectorSearchResult:
    """A single result from a vector similarity search."""
    chunk_id: str
    document_id: str
    patient_id: str
    content: str
    chunk_type: str
    page_number: int | None
    similarity_score: float
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    """
    Abstract interface for vector store backends.

    Implementations: pgvector_store.py, pinecone_store.py
    """

    @abstractmethod
    async def upsert(
        self,
        chunk_id: str,
        document_id: str,
        patient_id: str,
        content: str,
        embedding: list[float],
        chunk_type: str = "text",
        page_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a chunk with its embedding."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        patient_id: str | None = None,
        top_k: int = 10,
        chunk_types: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Cosine similarity search."""
        ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns deleted count."""
        ...

    @abstractmethod
    async def delete_patient(self, patient_id: str) -> int:
        """Delete all chunks belonging to a patient."""
        ...

    @abstractmethod
    async def count(self, patient_id: str | None = None) -> int:
        """Count stored vectors."""
        ...

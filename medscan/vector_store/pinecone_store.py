"""
Pinecone implementation of the vector store.

Uses Pinecone serverless for cloud-managed vector storage.
Namespace per patient for isolation.
"""

import logging
from typing import Any

from medscan.config import get_settings
from medscan.vector_store.base import BaseVectorStore, VectorSearchResult

logger = logging.getLogger(__name__)
settings = get_settings()


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone-backed vector store.

    Uses patient_id as Pinecone namespace for strict data isolation.
    Falls back to global namespace when patient_id is None.
    """

    def __init__(self) -> None:
        self._index = None
        self._initialized = False

    def _get_index(self) -> Any:
        """Lazy-initialize Pinecone index."""
        if self._index is None:
            from pinecone import Pinecone  # type: ignore[import]

            pc = Pinecone(api_key=settings.pinecone_api_key)
            self._index = pc.Index(settings.pinecone_index_name)
            logger.info(
                "Connected to Pinecone index: %s", settings.pinecone_index_name
            )
        return self._index

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
        """Upsert a chunk into Pinecone."""
        index = self._get_index()

        vector_metadata = {
            "document_id": document_id,
            "patient_id": patient_id,
            "content": content[:1000],  # Pinecone metadata size limit
            "chunk_type": chunk_type,
            "page_number": page_number or 0,
            **(metadata or {}),
        }

        # Use patient_id as namespace
        index.upsert(
            vectors=[{"id": chunk_id, "values": embedding, "metadata": vector_metadata}],
            namespace=patient_id,
        )

    async def search(
        self,
        query_embedding: list[float],
        patient_id: str | None = None,
        top_k: int = 10,
        chunk_types: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Query Pinecone with optional metadata filters."""
        index = self._get_index()

        filter_dict: dict[str, Any] = {}
        if chunk_types:
            filter_dict["chunk_type"] = {"$in": chunk_types}

        kwargs: dict[str, Any] = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
        }
        if patient_id:
            kwargs["namespace"] = patient_id
        if filter_dict:
            kwargs["filter"] = filter_dict

        response = index.query(**kwargs)

        results = []
        for match in response["matches"]:
            meta = match.get("metadata", {})
            results.append(VectorSearchResult(
                chunk_id=match["id"],
                document_id=meta.get("document_id", ""),
                patient_id=meta.get("patient_id", ""),
                content=meta.get("content", ""),
                chunk_type=meta.get("chunk_type", "text"),
                page_number=meta.get("page_number") or None,
                similarity_score=float(match["score"]),
                metadata=meta,
            ))

        return results

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document (Pinecone filter delete)."""
        logger.warning(
            "Pinecone delete by document_id requires scanning all namespaces. "
            "Consider using pgvector for frequent deletes."
        )
        return 0

    async def delete_patient(self, patient_id: str) -> int:
        """Delete an entire patient namespace."""
        index = self._get_index()
        index.delete(delete_all=True, namespace=patient_id)
        logger.info("Deleted Pinecone namespace for patient %s", patient_id)
        return -1  # Count unknown

    async def count(self, patient_id: str | None = None) -> int:
        """Get approximate vector count."""
        index = self._get_index()
        stats = index.describe_index_stats()
        if patient_id and patient_id in stats.get("namespaces", {}):
            return stats["namespaces"][patient_id]["vector_count"]
        return stats.get("total_vector_count", 0)

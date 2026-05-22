"""Vector store package — selects backend based on config."""

from medscan.config import get_settings
from medscan.vector_store.base import BaseVectorStore, VectorSearchResult

__all__ = ["BaseVectorStore", "VectorSearchResult", "get_vector_store"]


def get_vector_store(session=None) -> BaseVectorStore:
    """Factory: return the configured vector store backend."""
    settings = get_settings()
    if settings.vector_store_backend == "pinecone":
        from medscan.vector_store.pinecone_store import PineconeVectorStore
        return PineconeVectorStore()
    else:
        from medscan.vector_store.pgvector_store import PgVectorStore
        if session is None:
            raise ValueError("pgvector store requires a database session")
        return PgVectorStore(session=session)

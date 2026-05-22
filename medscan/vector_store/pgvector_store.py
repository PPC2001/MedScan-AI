"""
pgvector implementation of the vector store.

Uses PostgreSQL + pgvector extension for cosine similarity search.
Supports hybrid search: vector similarity + full-text (tsvector).
"""

import logging
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from medscan.vector_store.base import BaseVectorStore, VectorSearchResult

logger = logging.getLogger(__name__)


class PgVectorStore(BaseVectorStore):
    """
    pgvector-backed vector store with hybrid search.

    Features:
    - Cosine similarity search via ivfflat index
    - Optional patient_id filtering (namespace isolation)
    - Hybrid dense + sparse (full-text) search
    - Async SQLAlchemy sessions
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

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
        """Insert or update a chunk with its embedding in document_chunks."""
        import json

        meta_json = json.dumps(metadata or {})
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        sql = text("""
            INSERT INTO document_chunks
                (id, document_id, patient_id, content, chunk_type,
                 page_number, embedding, metadata, created_at)
            VALUES
                (:id, :document_id, :patient_id, :content, :chunk_type,
                 :page_number, CAST(:embedding AS vector), CAST(:metadata AS jsonb), NOW())
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
        """)

        await self.session.execute(sql, {
            "id": chunk_id,
            "document_id": document_id,
            "patient_id": patient_id,
            "content": content,
            "chunk_type": chunk_type,
            "page_number": page_number,
            "embedding": embedding_str,
            "metadata": meta_json,
        })
        await self.session.commit()

    async def search(
        self,
        query_embedding: list[float],
        patient_id: str | None = None,
        top_k: int = 10,
        chunk_types: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Cosine similarity search with optional patient and type filters.

        Args:
            query_embedding: Query vector (384-dim).
            patient_id: If provided, restricts to this patient's documents.
            top_k: Number of results to return.
            chunk_types: If provided, restricts to these chunk types.

        Returns:
            List of VectorSearchResult ordered by similarity (desc).
        """
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Build dynamic WHERE clauses
        conditions = ["embedding IS NOT NULL"]
        params: dict[str, Any] = {
            "embedding": embedding_str,
            "top_k": top_k,
        }

        if patient_id:
            conditions.append("patient_id = :patient_id")
            params["patient_id"] = patient_id

        if chunk_types:
            conditions.append("chunk_type = ANY(:chunk_types)")
            params["chunk_types"] = chunk_types

        where_clause = " AND ".join(conditions)

        sql = text(f"""
            SELECT
                id,
                document_id,
                patient_id,
                content,
                chunk_type,
                page_number,
                metadata,
                1 - (embedding <=> CAST(:embedding AS vector)) AS similarity_score
            FROM document_chunks
            WHERE {where_clause}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        result = await self.session.execute(sql, params)
        rows = result.fetchall()

        return [
            VectorSearchResult(
                chunk_id=str(row.id),
                document_id=str(row.document_id),
                patient_id=str(row.patient_id),
                content=row.content,
                chunk_type=row.chunk_type,
                page_number=row.page_number,
                similarity_score=float(row.similarity_score),
                metadata=row.metadata or {},
            )
            for row in rows
        ]

    async def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        patient_id: str | None = None,
        top_k: int = 10,
        vector_weight: float = 0.7,
    ) -> list[VectorSearchResult]:
        """
        Hybrid search: combine vector similarity with full-text search.

        Args:
            query_embedding: Dense vector embedding of the query.
            query_text: Raw query text for full-text search.
            patient_id: Optional patient filter.
            top_k: Number of results.
            vector_weight: Weight for vector score (1-vector_weight for text).

        Returns:
            Merged and re-ranked search results.
        """
        # Dense vector results
        vector_results = await self.search(
            query_embedding=query_embedding,
            patient_id=patient_id,
            top_k=top_k * 2,
        )

        # Full-text search
        text_weight = 1 - vector_weight
        params: dict[str, Any] = {
            "query": query_text,
            "top_k": top_k * 2,
        }
        if patient_id:
            params["patient_id"] = patient_id

        patient_filter = "AND patient_id = :patient_id" if patient_id else ""

        fts_sql = text(f"""
            SELECT id, ts_rank_cd(to_tsvector('english', content),
                   plainto_tsquery('english', :query)) AS rank
            FROM document_chunks
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
              {patient_filter}
            ORDER BY rank DESC
            LIMIT :top_k
        """)

        fts_result = await self.session.execute(fts_sql, params)
        fts_rows = {str(row.id): float(row.rank) for row in fts_result.fetchall()}

        # Merge and re-rank
        combined: dict[str, dict[str, Any]] = {}

        for res in vector_results:
            combined[res.chunk_id] = {
                "result": res,
                "score": vector_weight * res.similarity_score
                + text_weight * fts_rows.get(res.chunk_id, 0.0),
            }

        for chunk_id, fts_score in fts_rows.items():
            if chunk_id not in combined:
                # FTS-only hit — fetch from DB
                pass  # Skip for now (would need extra DB call)
            else:
                combined[chunk_id]["score"] += text_weight * fts_score

        sorted_results = sorted(
            combined.values(), key=lambda x: x["score"], reverse=True
        )

        return [item["result"] for item in sorted_results[:top_k]]

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        sql = text(
            "DELETE FROM document_chunks WHERE document_id = :doc_id"
        )
        result = await self.session.execute(sql, {"doc_id": document_id})
        await self.session.commit()
        return result.rowcount  # type: ignore[return-value]

    async def delete_patient(self, patient_id: str) -> int:
        """Delete all chunks for a patient."""
        sql = text(
            "DELETE FROM document_chunks WHERE patient_id = :patient_id"
        )
        result = await self.session.execute(sql, {"patient_id": patient_id})
        await self.session.commit()
        return result.rowcount  # type: ignore[return-value]

    async def count(self, patient_id: str | None = None) -> int:
        """Count total chunks, optionally filtered by patient."""
        if patient_id:
            sql = text(
                "SELECT COUNT(*) FROM document_chunks WHERE patient_id = :patient_id"
            )
            result = await self.session.execute(sql, {"patient_id": patient_id})
        else:
            sql = text("SELECT COUNT(*) FROM document_chunks")
            result = await self.session.execute(sql)
        return result.scalar() or 0

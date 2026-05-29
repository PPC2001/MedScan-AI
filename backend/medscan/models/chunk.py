"""
DocumentChunk model — a piece of a document stored with its embedding.

Supports pgvector for efficient cosine similarity search.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Index, JSON, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE_DESCRIPTION = "image_description"
    OCR_TEXT = "ocr_text"
    HANDWRITING = "handwriting"
    STRUCTURED = "structured"  # extracted lab values, vitals, etc.


EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension


class DocumentChunk(SQLModel, table=True):
    __tablename__ = "document_chunks"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        nullable=False,
    )
    document_id: uuid.UUID = Field(foreign_key="documents.id", index=True)
    patient_id: uuid.UUID = Field(foreign_key="patients.id", index=True)

    # Chunk content
    content: str = Field(sa_column=Column(Text, nullable=False))
    chunk_type: ChunkType = ChunkType.TEXT

    # Source location
    page_number: int | None = None
    chunk_index: int = 0  # position within document

    # pgvector embedding (384-dim for all-MiniLM-L6-v2)
    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(Vector(EMBEDDING_DIM)),
    )

    # Rich metadata (bounding boxes, confidence scores, source model, etc.)
    chunk_metadata: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSON().with_variant(JSONB, "postgresql")),
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Full-text search support (tsvector is managed via trigger in Postgres)
    # Cosine similarity index created below
    __table_args__ = (
        Index(
            "ix_chunks_embedding_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_chunks_document_id",
            "document_id",
        ),
        Index(
            "ix_chunks_patient_id",
            "patient_id",
        ),
    )

"""Models package — imports all SQLModel tables so Alembic/SQLModel can discover them."""

from medscan.models.chunk import ChunkType, DocumentChunk
from medscan.models.document import Document, DocumentCreate, DocumentRead, DocumentStatus, DocumentType
from medscan.models.patient import Patient, PatientCreate, PatientRead, PatientUpdate

__all__ = [
    "Patient", "PatientCreate", "PatientRead", "PatientUpdate",
    "Document", "DocumentCreate", "DocumentRead", "DocumentStatus", "DocumentType",
    "DocumentChunk", "ChunkType",
]

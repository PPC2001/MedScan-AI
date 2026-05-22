"""Document model — represents an ingested medical document."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Column, JSON, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class DocumentType(str, Enum):
    LAB_REPORT = "lab_report"
    PHYSICIAN_NOTE = "physician_note"
    IMAGING = "imaging"
    PRESCRIPTION = "prescription"
    DISCHARGE_SUMMARY = "discharge_summary"
    CLINICAL_PHOTO = "clinical_photo"
    HL7_FHIR = "hl7_fhir"
    OTHER = "other"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentBase(SQLModel):
    patient_id: uuid.UUID = Field(foreign_key="patients.id", index=True)
    filename: str
    original_filename: str
    file_path: str
    file_size_bytes: int
    mime_type: str
    doc_type: DocumentType = DocumentType.OTHER
    status: DocumentStatus = DocumentStatus.PENDING


class Document(DocumentBase, table=True):
    __tablename__ = "documents"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        nullable=False,
    )
    # Raw extracted text (full document)
    raw_text: str | None = Field(default=None, sa_column=Column(Text))

    # Structured data extracted by Instructor (labs, vitals, diagnoses, etc.)
    structured_data: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON().with_variant(JSONB, "postgresql"))
    )

    # Metadata from pipeline (page count, OCR confidence, etc.)
    pipeline_metadata: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON().with_variant(JSONB, "postgresql"))
    )

    # Error info if processing failed
    error_message: str | None = None

    # Celery task ID for status tracking
    task_id: str | None = None

    # Urgency / severity from text classification
    urgency_score: float | None = None
    urgency_label: str | None = None

    # Document type confidence from zero-shot classification
    doc_type_confidence: float | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: datetime | None = None


class DocumentCreate(SQLModel):
    patient_id: uuid.UUID
    original_filename: str
    doc_type: DocumentType = DocumentType.OTHER


class DocumentRead(DocumentBase):
    id: uuid.UUID
    raw_text: str | None
    structured_data: dict[str, Any] | None
    pipeline_metadata: dict[str, Any] | None
    urgency_score: float | None
    urgency_label: str | None
    task_id: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None

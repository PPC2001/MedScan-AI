"""
Pydantic schemas for API request/response bodies.
Separates API contracts from SQLModel table definitions.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Upload & Processing
# ---------------------------------------------------------------------------

class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    task_id: str
    status: str
    message: str


class DocumentStatusResponse(BaseModel):
    document_id: uuid.UUID
    status: str
    task_id: str | None
    error_message: str | None
    processed_at: datetime | None


# ---------------------------------------------------------------------------
# Structured Extraction Schemas (output of Instructor pipeline)
# ---------------------------------------------------------------------------

class LabValue(BaseModel):
    test_name: str
    value: str
    unit: str | None = None
    reference_range: str | None = None
    flag: str | None = None  # H (high), L (low), C (critical)
    icd10_code: str | None = None


class Vital(BaseModel):
    measurement: str  # e.g. "blood_pressure", "heart_rate"
    value: str
    unit: str | None = None
    timestamp: str | None = None


class Diagnosis(BaseModel):
    description: str
    icd10_code: str | None = None
    snomed_code: str | None = None
    certainty: str = "confirmed"  # confirmed | suspected | ruled_out


class Medication(BaseModel):
    drug_name: str
    dose: str | None = None
    frequency: str | None = None
    route: str | None = None
    duration: str | None = None
    rxnorm_code: str | None = None


class ExtractedClinicalData(BaseModel):
    """Root schema returned by Instructor extraction from a document."""
    lab_values: list[LabValue] = Field(default_factory=list)
    vitals: list[Vital] = Field(default_factory=list)
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    clinical_summary: str | None = None
    patient_name: str | None = None
    encounter_date: str | None = None
    physician_name: str | None = None
    facility_name: str | None = None


# ---------------------------------------------------------------------------
# Query / Agent
# ---------------------------------------------------------------------------

class ClinicalQueryRequest(BaseModel):
    query: str = Field(
        description="Natural language clinical question",
        examples=["What were the patient's latest HbA1c results?"],
    )
    patient_id: uuid.UUID = Field(description="Target patient UUID")
    include_sources: bool = True
    max_chunks: int = Field(default=10, ge=1, le=50)


class SourceChunk(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    chunk_type: str
    page_number: int | None
    similarity_score: float | None


class ClinicalQueryResponse(BaseModel):
    query: str
    patient_id: uuid.UUID
    answer: str
    query_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SourceChunk] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "This information is AI-generated and must be reviewed by a "
        "qualified healthcare professional before clinical use."
    )
    latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvaluationSample(BaseModel):
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]


class EvaluationReport(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    samples_evaluated: int
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]

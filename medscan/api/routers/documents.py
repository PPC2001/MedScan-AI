"""
Documents router — upload, status polling, and retrieval.
"""

import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from medscan.api.dependencies import AuthDep, SessionDep
from medscan.config import get_settings
from medscan.models.document import Document, DocumentCreate, DocumentRead, DocumentStatus, DocumentType
from medscan.models.schemas import DocumentStatusResponse, DocumentUploadResponse
from medscan.tasks.document_tasks import process_document

router = APIRouter()
settings = get_settings()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a medical document for processing",
)
async def upload_document(
    session: SessionDep,
    _auth: AuthDep,
    patient_id: uuid.UUID = Form(..., description="Target patient UUID"),
    doc_type: DocumentType = Form(default=DocumentType.OTHER),
    file: UploadFile = File(..., description="Medical document (PDF, JPG, PNG, DOCX)"),
) -> DocumentUploadResponse:
    """
    Upload a medical document and trigger async processing.

    Supported formats: PDF, JPG, PNG, DOCX, TXT

    Returns immediately with a document_id and task_id for polling status.
    """
    # Validate file size
    file_size = 0
    content = await file.read()
    file_size = len(content)

    if file_size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file not allowed",
        )

    # Validate MIME type
    allowed_types = {
        "application/pdf", "image/jpeg", "image/png", "image/tiff",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    # Save file to upload directory
    doc_id = uuid.uuid4()
    suffix = Path(file.filename or "upload").suffix
    saved_filename = f"{doc_id}{suffix}"
    file_path = settings.upload_dir / saved_filename

    settings.ensure_upload_dir()
    file_path.write_bytes(content)

    # Create DB record
    doc = Document(
        id=doc_id,
        patient_id=patient_id,
        filename=saved_filename,
        original_filename=file.filename or saved_filename,
        file_path=str(file_path),
        file_size_bytes=file_size,
        mime_type=file.content_type or "application/octet-stream",
        doc_type=doc_type,
        status=DocumentStatus.PENDING,
    )
    session.add(doc)
    await session.commit()

    # Dispatch async processing task
    task = process_document.delay(str(doc_id), str(file_path))

    # Store task_id in DB
    await session.execute(
        update(Document)
        .where(Document.id == doc_id)
        .values(task_id=task.id)
    )
    await session.commit()

    return DocumentUploadResponse(
        document_id=doc_id,
        task_id=task.id,
        status="pending",
        message=f"Document '{file.filename}' uploaded successfully. Processing started.",
    )


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get document processing status",
)
async def get_document_status(
    document_id: uuid.UUID,
    session: SessionDep,
    _auth: AuthDep,
) -> DocumentStatusResponse:
    """Poll the processing status of an uploaded document."""
    result = await session.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return DocumentStatusResponse(
        document_id=doc.id,
        status=doc.status,
        task_id=doc.task_id,
        error_message=doc.error_message,
        processed_at=doc.processed_at,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Get processed document with extracted data",
)
async def get_document(
    document_id: uuid.UUID,
    session: SessionDep,
    _auth: AuthDep,
) -> DocumentRead:
    """Retrieve a processed document including structured extracted data."""
    result = await session.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return doc


@router.get(
    "/patient/{patient_id}",
    response_model=list[DocumentRead],
    summary="List all documents for a patient",
)
async def list_patient_documents(
    patient_id: uuid.UUID,
    session: SessionDep,
    _auth: AuthDep,
    limit: int = 50,
    offset: int = 0,
) -> list[DocumentRead]:
    """List all documents uploaded for a specific patient."""
    result = await session.execute(
        select(Document)
        .where(Document.patient_id == patient_id)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    docs = result.scalars().all()
    return list(docs)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
async def delete_document(
    document_id: uuid.UUID,
    session: SessionDep,
    _auth: AuthDep,
) -> None:
    """Delete a document and its associated chunks."""
    result = await session.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Delete file from disk
    try:
        Path(doc.file_path).unlink(missing_ok=True)
    except Exception:
        pass

    await session.delete(doc)
    await session.commit()

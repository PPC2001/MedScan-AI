"""
Celery tasks for async document processing.

Tasks are dispatched when a document is uploaded and run the full
ingestion pipeline in the background, updating document status in DB.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path

from celery import Task
from sqlalchemy import create_engine, update
from sqlmodel import Session

from medscan.config import get_settings
from medscan.models.document import Document, DocumentStatus
from medscan.pipeline.ingestion import DocumentIngestionPipeline
from medscan.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()


class PipelineTask(Task):
    """Base task class that holds a shared pipeline instance."""

    _pipeline: DocumentIngestionPipeline | None = None

    @property
    def pipeline(self) -> DocumentIngestionPipeline:
        """Lazy-load the pipeline (shared across task invocations in same worker)."""
        if self._pipeline is None:
            logger.info("Initializing DocumentIngestionPipeline in worker...")
            self._pipeline = DocumentIngestionPipeline()
        return self._pipeline


@celery_app.task(
    bind=True,
    base=PipelineTask,
    name="medscan.tasks.document_tasks.process_document",
    max_retries=2,
    default_retry_delay=30,
)
def process_document(self: PipelineTask, document_id: str, file_path: str) -> dict:
    """
    Process a newly uploaded document through the full ingestion pipeline.

    Args:
        document_id: UUID of the Document record in the database.
        file_path: Absolute path to the uploaded file.

    Returns:
        Summary dict with processing results.
    """
    logger.info("Starting processing for document %s", document_id)

    # Sync engine for Celery (cannot use async in a sync task easily)
    engine = create_engine(settings.sync_database_url)

    def update_status(status: DocumentStatus, **kwargs) -> None:
        """Update document status in DB."""
        with Session(engine) as session:
            stmt = (
                update(Document)
                .where(Document.id == uuid.UUID(document_id))
                .values(status=status, updated_at=datetime.utcnow(), **kwargs)
            )
            session.execute(stmt)
            session.commit()

    # Mark as processing
    update_status(DocumentStatus.PROCESSING)

    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Run the full ingestion pipeline
        result = self.pipeline.process(
            file_path=path,
            document_id=document_id,
        )

        if not result.success:
            raise RuntimeError(result.error or "Pipeline returned failure")

        # Store results back to DB
        import json
        with Session(engine) as session:
            stmt = (
                update(Document)
                .where(Document.id == uuid.UUID(document_id))
                .values(
                    status=DocumentStatus.COMPLETED,
                    raw_text=result.raw_text[:50000] if result.raw_text else None,
                    structured_data=result.structured_data,
                    pipeline_metadata=result.pipeline_metadata,
                    urgency_score=result.urgency_score,
                    urgency_label=result.urgency_label,
                    doc_type_confidence=result.doc_type_confidence,
                    processed_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )
            session.execute(stmt)
            session.commit()

        # Store chunks to vector store (sync version via pgvector directly)
        _store_chunks_sync(engine, result, document_id)

        logger.info(
            "Document %s processed: %d chunks, urgency=%s",
            document_id, len(result.chunks), result.urgency_label,
        )

        return {
            "document_id": document_id,
            "status": "completed",
            "chunks": len(result.chunks),
            "urgency": result.urgency_label,
            "tables": len(result.tables),
            "images": len(result.image_descriptions),
        }

    except Exception as exc:
        logger.exception("Processing failed for document %s: %s", document_id, exc)
        update_status(
            DocumentStatus.FAILED,
            error_message=str(exc)[:1000],
        )
        # Retry on transient errors
        self.retry(exc=exc, countdown=30)


def _store_chunks_sync(engine, result, document_id: str) -> None:
    """Store embedded chunks to the database (sync version for Celery)."""
    from sqlmodel import Session
    from medscan.models.chunk import DocumentChunk, ChunkType
    from medscan.models.document import Document

    with Session(engine) as session:
        doc = session.get(Document, uuid.UUID(document_id))
        if not doc:
            logger.error("Document %s not found in database, cannot store chunks", document_id)
            return
        patient_id = doc.patient_id

        for chunk in result.chunks:
            embedding = chunk.metadata.get("embedding")
            if not embedding:
                continue

            db_chunk = DocumentChunk(
                id=uuid.uuid4(),
                document_id=uuid.UUID(document_id),
                patient_id=patient_id,
                content=chunk.content,
                chunk_type=chunk.chunk_type,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                embedding=embedding,
                chunk_metadata={
                    k: v for k, v in chunk.metadata.items() if k != "embedding"
                },
            )
            session.add(db_chunk)

        session.commit()
        logger.debug("Stored %d chunks for document %s", len(result.chunks), document_id)

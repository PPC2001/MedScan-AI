"""
Clinical Query router — the main Q&A endpoint using the LangGraph agent.
"""

import time
import uuid

from fastapi import APIRouter, HTTPException, status

from medscan.agents.graph import run_clinical_query
from medscan.api.dependencies import AuthDep, SessionDep
from medscan.models.schemas import (
    ClinicalQueryRequest,
    ClinicalQueryResponse,
    SourceChunk,
)
from medscan.pipeline.embedder import DocumentEmbedder
from medscan.vector_store import get_vector_store

router = APIRouter()

# Shared embedder for query encoding
_embedder: DocumentEmbedder | None = None


def get_embedder() -> DocumentEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = DocumentEmbedder()
    return _embedder


@router.post(
    "/",
    response_model=ClinicalQueryResponse,
    summary="Run a clinical query against a patient's documents",
)
async def clinical_query(
    request: ClinicalQueryRequest,
    session: SessionDep,
    _auth: AuthDep,
) -> ClinicalQueryResponse:
    """
    Answer a natural language clinical question by searching the patient's
    medical documents and reasoning across text, tables, and images.

    Examples:
    - "What were the patient's latest HbA1c and fasting glucose results?"
    - "Summarize all diagnoses found in the uploaded documents"
    - "What does the chest X-ray show?"
    - "List all medications with their doses"
    - "Are there any critical or abnormal lab values?"
    """
    start = time.monotonic()

    # Step 1: Embed the query
    embedder = get_embedder()
    try:
        query_embedding = embedder.embed_single(text=request.query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode query: {e}",
        )

    # Step 2: Retrieve relevant chunks from vector store
    vector_store = get_vector_store(session=session)
    try:
        search_results = await vector_store.search(
            query_embedding=query_embedding,
            patient_id=str(request.patient_id),
            top_k=request.max_chunks,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search failed: {e}",
        )

    # Convert search results to state format
    retrieved_chunks = [
        {
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "content": r.content,
            "chunk_type": r.chunk_type,
            "page_number": r.page_number,
            "similarity_score": r.similarity_score,
            **r.metadata,
        }
        for r in search_results
    ]

    # Separate table and image chunks for specialized agents
    table_chunks = [c for c in retrieved_chunks if c["chunk_type"] == "table"]
    image_chunks = [c for c in retrieved_chunks if c["chunk_type"] == "image_description"]

    # Format tables for TAPAS
    retrieved_tables = []
    for chunk in table_chunks:
        table_data = chunk.get("metadata", {})
        retrieved_tables.append({
            "table_type": table_data.get("table_type", "unknown"),
            "page_number": chunk.get("page_number"),
            "tapas_format": table_data.get("tapas_format", {}),
        })

    # Step 3: Run the multi-agent graph
    try:
        final_state = await run_clinical_query(
            query=request.query,
            patient_id=str(request.patient_id),
            retrieved_chunks=retrieved_chunks,
            retrieved_tables=retrieved_tables,
            retrieved_images=[],  # Image objects not serialized for now
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {e}",
        )

    # Step 4: Build response
    elapsed_ms = (time.monotonic() - start) * 1000

    sources = []
    if request.include_sources:
        for r in search_results[:5]:
            sources.append(SourceChunk(
                chunk_id=uuid.UUID(r.chunk_id),
                document_id=uuid.UUID(r.document_id),
                content=r.content[:500],
                chunk_type=r.chunk_type,
                page_number=r.page_number,
                similarity_score=round(r.similarity_score, 4),
            ))

    return ClinicalQueryResponse(
        query=request.query,
        patient_id=request.patient_id,
        answer=final_state.get("final_answer", "No answer generated."),
        query_type=final_state.get("query_type", "general"),
        confidence=final_state.get("confidence", 0.0),
        sources=sources,
        reasoning_trace=final_state.get("reasoning_trace", []),
        disclaimer=final_state.get("disclaimer", ""),
        latency_ms=round(elapsed_ms, 2),
    )

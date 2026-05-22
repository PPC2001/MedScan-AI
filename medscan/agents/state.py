"""
LangGraph Agent State — shared state schema for all nodes in the clinical
multi-agent graph.
"""

import uuid
from typing import Any, TypedDict


class ClinicalQueryState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph StateGraph.

    All fields are optional (total=False) to allow partial updates
    from individual nodes.
    """

    # Input
    query: str
    patient_id: str
    session_id: str

    # Routing
    query_type: str
    # One of: text_rag | table_qa | visual_reasoning | hybrid | general

    # Retrieval
    retrieved_chunks: list[dict[str, Any]]
    retrieved_tables: list[dict[str, Any]]
    retrieved_images: list[dict[str, Any]]

    # Reasoning
    reasoning_trace: list[str]
    intermediate_answers: list[str]

    # Final output
    final_answer: str
    confidence: float
    sources: list[dict[str, Any]]
    disclaimer: str

    # Safety
    is_safe: bool
    safety_flags: list[str]
    hallucination_risk: str  # low | medium | high

    # Metadata
    latency_ms: float
    agent_path: list[str]  # which nodes were visited
    error: str | None

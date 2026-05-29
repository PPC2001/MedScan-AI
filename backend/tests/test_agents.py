"""Tests for the LangGraph multi-agent system."""

import pytest

from medscan.agents.state import ClinicalQueryState


@pytest.mark.asyncio
async def test_router_text_classification() -> None:
    """Router correctly classifies lab-related query."""
    from medscan.agents.nodes.router import router_node

    state: ClinicalQueryState = {
        "query": "What were the patient's HbA1c and glucose levels?",
        "patient_id": "00000000-0000-0000-0000-000000000001",
        "reasoning_trace": [],
        "agent_path": [],
    }

    result = router_node(state)
    assert "query_type" in result
    assert result["query_type"] in ("text_rag", "table_qa", "visual_reasoning", "hybrid")
    assert "router" in result["agent_path"]


@pytest.mark.asyncio
async def test_router_imaging_query() -> None:
    """Router routes imaging query to visual_reasoning."""
    from medscan.agents.nodes.router import router_node

    state: ClinicalQueryState = {
        "query": "What does the chest X-ray show?",
        "patient_id": "00000000-0000-0000-0000-000000000001",
        "reasoning_trace": [],
        "agent_path": [],
    }

    result = router_node(state)
    # Should route to visual_reasoning or text_rag
    assert result["query_type"] in ("visual_reasoning", "text_rag")


@pytest.mark.asyncio
async def test_safety_guard_injects_disclaimer() -> None:
    """Safety guard always injects disclaimer."""
    from medscan.agents.nodes.safety_guard import safety_guard_node, DISCLAIMER

    state: ClinicalQueryState = {
        "query": "What is the patient's diagnosis?",
        "patient_id": "test",
        "final_answer": "The patient has Type 2 Diabetes Mellitus.",
        "retrieved_chunks": [{"content": "Diabetes diagnosis confirmed", "chunk_type": "text"}],
        "confidence": 0.8,
        "reasoning_trace": [],
        "agent_path": [],
    }

    result = safety_guard_node(state)
    assert DISCLAIMER in result["final_answer"]
    assert "hallucination_risk" in result
    assert "safety_guard" in result["agent_path"]


@pytest.mark.asyncio
async def test_safety_guard_flags_empty_answer() -> None:
    """Safety guard flags too-short answers."""
    from medscan.agents.nodes.safety_guard import safety_guard_node

    state: ClinicalQueryState = {
        "query": "What is the diagnosis?",
        "patient_id": "test",
        "final_answer": "N/A",
        "retrieved_chunks": [],
        "confidence": 0.9,
        "reasoning_trace": [],
        "agent_path": [],
    }

    result = safety_guard_node(state)
    assert "ANSWER_TOO_SHORT" in result["safety_flags"]
    assert result["hallucination_risk"] in ("medium", "high")


@pytest.mark.asyncio
async def test_full_graph_no_chunks() -> None:
    """Full graph runs without crashing when no chunks retrieved."""
    from medscan.agents.graph import run_clinical_query

    state = await run_clinical_query(
        query="What are the patient's latest lab results?",
        patient_id="00000000-0000-0000-0000-000000000001",
        retrieved_chunks=[],
        retrieved_tables=[],
        retrieved_images=[],
    )

    assert "final_answer" in state
    assert "confidence" in state
    assert "agent_path" in state
    assert len(state["agent_path"]) > 0

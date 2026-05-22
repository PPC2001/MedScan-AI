"""
LangGraph StateGraph — the main multi-agent clinical query graph.

Graph topology:
    START
      │
      ▼
   [router]  ──────────────────────────────────────┐
      │                                            │
   (conditional edge)                              │
      ├── text_rag ──► [text_rag_agent]             │
      ├── table_qa ──► [table_qa_agent]             │
      └── visual  ──► [visual_reasoning_agent]      │
                         │                          │
                         ▼                          │
                   [synthesizer]  ◄─────────────────┘
                         │
                         ▼
                  [safety_guard]
                         │
                        END
"""

import logging
import time
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from medscan.agents.nodes.router import route_decision, router_node
from medscan.agents.nodes.safety_guard import safety_guard_node
from medscan.agents.nodes.synthesizer import synthesizer_node
from medscan.agents.nodes.table_qa import table_qa_agent_node
from medscan.agents.nodes.text_rag import text_rag_agent_node
from medscan.agents.nodes.visual_reasoning import visual_reasoning_agent_node
from medscan.agents.state import ClinicalQueryState

logger = logging.getLogger(__name__)


def build_clinical_graph() -> Any:
    """
    Build and compile the clinical query StateGraph.

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    graph = StateGraph(ClinicalQueryState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("text_rag_agent", text_rag_agent_node)
    graph.add_node("table_qa_agent", table_qa_agent_node)
    graph.add_node("visual_reasoning_agent", visual_reasoning_agent_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("safety_guard", safety_guard_node)

    # Entry point
    graph.add_edge(START, "router")

    # Conditional routing from router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "text_rag_agent": "text_rag_agent",
            "table_qa_agent": "table_qa_agent",
            "visual_reasoning_agent": "visual_reasoning_agent",
        },
    )

    # All agents converge to synthesizer
    graph.add_edge("text_rag_agent", "synthesizer")
    graph.add_edge("table_qa_agent", "synthesizer")
    graph.add_edge("visual_reasoning_agent", "synthesizer")

    # Synthesizer → safety guard → end
    graph.add_edge("synthesizer", "safety_guard")
    graph.add_edge("safety_guard", END)

    return graph.compile()


# Module-level compiled graph (lazy singleton)
_graph = None


def get_graph() -> Any:
    """Return the compiled clinical query graph (singleton)."""
    global _graph
    if _graph is None:
        logger.info("Compiling clinical query graph...")
        _graph = build_clinical_graph()
    return _graph


async def run_clinical_query(
    query: str,
    patient_id: str,
    retrieved_chunks: list[dict[str, Any]] | None = None,
    retrieved_tables: list[dict[str, Any]] | None = None,
    retrieved_images: list[dict[str, Any]] | None = None,
) -> ClinicalQueryState:
    """
    Run a clinical query through the full multi-agent graph.

    Args:
        query: Natural language clinical question.
        patient_id: Patient UUID string.
        retrieved_chunks: Pre-retrieved text chunks (from RAG).
        retrieved_tables: Pre-retrieved table data.
        retrieved_images: Pre-retrieved image data.

    Returns:
        Final ClinicalQueryState with answer, confidence, and sources.
    """
    start_time = time.monotonic()

    initial_state: ClinicalQueryState = {
        "query": query,
        "patient_id": patient_id,
        "session_id": str(uuid.uuid4()),
        "retrieved_chunks": retrieved_chunks or [],
        "retrieved_tables": retrieved_tables or [],
        "retrieved_images": retrieved_images or [],
        "reasoning_trace": [],
        "intermediate_answers": [],
        "agent_path": [],
        "final_answer": "",
        "confidence": 0.0,
        "sources": [],
        "is_safe": True,
        "safety_flags": [],
        "hallucination_risk": "low",
        "error": None,
    }

    try:
        graph = get_graph()
        final_state = await graph.ainvoke(initial_state)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        final_state["latency_ms"] = round(elapsed_ms, 2)

        logger.info(
            "Graph complete: query_type=%s, confidence=%.2f, latency=%.0fms, path=%s",
            final_state.get("query_type"),
            final_state.get("confidence", 0),
            elapsed_ms,
            " → ".join(final_state.get("agent_path", [])),
        )

        return final_state

    except Exception as e:
        logger.exception("Graph execution failed: %s", e)
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return {
            **initial_state,
            "final_answer": f"An error occurred processing your query: {e}",
            "confidence": 0.0,
            "is_safe": False,
            "error": str(e),
            "latency_ms": round(elapsed_ms, 2),
        }

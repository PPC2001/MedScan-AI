"""
Router Node — classifies incoming clinical query and routes to appropriate agent.

Routes to:
- text_rag: Narrative queries (history, notes, summaries)
- table_qa: Lab values, vitals, structured data queries
- visual_reasoning: Imaging, X-ray, clinical photo queries
- hybrid: Complex cross-modal queries
"""

import logging
from typing import Any

from medscan.agents.state import ClinicalQueryState
from medscan.hf.zero_shot import ZeroShotClassificationPipeline

logger = logging.getLogger(__name__)

# Query type labels for zero-shot classification
QUERY_TYPE_LABELS = [
    "laboratory results and blood test values",
    "radiology imaging X-ray CT MRI scan",
    "clinical notes physician narrative history",
    "medication prescription drug dosage",
    "vital signs blood pressure heart rate",
    "diagnosis condition disease",
    "clinical photograph wound skin lesion",
    "multiple data types complex reasoning",
]

LABEL_TO_ROUTE = {
    "laboratory results and blood test values": "table_qa",
    "radiology imaging X-ray CT MRI scan": "visual_reasoning",
    "clinical notes physician narrative history": "text_rag",
    "medication prescription drug dosage": "text_rag",
    "vital signs blood pressure heart rate": "table_qa",
    "diagnosis condition disease": "text_rag",
    "clinical photograph wound skin lesion": "visual_reasoning",
    "multiple data types complex reasoning": "hybrid",
}

_zero_shot: ZeroShotClassificationPipeline | None = None


def get_zero_shot() -> ZeroShotClassificationPipeline:
    global _zero_shot
    if _zero_shot is None:
        _zero_shot = ZeroShotClassificationPipeline()
    return _zero_shot


def router_node(state: ClinicalQueryState) -> ClinicalQueryState:
    """
    Classify the query and set query_type for routing.

    Uses zero-shot classification against predefined query type labels.
    Falls back to 'text_rag' if classification fails.
    """
    query = state.get("query", "")
    logger.info("Router node: classifying query: %s", query[:100])

    try:
        classifier = get_zero_shot()
        result = classifier.run(
            text=query,
            candidate_labels=QUERY_TYPE_LABELS,
        )
        top_label = result["top_label"]
        query_type = LABEL_TO_ROUTE.get(top_label, "text_rag")
        logger.info("Router: classified as '%s' → route='%s'", top_label, query_type)
    except Exception as e:
        logger.warning("Router classification failed: %s — defaulting to text_rag", e)
        query_type = "text_rag"

    return {
        **state,
        "query_type": query_type,
        "agent_path": [*state.get("agent_path", []), "router"],
        "reasoning_trace": [
            *state.get("reasoning_trace", []),
            f"Query classified as: {query_type}",
        ],
    }


def route_decision(state: ClinicalQueryState) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node to visit.
    """
    query_type = state.get("query_type", "text_rag")
    route_map = {
        "table_qa": "table_qa_agent",
        "visual_reasoning": "visual_reasoning_agent",
        "hybrid": "text_rag_agent",  # Hybrid starts with text_rag then visual
        "text_rag": "text_rag_agent",
    }
    return route_map.get(query_type, "text_rag_agent")

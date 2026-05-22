"""
Table QA Agent Node — answers quantitative questions against structured lab/vital tables.

Uses TAPAS (HuggingFace Table QA) + LLM reasoning for complex aggregations.
"""

import logging
from typing import Any

from medscan.agents.state import ClinicalQueryState
from medscan.hf.table_qa import TableQAPipeline

logger = logging.getLogger(__name__)

_tapas: TableQAPipeline | None = None


def get_tapas() -> TableQAPipeline:
    global _tapas
    if _tapas is None:
        _tapas = TableQAPipeline()
    return _tapas


def table_qa_agent_node(state: ClinicalQueryState) -> ClinicalQueryState:
    """
    Table QA agent: uses TAPAS to query structured tables extracted from documents.

    Handles:
    - "What was the patient's HbA1c result?"
    - "List all medications with dose > 500mg"
    - "What are the critical lab values?"
    - "Show me all abnormal findings"
    """
    query = state.get("query", "")
    tables = state.get("retrieved_tables", [])
    chunks = state.get("retrieved_chunks", [])

    logger.info(
        "Table QA agent: %d tables, %d chunks, query='%s'",
        len(tables), len(chunks), query[:80],
    )

    answers = []

    # Try TAPAS on each retrieved table
    tapas = get_tapas()
    for table in tables[:5]:  # Max 5 tables
        try:
            tapas_table = table.get("tapas_format", {})
            if tapas_table:
                result = tapas.run(table=tapas_table, query=query)
                answer = result["answer"]
                if answer and answer.lower() not in ("", "none", "n/a"):
                    table_type = table.get("table_type", "table")
                    answers.append(
                        f"From {table_type} (page {table.get('page_number', '?')}): {answer}"
                    )
        except Exception as e:
            logger.warning("TAPAS query failed on table: %s", e)

    # Also check text chunks for tabular data patterns
    if not answers:
        for chunk in chunks[:5]:
            if chunk.get("chunk_type") == "table":
                answers.append(
                    f"Found table content: {chunk['content'][:500]}"
                )

    if not answers:
        answers.append(
            "No structured table data found matching this query. "
            "The information may be in narrative text form."
        )

    combined_answer = "\n\n".join(answers)

    return {
        **state,
        "intermediate_answers": [
            *state.get("intermediate_answers", []),
            combined_answer,
        ],
        "agent_path": [*state.get("agent_path", []), "table_qa_agent"],
        "reasoning_trace": [
            *state.get("reasoning_trace", []),
            f"Table QA: Queried {len(tables)} tables using TAPAS.",
        ],
    }

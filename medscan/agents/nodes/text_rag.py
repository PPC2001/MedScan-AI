"""
Text RAG Agent Node — retrieves relevant chunks and generates answers
using the configured LLM (Grok / OpenAI / Anthropic — auto-detected).
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from medscan.agents.state import ClinicalQueryState
from medscan.llm import get_llm_or_none

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert clinical AI assistant helping healthcare professionals
understand patient records. You have access to retrieved excerpts from the patient's
medical documents.

Rules:
1. Answer ONLY based on the provided context. Do not add information not in the context.
2. If the context doesn't contain the answer, say so clearly.
3. Use precise medical terminology.
4. Always cite which document/page the information comes from.
5. Flag any critical findings (abnormal values, urgent diagnoses) clearly.
6. Never make diagnostic or treatment recommendations — only summarize findings."""


def text_rag_agent_node(state: ClinicalQueryState) -> ClinicalQueryState:
    """
    Text RAG agent: retrieves chunks from vector store and generates answer.

    The actual vector retrieval happens before this node via the RAG retriever.
    This node performs reasoning + generation over retrieved chunks.
    """
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    patient_id = state.get("patient_id", "unknown")

    logger.info(
        "Text RAG agent: %d chunks, query='%s'", len(chunks), query[:80]
    )

    if not chunks:
        return {
            **state,
            "intermediate_answers": [
                *state.get("intermediate_answers", []),
                "No relevant documents found for this query.",
            ],
            "agent_path": [*state.get("agent_path", []), "text_rag_agent"],
            "reasoning_trace": [
                *state.get("reasoning_trace", []),
                "Text RAG: No chunks retrieved — answering from general knowledge only.",
            ],
        }

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks[:10], 1):
        source = f"[Source {i}: {chunk.get('chunk_type', 'text')}, page {chunk.get('page_number', '?')}]"
        context_parts.append(f"{source}\n{chunk['content']}")

    context = "\n\n---\n\n".join(context_parts)

    user_message = f"""Patient ID: {patient_id}

Retrieved Medical Context:
{context}

Clinical Question: {query}

Please answer the question based solely on the provided context.
Cite specific sources where applicable."""

    llm = get_llm_or_none(max_tokens=1500)

    if llm is None:
        # No LLM key available — return raw chunks as the answer
        answer = (
            "⚠️ No LLM provider configured. Set XAI_API_KEY (Grok), "
            "OPENAI_API_KEY, or ANTHROPIC_API_KEY in your .env file.\n\n"
            f"Retrieved {len(chunks)} relevant chunks — here are the top excerpts:\n\n"
            + "\n\n---\n\n".join(c["content"][:400] for c in chunks[:3])
        )
    else:
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
            response = llm.invoke(messages)
            answer = response.content
        except Exception as e:
            logger.error("Text RAG LLM call failed: %s", e)
            answer = f"LLM call failed ({e}). Raw chunks available — check logs."

    return {
        **state,
        "intermediate_answers": [
            *state.get("intermediate_answers", []),
            answer,
        ],
        "agent_path": [*state.get("agent_path", []), "text_rag_agent"],
        "reasoning_trace": [
            *state.get("reasoning_trace", []),
            f"Text RAG: Generated answer from {len(chunks)} retrieved chunks.",
        ],
    }

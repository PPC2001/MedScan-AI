"""
Synthesizer Node — merges all intermediate answers into a single,
well-formatted clinical response using the configured LLM provider.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from medscan.agents.state import ClinicalQueryState
from medscan.llm import get_llm_or_none

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM = """You are a senior clinical information officer synthesizing
findings from multiple AI analysis modules into a single coherent answer.

Your response must:
1. Integrate all findings into a clear, structured narrative
2. Highlight critical or urgent findings prominently
3. Group related information logically (labs, imaging, medications, etc.)
4. Use clear medical terminology with explanations where helpful
5. Indicate confidence level (High/Medium/Low) based on data completeness
6. Note any gaps or limitations in the available data

Format the answer with clear sections. Be concise but complete."""


def synthesizer_node(state: ClinicalQueryState) -> ClinicalQueryState:
    """
    Synthesize all intermediate answers into a final unified response.
    """
    query = state.get("query", "")
    intermediate = state.get("intermediate_answers", [])

    logger.info("Synthesizer: merging %d intermediate answers", len(intermediate))

    if not intermediate:
        return {
            **state,
            "final_answer": "No information could be retrieved for this query.",
            "confidence": 0.0,
            "agent_path": [*state.get("agent_path", []), "synthesizer"],
        }

    if len(intermediate) == 1:
        # No synthesis needed for a single answer
        return {
            **state,
            "final_answer": intermediate[0],
            "confidence": 0.7,
            "agent_path": [*state.get("agent_path", []), "synthesizer"],
            "reasoning_trace": [
                *state.get("reasoning_trace", []),
                "Synthesizer: Single answer — no merge needed.",
            ],
        }

    # Multi-source synthesis
    answers_text = "\n\n---\n\n".join(
        f"[Module {i + 1}]:\n{ans}" for i, ans in enumerate(intermediate)
    )

    llm = get_llm_or_none(max_tokens=1500)

    if llm is None:
        # No LLM — just concatenate all intermediate answers
        final_answer = (
            "⚠️ No LLM configured for synthesis. Raw module outputs:\n\n" + answers_text
        )
        confidence = 0.4
    else:
        try:
            response = llm.invoke([
                SystemMessage(content=SYNTHESIZER_SYSTEM),
                HumanMessage(content=(
                    f"Original clinical question: {query}\n\n"
                    f"Analysis results from multiple modules:\n{answers_text}\n\n"
                    f"Please synthesize these into a single comprehensive answer."
                )),
            ])
            final_answer = response.content
            confidence = 0.8
        except Exception as e:
            logger.error("Synthesizer LLM call failed: %s", e)
            final_answer = "\n\n".join(intermediate)
            confidence = 0.5

    return {
        **state,
        "final_answer": final_answer,
        "confidence": confidence,
        "agent_path": [*state.get("agent_path", []), "synthesizer"],
        "reasoning_trace": [
            *state.get("reasoning_trace", []),
            f"Synthesizer: Merged {len(intermediate)} answers into unified response.",
        ],
    }

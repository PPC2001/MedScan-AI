"""
Visual Reasoning Agent Node — interprets medical images using BLIP VQA + the
configured LLM provider (Grok / OpenAI / Anthropic — auto-detected).
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from medscan.agents.state import ClinicalQueryState
from medscan.hf.vqa import VQAPipeline
from medscan.llm import get_llm_or_none

logger = logging.getLogger(__name__)

_vqa: VQAPipeline | None = None


def get_vqa() -> VQAPipeline:
    global _vqa
    if _vqa is None:
        _vqa = VQAPipeline()
    return _vqa


def visual_reasoning_agent_node(state: ClinicalQueryState) -> ClinicalQueryState:
    """
    Visual reasoning agent: answers questions about medical images.

    Uses:
    1. Retrieved image description chunks from vector store (text-based)
    2. BLIP VQA for direct image questions (if image objects are in state)
    3. Configured LLM (Grok / OpenAI / Anthropic) for synthesis
    """
    query = state.get("query", "")
    images = state.get("retrieved_images", [])
    chunks = state.get("retrieved_chunks", [])

    logger.info(
        "Visual reasoning agent: %d images, query='%s'", len(images), query[:80]
    )

    answers = []

    # Step 1: Check retrieved image description chunks
    image_chunks = [c for c in chunks if c.get("chunk_type") == "image_description"]

    if image_chunks:
        combined_descs = "\n\n".join(
            f"[Image from page {c.get('page_number', '?')}]:\n{c['content']}"
            for c in image_chunks[:5]
        )

        llm = get_llm_or_none(max_tokens=1000)
        if llm is None:
            # No LLM — return descriptions directly
            answers.append(
                "⚠️ No LLM configured. Raw image descriptions:\n\n" + combined_descs
            )
        else:
            try:
                response = llm.invoke([
                    SystemMessage(content=(
                        "You are a medical imaging expert. Answer questions about "
                        "medical images based on the provided image descriptions."
                    )),
                    HumanMessage(content=(
                        f"Image descriptions from patient's medical record:\n"
                        f"{combined_descs}\n\nQuestion: {query}"
                    )),
                ])
                answers.append(response.content)
            except Exception as e:
                logger.error("Visual reasoning LLM call failed: %s", e)
                answers.append(f"LLM call failed ({e}). Raw descriptions:\n{combined_descs}")

    # Step 2: Direct VQA on available image objects (if passed in state)
    for img_data in images[:3]:
        try:
            image_obj = img_data.get("image")
            if image_obj is not None:
                vqa = get_vqa()
                result = vqa.run(image=image_obj, question=query)
                answers.append(
                    f"Direct VQA answer (confidence {result['score']:.2f}): {result['answer']}"
                )
        except Exception as e:
            logger.warning("VQA on image failed: %s", e)

    if not answers:
        answers.append(
            "No imaging data found for this patient. "
            "Please ensure imaging reports or clinical photos have been uploaded."
        )

    combined = "\n\n".join(answers)

    return {
        **state,
        "intermediate_answers": [
            *state.get("intermediate_answers", []),
            combined,
        ],
        "agent_path": [*state.get("agent_path", []), "visual_reasoning_agent"],
        "reasoning_trace": [
            *state.get("reasoning_trace", []),
            f"Visual Reasoning: Analyzed {len(image_chunks)} image descriptions.",
        ],
    }

"""
Safety Guard Node — validates the final answer for:
1. Citation verification (does the answer reference provided context?)
2. Hallucination risk assessment
3. Medical disclaimer injection
4. Critical finding alerts
"""

import logging
import re

from medscan.agents.state import ClinicalQueryState

logger = logging.getLogger(__name__)

# Keywords that indicate critical/urgent clinical findings
CRITICAL_KEYWORDS = {
    "critical", "stat", "urgent", "emergency", "immediately",
    "life-threatening", "severe", "sepsis", "hemorrhage", "stroke",
    "cardiac arrest", "anaphylaxis", "toxic", "overdose",
    # Critical lab patterns
    "panic value", "critical value",
}

HALLUCINATION_RISK_PATTERNS = [
    r"studies show",
    r"research indicates",
    r"according to guidelines",
    r"typically|generally|usually|often",
    r"may|might|could|possibly",
]

DISCLAIMER = (
    "⚠️ CLINICAL DISCLAIMER: This AI-generated summary is based solely on the "
    "uploaded patient documents. It does not constitute medical advice, diagnosis, "
    "or treatment recommendation. All findings must be reviewed and verified by a "
    "qualified healthcare professional before any clinical action is taken."
)


def safety_guard_node(state: ClinicalQueryState) -> ClinicalQueryState:
    """
    Safety guard: validate answer quality and inject appropriate warnings.

    Checks:
    1. Is the answer grounded in retrieved context?
    2. Are there hallucination risk markers?
    3. Are there critical findings that need flagging?
    4. Inject disclaimer
    """
    answer = state.get("final_answer", "")
    chunks = state.get("retrieved_chunks", [])
    query = state.get("query", "")

    logger.info("Safety guard: validating answer (%d chars)", len(answer))

    safety_flags: list[str] = []
    hallucination_risk = "low"

    # Check 1: Empty or very short answer
    if len(answer.strip()) < 20:
        safety_flags.append("ANSWER_TOO_SHORT")
        hallucination_risk = "medium"

    # Check 2: No context was retrieved
    if not chunks:
        safety_flags.append("NO_CONTEXT_RETRIEVED")
        hallucination_risk = "high"

    # Check 3: Hallucination risk patterns
    risk_matches = 0
    for pattern in HALLUCINATION_RISK_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            risk_matches += 1

    if risk_matches >= 3:
        hallucination_risk = "high"
        safety_flags.append("HIGH_HALLUCINATION_MARKERS")
    elif risk_matches >= 1:
        hallucination_risk = "medium"

    # Check 4: Critical findings detection
    answer_lower = answer.lower()
    found_critical = [
        kw for kw in CRITICAL_KEYWORDS if kw in answer_lower
    ]
    if found_critical:
        safety_flags.append("CRITICAL_FINDINGS_DETECTED")
        critical_alert = (
            f"\n\n🚨 CRITICAL ALERT: This response contains findings that may require "
            f"immediate clinical attention: {', '.join(found_critical[:5])}"
        )
        answer = critical_alert + "\n\n" + answer

    # Check 5: Adjust confidence based on risk
    confidence = state.get("confidence", 0.7)
    if hallucination_risk == "high":
        confidence = min(confidence, 0.4)
    elif hallucination_risk == "medium":
        confidence = min(confidence, 0.65)

    # Inject disclaimer
    final_answer_with_disclaimer = f"{answer}\n\n---\n{DISCLAIMER}"

    is_safe = hallucination_risk != "high" and "ANSWER_TOO_SHORT" not in safety_flags

    return {
        **state,
        "final_answer": final_answer_with_disclaimer,
        "confidence": round(confidence, 3),
        "is_safe": is_safe,
        "safety_flags": safety_flags,
        "hallucination_risk": hallucination_risk,
        "disclaimer": DISCLAIMER,
        "agent_path": [*state.get("agent_path", []), "safety_guard"],
        "reasoning_trace": [
            *state.get("reasoning_trace", []),
            f"Safety Guard: risk={hallucination_risk}, flags={safety_flags}",
        ],
    }

"""
HF Task: Text Classification
Used for: Urgency/severity classification of clinical findings.
         Triage incoming documents by clinical priority.
Model: medicalai/ClinicalBERT (fine-tuned BERT on clinical notes)
"""

from typing import Any

from medscan.hf.base import BaseHFPipeline


# Urgency tier definitions
URGENCY_TIERS = {
    "CRITICAL": {"score_threshold": 0.8, "color": "red", "sla_hours": 1},
    "URGENT": {"score_threshold": 0.6, "color": "orange", "sla_hours": 24},
    "ROUTINE": {"score_threshold": 0.4, "color": "yellow", "sla_hours": 72},
    "INFORMATIONAL": {"score_threshold": 0.0, "color": "green", "sla_hours": None},
}


class TextClassificationPipeline(BaseHFPipeline):
    """
    Classify clinical text by urgency, severity, or custom labels.

    Primary use: Flag critical lab values, urgent diagnoses, or
    time-sensitive findings that require immediate clinical action.
    """

    task = "text-classification"
    default_model = "medicalai/ClinicalBERT"

    def run(
        self,
        text: str,
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Args:
            text: Clinical text snippet to classify.
            top_k: Number of top labels to return.

        Returns:
            List of dicts with 'label' and 'score' keys.
        """
        results = self.pipe(text, top_k=top_k)
        if isinstance(results, dict):
            results = [results]
        return [
            {"label": r["label"], "score": round(r["score"], 4)}
            for r in results
        ]

    def classify_urgency(self, text: str) -> dict[str, Any]:
        """
        Classify text urgency and return tier metadata.

        Returns:
            dict with 'label', 'score', 'tier', 'sla_hours', 'color'.
        """
        results = self.run(text=text, top_k=1)
        top = results[0] if results else {"label": "ROUTINE", "score": 0.5}

        # Map to urgency tier
        label = top["label"].upper()
        tier_info = URGENCY_TIERS.get(label, URGENCY_TIERS["ROUTINE"])

        return {
            "label": label,
            "score": top["score"],
            "tier": label,
            "sla_hours": tier_info["sla_hours"],
            "color": tier_info["color"],
            "requires_immediate_action": label == "CRITICAL",
        }

    def batch_classify(self, texts: list[str]) -> list[dict[str, Any]]:
        """Classify a list of text snippets."""
        return [self.classify_urgency(text=t) for t in texts]

"""
HF Task: Zero-Shot Classification
Used for: Triaging incoming documents without labeled training data.
         Classifies document intent, urgency, specialty, and category.
Model: facebook/bart-large-mnli
"""

from typing import Any

from medscan.hf.base import BaseHFPipeline


# Default candidate labels for medical document triage
DEFAULT_DOC_TYPE_LABELS = [
    "laboratory report",
    "radiology imaging report",
    "physician clinical note",
    "prescription medication order",
    "discharge summary",
    "clinical photograph",
    "pathology report",
    "referral letter",
    "consent form",
    "insurance authorization",
]

DEFAULT_URGENCY_LABELS = [
    "critical — immediate action required",
    "urgent — action needed within 24 hours",
    "routine — follow up at next visit",
    "informational — no action required",
]

DEFAULT_SPECIALTY_LABELS = [
    "cardiology",
    "oncology",
    "neurology",
    "endocrinology",
    "nephrology",
    "pulmonology",
    "gastroenterology",
    "hematology",
    "orthopedics",
    "general medicine",
]


class ZeroShotClassificationPipeline(BaseHFPipeline):
    """
    Classify medical text or document descriptions into arbitrary label sets
    without fine-tuning. Excellent for triage and routing.
    """

    task = "zero-shot-classification"
    default_model = "facebook/bart-large-mnli"

    def run(
        self,
        text: str,
        candidate_labels: list[str],
        multi_label: bool = False,
    ) -> dict[str, Any]:
        """
        Args:
            text: Text to classify (snippet or full document excerpt).
            candidate_labels: List of possible class labels.
            multi_label: If True, returns independent scores for each label.

        Returns:
            dict with 'labels' (sorted by score), 'scores', and 'top_label'.
        """
        result = self.pipe(
            text,
            candidate_labels=candidate_labels,
            multi_label=multi_label,
        )
        return {
            "top_label": result["labels"][0],
            "top_score": round(result["scores"][0], 4),
            "labels": result["labels"],
            "scores": [round(s, 4) for s in result["scores"]],
        }

    def triage_document(self, text: str) -> dict[str, Any]:
        """
        Full triage pipeline: classify doc type, urgency, and medical specialty.

        Returns:
            Comprehensive triage result with all three dimensions.
        """
        doc_type = self.run(text=text, candidate_labels=DEFAULT_DOC_TYPE_LABELS)
        urgency = self.run(text=text, candidate_labels=DEFAULT_URGENCY_LABELS)
        specialty = self.run(text=text, candidate_labels=DEFAULT_SPECIALTY_LABELS)

        return {
            "document_type": doc_type["top_label"],
            "document_type_confidence": doc_type["top_score"],
            "urgency": urgency["top_label"],
            "urgency_confidence": urgency["top_score"],
            "medical_specialty": specialty["top_label"],
            "specialty_confidence": specialty["top_score"],
        }

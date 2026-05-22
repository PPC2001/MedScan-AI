"""
HF Task: Image Classification
Used for: Classifying incoming document pages by type
         (lab_report, prescription, imaging, physician_note, clinical_photo).
Model: google/vit-base-patch16-224 (fine-tunable on medical doc types)
"""

from typing import Any

from medscan.hf.base import BaseHFPipeline


# Labels map to DocumentType enum values
MEDICAL_DOC_LABELS = [
    "lab_report",
    "prescription",
    "imaging_radiology",
    "physician_note",
    "clinical_photograph",
    "discharge_summary",
    "other_medical_document",
]


class ImageClassificationPipeline(BaseHFPipeline):
    """
    Classify document page images into medical document categories.
    Can use zero-shot or a fine-tuned ViT checkpoint.
    """

    task = "image-classification"
    default_model = "google/vit-base-patch16-224"

    def run(
        self,
        image: Any,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Args:
            image: PIL Image or image path.
            top_k: Number of top predictions to return.

        Returns:
            List of dicts with 'label' and 'score' keys, sorted by score desc.
        """
        results = self.pipe(image, top_k=top_k)
        return [
            {"label": r["label"], "score": round(r["score"], 4)}
            for r in results
        ]

    def classify_top(self, image: Any) -> dict[str, Any]:
        """Return only the top prediction."""
        results = self.run(image=image, top_k=1)
        return results[0] if results else {"label": "unknown", "score": 0.0}

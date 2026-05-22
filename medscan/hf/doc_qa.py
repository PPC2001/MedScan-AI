"""
HF Task: Document Question Answering
Used for: Core query interface against clinical documents.
Model: impira/layoutlm-document-qa (understands layout + text)
"""

from typing import Any

from medscan.hf.base import BaseHFPipeline


class DocumentQAPipeline(BaseHFPipeline):
    """
    Answer questions directly from a document image (LayoutLM-based).
    Useful for structured forms, lab report tables, prescription pads.
    """

    task = "document-question-answering"
    default_model = "impira/layoutlm-document-qa"

    def run(self, image: Any, question: str) -> dict[str, Any]:
        """
        Args:
            image: PIL Image or path to image file.
            question: Natural language question about the document.

        Returns:
            dict with 'answer', 'score', 'start', 'end' keys.
        """
        result = self.pipe(image=image, question=question)
        if isinstance(result, list):
            result = result[0]
        return {
            "answer": result.get("answer", ""),
            "confidence": round(result.get("score", 0.0), 4),
            "start": result.get("start"),
            "end": result.get("end"),
        }

    def run_batch(
        self, image: Any, questions: list[str]
    ) -> list[dict[str, Any]]:
        """Answer multiple questions against the same document image."""
        return [self.run(image=image, question=q) for q in questions]

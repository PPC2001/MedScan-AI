"""
HF Task: Visual Question Answering
Used for: "What does this X-ray show?", "Is there consolidation in this CT?",
          "What anatomical region is shown?", "Are there any abnormalities?"
Model: Salesforce/blip-vqa-base
"""

from typing import Any

from medscan.hf.base import BaseHFPipeline


# Pre-built clinical VQA question templates
CLINICAL_VQA_QUESTIONS = {
    "anatomy": "What anatomical region or body part is shown in this image?",
    "modality": "What imaging modality is this? (X-ray, CT, MRI, ultrasound, photograph)",
    "findings": "What are the main findings or abnormalities visible in this image?",
    "quality": "Is the image quality adequate for clinical interpretation?",
    "laterality": "Is this the left side, right side, or bilateral?",
    "contrast": "Is contrast enhancement used in this image?",
}


class VQAPipeline(BaseHFPipeline):
    """
    Answer clinical questions about medical images.

    Supports: X-rays, CT slices, MRI images, ultrasound frames,
    clinical photographs, histopathology slides.
    """

    task = "visual-question-answering"
    default_model = "Salesforce/blip-vqa-base"

    def run(
        self,
        image: Any,
        question: str,
    ) -> dict[str, Any]:
        """
        Args:
            image: PIL Image, URL, or file path.
            question: Clinical question about the image.

        Returns:
            dict with 'answer' and 'score' keys.
        """
        result = self.pipe(image=image, question=question)
        if isinstance(result, list):
            result = result[0]

        return {
            "answer": result.get("answer", ""),
            "score": round(result.get("score", 0.0), 4),
            "question": question,
        }

    def clinical_analysis(self, image: Any) -> dict[str, Any]:
        """
        Run a battery of standard clinical VQA questions against an image.

        Returns:
            Dict mapping question_type → {'answer', 'score'} pairs.
        """
        analysis: dict[str, Any] = {}
        for question_type, question in CLINICAL_VQA_QUESTIONS.items():
            analysis[question_type] = self.run(image=image, question=question)
        return analysis

    def ask_about_xray(self, image: Any) -> str:
        """Convenience: generate a structured X-ray description."""
        results = self.clinical_analysis(image=image)
        lines = [
            f"Modality: {results['modality']['answer']}",
            f"Region: {results['anatomy']['answer']}",
            f"Laterality: {results['laterality']['answer']}",
            f"Findings: {results['findings']['answer']}",
        ]
        return "\n".join(lines)

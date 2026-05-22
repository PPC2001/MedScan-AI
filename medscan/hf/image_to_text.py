"""
HF Task: Image-to-Text (Captioning + Clinical Photo Description)
Used for: Processing clinical photographs, X-ray descriptions, wound images.
Model: Salesforce/blip-image-captioning-large
"""

from typing import Any

from medscan.hf.base import BaseHFPipeline


class ImageToTextPipeline(BaseHFPipeline):
    """
    Generate textual descriptions of medical images.
    Used for clinical photos, wound images, skin conditions, etc.
    """

    task = "image-to-text"
    default_model = "Salesforce/blip-image-captioning-large"

    def run(
        self,
        image: Any,
        max_new_tokens: int = 200,
        context_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Args:
            image: PIL Image, URL, or file path.
            max_new_tokens: Maximum caption length.
            context_prompt: Optional medical context (e.g. "A clinical photograph of")

        Returns:
            dict with 'caption' and 'model' keys.
        """
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if context_prompt:
            kwargs["prompt"] = context_prompt

        results = self.pipe(image, **kwargs)
        caption = results[0]["generated_text"] if results else ""

        return {
            "caption": caption.strip(),
            "model": self.model_name,
            "max_new_tokens": max_new_tokens,
        }

    def describe_medical_image(self, image: Any) -> str:
        """Convenience method with a clinical context prompt."""
        result = self.run(
            image=image,
            max_new_tokens=300,
            context_prompt="A medical/clinical image showing",
        )
        return result["caption"]

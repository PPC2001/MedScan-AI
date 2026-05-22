"""
Vision Pipeline — GPT-4o multimodal for complex image understanding.

Handles:
- X-ray and imaging interpretation
- Table extraction from embedded images
- Clinical photograph analysis
- Embedded chart / graph interpretation
- Handwriting assistance (for complex cases TrOCR can't handle)
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image  # type: ignore[import]
from tenacity import retry, stop_after_attempt, wait_exponential

from medscan.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string for API calls."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _load_image(source: Any) -> Image.Image:
    """Load image from PIL Image, file path, or bytes."""
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, (str, Path)):
        return Image.open(source).convert("RGB")
    if isinstance(source, bytes):
        return Image.open(BytesIO(source)).convert("RGB")
    raise ValueError(f"Unsupported image source type: {type(source)}")


class GPT4oVisionProcessor:
    """
    Multimodal vision processor for medical image understanding.

    Automatically uses the best available vision-capable provider:
    - Grok  → xAI endpoint (OpenAI-compatible) with grok-3 vision
    - OpenAI → GPT-4o
    - No key → graceful fallback (returns empty/placeholder results)
    """

    def __init__(self) -> None:
        from medscan.config import get_settings
        cfg = get_settings()

        # Prefer Grok (xAI), then OpenAI
        if cfg.xai_api_key:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=cfg.xai_api_key,
                base_url=cfg.grok_base_url,
            )
            self.model = cfg.grok_model
            self._provider = "grok"
        elif cfg.openai_api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=cfg.openai_api_key)
            self.model = cfg.openai_model
            self._provider = "openai"
        else:
            self.client = None  # type: ignore[assignment]
            self.model = ""
            self._provider = "none"
            logger.warning(
                "No vision-capable LLM key configured (XAI_API_KEY or OPENAI_API_KEY). "
                "Image analysis will return placeholder results."
            )

    @property
    def available(self) -> bool:
        """True if a vision LLM is configured."""
        return self._provider != "none"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def analyze_image(
        self,
        image: Any,
        prompt: str,
        max_tokens: int = 1000,
        detail: str = "high",
    ) -> str:
        """
        Send an image with a prompt to the vision LLM.

        Returns empty string if no vision provider is configured.
        """
        if not self.available:
            logger.debug("analyze_image skipped — no vision LLM configured")
            return ""
        img = _load_image(image)
        b64 = _image_to_base64(img)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": detail,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def describe_medical_image(self, image: Any) -> dict[str, Any]:
        """Generate a structured clinical description of a medical image."""
        prompt = """You are an expert radiologist and clinical physician.
Analyze this medical image and provide a structured report with:
1. Image type/modality (X-ray, CT, MRI, photograph, etc.)
2. Anatomical region
3. Key findings (normal and abnormal)
4. Impression/assessment
5. Recommended follow-up (if any)

Be concise but clinically precise. Format as structured text."""

        description = self.analyze_image(image=image, prompt=prompt, detail="high")
        return {
            "description": description,
            "model": self.model,
            "type": "medical_image_analysis",
        }

    def extract_table_from_image(self, image: Any) -> dict[str, Any]:
        """Extract tabular data from an image of a table."""
        prompt = """Extract all data from this table/chart image.
Return the data as a structured JSON object with:
- "headers": list of column names
- "rows": list of row arrays
- "notes": any footnotes or special values
- "table_type": what type of table this is (lab results, vitals, medications, etc.)

If values have units, include them in the cell values."""

        import json

        raw = self.analyze_image(image=image, prompt=prompt, detail="high")

        # Attempt to parse JSON from response
        try:
            # Find JSON block
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

        return {
            "raw_extraction": raw,
            "headers": [],
            "rows": [],
            "notes": "",
            "table_type": "unknown",
        }

    def transcribe_handwriting(self, image: Any) -> str:
        """Transcribe handwritten physician notes using GPT-4o vision."""
        prompt = """Transcribe all handwritten text in this image exactly as written.
Preserve:
- Abbreviations (do not expand them)
- Medical terminology
- Numbers and units
- Paragraph breaks

If text is unclear, mark it as [illegible].
Output only the transcription, nothing else."""

        return self.analyze_image(image=image, prompt=prompt, detail="high")

    def answer_visual_question(self, image: Any, question: str) -> str:
        """Answer a clinical question about a medical image."""
        prompt = f"""You are a medical image expert.
Answer the following clinical question about this image:

Question: {question}

Provide a concise, accurate clinical answer. If you cannot determine 
the answer from the image alone, say so explicitly."""

        return self.analyze_image(image=image, prompt=prompt, detail="high")

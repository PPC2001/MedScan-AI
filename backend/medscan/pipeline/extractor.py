"""
Structured Clinical Data Extractor — Instructor + Pydantic + GPT-4o.

Extracts structured clinical entities from raw OCR text:
- Lab values with flags and ICD-10 codes
- Vitals (BP, HR, temp, SpO2, weight, height)
- Diagnoses with ICD-10 / SNOMED codes
- Medications with dose, frequency, route, RxNorm
- Patient identifiers and encounter metadata
"""

import logging
from typing import Any

import instructor
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from medscan.config import get_settings
from medscan.models.schemas import (
    Diagnosis,
    ExtractedClinicalData,
    LabValue,
    Medication,
    Vital,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class ClinicalExtractor:
    """
    Uses Instructor + the configured LLM to extract structured clinical data.

    Works with Grok (via OpenAI-compatible endpoint), OpenAI, or gracefully
    returns empty results when no API key is configured.
    """

    def __init__(self) -> None:
        from medscan.config import get_settings
        cfg = get_settings()

        # Instructor patches an OpenAI-compatible client
        # Grok exposes the same /v1/chat/completions API
        if cfg.xai_api_key:
            raw_client = OpenAI(
                api_key=cfg.xai_api_key,
                base_url=cfg.grok_base_url,
            )
            self.model = cfg.grok_model
            self._provider = "grok"
        elif cfg.openai_api_key:
            raw_client = OpenAI(api_key=cfg.openai_api_key)
            self.model = cfg.openai_model
            self._provider = "openai"
        else:
            self.client = None  # type: ignore[assignment]
            self.model = ""
            self._provider = "none"
            logger.warning(
                "No API key configured for clinical extraction. "
                "Structured data will not be extracted."
            )
            return

        self.client = instructor.patch(raw_client)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def extract(self, text: str, doc_type: str = "unknown") -> ExtractedClinicalData:
        """Extract all structured clinical entities from raw document text."""
        # No provider configured — return empty result
        if self._provider == "none" or self.client is None:
            logger.debug("extract() skipped — no LLM provider configured")
            return ExtractedClinicalData()
        system_prompt = """You are a clinical data extraction specialist.
Extract ALL structured clinical information from the provided medical document text.

Rules:
- Extract every lab value with its numeric result, unit, and reference range
- Flag abnormal values as H (high), L (low), or C (critical)
- Extract all medications with full prescribing details
- Include ICD-10 codes where you can determine them with confidence
- For dates, use ISO format (YYYY-MM-DD)
- If information is ambiguous or unclear, omit it rather than guess
- Preserve numeric precision exactly as written"""

        user_prompt = f"""Document type: {doc_type}

Document text:
{text[:8000]}  # Limit to avoid token overflow

Extract all clinical data from this document."""

        try:
            return self.client.chat.completions.create(
                model=self.model,
                response_model=ExtractedClinicalData,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2000,
            )
        except Exception as e:
            logger.error("Extraction failed: %s", e)
            return ExtractedClinicalData()

    def extract_labs_only(self, text: str) -> list[LabValue]:
        """Targeted extraction of lab values only (faster, cheaper)."""
        result = self.extract(text=text, doc_type="lab_report")
        return result.lab_values

    def extract_medications_only(self, text: str) -> list[Medication]:
        """Targeted extraction of medications only."""
        result = self.extract(text=text, doc_type="prescription")
        return result.medications

    def to_dict(self, data: ExtractedClinicalData) -> dict[str, Any]:
        """Convert extraction result to a JSON-serializable dict."""
        return data.model_dump(exclude_none=True)

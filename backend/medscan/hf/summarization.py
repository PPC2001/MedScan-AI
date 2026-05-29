"""
HF Task: Summarization
Used for: Generating patient timelines, discharge summaries,
         episode-of-care summaries, and medication review narratives.
Model: Falconsai/medical_summarization (domain-adapted BART/T5)
"""

import logging
from typing import Any

from medscan.hf.base import BaseHFPipeline

logger = logging.getLogger(__name__)


class SummarizationPipeline(BaseHFPipeline):
    """
    Summarize clinical text into concise, structured narratives.

    Used for:
    - Patient timeline generation across multiple documents
    - Discharge summary drafting
    - Medication history consolidation
    - Progress note summarization
    """

    task = "summarization"
    default_model = "Falconsai/medical_summarization"

    def _load(self) -> Any:
        """Override to load model and tokenizer directly instead of using pipeline."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        logger.info(
            "Loading Seq2Seq model for summarization: %s device=%s",
            self.model_name,
            self.device,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if self.device != "cpu":
            model = model.to(self.device)
        return {"model": model, "tokenizer": tokenizer}

    def run(
        self,
        text: str,
        max_length: int = 512,
        min_length: int = 64,
        do_sample: bool = False,
    ) -> dict[str, Any]:
        """
        Args:
            text: Clinical text to summarize (can be long — chunked internally).
            max_length: Maximum token length of summary.
            min_length: Minimum token length of summary.
            do_sample: Whether to use sampling (False = deterministic).

        Returns:
            dict with 'summary' and 'original_length' keys.
        """
        import torch

        # Truncate input to avoid OOM on very long docs (model's max context)
        max_input_chars = 4000
        truncated = text[:max_input_chars]

        loaded = self.pipe
        model = loaded["model"]
        tokenizer = loaded["tokenizer"]

        inputs = tokenizer(truncated, return_tensors="pt", truncation=True, max_length=1024)
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "summary": summary.strip(),
            "original_length": len(text),
            "truncated": len(text) > max_input_chars,
        }

    def summarize_patient_timeline(
        self, documents: list[dict[str, str]]
    ) -> str:
        """
        Summarize across multiple document texts into a patient timeline.

        Args:
            documents: List of dicts with 'date', 'doc_type', 'text' keys.

        Returns:
            Consolidated timeline summary string.
        """
        combined = "\n\n".join(
            f"[{doc.get('date', 'Unknown date')} — {doc.get('doc_type', 'Document')}]\n"
            f"{doc.get('text', '')}"
            for doc in documents
        )
        result = self.run(text=combined, max_length=800, min_length=100)
        return result["summary"]

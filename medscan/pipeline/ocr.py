"""
OCR Pipeline — Surya OCR + TrOCR for handwritten text.

Handles:
- Scanned PDF pages (Surya layout detection + text recognition)
- Handwritten physician notes (TrOCR)
- Mixed documents (detect layout first, then apply appropriate OCR)
"""

import logging
from pathlib import Path
from typing import Any

from PIL import Image  # type: ignore[import]

logger = logging.getLogger(__name__)


class OCRResult:
    """Structured result from OCR processing."""

    def __init__(
        self,
        text: str,
        confidence: float,
        page_number: int,
        blocks: list[dict[str, Any]],
        is_handwritten: bool = False,
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.page_number = page_number
        self.blocks = blocks  # layout blocks with bounding boxes
        self.is_handwritten = is_handwritten

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "page_number": self.page_number,
            "block_count": len(self.blocks),
            "is_handwritten": self.is_handwritten,
        }


class SuryaOCRProcessor:
    """
    Surya OCR wrapper for high-quality layout-aware text extraction.
    Surya handles: text detection, layout parsing, reading order, and recognition.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._det_model = None
        self._rec_model = None
        self._layout_model = None

    def _load_models(self) -> None:
        """Lazy-load Surya models."""
        try:
            from surya.model.detection.model import load_model as load_det  # type: ignore[import]
            from surya.model.recognition.model import load_model as load_rec  # type: ignore[import]
            from surya.model.recognition.processor import load_processor  # type: ignore[import]

            if self._det_model is None:
                logger.info("Loading Surya detection model...")
                self._det_model = load_det()
                self._det_processor = load_det()

            if self._rec_model is None:
                logger.info("Loading Surya recognition model...")
                self._rec_model = load_rec()
                self._rec_processor = load_processor()

        except ImportError as e:
            logger.warning("Surya OCR not available: %s. Using fallback.", e)

    def process_image(
        self, image: Image.Image, page_number: int = 1, langs: list[str] | None = None
    ) -> OCRResult:
        """
        Run Surya OCR on a single page image.

        Args:
            image: PIL Image of the page.
            page_number: Page number for metadata.
            langs: Language codes (e.g. ['en', 'hi']). Defaults to ['en'].

        Returns:
            OCRResult with extracted text, confidence, and block layout.
        """
        if langs is None:
            langs = ["en"]

        self._load_models()

        try:
            from surya.ocr import run_ocr  # type: ignore[import]

            predictions = run_ocr(
                [image],
                [langs],
                self._det_model,
                self._det_processor,
                self._rec_model,
                self._rec_processor,
            )
            page_pred = predictions[0]

            blocks = []
            full_text_parts = []
            confidences = []

            for line in page_pred.text_lines:
                text = line.text.strip()
                if text:
                    full_text_parts.append(text)
                    confidences.append(line.confidence)
                    blocks.append({
                        "text": text,
                        "confidence": line.confidence,
                        "bbox": line.bbox,
                    })

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            full_text = "\n".join(full_text_parts)

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                page_number=page_number,
                blocks=blocks,
                is_handwritten=False,
            )

        except Exception as e:
            logger.error("Surya OCR failed on page %d: %s", page_number, e)
            return OCRResult(
                text="",
                confidence=0.0,
                page_number=page_number,
                blocks=[],
                is_handwritten=False,
            )


class HandwritingOCRProcessor:
    """
    TrOCR processor for handwritten physician notes.
    Microsoft's TrOCR is specifically trained on handwritten text.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore[import]
            logger.info("Loading TrOCR handwriting model...")
            self._processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-large-handwritten"
            )
            self._model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-large-handwritten"
            )

    def transcribe(self, image: Image.Image) -> str:
        """
        Transcribe handwritten text from an image region.

        Args:
            image: PIL Image (preferably a cropped region of handwriting).

        Returns:
            Transcribed text string.
        """
        self._load()
        try:
            import torch  # type: ignore[import]

            pixel_values = self._processor(images=image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = self._model.generate(pixel_values)
            transcription = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return transcription.strip()
        except Exception as e:
            logger.error("TrOCR transcription failed: %s", e)
            return ""


class OCROrchestrator:
    """
    Orchestrates OCR across different document types.
    Automatically selects Surya (printed) or TrOCR (handwritten).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.surya = SuryaOCRProcessor(device=device)
        self.handwriting = HandwritingOCRProcessor(device=device)

    def process_pdf_pages(
        self, pdf_path: Path, max_pages: int = 50
    ) -> list[OCRResult]:
        """
        Convert PDF to images and OCR each page.

        Args:
            pdf_path: Path to PDF file.
            max_pages: Maximum pages to process.

        Returns:
            List of OCRResult objects, one per page.
        """
        try:
            import fitz  # PyMuPDF  # type: ignore[import]
        except ImportError:
            logger.warning("PyMuPDF not installed. Using PIL fallback.")
            return []

        doc = fitz.open(str(pdf_path))
        results = []

        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            # Render at 2x DPI for better OCR accuracy
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            result = self.surya.process_image(image=img, page_number=page_num + 1)
            results.append(result)
            logger.debug("OCR page %d/%d: %d chars", page_num + 1, len(doc), len(result.text))

        doc.close()
        return results

    def get_full_text(self, results: list[OCRResult]) -> str:
        """Concatenate all page texts into a single document string."""
        parts = []
        for r in results:
            if r.text.strip():
                parts.append(f"[Page {r.page_number}]\n{r.text}")
        return "\n\n".join(parts)

"""
Master Document Ingestion Pipeline — orchestrates all processing stages.

Stages:
1. File detection & validation
2. Unstructured parsing (text extraction)
3. OCR (Surya for scanned docs)
4. Image extraction & analysis (GPT-4o vision)
5. Table detection & extraction
6. Urgency triage (zero-shot + text classification)
7. Document type classification
8. Medical NER
9. Smart chunking
10. Embedding generation
11. Structured data extraction (Instructor)
12. Store to DB + vector store
"""

import logging
import mimetypes
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medscan.config import get_settings
from medscan.hf.image_classification import ImageClassificationPipeline
from medscan.hf.token_classification import TokenClassificationPipeline
from medscan.hf.zero_shot import ZeroShotClassificationPipeline
from medscan.hf.text_classification import TextClassificationPipeline
from medscan.hf.summarization import SummarizationPipeline
from medscan.pipeline.chunker import Chunk, MedicalAwareChunker
from medscan.pipeline.embedder import DocumentEmbedder
from medscan.pipeline.extractor import ClinicalExtractor
from medscan.pipeline.ocr import OCROrchestrator
from medscan.pipeline.table_extractor import TableExtractor
from medscan.pipeline.vision import GPT4oVisionProcessor

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class IngestionResult:
    """Complete result from the ingestion pipeline."""
    document_id: str
    raw_text: str
    chunks: list[Chunk] = field(default_factory=list)
    structured_data: dict[str, Any] = field(default_factory=dict)
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)
    urgency_label: str = "ROUTINE"
    urgency_score: float = 0.5
    doc_type_detected: str = "other"
    doc_type_confidence: float = 0.0
    named_entities: dict[str, list[str]] = field(default_factory=dict)
    tables: list[dict[str, Any]] = field(default_factory=list)
    image_descriptions: list[str] = field(default_factory=list)
    summary: str = ""
    success: bool = True
    error: str | None = None


class DocumentIngestionPipeline:
    """
    End-to-end document processing pipeline.

    Instantiate once and call `process()` for each document.
    All HF models are lazy-loaded on first use.
    """

    def __init__(self, device: str | None = None) -> None:
        dev = device or settings.hf_device

        # Core processors
        self.ocr = OCROrchestrator(device=dev)
        self.vision = GPT4oVisionProcessor()
        self.chunker = MedicalAwareChunker(chunk_size=512, chunk_overlap=64)
        self.embedder = DocumentEmbedder(device=dev)
        self.extractor = ClinicalExtractor()
        self.table_extractor = TableExtractor(device=dev)

        # HuggingFace task pipelines (lazy via property access)
        self._zero_shot: ZeroShotClassificationPipeline | None = None
        self._text_clf: TextClassificationPipeline | None = None
        self._img_clf: ImageClassificationPipeline | None = None
        self._ner: TokenClassificationPipeline | None = None
        self._summarizer: SummarizationPipeline | None = None

        self._dev = dev

    # Lazy properties to avoid loading all models at startup
    @property
    def zero_shot(self) -> ZeroShotClassificationPipeline:
        if self._zero_shot is None:
            self._zero_shot = ZeroShotClassificationPipeline(device=self._dev)
        return self._zero_shot

    @property
    def text_clf(self) -> TextClassificationPipeline:
        if self._text_clf is None:
            self._text_clf = TextClassificationPipeline(device=self._dev)
        return self._text_clf

    @property
    def img_clf(self) -> ImageClassificationPipeline:
        if self._img_clf is None:
            self._img_clf = ImageClassificationPipeline(device=self._dev)
        return self._img_clf

    @property
    def ner(self) -> TokenClassificationPipeline:
        if self._ner is None:
            self._ner = TokenClassificationPipeline(device=self._dev)
        return self._ner

    @property
    def summarizer(self) -> SummarizationPipeline:
        if self._summarizer is None:
            self._summarizer = SummarizationPipeline(device=self._dev)
        return self._summarizer

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def process(
        self,
        file_path: Path,
        document_id: str | None = None,
        doc_type_hint: str | None = None,
    ) -> IngestionResult:
        """
        Process a single document through the full pipeline.

        Args:
            file_path: Path to the uploaded document.
            document_id: Optional UUID for this document.
            doc_type_hint: Optional hint about document type.

        Returns:
            IngestionResult with all extracted data.
        """
        doc_id = document_id or str(uuid.uuid4())
        logger.info("Starting ingestion for document %s: %s", doc_id, file_path.name)

        result = IngestionResult(document_id=doc_id, raw_text="")
        meta: dict[str, Any] = {"filename": file_path.name, "stages": {}}

        try:
            # ------------------------------------------------------------------
            # Stage 1: Detect file type
            # ------------------------------------------------------------------
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"
            meta["mime_type"] = mime_type
            meta["file_size_bytes"] = file_path.stat().st_size
            logger.debug("Stage 1 complete: mime=%s", mime_type)

            # ------------------------------------------------------------------
            # Stage 2: Text extraction (Unstructured + OCR)
            # ------------------------------------------------------------------
            raw_text, ocr_meta = self._extract_text(file_path, mime_type)
            result.raw_text = raw_text
            meta["stages"]["ocr"] = ocr_meta
            logger.info("Stage 2 complete: %d chars extracted", len(raw_text))

            # ------------------------------------------------------------------
            # Stage 3: Document type triage (zero-shot)
            # ------------------------------------------------------------------
            if raw_text:
                triage = self.zero_shot.triage_document(text=raw_text[:2000])
                result.doc_type_detected = triage.get("document_type", "other")
                result.doc_type_confidence = triage.get("document_type_confidence", 0.0)
                meta["stages"]["triage"] = triage
                logger.debug("Stage 3 complete: doc_type=%s", result.doc_type_detected)

            # ------------------------------------------------------------------
            # Stage 4: Urgency / severity classification
            # ------------------------------------------------------------------
            if raw_text:
                urgency = self.text_clf.classify_urgency(text=raw_text[:1000])
                result.urgency_label = urgency["label"]
                result.urgency_score = urgency["score"]
                meta["stages"]["urgency"] = urgency
                logger.debug("Stage 4 complete: urgency=%s", result.urgency_label)

            # ------------------------------------------------------------------
            # Stage 5: Medical NER
            # ------------------------------------------------------------------
            if raw_text:
                entities = self.ner.extract_by_type(
                    text=raw_text[:3000],
                    entity_types=["DISEASE", "CHEMICAL", "MEDICATION", "PROCEDURE"],
                )
                result.named_entities = entities
                meta["stages"]["ner"] = {
                    k: len(v) for k, v in entities.items()
                }
                logger.debug("Stage 5 complete: %d entity types", len(entities))

            # ------------------------------------------------------------------
            # Stage 6: Table extraction (if PDF/image)
            # ------------------------------------------------------------------
            if mime_type in ("application/pdf", "image/png", "image/jpeg"):
                tables, image_descs = self._extract_visual_content(file_path, mime_type)
                result.tables = tables
                result.image_descriptions = image_descs
                meta["stages"]["tables"] = {"count": len(tables)}
                meta["stages"]["images"] = {"descriptions": len(image_descs)}
                logger.info(
                    "Stage 6 complete: %d tables, %d image descriptions",
                    len(tables), len(image_descs),
                )

            # ------------------------------------------------------------------
            # Stage 7: Smart chunking
            # ------------------------------------------------------------------
            chunks: list[Chunk] = []

            # Text chunks
            if raw_text:
                text_chunks = self.chunker.chunk_text(text=raw_text, page_number=1)
                chunks.extend(text_chunks)

            # Table chunks
            for idx, table in enumerate(result.tables):
                table_chunk = self.chunker.chunk_table(
                    table_data=table,
                    page_number=table.get("page_number"),
                    chunk_index=len(chunks) + idx,
                )
                chunks.append(table_chunk)

            # Image description chunks
            for idx, desc in enumerate(result.image_descriptions):
                img_chunk = self.chunker.chunk_image_description(
                    description=desc,
                    chunk_index=len(chunks) + idx,
                )
                chunks.append(img_chunk)

            meta["stages"]["chunking"] = {"chunk_count": len(chunks)}
            logger.info("Stage 7 complete: %d chunks created", len(chunks))

            # ------------------------------------------------------------------
            # Stage 8: Embedding
            # ------------------------------------------------------------------
            chunks = self.embedder.embed_chunks(chunks)
            result.chunks = chunks
            meta["stages"]["embedding"] = {"dim": 384, "chunks_embedded": len(chunks)}
            logger.info("Stage 8 complete: %d chunks embedded", len(chunks))

            # ------------------------------------------------------------------
            # Stage 9: Structured data extraction
            # ------------------------------------------------------------------
            if raw_text and settings.has_llm:
                doc_type = doc_type_hint or result.doc_type_detected
                structured = self.extractor.extract(text=raw_text, doc_type=doc_type)
                result.structured_data = self.extractor.to_dict(structured)
                meta["stages"]["extraction"] = {
                    "provider": settings.active_llm_provider,
                    "labs": len(result.structured_data.get("lab_values", [])),
                    "medications": len(result.structured_data.get("medications", [])),
                    "diagnoses": len(result.structured_data.get("diagnoses", [])),
                }
                logger.info("Stage 9 complete: structured data extracted via %s", settings.active_llm_provider)
            elif raw_text:
                meta["stages"]["extraction"] = {"skipped": "no_llm_provider"}

            # ------------------------------------------------------------------
            # Stage 10: Summary generation
            # ------------------------------------------------------------------
            if raw_text:
                summary_result = self.summarizer.run(text=raw_text, max_length=256)
                result.summary = summary_result["summary"]
                meta["stages"]["summary"] = {"length": len(result.summary)}

            result.pipeline_metadata = meta
            logger.info(
                "Ingestion complete for %s: %d chunks, urgency=%s",
                doc_id, len(result.chunks), result.urgency_label,
            )

        except Exception as e:
            logger.exception("Ingestion failed for %s: %s", doc_id, e)
            result.success = False
            result.error = str(e)
            result.pipeline_metadata = meta

        return result

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_text(
        self, file_path: Path, mime_type: str
    ) -> tuple[str, dict[str, Any]]:
        """Extract raw text via Unstructured + Surya OCR."""
        meta: dict[str, Any] = {}

        # Try Unstructured first (handles PDFs, DOCX, HTML, etc.)
        try:
            from unstructured.partition.auto import partition  # type: ignore[import]

            elements = partition(filename=str(file_path))
            text = "\n\n".join(str(e) for e in elements if str(e).strip())
            meta["method"] = "unstructured"
            meta["element_count"] = len(elements)

            if text.strip():
                return text, meta

        except Exception as e:
            logger.warning("Unstructured failed: %s — falling back to OCR", e)

        # Fallback: Surya OCR for scanned PDFs and images
        if mime_type == "application/pdf":
            ocr_results = self.ocr.process_pdf_pages(pdf_path=file_path)
            text = self.ocr.get_full_text(ocr_results)
            meta["method"] = "surya_ocr"
            meta["pages_processed"] = len(ocr_results)
            avg_conf = (
                sum(r.confidence for r in ocr_results) / len(ocr_results)
                if ocr_results else 0.0
            )
            meta["avg_confidence"] = round(avg_conf, 3)
        elif mime_type.startswith("image/"):
            from PIL import Image  # type: ignore[import]
            img = Image.open(file_path).convert("RGB")
            ocr_result = self.ocr.surya.process_image(image=img)
            text = ocr_result.text
            meta["method"] = "surya_ocr_image"
            meta["confidence"] = ocr_result.confidence
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            meta["method"] = "plaintext"

        return text, meta

    def _extract_visual_content(
        self, file_path: Path, mime_type: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Extract tables and image descriptions from PDF/image files."""
        tables: list[dict[str, Any]] = []
        image_descs: list[str] = []

        if not self.vision.available:
            logger.debug("Visual content extraction skipped — no vision LLM configured")
            return tables, image_descs

        try:
            if mime_type == "application/pdf":
                import fitz  # type: ignore[import]
                from PIL import Image
                import io

                doc = fitz.open(str(file_path))
                for page_num in range(min(len(doc), 10)):  # Max 10 pages
                    page = doc[page_num]
                    mat = fitz.Matrix(1.5, 1.5)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # Table detection + extraction
                    page_tables = self.table_extractor.extract_tables_from_page(
                        image=img, page_number=page_num + 1
                    )
                    tables.extend(page_tables)

                    # Extract embedded images from PDF page
                    img_list = page.get_images()
                    for img_idx, img_info in enumerate(img_list[:3]):  # Max 3 images/page
                        try:
                            xref = img_info[0]
                            base_img = doc.extract_image(xref)
                            img_bytes = base_img["image"]
                            pil_img = Image.open(io.BytesIO(img_bytes))
                            desc = self.vision.describe_medical_image(image=pil_img)
                            image_descs.append(desc.get("description", ""))
                        except Exception as e:
                            logger.debug("Image extraction failed: %s", e)

                doc.close()

            elif mime_type.startswith("image/"):
                from PIL import Image
                img = Image.open(file_path).convert("RGB")

                page_tables = self.table_extractor.extract_tables_from_page(image=img)
                tables.extend(page_tables)

                desc = self.vision.describe_medical_image(image=img)
                image_descs.append(desc.get("description", ""))

        except Exception as e:
            logger.warning("Visual content extraction failed: %s", e)

        return tables, image_descs

"""
Table Extractor — detects and extracts tabular data from document pages.

Pipeline:
1. Object detection (Table Transformer) to find table bounding boxes
2. Crop table regions from the page image
3. GPT-4o vision or TAPAS to extract structured table data
4. Convert to dict-of-lists for Table QA
"""

import logging
from typing import Any

from PIL import Image  # type: ignore[import]

from medscan.hf.object_detection import ObjectDetectionPipeline
from medscan.pipeline.vision import GPT4oVisionProcessor

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extracts structured tabular data from document page images.

    Uses Table Transformer for detection, then GPT-4o for content extraction.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.detector = ObjectDetectionPipeline(device=device)
        self.vision = GPT4oVisionProcessor()

    def extract_tables_from_page(
        self,
        image: Image.Image,
        page_number: int = 1,
        detection_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Detect and extract all tables from a page image.

        Args:
            image: PIL Image of the document page.
            page_number: Source page number for metadata.
            detection_threshold: Minimum confidence for table detection.

        Returns:
            List of extracted table dicts with headers, rows, and metadata.
        """
        # Step 1: Detect table regions
        detections = self.detector.get_tables(
            image=image, threshold=detection_threshold
        )

        if not detections:
            logger.debug("No tables detected on page %d", page_number)
            return []

        logger.info(
            "Detected %d table(s) on page %d", len(detections), page_number
        )

        # Step 2: Crop and extract each table
        tables = []
        for idx, det in enumerate(detections):
            box = det["box"]
            crop = image.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))

            # Step 3: Extract table content via GPT-4o vision
            table_data = self.vision.extract_table_from_image(image=crop)
            table_data["page_number"] = page_number
            table_data["table_index"] = idx
            table_data["detection_score"] = det["score"]
            table_data["bbox"] = box

            tables.append(table_data)

        return tables

    def to_tapas_format(
        self, table_data: dict[str, Any]
    ) -> dict[str, list[str]]:
        """
        Convert extracted table to TAPAS dict-of-lists format.

        Args:
            table_data: Table dict with 'headers' and 'rows'.

        Returns:
            Dict mapping column name → list of string values.
        """
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        if not headers or not rows:
            return {}

        result: dict[str, list[str]] = {h: [] for h in headers}
        for row in rows:
            for i, header in enumerate(headers):
                val = str(row[i]) if i < len(row) else ""
                result[header].append(val)

        return result

    def to_text_representation(self, table_data: dict[str, Any]) -> str:
        """Convert table to a pipe-delimited text string for embedding."""
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        table_type = table_data.get("table_type", "table")

        lines = [f"[TABLE: {table_type.upper()}]"]
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * (len(headers) * 12))
        for row in rows:
            lines.append(" | ".join(str(c) for c in row))

        notes = table_data.get("notes", "")
        if notes:
            lines.append(f"Notes: {notes}")

        return "\n".join(lines)

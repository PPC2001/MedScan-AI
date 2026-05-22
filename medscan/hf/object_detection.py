"""
HF Task: Object Detection
Used for: Detecting tables, signatures, stamps, barcodes, seals in scanned documents.
Model: microsoft/table-transformer-detection (specialized for table detection)
"""

from typing import Any

from medscan.hf.base import BaseHFPipeline


class ObjectDetectionPipeline(BaseHFPipeline):
    """
    Detect structural elements in scanned medical documents.

    Primary use: Table detection for lab reports.
    Also detects: signatures, stamps, barcodes, headers, footers.
    """

    task = "object-detection"
    default_model = "microsoft/table-transformer-detection"

    def run(
        self,
        image: Any,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Args:
            image: PIL Image or image path.
            threshold: Minimum confidence score to include a detection.

        Returns:
            List of dicts with 'label', 'score', and 'box' (xmin,ymin,xmax,ymax).
        """
        results = self.pipe(image, threshold=threshold)
        return [
            {
                "label": r["label"],
                "score": round(r["score"], 4),
                "box": {
                    "xmin": int(r["box"]["xmin"]),
                    "ymin": int(r["box"]["ymin"]),
                    "xmax": int(r["box"]["xmax"]),
                    "ymax": int(r["box"]["ymax"]),
                },
            }
            for r in results
        ]

    def get_tables(self, image: Any, threshold: float = 0.7) -> list[dict[str, Any]]:
        """Return only table detections."""
        detections = self.run(image=image, threshold=threshold)
        return [d for d in detections if "table" in d["label"].lower()]

    def crop_detected_regions(
        self, image: Any, detections: list[dict[str, Any]]
    ) -> list[Any]:
        """
        Crop PIL image to each detected bounding box.
        Returns list of cropped PIL Images.
        """
        from PIL import Image  # type: ignore[import]

        if not isinstance(image, Image.Image):
            image = Image.open(image)

        crops = []
        for det in detections:
            box = det["box"]
            crop = image.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
            crops.append(crop)
        return crops

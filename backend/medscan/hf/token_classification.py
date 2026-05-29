"""
HF Task: Token Classification (Named Entity Recognition)
Used for: Medical NER — extracting diseases, medications, procedures,
         anatomical locations, lab tests, clinical findings.
Model: d4data/biomedical-ner-all (trained on PubMed + clinical notes)
"""

from typing import Any

from medscan.hf.base import BaseHFPipeline


# Standard biomedical NER entity types
MEDICAL_ENTITY_TYPES = {
    "DISEASE",
    "CHEMICAL",
    "GENE",
    "SPECIES",
    "MUTATION",
    "CELL_LINE",
    "CELL_TYPE",
    "DNA",
    "RNA",
    "PROTEIN",
    # Clinical entities
    "MEDICATION",
    "PROCEDURE",
    "ANATOMY",
    "LAB_TEST",
    "VITAL_SIGN",
}


class TokenClassificationPipeline(BaseHFPipeline):
    """
    Biomedical NER for extracting clinical entities from text.

    Identifies: diseases, drugs/chemicals, procedures, anatomy,
    lab tests, vital signs, and more.
    """

    task = "token-classification"
    default_model = "d4data/biomedical-ner-all"

    def run(
        self,
        text: str,
        aggregation_strategy: str = "simple",
    ) -> list[dict[str, Any]]:
        """
        Args:
            text: Clinical text to extract entities from.
            aggregation_strategy: 'none' | 'simple' | 'first' | 'average' | 'max'

        Returns:
            List of entity dicts with 'entity_group', 'word', 'score', 'start', 'end'.
        """
        results = self.pipe(text, aggregation_strategy=aggregation_strategy)
        return [
            {
                "entity_type": r["entity_group"],
                "text": r["word"],
                "score": round(r["score"], 4),
                "start": r["start"],
                "end": r["end"],
            }
            for r in results
        ]

    def extract_by_type(
        self, text: str, entity_types: list[str]
    ) -> dict[str, list[str]]:
        """
        Extract specific entity types.

        Returns:
            Dict mapping entity_type → list of extracted strings.
        """
        entities = self.run(text=text)
        result: dict[str, list[str]] = {etype: [] for etype in entity_types}
        for ent in entities:
            if ent["entity_type"] in entity_types:
                result[ent["entity_type"]].append(ent["text"])
        return result

    def extract_medications(self, text: str) -> list[str]:
        """Shortcut: extract medication names."""
        entities = self.run(text=text)
        return [e["text"] for e in entities if e["entity_type"] in ("CHEMICAL", "MEDICATION")]

    def extract_diseases(self, text: str) -> list[str]:
        """Shortcut: extract disease names."""
        entities = self.run(text=text)
        return [e["text"] for e in entities if e["entity_type"] == "DISEASE"]

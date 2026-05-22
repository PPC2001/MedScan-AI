"""
HF Task: Table Question Answering
Used for: Querying extracted lab result tables, medication lists, vital sign grids.
Model: google/tapas-large-finetuned-wtq (TAPAS — table parsing)
"""

from typing import Any

from medscan.hf.base import BaseHFPipeline


class TableQAPipeline(BaseHFPipeline):
    """
    Answer natural language questions against structured tables.

    Tables should be provided as a dict of column→list format (Pandas-style).
    Useful for: "What was the highest creatinine value?",
                "Which medications have dose > 500mg?"
    """

    task = "table-question-answering"
    default_model = "google/tapas-large-finetuned-wtq"

    def run(
        self,
        table: dict[str, list[str]],
        query: str,
    ) -> dict[str, Any]:
        """
        Args:
            table: Dict mapping column names → list of string cell values.
                   Example: {"Test": ["HbA1c", "Glucose"], "Value": ["7.2%", "126 mg/dL"]}
            query: Natural language question about the table.

        Returns:
            dict with 'answer', 'coordinates', 'cells', 'aggregator'.
        """
        result = self.pipe(table=table, query=query)
        return {
            "answer": result.get("answer", ""),
            "cells": result.get("cells", []),
            "aggregator": result.get("aggregator", "NONE"),
            "coordinates": result.get("coordinates", []),
        }

    def query_lab_table(
        self,
        lab_values: list[dict[str, str]],
        question: str,
    ) -> str:
        """
        Convenience wrapper for querying a list of lab value dicts.

        Args:
            lab_values: List of dicts with keys like 'test_name', 'value', 'unit', 'flag'.
            question: Natural language question.

        Returns:
            Answer string.
        """
        if not lab_values:
            return "No lab values available."

        # Transpose list-of-dicts to dict-of-lists
        keys = list(lab_values[0].keys())
        table = {k: [str(row.get(k, "")) for row in lab_values] for k in keys}

        result = self.run(table=table, query=question)
        return result["answer"]

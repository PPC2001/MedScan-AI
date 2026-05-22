"""
RAGAS Evaluation Harness — measure answer quality across key metrics.

Metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved chunks relevant?
- Context Recall: Are all relevant chunks retrieved?
"""

import logging
from datetime import datetime
from typing import Any

from medscan.models.schemas import EvaluationReport, EvaluationSample

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    RAGAS-based evaluator for the MedScan AI RAG pipeline.

    Requires OpenAI API key (RAGAS uses LLM as judge).
    """

    def __init__(self) -> None:
        self._initialized = False

    def _init(self) -> None:
        """Lazy import RAGAS to avoid startup cost."""
        if not self._initialized:
            try:
                from ragas import evaluate  # type: ignore[import]
                from ragas.metrics import (  # type: ignore[import]
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )
                self._evaluate = evaluate
                self._metrics = [
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ]
                self._initialized = True
            except ImportError as e:
                raise ImportError(f"RAGAS not installed: {e}") from e

    def evaluate(
        self,
        samples: list[EvaluationSample],
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """
        Evaluate a list of QA samples using RAGAS metrics.

        Args:
            samples: List of EvaluationSample with question, answer, ground_truth, contexts.
            metadata: Optional metadata to include in the report.

        Returns:
            EvaluationReport with metric scores.
        """
        self._init()

        if not samples:
            raise ValueError("No samples provided for evaluation")

        # Build RAGAS dataset
        from datasets import Dataset  # type: ignore[import]

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
            "contexts": [s.contexts for s in samples],
        }
        dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluation on %d samples...", len(samples))

        try:
            result = self._evaluate(dataset, metrics=self._metrics)
            scores = result.to_pandas()

            avg_scores = {
                "faithfulness": float(scores["faithfulness"].mean()),
                "answer_relevancy": float(scores["answer_relevancy"].mean()),
                "context_precision": float(scores["context_precision"].mean()),
                "context_recall": float(scores["context_recall"].mean()),
            }

            logger.info("RAGAS results: %s", avg_scores)

            return EvaluationReport(
                faithfulness=avg_scores["faithfulness"],
                answer_relevancy=avg_scores["answer_relevancy"],
                context_precision=avg_scores["context_precision"],
                context_recall=avg_scores["context_recall"],
                samples_evaluated=len(samples),
                timestamp=datetime.utcnow(),
                metadata=metadata or {},
            )

        except Exception as e:
            logger.exception("RAGAS evaluation failed: %s", e)
            raise

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> dict[str, float]:
        """Quick evaluation of a single QA pair."""
        sample = EvaluationSample(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
        )
        report = self.evaluate(samples=[sample])
        return {
            "faithfulness": report.faithfulness,
            "answer_relevancy": report.answer_relevancy,
            "context_precision": report.context_precision,
            "context_recall": report.context_recall,
        }

"""
RAGAS Evaluation Runner — benchmarks the full RAG pipeline.

Run: uv run python scripts/run_evaluation.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from medscan.evaluation.evaluator import RAGASEvaluator
from medscan.models.schemas import EvaluationSample

# Sample evaluation dataset (extend with real ground-truth pairs)
EVAL_SAMPLES = [
    EvaluationSample(
        question="What is the patient's HbA1c level?",
        answer="The patient's HbA1c is 8.2%, which is critically elevated above the normal range of <5.7%.",
        ground_truth="HbA1c is 8.2%, indicating poorly controlled diabetes.",
        contexts=[
            "HbA1c 8.2% (Reference: <5.7%) - CRITICAL H",
            "Elevated HbA1c consistent with poorly controlled Type 2 Diabetes Mellitus",
        ],
    ),
    EvaluationSample(
        question="What medications is the patient taking?",
        answer="The patient is on Metformin 1000mg BID, Lisinopril 10mg daily, Atorvastatin 40mg nightly, and Aspirin 81mg daily.",
        ground_truth="Metformin, Lisinopril, Atorvastatin, Aspirin",
        contexts=[
            "MEDICATIONS:\n1. Metformin 1000mg PO BID\n2. Lisinopril 10mg PO daily\n3. Atorvastatin 40mg PO nightly\n4. Aspirin 81mg PO daily",
        ],
    ),
    EvaluationSample(
        question="What are the patient's vital signs?",
        answer="BP 148/92 mmHg (elevated), HR 82 bpm, Temp 98.4°F, SpO2 98% on room air, Weight 189 lbs, BMI 32.4.",
        ground_truth="Blood Pressure 148/92, HR 82, SpO2 98%, BMI 32.4",
        contexts=[
            "VITAL SIGNS:\nBlood Pressure: 148/92 mmHg\nHeart Rate: 82 bpm\nTemperature: 98.4°F\nSpO2: 98% on room air\nWeight: 189 lbs\nBMI: 32.4",
        ],
    ),
]


async def run_evaluation() -> None:
    """Run RAGAS evaluation and print report."""
    print("📊 MedScan AI — RAGAS Evaluation Runner")
    print("=" * 50)

    evaluator = RAGASEvaluator()

    try:
        report = evaluator.evaluate(
            samples=EVAL_SAMPLES,
            metadata={"pipeline_version": "0.1.0", "model": "claude-3-5-sonnet"},
        )

        print(f"\n✅ Evaluation complete ({report.samples_evaluated} samples)")
        print(f"\n{'Metric':<30} {'Score':>10}")
        print("-" * 42)
        print(f"{'Faithfulness':<30} {report.faithfulness:>10.3f}")
        print(f"{'Answer Relevancy':<30} {report.answer_relevancy:>10.3f}")
        print(f"{'Context Precision':<30} {report.context_precision:>10.3f}")
        print(f"{'Context Recall':<30} {report.context_recall:>10.3f}")

        avg = (
            report.faithfulness
            + report.answer_relevancy
            + report.context_precision
            + report.context_recall
        ) / 4
        print("-" * 42)
        print(f"{'Overall Average':<30} {avg:>10.3f}")

        # Thresholds
        print("\n📋 Quality Gates:")
        gates = {
            "Faithfulness ≥ 0.8": report.faithfulness >= 0.8,
            "Answer Relevancy ≥ 0.75": report.answer_relevancy >= 0.75,
            "Context Precision ≥ 0.7": report.context_precision >= 0.7,
            "Context Recall ≥ 0.7": report.context_recall >= 0.7,
        }
        for gate, passed in gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}  {gate}")

    except ImportError:
        print("⚠️  RAGAS not available. Install with: uv add ragas")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())

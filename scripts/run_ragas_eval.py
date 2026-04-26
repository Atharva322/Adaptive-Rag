"""
Run RAGAS evaluation from a local JSON file.

Usage:
    python scripts/run_ragas_eval.py --dataset eval_dataset.json
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ragas_evaluator import EvalSample, evaluate_with_ragas


def _load_samples(dataset_path: Path) -> list[EvalSample]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("dataset")
    else:
        rows = None

    if not isinstance(rows, list):
        raise ValueError("Dataset must be a list or contain a top-level 'dataset' list.")

    samples = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset item at index {idx} is not an object.")
        samples.append(
            EvalSample(
                question=row["question"],
                ground_truth=row["ground_truth"],
                metadata_filter=row.get("metadata_filter"),
            )
        )
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptive RAG with RAGAS.")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric names (faithfulness answer_relevancy context_precision context_recall).",
    )
    parser.add_argument(
        "--no-per-sample",
        action="store_true",
        help="Skip per-sample scores in output.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    samples = _load_samples(dataset_path)
    result = evaluate_with_ragas(
        samples=samples,
        metrics=args.metrics,
        include_per_sample=not args.no_per_sample,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

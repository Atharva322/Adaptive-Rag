"""
RAGAS evaluation helpers for the Adaptive RAG graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings

from src.llms.openai import llm
from src.rag.graph_builder import builder


@dataclass
class EvalSample:
    question: str
    ground_truth: str | None = None
    answer: str | None = None
    contexts: list[str] | None = None
    relevant_contexts: list[str] | None = None
    metadata_filter: dict | None = None


DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
]


def _load_ragas_dependencies():
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import context_precision, context_recall, faithfulness
        try:
            from ragas.metrics import answer_relevancy
        except ImportError:
            from ragas.metrics import response_relevancy as answer_relevancy
        try:
            from ragas.metrics import answer_correctness
        except ImportError:
            answer_correctness = None
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS dependencies are not installed. Run: pip install -r requirements.txt"
        ) from exc

    metric_registry = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "response_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    if answer_correctness is not None:
        metric_registry["answer_correctness"] = answer_correctness
    return Dataset, evaluate, metric_registry, LangchainLLMWrapper, LangchainEmbeddingsWrapper


def _resolve_metrics(metric_names: list[str] | None, metric_registry: dict[str, Any]):
    requested = metric_names or DEFAULT_METRICS
    resolved = []
    invalid = []
    for name in requested:
        metric = metric_registry.get(name.strip().lower())
        if metric is None:
            invalid.append(name)
            continue
        resolved.append(metric)

    if invalid:
        raise ValueError(
            f"Unsupported RAGAS metrics: {invalid}. "
            f"Supported: {sorted(set(metric_registry.keys()))}"
        )

    unique = []
    seen = set()
    for metric in resolved:
        if metric.name not in seen:
            seen.add(metric.name)
            unique.append(metric)
    return unique


def _run_single_query(question: str, metadata_filter: dict | None):
    state = {
        "messages": [HumanMessage(content=question)],
        "latest_query": question,
        "rewrite_count": 0,
        "metadata_filter": metadata_filter,
    }
    result = builder.invoke(state, config={"recursion_limit": 50})

    answer = result.get("final_answer")
    if not answer:
        messages = result.get("messages") or []
        if messages:
            last = messages[-1]
            answer = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")
    answer = answer or ""

    contexts = result.get("retrieved_contexts") or []
    normalized_contexts = []
    for context in contexts:
        text = str(context).strip()
        if text:
            normalized_contexts.append(text)

    return answer, normalized_contexts


def _normalize_text(text: str) -> str:
    return " ".join(re.sub(r"\s+", " ", text.lower()).strip().split())


def _is_relevant(retrieved: str, relevant: str) -> bool:
    retrieved_n = _normalize_text(retrieved)
    relevant_n = _normalize_text(relevant)
    if not retrieved_n or not relevant_n:
        return False
    return relevant_n in retrieved_n or retrieved_n in relevant_n


def _compute_retrieval_metrics(
    retrieved_contexts: list[str], relevant_contexts: list[str]
) -> dict[str, float]:
    cleaned_retrieved = [ctx for ctx in (retrieved_contexts or []) if str(ctx).strip()]
    cleaned_relevant = [ctx for ctx in (relevant_contexts or []) if str(ctx).strip()]
    if not cleaned_relevant:
        return {}

    relevant_total = len(cleaned_relevant)
    first_rel_rank = None
    relevant_index_sets: list[set[int]] = []
    for idx, retrieved in enumerate(cleaned_retrieved, start=1):
        matched_indexes = {
            rel_idx
            for rel_idx, relevant in enumerate(cleaned_relevant)
            if _is_relevant(retrieved, relevant)
        }
        relevant_index_sets.append(matched_indexes)
        if matched_indexes and first_rel_rank is None:
            first_rel_rank = idx

    def _recall_at_k(k: int) -> float:
        seen: set[int] = set()
        for matches in relevant_index_sets[:k]:
            seen.update(matches)
        return len(seen) / relevant_total

    metrics = {
        "recall_at_3": _recall_at_k(3),
        "recall_at_5": _recall_at_k(5),
        "recall_at_10": _recall_at_k(10),
        "mrr": (1.0 / first_rel_rank) if first_rel_rank else 0.0,
    }
    return metrics


def evaluate_with_ragas(
    samples: list[EvalSample],
    metrics: list[str] | None = None,
    include_per_sample: bool = True,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Evaluation dataset is empty.")

    (
        Dataset,
        evaluate,
        metric_registry,
        LangchainLLMWrapper,
        LangchainEmbeddingsWrapper,
    ) = _load_ragas_dependencies()
    rows = []
    retrieval_scores = []

    has_ground_truth = True
    has_contexts = True
    for sample in samples:
        if sample.answer is not None:
            answer = str(sample.answer).strip()
            sample_contexts = sample.contexts or []
            contexts = [str(context).strip() for context in sample_contexts if str(context).strip()]
        else:
            answer, contexts = _run_single_query(sample.question, sample.metadata_filter)

        ground_truth = (sample.ground_truth or "").strip()
        if not ground_truth:
            has_ground_truth = False
        if not contexts:
            has_contexts = False

        row = {
            "question": sample.question,
            "answer": answer,
            "contexts": contexts,
        }
        if ground_truth:
            row["ground_truth"] = ground_truth

        rows.append(row)
        retrieval_scores.append(
            _compute_retrieval_metrics(
                retrieved_contexts=contexts,
                relevant_contexts=sample.relevant_contexts or [],
            )
        )

    if metrics:
        selected_metrics = _resolve_metrics(metrics, metric_registry)
    else:
        dynamic_defaults = ["answer_relevancy"]
        if has_contexts:
            dynamic_defaults.append("faithfulness")
        if has_contexts and has_ground_truth:
            dynamic_defaults.extend(["context_precision", "context_recall"])
        selected_metrics = _resolve_metrics(dynamic_defaults, metric_registry)

    dataset = Dataset.from_list(rows)
    score = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings()),
        # FastAPI/Uvicorn (uvloop) cannot run nested event loops.
        # This should remain False in server environments.
        allow_nest_asyncio=False,
    )

    score_df = score.to_pandas()
    metric_columns = [metric.name for metric in selected_metrics]
    aggregate = {
        col: float(score_df[col].mean()) for col in metric_columns if col in score_df.columns
    }

    response: dict[str, Any] = {
        "num_samples": len(rows),
        "metrics": metric_columns,
        "aggregate_scores": aggregate,
    }

    retrieval_metric_names = ["recall_at_3", "recall_at_5", "recall_at_10", "mrr"]
    available_retrieval_metric_names = [
        name for name in retrieval_metric_names if any(name in score for score in retrieval_scores)
    ]
    if available_retrieval_metric_names:
        retrieval_aggregate = {}
        for metric_name in available_retrieval_metric_names:
            values = [
                score[metric_name]
                for score in retrieval_scores
                if metric_name in score
            ]
            if values:
                retrieval_aggregate[metric_name] = float(sum(values) / len(values))
        response["aggregate_scores"].update(retrieval_aggregate)
        response["metrics"] = metric_columns + available_retrieval_metric_names

    if include_per_sample:
        base_columns = ["question", "answer", "contexts", "ground_truth"]
        available = [col for col in base_columns + metric_columns if col in score_df.columns]
        per_sample_rows = score_df[available].to_dict(orient="records")
        for idx, row in enumerate(per_sample_rows):
            if idx < len(retrieval_scores):
                row.update(retrieval_scores[idx])
        response["per_sample_scores"] = per_sample_rows

    return response

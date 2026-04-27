"""
RAGAS evaluation helpers for the Adaptive RAG graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    metadata_filter: dict | None = None


DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
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

    if include_per_sample:
        base_columns = ["question", "answer", "contexts", "ground_truth"]
        available = [col for col in base_columns + metric_columns if col in score_df.columns]
        response["per_sample_scores"] = score_df[available].to_dict(orient="records")

    return response

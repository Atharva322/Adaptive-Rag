"""
Request models for RAGAS evaluation.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


RagasMetricName = Literal[
    "faithfulness",
    "answer_relevancy",
    "response_relevancy",
    "context_precision",
    "context_recall",
]


class RagasSample(BaseModel):
    question: str = Field(..., description="User question to evaluate.")
    ground_truth: str = Field(..., description="Expected reference answer.")
    metadata_filter: Optional[dict] = Field(
        default=None,
        description="Optional retrieval metadata filter for this sample.",
    )


class RagasEvalRequest(BaseModel):
    dataset: list[RagasSample]
    metrics: Optional[list[RagasMetricName]] = Field(
        default=None,
        description="Optional metric names. Defaults to core RAGAS metrics.",
        examples=[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]],
    )
    include_per_sample: bool = Field(
        default=True,
        description="Include per-question metric rows in the response.",
    )

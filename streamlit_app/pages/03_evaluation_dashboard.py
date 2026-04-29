"""
Benchmark evaluation dashboard for RAGAS + retrieval metrics.
"""

import json
import os
import uuid
import sys

import pandas as pd
import streamlit as st

sys.path.append("..")
from utils.api_client import evaluate_ragas_dataset


st.set_page_config(
    page_title="Evaluation Dashboard - Adaptive RAG",
    page_icon="📊",
    layout="wide",
)

# Check authentication
if "jwt_token" not in st.session_state or st.session_state.jwt_token is None:
    _disable_auth = os.getenv("DISABLE_AUTH", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    if _disable_auth:
        st.session_state.jwt_token = "local_dev_token"
        st.session_state.username = st.session_state.get("username") or "local_dev"
    else:
        st.warning("Please log in first")
        st.stop()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

st.title("Evaluation Dashboard")
st.markdown("Run benchmark evaluation (50-60 samples) and inspect quality metrics.")

if "eval_result" not in st.session_state:
    st.session_state["eval_result"] = None

default_metrics = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
]

with st.sidebar:
    st.header("Run Config")
    selected_metrics = st.multiselect(
        "RAGAS metrics",
        options=default_metrics,
        default=default_metrics,
    )
    include_per_sample = st.checkbox("Include per-sample rows", value=True)
    st.caption("Retrieval metrics (recall_at_3/5/10, mrr) are auto-computed when relevant_contexts are provided.")

uploaded_json = st.file_uploader(
    "Upload benchmark dataset JSON",
    type=["json"],
    help="Use list format or {'dataset': [...]} format.",
)

with st.expander("Dataset format example", expanded=False):
    st.code(
        """[
  {
    "question": "What is ...?",
    "ground_truth": "Expected answer",
    "relevant_contexts": ["Ground-truth relevant chunk A", "Ground-truth relevant chunk B"],
    "metadata_filter": {"document_name": "my_doc.pdf"}
  }
]""",
        language="json",
    )

if st.button("Run Benchmark Evaluation", use_container_width=True):
    if not uploaded_json:
        st.error("Upload a dataset JSON file first.")
    else:
        try:
            payload = json.loads(uploaded_json.getvalue().decode("utf-8"))
            if isinstance(payload, dict):
                dataset = payload.get("dataset")
            else:
                dataset = payload

            if not isinstance(dataset, list) or not dataset:
                st.error("Dataset must be a non-empty list or contain a top-level 'dataset' list.")
            else:
                with st.spinner("Running evaluation..."):
                    result = evaluate_ragas_dataset(
                        dataset=dataset,
                        include_per_sample=include_per_sample,
                        metrics=selected_metrics,
                    )
                if result.get("status") == "success":
                    st.session_state["eval_result"] = result["data"]
                    st.success("Evaluation completed.")
                else:
                    st.error(f"Evaluation failed: {result.get('message', 'unknown error')}")
        except json.JSONDecodeError:
            st.error("Invalid JSON file.")
        except Exception as exc:
            st.error(f"Failed to run evaluation: {exc}")

eval_result = st.session_state.get("eval_result")
if eval_result:
    st.divider()
    st.subheader("Aggregate Metrics")

    aggregate_scores = eval_result.get("aggregate_scores", {})
    metric_items = sorted(aggregate_scores.items(), key=lambda x: x[0])
    if metric_items:
        cols = st.columns(min(4, len(metric_items)))
        for idx, (metric_name, metric_value) in enumerate(metric_items):
            with cols[idx % len(cols)]:
                st.metric(metric_name, f"{metric_value:.3f}")
    else:
        st.info("No aggregate scores available.")

    st.markdown("---")
    st.subheader("Metric Comparison")
    metric_df = pd.DataFrame(metric_items, columns=["metric", "score"])
    if not metric_df.empty:
        st.bar_chart(metric_df.set_index("metric"))

    per_sample = eval_result.get("per_sample_scores", [])
    if per_sample:
        st.markdown("---")
        st.subheader("Per-Sample Scores")
        per_sample_df = pd.DataFrame(per_sample)
        st.dataframe(per_sample_df, use_container_width=True)

        csv_data = per_sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Per-Sample CSV",
            data=csv_data,
            file_name="rag_eval_per_sample_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )

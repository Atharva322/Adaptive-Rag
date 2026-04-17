from __future__ import annotations
import logging
import numpy as np
from typing import List, Optional
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

_bm25_retriever: Optional[BM25Retriever] = None


def mmr_rerank(
    query: str,
    candidates: List[Document],
    k: int = 4,
    lambda_mult: float = 0.5,
    embeddings=None,
) -> List[Document]:
    if len(candidates) <= k:
        return candidates
    q_emb = np.array(embeddings.embed_query(query))
    d_embs = np.array(embeddings.embed_documents([d.page_content for d in candidates]))
    selected, remaining = [], list(range(len(candidates)))
    for _ in range(k):
        if not remaining:
            break
        rel = np.dot(d_embs[remaining], q_emb) / (
            np.linalg.norm(d_embs[remaining], axis=1) * np.linalg.norm(q_emb) + 1e-9
        )
        if not selected:
            pick = remaining[int(np.argmax(rel))]
        else:
            red = np.max(
                np.dot(d_embs[remaining], d_embs[selected].T) /
                (np.linalg.norm(d_embs[remaining], axis=1, keepdims=True)
                 * np.linalg.norm(d_embs[selected], axis=1) + 1e-9),
                axis=1,
            )
            pick = remaining[int(np.argmax(lambda_mult * rel - (1 - lambda_mult) * red))]
        selected.append(pick)
        remaining.remove(pick)
    return [candidates[i] for i in selected]


def build_bm25_retriever(documents: List[Document], k: int = 4) -> BM25Retriever:
    global _bm25_retriever
    _bm25_retriever = BM25Retriever.from_documents(documents, k=k)
    logger.info(f"BM25 index built with {len(documents)} documents")
    return _bm25_retriever


def get_bm25_retriever() -> Optional[BM25Retriever]:
    return _bm25_retriever


def clear_bm25_retriever() -> None:
    global _bm25_retriever
    _bm25_retriever = None


def get_hybrid_retriever(
    vector_retriever: BaseRetriever,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> EnsembleRetriever:
    bm25 = get_bm25_retriever()
    if bm25 is None:
        logger.warning("BM25 not initialized, falling back to vector-only")
        return vector_retriever  # type: ignore
    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )

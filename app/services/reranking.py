"""
Reranking service - reranks retrieval candidates to improve relevance ordering.
"""

import time
from typing import Any

_reranker_model: Any | None = None


def configure_reranker(*, reranker_model: Any | None = None) -> None:
    """Register shared runtime dependencies for retrieval."""
    global _reranker_model
    _reranker_model = reranker_model

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int,
) -> tuple[list[dict], float]:
    start = time.perf_counter()

    if not candidates:
        return []
    
    # Create query-document pairs for cross-encoder
    # We use the original question text, not the embedding text
    pairs = [(query, cand["text"]) for cand in candidates]
    
    # Get cross-encoder scores
    cross_scores = _reranker_model.predict(pairs)
    
    # Add scores to candidates
    for candidate, score in zip(candidates, cross_scores):
        candidate["rerank_score"] = float(score)
    
    # Sort by rerank score (higher is better)
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    
    # reranked: list[dict] = candidates[:top_k]  # Passthrough stub — replace this

    elapsed_ms = (time.perf_counter() - start) * 1000

    return reranked[:top_k], elapsed_ms

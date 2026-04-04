"""
Reranking service - reranks retrieval candidates to improve relevance ordering.
"""

import time


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int,
) -> tuple[list[dict], float]:
    start = time.perf_counter()

    # TODO: Implement reranking logic
    reranked: list[dict] = candidates[:top_k]  # Passthrough stub — replace this

    elapsed_ms = (time.perf_counter() - start) * 1000

    return reranked, elapsed_ms

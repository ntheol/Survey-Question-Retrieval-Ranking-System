"""
Retrieval service - embeds queries and searches the vector store.
"""

import time

from models.schemas import SearchFilters


def retrieve(
    query: str,
    top_k: int,
    filters: SearchFilters | None = None,
    retrieval_multiplier: int | None = None,
) -> tuple[list[dict], float]:
    start = time.perf_counter()

    # TODO: Implement semantic retrieval
    results: list[dict] = []

    elapsed_ms = (time.perf_counter() - start) * 1000

    return results, elapsed_ms

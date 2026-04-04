"""
Retrieval service - embeds queries and searches the vector store.
"""

from __future__ import annotations

import time
from typing import Any

from app.models.schemas import SearchFilters
from app.services.lib import EMBEDDING_MODEL


_client: Any | None = None
_collection: Any | None = None


def configure_retrieval(*, client: Any, collection: Any | None = None) -> None:
    """Register shared runtime dependencies for retrieval."""
    global _client, _collection
    _client = client
    _collection = collection


def _build_where_filter(filters: SearchFilters | None) -> dict[str, Any] | None:
    if filters is None:
        return None

    where_conditions: list[dict[str, Any]] = []

    if filters.category:
        where_conditions.append({"category": {"$eq": filters.category}})

    if filters.demographic_focus:
        where_conditions.append({"demographic_focus": {"$eq": filters.demographic_focus}})

    if len(where_conditions) == 1:
        return where_conditions[0]

    if len(where_conditions) > 1:
        return {"$and": where_conditions}

    return None


def retrieve(
    query: str,
    top_k: int,
    filters: SearchFilters | None = None,
    retrieval_multiplier: int | None = None,
) -> tuple[list[dict], float]:
    start = time.perf_counter()

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if _client is None or _collection is None:
        raise RuntimeError("Retrieval service is not initialized")

    n_results = top_k * retrieval_multiplier if retrieval_multiplier else top_k

    query_embedding = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    ).data[0].embedding

    where_filter = _build_where_filter(filters)

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["metadatas", "distances"],
    )

    formatted_results: list[dict] = []
    for question_id, metadata, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity_score = 1 / (1 + distance)

        formatted_results.append(
            {
                "question_id": question_id,
                "text": metadata["text"],
                "category": metadata["category"],
                "subcategory": metadata["subcategory"],
                "demographic_focus": metadata["demographic_focus"],
                "answer_type": metadata["answer_type"],
                "tags": metadata["tags"].split(","),
                "retrieval_score": similarity_score,
                "distance": distance,
            }
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return formatted_results, elapsed_ms

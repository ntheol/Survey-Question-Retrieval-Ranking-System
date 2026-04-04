"""API route definitions for the Survey Question Retrieval & Ranking System."""

from fastapi import APIRouter
from models.schemas import (
    HealthResponse,
    ModelInfo,
    SearchRequest,
    SearchResponse,
)

router = APIRouter()

##### TODO (Optional): Feel free to add more fields to the response models as you see fit

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint. Returns service status and index info."""
    return HealthResponse(
        status="healthy",
        version="0.0.0",
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Search for relevant profiling questions given a survey brief.

    Runs the retrieval pipeline, optionally followed by reranking.
    """

    return SearchResponse(
        query=request.query,
        results=[],
        retrieval_time_ms=0.0,
        rerank_time_ms=None,
        model_info=ModelInfo(embedding_model="", reranker_model=""),
    )

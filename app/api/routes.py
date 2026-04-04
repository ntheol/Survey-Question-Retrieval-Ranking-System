"""API route definitions for the Survey Question Retrieval & Ranking System."""

from fastapi import APIRouter, HTTPException, Request
from app.models.schemas import (
    HealthResponse,
    ModelInfo,
    QuestionResult,
    SearchRequest,
    SearchResponse,
)

from app.services.lib import EMBEDDING_MODEL
from app.services.retrieval import retrieve
from app.services.reranking import rerank
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

router = APIRouter()

##### TODO (Optional): Feel free to add more fields to the response models as you see fit

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint. Returns service status and index info."""
    has_client = hasattr(request.app.state, "client")
    has_collection = hasattr(request.app.state, "collection")

    if not has_client or not has_collection:
        raise HTTPException(
            status_code=503,
            detail="Service dependencies are not initialized",
        )

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        service="survey-question-retrieval-ranking-api",
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Runs the retrieval pipeline, optionally followed by reranking.
    """

    retrieved_results, retrieval_time_ms = retrieve(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
    )

    if rerank:
        reranked_results, rerank_time_ms = rerank(
            query=request.query,
            candidates=retrieved_results,
            top_k=request.top_k,
        )
    else:
        rerank_time_ms = None


    response_results = [
        QuestionResult(
            question_id=result["question_id"],
            text=result["text"],
            category=result["category"],
            relevance_score=result["retrieval_score"],
            rank=rank,
        )
        for rank, result in enumerate(retrieved_results if not request.rerank else reranked_results, start=1)
    ]

    return SearchResponse(
        query=request.query,
        results=response_results,
        total_results=len(response_results),
        rerank_applied=request.rerank,
        retrieval_time_ms=retrieval_time_ms,
        rerank_time_ms=rerank_time_ms,
        model_info=ModelInfo(
            embedding_model=EMBEDDING_MODEL,
            reranker_model= RERANK_MODEL if request.rerank else None,
        ),
    )

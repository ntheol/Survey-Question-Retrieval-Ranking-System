"""Pydantic models for API request/response contracts."""

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Optional filters for narrowing search results."""

    category: str | None = Field(
        default=None,
        description="Filter by question category",
        examples=["Financial Behavior"],
    )
    demographic_focus: str | None = Field(
        default=None,
        description="Filter by demographic focus",
        examples=["millennials"],
    )


class SearchRequest(BaseModel):
    """POST /search request body."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language survey brief describing target audience or research goal",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    filters: SearchFilters | None = Field(
        default=None,
        description="Optional metadata filters",
    )
    rerank: bool = Field(
        default=True,
        description="Whether to apply reranking stage",
    )


class QuestionResult(BaseModel):
    """A single ranked question in the search response."""

    question_id: str = Field(..., description="Question identifier", examples=["q_012"])
    text: str = Field(..., description="Question text")
    category: str = Field(..., description="Question category")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final relevance score after retrieval/reranking",
    )
    rank: int = Field(..., ge=1, description="1-based rank position")


class ModelInfo(BaseModel):
    """Metadata about models used in the pipeline."""

    embedding_model: str = Field(..., description="Embedding model identifier")
    reranker_model: str | None = Field(
        default=None,
        description="Reranker model identifier, null if reranking disabled",
    )


class SearchResponse(BaseModel):
    """POST /search response body."""

    query: str = Field(..., description="Echo of the input query")
    results: list[QuestionResult] = Field(
        default_factory=list, description="Ranked list of matching questions"
    )
    retrieval_time_ms: float = Field(
        ..., ge=0, description="Time spent on retrieval stage in milliseconds"
    )
    rerank_time_ms: float | None = Field(
        default=None,
        ge=0,
        description="Time spent on reranking in milliseconds, null if reranking disabled",
    )
    model_info: ModelInfo


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str = Field(default="healthy")
    version: str = Field(..., description="Service version")

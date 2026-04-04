"""FastAPI application entry point."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
import uvicorn
from fastapi import FastAPI
from openai import OpenAI
from sentence_transformers import CrossEncoder

from app.api import routes
from app.services.lib import (
    build_embedding_text,
    create_chroma_collection,
    embed_texts,
    prepare_data_for_chroma,
)
from app.services.retrieval import configure_retrieval
from app.services.reranking import configure_reranker

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "profiling_questions.json"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "survey_questions"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 


def _load_or_create_collection(client: OpenAI, chroma_client: chromadb.PersistentClient):
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"Loaded existing ChromaDB collection: {COLLECTION_NAME}")
        return collection
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found. Building it from source data.")

    with DATA_PATH.open() as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions from {DATA_PATH}")

    document_texts = [build_embedding_text(q) for q in questions]
    document_embeddings = embed_texts(client, document_texts)

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Survey profiling questions with semantic search"},
    )

    ids, documents, embeddings, metadatas = prepare_data_for_chroma(
        questions,
        document_texts,
        document_embeddings,
    )
    create_chroma_collection(collection, ids, documents, embeddings, metadatas)
    print(f"Created ChromaDB collection: {COLLECTION_NAME}")
    return collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up")

    app.state.client = OpenAI()
    app.state.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    app.state.reranker_model = CrossEncoder(RERANK_MODEL)
    app.state.collection = _load_or_create_collection(
        client=app.state.client,
        chroma_client=app.state.chroma_client,
    )

    configure_retrieval(
        client=app.state.client,
        collection=app.state.collection,
    )

    configure_reranker(
        reranker_model=app.state.reranker_model,
    )

    yield
    print("Shutting down")


app = FastAPI(
    title="Survey Question Retrieval & Ranking API",
    description="RAG-based retrieval and reranking service for survey profiling questions",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(routes.router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

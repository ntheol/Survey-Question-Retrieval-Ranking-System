"""FastAPI application entry point"""

from contextlib import asynccontextmanager

import uvicorn
from api.routes import router
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up")
    yield
    print("Shutting down")


app = FastAPI(
    title="Survey Question Retrieval & Ranking API",
    description="RAG-based retrieval and reranking service for survey profiling questions",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Survey Question Retrieval & Ranking System

RAG-based retrieval and reranking service for survey profiling questions, built with FastAPI.

## Overview

In this project we create a RAG-based retrieval and reranking service, exposed through a
FastAPI application.

The system uses:
- OpenAI embeddings (`text-embedding-3-small`) for semantic retrieval
- ChromaDB as the persistent vector store
- FastAPI for API serving
- A cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) as a second-stage ranking model

On startup, we load the dataset, build embedding-friendly text for each question, generate embeddings and index the documents into ChromDB. If there is already an existing ChromDB collection all the above steps are skipped and the existing collection is loaded. At query time, the API embeds the incoming query, retrieves the nearest matches, and can optionally rerank them before returning the final top-K results.

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key exported as `OPENAI_API_KEY`

### Installation

1. Create and activate a Python 3.11+ virtual environment.

Using python:
```bash
python -m venv .venv
source .venv/bin/activate
```

Using conda:
```bash
conda create -n venv python=3.11 
conda activate venv
```

2. Install the base requirements.

```bash
pip install -r requirements.txt
```

or

```bash
pip install -e .
```
3. Set your OpenAI API key in the environment.

```bash
export OPENAI_API_KEY="your_api_key_here"
```

You can also place the key in an environment file and load it before starting the app, depending on your local setup.

### Running the Application

Start the FastAPI server from the project root:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "health-conscious millennials in urban areas", "top_k": 5, "rerank": true}'
```

## Architecture / Design Decisions

The project is structured around a simple two-stage retrieval pipeline:

1. Indexing / startup initialization
- The dataset is loaded from `data/profiling_questions.json`.
- Each question is transformed into an embedding-friendly text representation that combines the raw question with metadata such as category, subcategory, demographic focus, and tags.
- Embeddings are generated once and stored in ChromaDB.

2. Retrieval
- For each `/search` request, the query is embedded with the same embedding model.
- ChromaDB retrieves the nearest questions, optionally filtered by metadata such as `category` or `demographic_focus`.

3. Reranking
- A cross-encoder reranker is loaded once at startup.
- When reranking is enabled, the retrieved candidates are rescored using the full query-question pair and returned in improved order.

Why these choices:
- ChromaDB keeps the implementation simple while still giving persistence and metadata filtering.
- OpenAI embeddings provide strong semantic retrieval quality with very little custom modeling work.
- Cross-encoder reranking is more accurate than raw cosine similarity alone and is a common second-stage ranking pattern in retrieval systems.
- FastAPI keeps the serving layer lightweight and easy to test.

# Survey Question Retrieval & Ranking System

RAG-based retrieval and reranking service for survey profiling questions, built with FastAPI.

## Overview

<!-- TODO: Overview of your project -->

## Quick Start

### Prerequisites

- Python 3.11+

### Installation

<!-- TODO: Installation steps of the project -->

### Running the Application

<!-- TODO: Steps to get the project up and running -->

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

<!-- TODO: (OPTIONAL) Describe your architecture and design decisions -->
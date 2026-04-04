from typing import List, Dict, Optional, Tuple

EMBEDDING_MODEL = "text-embedding-3-small"

def build_embedding_text(item: dict) -> str:
    """Create natural-language text for embedding that includes metadata context."""
    tags = ", ".join(item["tags"])
    
    # More natural format that reads like a description
    return (
        f"{item['text']} "
        f"This question is about {item['subcategory']} ({item['category']}), "
        f"targeting {item['demographic_focus']} audiences. "
        f"Related topics: {tags}."
    )

def batched(items: list, batch_size: int):
    """Yield successive batches from items."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_texts(
    client,
    texts: List[str], 
    model: str = EMBEDDING_MODEL, 
    batch_size: int = 100
) -> List[List[float]]:
    """Generate embeddings for texts using OpenAI API with batching."""
    vectors: List[List[float]] = []

    for i, batch in enumerate(batched(texts, batch_size)):
        print(f"Processing batch {i+1}/{(len(texts)-1)//batch_size + 1}...", end="\r")
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    
    print("\n✓ Embedding generation complete")
    return vectors

def prepare_data_for_chroma(questions: List[Dict],
                            document_texts: List[str],
                            document_embeddings: List[List[float]],
                            ) -> Tuple[List[str], List[str], List[List[float]], List[Dict]]:
    
    """Prepare data for ChromaDB ingestion."""
    ids = [q["id"] for q in questions]
    documents = document_texts
    embeddings = document_embeddings
    metadatas = [
        {
            "text": q["text"],
            "category": q["category"],
            "subcategory": q["subcategory"],
            "demographic_focus": q["demographic_focus"],
            "answer_type": q["answer_type"],
            "tags": ",".join(q["tags"])  # ChromaDB doesn't support list metadata
        }
        for q in questions
    ]
    return ids, documents, embeddings, metadatas

def create_chroma_collection(collection, 
                             ids: List[str], 
                             documents: List[str], 
                             embeddings: List[List[float]], 
                             metadatas: List[Dict]):
    """Create a ChromaDB collection with metadata."""
    BATCH_SIZE = 100
    for i in range(0, len(ids), BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(ids))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        print(f"Added {end_idx}/{len(ids)} documents to ChromaDB", end="\r")
    return collection

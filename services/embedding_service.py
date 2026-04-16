"""Embedding Service — schema manager + vector index.

Owns two pieces of state, both keyed by schema name:

* **Schemas** — ``{name, dimensions}`` contracts that incoming vectors
  must match.  Rejecting shape drift at index time prevents silently
  mixing incompatible vectors from different models.
* **Vector index** — ``(schema_name, image_id) → vector`` with a cosine
  similarity search.

Subscribes: inference.completed, search.requested
Publishes:  embedding.indexed, search.completed
HTTP:       /schemas, /embeddings/{schema}/{id}, /search/similar, /stats
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from events import (
    EMBEDDING_INDEXED,
    INFERENCE_COMPLETED,
    SEARCH_COMPLETED,
    SEARCH_REQUESTED,
    EmbeddingIndexedPayload,
    SearchCompletedPayload,
    SearchHit,
    make_event,
    validate_payload,
)
from messaging import MessageBus

app = FastAPI(title="Embedding Service")
_bus: MessageBus | None = None


def set_bus(bus: MessageBus) -> None:
    global _bus
    _bus = bus


class VectorSchema(BaseModel):
    name: str
    dimensions: int


# schema_name → VectorSchema
_schemas: dict[str, VectorSchema] = {}
# schema_name → { image_id → vector }
_index: dict[str, dict[str, list[float]]] = {}


def register_schema(schema: VectorSchema) -> None:
    existing = _schemas.get(schema.name)
    if existing is not None and existing.dimensions != schema.dimensions:
        raise ValueError(
            f"Schema {schema.name!r} already registered with different dimensions"
        )
    _schemas[schema.name] = schema
    _index.setdefault(schema.name, {})


def _ensure_schema(name: str, dimensions: int) -> VectorSchema:
    """Look up (and lazily create) a schema for an incoming vector."""
    schema = _schemas.get(name)
    if schema is None:
        schema = VectorSchema(name=name, dimensions=dimensions)
        register_schema(schema)
        return schema
    if schema.dimensions != dimensions:
        raise ValueError(
            f"Schema {name!r} expects dim {schema.dimensions}, got {dimensions}"
        )
    return schema


def reset_state() -> None:
    """Test helper — clear all schemas and vectors."""
    _schemas.clear()
    _index.clear()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _store_vector(schema_name: str, image_id: str, vector: list[float]) -> VectorSchema:
    schema = _ensure_schema(schema_name, len(vector))
    _index[schema_name][image_id] = vector
    return schema


def _search(schema_name: str, query: list[float], top_k: int) -> list[SearchHit]:
    if schema_name not in _schemas:
        raise KeyError(f"Unknown schema: {schema_name}")
    expected = _schemas[schema_name].dimensions
    if len(query) != expected:
        raise ValueError(
            f"Query dim {len(query)} does not match schema dim {expected}"
        )
    hits = [
        SearchHit(image_id=image_id, similarity=round(_cosine_similarity(query, v), 4))
        for image_id, v in _index[schema_name].items()
    ]
    hits.sort(key=lambda h: h.similarity, reverse=True)
    return hits[:top_k]


def handle_inference_completed(event: dict[str, Any]) -> None:
    payload = validate_payload(INFERENCE_COMPLETED, event["payload"])
    schema = _store_vector(
        payload.schema_name, payload.image_id, payload.embedding_vector
    )
    if _bus is not None:
        indexed = EmbeddingIndexedPayload(
            image_id=payload.image_id,
            schema_name=schema.name,
            dimensions=schema.dimensions,
        )
        _bus.publish(
            EMBEDDING_INDEXED,
            make_event(
                EMBEDDING_INDEXED,
                indexed,
                correlation_id=event.get("correlation_id"),
            ),
        )


def handle_search_requested(event: dict[str, Any]) -> None:
    payload = validate_payload(SEARCH_REQUESTED, event["payload"])
    try:
        hits = _search(payload.schema_name, payload.vector, payload.top_k)
    except (KeyError, ValueError):
        hits = []
    completed = SearchCompletedPayload(
        query_id=payload.query_id,
        schema_name=payload.schema_name,
        results=hits,
    )
    if _bus is not None:
        _bus.publish(
            SEARCH_COMPLETED,
            make_event(
                SEARCH_COMPLETED,
                completed,
                correlation_id=event.get("correlation_id"),
            ),
        )


class SimilarityQuery(BaseModel):
    vector: list[float]
    top_k: int = 5
    schema_name: str = "default"


@app.get("/schemas")
def list_schemas() -> dict[str, Any]:
    return {"schemas": [s.model_dump() for s in _schemas.values()]}


@app.post("/schemas")
def create_schema(schema: VectorSchema) -> dict[str, Any]:
    try:
        register_schema(schema)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return schema.model_dump()


@app.get("/embeddings/{schema_name}/{image_id}")
def get_embedding(schema_name: str, image_id: str) -> dict[str, Any]:
    shard = _index.get(schema_name) or {}
    vec = shard.get(image_id)
    if vec is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return {
        "image_id": image_id,
        "schema_name": schema_name,
        "vector": vec,
        "dimensions": len(vec),
    }


@app.post("/search/similar")
def search_similar(query: SimilarityQuery) -> dict[str, Any]:
    try:
        hits = _search(query.schema_name, query.vector, query.top_k)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"results": [h.model_dump() for h in hits]}


@app.get("/stats")
def stats() -> dict[str, Any]:
    return {
        "schemas": len(_schemas),
        "total_embeddings": sum(len(s) for s in _index.values()),
        "by_schema": {name: len(s) for name, s in _index.items()},
    }


def register(bus: MessageBus) -> None:
    set_bus(bus)
    bus.subscribe(INFERENCE_COMPLETED, handle_inference_completed)
    bus.subscribe(SEARCH_REQUESTED, handle_search_requested)


if __name__ == "__main__":  # pragma: no cover
    import threading
    import uvicorn
    from messaging import make_default_bus

    bus = make_default_bus()
    register(bus)
    threading.Thread(target=bus.run_forever, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8004)

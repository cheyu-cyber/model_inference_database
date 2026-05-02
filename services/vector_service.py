"""Vector Service — FAISS-backed vector index + similarity search.

The vector index lives here, not in the Embedding Service. Embedding
turns tags or text into a vector and ships it on the bus; this service
adds vectors to a per-schema FAISS index and answers similarity queries.

FAISS choice
------------
``IndexIDMap2`` over ``IndexFlatIP`` (inner product). Vectors arriving
on the bus are unit-length (built by the Embedding Service), so inner
product == cosine similarity. ``IDMap2`` lets us re-index by stable
``image_id`` — re-uploads remove and replace the old entry rather than
duplicating it.

Owns
----
* **Schemas** — ``{name, dimensions}`` contracts. The default
  ``"semantic"`` schema is created lazily on the first vector.
* **FAISS indices** — one per schema.

Subscribes: vector.computed, vector.search.requested
Publishes:  embedding.indexed, search.completed
HTTP:       /schemas, /embeddings/{schema}/{id}, /search/similar, /stats
"""

from __future__ import annotations

import math
import os
import sys
import threading
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from DB.model_inference_database.events import (
    EMBEDDING_INDEXED,
    SEARCH_COMPLETED,
    VECTOR_COMPUTED,
    VECTOR_SEARCH_REQUESTED,
    EmbeddingIndexedPayload,
    SearchCompletedPayload,
    SearchHit,
    make_event,
    validate_payload,
)
from DB.model_inference_database.messaging import MessageBus

app = FastAPI(title="Vector Service")
_bus: MessageBus | None = None

DEFAULT_SCHEMA = "semantic"

# Pre-registered dimension for the default schema. Must agree with the
# Embedding Service's vocabulary length — the two services share the
# topic contract, not Python imports.
DEFAULT_SCHEMA_DIMENSIONS = 8


def set_bus(bus: MessageBus | None) -> None:
    global _bus
    _bus = bus


# ── FAISS shard per schema ───────────────────────────────────────────

class VectorSchema(BaseModel):
    name: str
    dimensions: int


class _FaissShard:
    """A FAISS index plus the bookkeeping to address it by ``image_id``.

    FAISS only knows int64 ids, so we keep a parallel ``image_id ↔ int``
    map. ``upsert`` removes the old int id (if any) before adding, so
    re-indexing the same image overwrites the previous vector.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._next_int = 0
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._lock = threading.Lock()

    def upsert(self, image_id: str, vector: list[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(
                f"Vector dim {len(vector)} does not match schema dim {self.dim}"
            )
        vec = np.asarray(vector, dtype="float32").reshape(1, -1)
        with self._lock:
            if image_id in self._id_to_int:
                old = self._id_to_int[image_id]
                self._index.remove_ids(np.asarray([old], dtype="int64"))
                del self._int_to_id[old]
            new_int = self._next_int
            self._next_int += 1
            self._id_to_int[image_id] = new_int
            self._int_to_id[new_int] = image_id
            self._index.add_with_ids(vec, np.asarray([new_int], dtype="int64"))

    def get(self, image_id: str) -> list[float] | None:
        with self._lock:
            int_id = self._id_to_int.get(image_id)
            if int_id is None:
                return None
            return self._index.reconstruct(int_id).tolist()

    def search(self, query: list[float], top_k: int) -> list[SearchHit]:
        if len(query) != self.dim:
            raise ValueError(
                f"Query dim {len(query)} does not match schema dim {self.dim}"
            )
        with self._lock:
            ntotal = self._index.ntotal
            if ntotal == 0 or top_k <= 0:
                return []
            k = min(top_k, ntotal)
            q = np.asarray(query, dtype="float32").reshape(1, -1)
            sims, ids = self._index.search(q, k)
            hits: list[SearchHit] = []
            for sim, int_id in zip(sims[0], ids[0]):
                if int_id == -1:
                    continue
                hits.append(SearchHit(
                    image_id=self._int_to_id[int(int_id)],
                    similarity=round(float(sim), 4),
                ))
            return hits

    def size(self) -> int:
        with self._lock:
            return int(self._index.ntotal)


_schemas: dict[str, VectorSchema] = {}
_shards: dict[str, _FaissShard] = {}


def register_schema(schema: VectorSchema) -> None:
    existing = _schemas.get(schema.name)
    if existing is not None and existing.dimensions != schema.dimensions:
        raise ValueError(
            f"Schema {schema.name!r} already registered with different dimensions"
        )
    _schemas[schema.name] = schema
    _shards.setdefault(schema.name, _FaissShard(schema.dimensions))


def _ensure_schema(name: str, dimensions: int) -> VectorSchema:
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


def _init_default_schema() -> None:
    register_schema(VectorSchema(
        name=DEFAULT_SCHEMA, dimensions=DEFAULT_SCHEMA_DIMENSIONS,
    ))


def reset_state() -> None:
    """Test helper — clear all schemas and FAISS indices, then re-register default."""
    _schemas.clear()
    _shards.clear()
    _init_default_schema()


_init_default_schema()


# ── Cosine similarity (utility kept for parity with old API) ─────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# Back-compat alias for tests written against the old embedding service.
_cosine_similarity = cosine_similarity


# ── Event handlers ───────────────────────────────────────────────────

def handle_vector_computed(event: dict[str, Any]) -> None:
    """Insert a vector arriving from the Embedding Service into FAISS."""
    payload = validate_payload(VECTOR_COMPUTED, event["payload"])
    schema = _ensure_schema(payload.schema_name, len(payload.vector))  # type: ignore[attr-defined]
    _shards[schema.name].upsert(payload.image_id, payload.vector)  # type: ignore[attr-defined]

    if _bus is not None:
        indexed = EmbeddingIndexedPayload(
            image_id=payload.image_id,  # type: ignore[attr-defined]
            schema_name=schema.name,
            dimensions=schema.dimensions,
        )
        _bus.publish(
            EMBEDDING_INDEXED,
            make_event(EMBEDDING_INDEXED, indexed, correlation_id=event.get("correlation_id")),
        )


def handle_vector_search_requested(event: dict[str, Any]) -> None:
    """Run a FAISS similarity search for a pre-vectorized query."""
    payload = validate_payload(VECTOR_SEARCH_REQUESTED, event["payload"])
    hits: list[SearchHit] = []
    shard = _shards.get(payload.schema_name)  # type: ignore[attr-defined]
    if shard is not None:
        try:
            hits = shard.search(payload.vector, payload.top_k)  # type: ignore[attr-defined]
        except ValueError:
            hits = []

    completed = SearchCompletedPayload(
        query_id=payload.query_id,  # type: ignore[attr-defined]
        schema_name=payload.schema_name,  # type: ignore[attr-defined]
        results=hits,
    )
    if _bus is not None:
        _bus.publish(
            SEARCH_COMPLETED,
            make_event(SEARCH_COMPLETED, completed, correlation_id=event.get("correlation_id")),
        )


# ── HTTP API ─────────────────────────────────────────────────────────

class SimilarityQuery(BaseModel):
    vector: list[float]
    top_k: int = 5
    schema_name: str = DEFAULT_SCHEMA


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
    shard = _shards.get(schema_name)
    vec = shard.get(image_id) if shard is not None else None
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
    shard = _shards.get(query.schema_name)
    if shard is None:
        raise HTTPException(status_code=400, detail=f"Unknown schema: {query.schema_name}")
    try:
        hits = shard.search(query.vector, query.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"query_vector": query.vector, "results": [h.model_dump() for h in hits]}


@app.get("/stats")
def stats() -> dict[str, Any]:
    sizes = {name: shard.size() for name, shard in _shards.items()}
    return {
        "schemas": len(_schemas),
        "total_embeddings": sum(sizes.values()),
        "by_schema": sizes,
    }


def register(bus: MessageBus) -> None:
    set_bus(bus)
    bus.subscribe(VECTOR_COMPUTED, handle_vector_computed)
    bus.subscribe(VECTOR_SEARCH_REQUESTED, handle_vector_search_requested)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    from DB.model_inference_database.messaging import make_default_bus

    bus = make_default_bus()
    register(bus)
    threading.Thread(target=bus.run_forever, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8005)

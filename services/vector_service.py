"""Vector Service — FAISS-backed vector index + similarity search.

Pure pub/sub: no HTTP. The vector index lives here, not in the
Embedding Service.  Embedding turns tags or text into a vector and ships
it on the bus; this service adds vectors to a per-schema FAISS index
and answers similarity queries.

FAISS choice
------------
``IndexIDMap2`` over ``IndexFlatIP`` (inner product). Vectors arriving
on the bus are unit-length (built by the Embedding Service), so inner
product == cosine similarity. ``IDMap2`` lets us re-index by stable
``image_id`` — re-uploads remove and replace the old entry.

Owns
----
* **Schemas** — ``{name, dimensions}`` contracts. The default
  ``"semantic"`` schema is registered eagerly.
* **FAISS indices** — one per schema.

Subscribes
----------
* ``vector.computed``           → upsert into FAISS, publish ``embedding.indexed``
* ``vector.search.requested``   → cosine search, publish ``search.completed``
* ``schemas.list.requested``    → publish ``schemas.list.completed``
* ``schemas.create.requested``  → publish ``schemas.create.completed``
* ``embeddings.get.requested``  → publish ``embeddings.get.completed``
* ``stats.requested``           → publish ``stats.completed``
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
from pydantic import BaseModel

from DB.model_inference_database.events import (
    EMBEDDING_GET_COMPLETED,
    EMBEDDING_GET_REQUESTED,
    EMBEDDING_INDEXED,
    SCHEMA_CREATE_COMPLETED,
    SCHEMA_CREATE_REQUESTED,
    SCHEMAS_LIST_COMPLETED,
    SCHEMAS_LIST_REQUESTED,
    SEARCH_COMPLETED,
    STATS_COMPLETED,
    STATS_REQUESTED,
    VECTOR_COMPUTED,
    VECTOR_SEARCH_REQUESTED,
    EmbeddingGetCompletedPayload,
    EmbeddingIndexedPayload,
    SchemaCreateCompletedPayload,
    SchemaSpec,
    SchemasListCompletedPayload,
    SearchCompletedPayload,
    SearchHit,
    StatsCompletedPayload,
    make_event,
    validate_payload,
)
from DB.model_inference_database.messaging import MessageBus

log = logging.getLogger(__name__)

_bus: MessageBus | None = None

DEFAULT_SCHEMA = "semantic"
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


# ── Cosine similarity (utility) ──────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


_cosine_similarity = cosine_similarity  # back-compat alias


# ── Shared helpers ───────────────────────────────────────────────────

def _publish(topic: str, payload, correlation_id: str | None) -> None:
    if _bus is None:
        return
    _bus.publish(topic, make_event(topic, payload, correlation_id=correlation_id))


# ── Pipeline handlers ────────────────────────────────────────────────

def handle_vector_computed(event: dict[str, Any]) -> None:
    """Insert a vector arriving from the Embedding Service into FAISS."""
    payload = validate_payload(VECTOR_COMPUTED, event["payload"])
    schema = _ensure_schema(payload.schema_name, len(payload.vector))  # type: ignore[attr-defined]
    _shards[schema.name].upsert(payload.image_id, payload.vector)  # type: ignore[attr-defined]

    _publish(
        EMBEDDING_INDEXED,
        EmbeddingIndexedPayload(
            image_id=payload.image_id,  # type: ignore[attr-defined]
            schema_name=schema.name,
            dimensions=schema.dimensions,
        ),
        correlation_id=event.get("correlation_id"),
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

    _publish(
        SEARCH_COMPLETED,
        SearchCompletedPayload(
            query_id=payload.query_id,  # type: ignore[attr-defined]
            schema_name=payload.schema_name,  # type: ignore[attr-defined]
            results=hits,
        ),
        correlation_id=event.get("correlation_id"),
    )


# ── Read request/reply handlers ──────────────────────────────────────

def handle_schemas_list_requested(event: dict[str, Any]) -> None:
    validate_payload(SCHEMAS_LIST_REQUESTED, event["payload"])
    reply = SchemasListCompletedPayload(
        schemas=[
            SchemaSpec(name=s.name, dimensions=s.dimensions)
            for s in _schemas.values()
        ],
    )
    _publish(SCHEMAS_LIST_COMPLETED, reply, event.get("correlation_id"))


def handle_schema_create_requested(event: dict[str, Any]) -> None:
    payload = validate_payload(SCHEMA_CREATE_REQUESTED, event["payload"])
    try:
        register_schema(VectorSchema(
            name=payload.name,  # type: ignore[attr-defined]
            dimensions=payload.dimensions,  # type: ignore[attr-defined]
        ))
        reply = SchemaCreateCompletedPayload(
            name=payload.name,  # type: ignore[attr-defined]
            dimensions=payload.dimensions,  # type: ignore[attr-defined]
        )
    except ValueError as exc:
        reply = SchemaCreateCompletedPayload(error=str(exc))
    _publish(SCHEMA_CREATE_COMPLETED, reply, event.get("correlation_id"))


def handle_embedding_get_requested(event: dict[str, Any]) -> None:
    payload = validate_payload(EMBEDDING_GET_REQUESTED, event["payload"])
    shard = _shards.get(payload.schema_name)  # type: ignore[attr-defined]
    vec = shard.get(payload.image_id) if shard is not None else None  # type: ignore[attr-defined]
    if vec is None:
        reply = EmbeddingGetCompletedPayload(error="not_found")
    else:
        reply = EmbeddingGetCompletedPayload(
            schema_name=payload.schema_name,  # type: ignore[attr-defined]
            image_id=payload.image_id,  # type: ignore[attr-defined]
            vector=vec,
            dimensions=len(vec),
        )
    _publish(EMBEDDING_GET_COMPLETED, reply, event.get("correlation_id"))


def handle_stats_requested(event: dict[str, Any]) -> None:
    validate_payload(STATS_REQUESTED, event["payload"])
    sizes = {name: shard.size() for name, shard in _shards.items()}
    reply = StatsCompletedPayload(
        schemas=len(_schemas),
        total_embeddings=sum(sizes.values()),
        by_schema=sizes,
    )
    _publish(STATS_COMPLETED, reply, event.get("correlation_id"))


def register(bus: MessageBus) -> None:
    set_bus(bus)
    bus.subscribe(VECTOR_COMPUTED, handle_vector_computed)
    bus.subscribe(VECTOR_SEARCH_REQUESTED, handle_vector_search_requested)
    bus.subscribe(SCHEMAS_LIST_REQUESTED, handle_schemas_list_requested)
    bus.subscribe(SCHEMA_CREATE_REQUESTED, handle_schema_create_requested)
    bus.subscribe(EMBEDDING_GET_REQUESTED, handle_embedding_get_requested)
    bus.subscribe(STATS_REQUESTED, handle_stats_requested)


if __name__ == "__main__":  # pragma: no cover
    from DB.model_inference_database.messaging import make_default_bus

    bus = make_default_bus()
    register(bus)
    print("[vector] subscribed; waiting for events…")
    bus.run_forever()

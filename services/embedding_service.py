"""Embedding Service — schema manager + semantic vector index.

This service owns *both* the vocabulary that defines the vector space and
the vectors themselves.  Inference produces tags (``person``,
``pedestrian``, ``car``…); the Embedding Service maps synonyms into the
same dimension of a semantic vector and indexes it.  Text queries go
through the same mapping, so "pedestrians and 4 wheeler" searches find
images whose tags include ``person`` and ``car``.

Owns
----
* **Schemas** — ``{name, dimensions}`` contracts.  The built-in
  ``"semantic"`` schema has one dimension per category in
  :data:`SEMANTIC_CATEGORIES`.
* **Vector index** — ``(schema_name, image_id) → vector``.

Subscribes: inference.completed, search.requested
Publishes:  embedding.indexed, search.completed
HTTP:       /schemas, /embeddings/{schema}/{id}, /search/similar, /stats
"""

from __future__ import annotations

import math
import os
import re
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

DEFAULT_SCHEMA = "semantic"


def set_bus(bus: MessageBus) -> None:
    global _bus
    _bus = bus


# ── Semantic vocabulary ──────────────────────────────────────────────
#
# Each entry is (category_name, synonyms).  Synonyms in the same entry
# collapse to the same dimension, so "person" and "pedestrian" produce
# identical vectors and cosine-search treats them as equivalent.
#
# This is a deliberately small, hand-curated vocabulary — it is enough
# to demonstrate the concept for the assignment and leaves room for
# swapping in a real text-embedding model (e.g. sentence-transformers)
# later without touching any other service.
SEMANTIC_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("person",       ("person", "persons", "people", "human", "humans",
                       "pedestrian", "pedestrians", "man", "men",
                       "woman", "women", "child", "children")),
    ("four_wheeler", ("car", "cars", "truck", "trucks", "bus", "buses",
                       "van", "suv", "vehicle", "vehicles", "automobile",
                       "4 wheeler", "four wheeler")),
    ("two_wheeler",  ("bicycle", "bicycles", "bike", "bikes",
                       "motorcycle", "motorcycles", "motorbike", "scooter",
                       "2 wheeler", "two wheeler")),
    ("animal",       ("dog", "dogs", "cat", "cats", "horse", "cow",
                       "sheep", "bird", "birds", "animal", "animals")),
    ("nature",       ("tree", "trees", "plant", "plants", "flower",
                       "flowers", "grass")),
    ("building",     ("building", "buildings", "house", "houses",
                       "skyscraper", "tower")),
    ("food",         ("food", "pizza", "burger", "apple", "banana",
                       "sandwich", "cake")),
    ("traffic",      ("traffic light", "stop sign", "sign", "signs")),
]

SEMANTIC_DIM = len(SEMANTIC_CATEGORIES)


def _category_index(term: str) -> int | None:
    term = term.lower().strip()
    if not term:
        return None
    for i, (_, synonyms) in enumerate(SEMANTIC_CATEGORIES):
        if term in synonyms:
            return i
    return None


def semantic_vector(terms: list[str]) -> list[float]:
    """Build a unit-length vector from a list of tags.

    Unknown terms are ignored.  An all-unknown input returns a zero
    vector (cosine similarity against it is 0 — that is the correct
    "no signal" behaviour).
    """
    vec = [0.0] * SEMANTIC_DIM
    for term in terms:
        idx = _category_index(term)
        if idx is not None:
            vec[idx] += 1.0
    mag = math.sqrt(sum(x * x for x in vec))
    if mag == 0:
        return vec
    return [x / mag for x in vec]


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def text_to_vector(text: str) -> list[float]:
    """Tokenize free-form text and semantically vectorize it.

    Greedily prefers bigram matches ("stop sign", "4 wheeler") over
    their constituent unigrams — so "stop sign" contributes once to the
    traffic category, not twice.
    """
    words = _TOKEN_RE.findall(text.lower())
    consumed = [False] * len(words)
    terms: list[str] = []
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        if _category_index(bigram) is not None:
            terms.append(bigram)
            consumed[i] = consumed[i + 1] = True
    for i, word in enumerate(words):
        if not consumed[i]:
            terms.append(word)
    return semantic_vector(terms)


def tags_from_annotations(annotations: dict[str, Any]) -> list[str]:
    """Walk ``annotations.objects[*].tags`` and flatten to a tag list."""
    tags: list[str] = []
    for obj in annotations.get("objects", []) or []:
        for tag in obj.get("tags", []) or []:
            tags.append(str(tag))
    return tags


# ── Schema registry + vector index ───────────────────────────────────

class VectorSchema(BaseModel):
    name: str
    dimensions: int


_schemas: dict[str, VectorSchema] = {}
_index: dict[str, dict[str, list[float]]] = {}  # schema → image_id → vector


def register_schema(schema: VectorSchema) -> None:
    existing = _schemas.get(schema.name)
    if existing is not None and existing.dimensions != schema.dimensions:
        raise ValueError(
            f"Schema {schema.name!r} already registered with different dimensions"
        )
    _schemas[schema.name] = schema
    _index.setdefault(schema.name, {})


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
    register_schema(VectorSchema(name=DEFAULT_SCHEMA, dimensions=SEMANTIC_DIM))


def reset_state() -> None:
    """Test helper — clear all schemas and vectors, then re-register default."""
    _schemas.clear()
    _index.clear()
    _init_default_schema()


_init_default_schema()


# ── Cosine search ────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


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


# ── Event handlers ───────────────────────────────────────────────────

def handle_inference_completed(event: dict[str, Any]) -> None:
    """Derive a semantic vector from the inference tags and index it."""
    payload = validate_payload(INFERENCE_COMPLETED, event["payload"])
    tags = tags_from_annotations(payload.annotations)  # type: ignore[attr-defined]
    vector = semantic_vector(tags)
    schema = _ensure_schema(payload.schema_name, len(vector))  # type: ignore[attr-defined]
    _index[schema.name][payload.image_id] = vector  # type: ignore[attr-defined]

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


def handle_search_requested(event: dict[str, Any]) -> None:
    """Run a similarity search from either a raw vector or a text query."""
    payload = validate_payload(SEARCH_REQUESTED, event["payload"])
    query_vec = _resolve_query_vector(payload)  # type: ignore[arg-type]
    try:
        hits = _search(payload.schema_name, query_vec, payload.top_k)  # type: ignore[attr-defined]
    except (KeyError, ValueError):
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


def _resolve_query_vector(payload) -> list[float]:
    """A search request supplies either a raw vector or free-form text."""
    if payload.vector is not None:
        return payload.vector
    if payload.query_text:
        return text_to_vector(payload.query_text)
    return []


# ── HTTP API ─────────────────────────────────────────────────────────

class SimilarityQuery(BaseModel):
    vector: list[float] | None = None
    query_text: str | None = None
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


@app.get("/vocabulary")
def get_vocabulary() -> dict[str, Any]:
    """Expose the semantic vocabulary so clients can see what maps to what."""
    return {
        "dimensions": SEMANTIC_DIM,
        "categories": [
            {"index": i, "name": name, "synonyms": list(syns)}
            for i, (name, syns) in enumerate(SEMANTIC_CATEGORIES)
        ],
    }


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
    if query.vector is None and not query.query_text:
        raise HTTPException(status_code=400, detail="Provide vector or query_text")
    vec = query.vector if query.vector is not None else text_to_vector(query.query_text or "")
    try:
        hits = _search(query.schema_name, vec, query.top_k)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"query_vector": vec, "results": [h.model_dump() for h in hits]}


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

"""Embedding Service — semantic vocabulary + vectorization.

This service owns the vocabulary that defines the semantic vector space
and converts tags / free-form text into unit-length vectors. It does
*not* store vectors or run similarity search — that lives in the Vector
Service. The two communicate over the bus:

    inference.completed → Embedding → vector.computed → Vector
    search.requested    → Embedding → vector.search.requested → Vector

Owns
----
* **Vocabulary** — the semantic categories and their synonyms.
* **Tag/text → vector** — the deterministic mapping that makes
  ``"pedestrian"`` and ``"person"`` collapse to the same dimension.

Subscribes: inference.completed, search.requested
Publishes:  vector.computed, vector.search.requested
HTTP:       /vocabulary, /embed/text, /embed/tags
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

from DB.model_inference_database.events import (
    INFERENCE_COMPLETED,
    SEARCH_REQUESTED,
    VECTOR_COMPUTED,
    VECTOR_SEARCH_REQUESTED,
    VectorComputedPayload,
    VectorSearchRequestedPayload,
    make_event,
    validate_payload,
)
from DB.model_inference_database.messaging import MessageBus

app = FastAPI(title="Embedding Service")
_bus: MessageBus | None = None

DEFAULT_SCHEMA = "semantic"


def set_bus(bus: MessageBus | None) -> None:
    global _bus
    _bus = bus


# ── Semantic vocabulary ──────────────────────────────────────────────
#
# Each entry is (category_name, synonyms).  Synonyms in the same entry
# collapse to the same dimension, so "person" and "pedestrian" produce
# identical vectors and cosine-search treats them as equivalent.
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


# ── Event handlers ───────────────────────────────────────────────────

def handle_inference_completed(event: dict[str, Any]) -> None:
    """Derive a semantic vector from the inference tags and emit it.

    The Vector Service consumes ``vector.computed`` and owns the index;
    this service intentionally has no idea where the vector goes.
    """
    payload = validate_payload(INFERENCE_COMPLETED, event["payload"])
    tags = tags_from_annotations(payload.annotations)  # type: ignore[attr-defined]
    vector = semantic_vector(tags)

    if _bus is not None:
        computed = VectorComputedPayload(
            image_id=payload.image_id,  # type: ignore[attr-defined]
            schema_name=payload.schema_name,  # type: ignore[attr-defined]
            vector=vector,
        )
        _bus.publish(
            VECTOR_COMPUTED,
            make_event(VECTOR_COMPUTED, computed, correlation_id=event.get("correlation_id")),
        )


def handle_search_requested(event: dict[str, Any]) -> None:
    """Vectorize a search query and forward it to the Vector Service."""
    payload = validate_payload(SEARCH_REQUESTED, event["payload"])
    query_vec = _resolve_query_vector(payload)

    if _bus is not None:
        forwarded = VectorSearchRequestedPayload(
            query_id=payload.query_id,  # type: ignore[attr-defined]
            schema_name=payload.schema_name,  # type: ignore[attr-defined]
            vector=query_vec,
            top_k=payload.top_k,  # type: ignore[attr-defined]
        )
        _bus.publish(
            VECTOR_SEARCH_REQUESTED,
            make_event(
                VECTOR_SEARCH_REQUESTED,
                forwarded,
                correlation_id=event.get("correlation_id"),
            ),
        )


def _resolve_query_vector(payload) -> list[float]:
    """A search request supplies either a raw vector or free-form text."""
    if payload.vector is not None:
        return payload.vector
    if payload.query_text:
        return text_to_vector(payload.query_text)
    return [0.0] * SEMANTIC_DIM


# ── HTTP API ─────────────────────────────────────────────────────────

class TextEmbedRequest(BaseModel):
    text: str


class TagsEmbedRequest(BaseModel):
    tags: list[str]


@app.get("/vocabulary")
def get_vocabulary() -> dict[str, Any]:
    """Expose the semantic vocabulary so clients can see what maps to what."""
    return {
        "dimensions": SEMANTIC_DIM,
        "schema_name": DEFAULT_SCHEMA,
        "categories": [
            {"index": i, "name": name, "synonyms": list(syns)}
            for i, (name, syns) in enumerate(SEMANTIC_CATEGORIES)
        ],
    }


@app.post("/embed/text")
def embed_text(req: TextEmbedRequest) -> dict[str, Any]:
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required")
    vec = text_to_vector(req.text)
    return {
        "schema_name": DEFAULT_SCHEMA,
        "vector": vec,
        "dimensions": len(vec),
    }


@app.post("/embed/tags")
def embed_tags(req: TagsEmbedRequest) -> dict[str, Any]:
    vec = semantic_vector(req.tags)
    return {
        "schema_name": DEFAULT_SCHEMA,
        "vector": vec,
        "dimensions": len(vec),
    }


def register(bus: MessageBus) -> None:
    set_bus(bus)
    bus.subscribe(INFERENCE_COMPLETED, handle_inference_completed)
    bus.subscribe(SEARCH_REQUESTED, handle_search_requested)


if __name__ == "__main__":  # pragma: no cover
    import threading
    import uvicorn
    from DB.model_inference_database.messaging import make_default_bus

    bus = make_default_bus()
    register(bus)
    threading.Thread(target=bus.run_forever, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8004)

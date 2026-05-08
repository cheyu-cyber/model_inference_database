"""Embedding Service — semantic vocabulary + tag/text → vector.

Pure pub/sub: no HTTP. Owns the vocabulary that defines the semantic
vector space and converts tags or free-form text into unit-length
vectors.  It does *not* store vectors or run similarity search — that
lives in the Vector Service.

Subscribes
----------
* ``inference.completed``   → derive vector from tags, publish ``vector.computed``
* ``search.requested``      → vectorize query, publish ``vector.search.requested``
* ``vocabulary.requested``  → publish ``vocabulary.completed``
* ``embed.text.requested``  → publish ``embed.text.completed``
* ``embed.tags.requested``  → publish ``embed.tags.completed``

Publishes
---------
* ``vector.computed``
* ``vector.search.requested``
* ``vocabulary.completed``
* ``embed.text.completed``
* ``embed.tags.completed``
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DB.model_inference_database.events import (
    EMBED_TAGS_COMPLETED,
    EMBED_TAGS_REQUESTED,
    EMBED_TEXT_COMPLETED,
    EMBED_TEXT_REQUESTED,
    INFERENCE_COMPLETED,
    SEARCH_REQUESTED,
    VECTOR_COMPUTED,
    VECTOR_SEARCH_REQUESTED,
    VOCABULARY_COMPLETED,
    VOCABULARY_REQUESTED,
    EmbedCompletedPayload,
    VectorComputedPayload,
    VectorSearchRequestedPayload,
    VocabularyCategory,
    VocabularyCompletedPayload,
    make_event,
    validate_payload,
)
from DB.model_inference_database.messaging import MessageBus

log = logging.getLogger(__name__)

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
    """Build a unit-length vector from a list of tags."""
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
    their constituent unigrams.
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


def _publish(topic: str, payload, correlation_id: str | None) -> None:
    if _bus is None:
        return
    _bus.publish(topic, make_event(topic, payload, correlation_id=correlation_id))


# ── Pipeline handlers ────────────────────────────────────────────────

def handle_inference_completed(event: dict[str, Any]) -> None:
    """Derive a semantic vector from inference tags and emit it."""
    payload = validate_payload(INFERENCE_COMPLETED, event["payload"])
    tags = tags_from_annotations(payload.annotations)  # type: ignore[attr-defined]
    vector = semantic_vector(tags)

    _publish(
        VECTOR_COMPUTED,
        VectorComputedPayload(
            image_id=payload.image_id,  # type: ignore[attr-defined]
            schema_name=payload.schema_name,  # type: ignore[attr-defined]
            vector=vector,
        ),
        correlation_id=event.get("correlation_id"),
    )


def handle_search_requested(event: dict[str, Any]) -> None:
    """Vectorize a search query and forward it to the Vector Service."""
    payload = validate_payload(SEARCH_REQUESTED, event["payload"])
    query_vec = _resolve_query_vector(payload)

    _publish(
        VECTOR_SEARCH_REQUESTED,
        VectorSearchRequestedPayload(
            query_id=payload.query_id,  # type: ignore[attr-defined]
            schema_name=payload.schema_name,  # type: ignore[attr-defined]
            vector=query_vec,
            top_k=payload.top_k,  # type: ignore[attr-defined]
        ),
        correlation_id=event.get("correlation_id"),
    )


def _resolve_query_vector(payload) -> list[float]:
    """A search request supplies either a raw vector or free-form text."""
    if payload.vector is not None:
        return payload.vector
    if payload.query_text:
        return text_to_vector(payload.query_text)
    return [0.0] * SEMANTIC_DIM


# ── Read request/reply handlers ──────────────────────────────────────

def handle_vocabulary_requested(event: dict[str, Any]) -> None:
    validate_payload(VOCABULARY_REQUESTED, event["payload"])
    reply = VocabularyCompletedPayload(
        dimensions=SEMANTIC_DIM,
        schema_name=DEFAULT_SCHEMA,
        categories=[
            VocabularyCategory(index=i, name=name, synonyms=list(syns))
            for i, (name, syns) in enumerate(SEMANTIC_CATEGORIES)
        ],
    )
    _publish(VOCABULARY_COMPLETED, reply, event.get("correlation_id"))


def handle_embed_text_requested(event: dict[str, Any]) -> None:
    payload = validate_payload(EMBED_TEXT_REQUESTED, event["payload"])
    if not payload.text:  # type: ignore[attr-defined]
        reply = EmbedCompletedPayload(error="text is required")
    else:
        vec = text_to_vector(payload.text)  # type: ignore[attr-defined]
        reply = EmbedCompletedPayload(
            schema_name=DEFAULT_SCHEMA, vector=vec, dimensions=len(vec),
        )
    _publish(EMBED_TEXT_COMPLETED, reply, event.get("correlation_id"))


def handle_embed_tags_requested(event: dict[str, Any]) -> None:
    payload = validate_payload(EMBED_TAGS_REQUESTED, event["payload"])
    vec = semantic_vector(payload.tags)  # type: ignore[attr-defined]
    reply = EmbedCompletedPayload(
        schema_name=DEFAULT_SCHEMA, vector=vec, dimensions=len(vec),
    )
    _publish(EMBED_TAGS_COMPLETED, reply, event.get("correlation_id"))


def register(bus: MessageBus) -> None:
    set_bus(bus)
    bus.subscribe(INFERENCE_COMPLETED, handle_inference_completed)
    bus.subscribe(SEARCH_REQUESTED, handle_search_requested)
    bus.subscribe(VOCABULARY_REQUESTED, handle_vocabulary_requested)
    bus.subscribe(EMBED_TEXT_REQUESTED, handle_embed_text_requested)
    bus.subscribe(EMBED_TAGS_REQUESTED, handle_embed_tags_requested)


if __name__ == "__main__":  # pragma: no cover
    from DB.model_inference_database.messaging import make_default_bus

    bus = make_default_bus()
    register(bus)
    print("[embedding] subscribed; waiting for events…")
    bus.run_forever()

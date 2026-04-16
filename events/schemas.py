"""Pydantic message schemas — the contracts for every event on the bus.

Every event published through the :class:`MessageBus` is wrapped in an
:class:`EventEnvelope` with a stable shape:

    {
        "event_id":       unique UUID for this event,
        "event_type":     topic name (see events.topics),
        "timestamp":      ISO-8601 UTC,
        "correlation_id": UUID that threads a request across services,
        "payload":        topic-specific payload dict
    }

The envelope is independent of the payload — services receive the full
envelope and use the `payload` field plus a payload schema to interpret it.
Correlation IDs let us trace a single upload → inference → index chain
through logs and tests.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Envelope
# ----------------------------------------------------------------------

class EventEnvelope(BaseModel):
    """Wrapper placed around every payload on the bus."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    payload: dict[str, Any]


def make_event(
    event_type: str,
    payload: BaseModel | dict[str, Any],
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Build a bus-ready envelope dict from a payload.

    Accepts either a Pydantic model (recommended — validates at publish
    time) or a plain dict (escape hatch for tests and fault injection).
    """
    if isinstance(payload, BaseModel):
        payload_dict = payload.model_dump()
    else:
        payload_dict = dict(payload)

    envelope = EventEnvelope(
        event_type=event_type,
        payload=payload_dict,
        correlation_id=correlation_id or str(uuid.uuid4()),
    )
    return envelope.model_dump()


# ----------------------------------------------------------------------
# Payload schemas — one per topic
# ----------------------------------------------------------------------

class ImageUploadedPayload(BaseModel):
    """Published by Upload Service after a file lands on disk."""

    image_id: str
    file_path: str
    file_size_bytes: int
    mime_type: str


class InferenceCompletedPayload(BaseModel):
    """Published by Inference Service after a model returns.

    `annotations` is an unconstrained dict on purpose: different models
    produce different nested shapes, which is exactly why the downstream
    store is a document DB.
    """

    image_id: str
    model_name: str
    annotations: dict[str, Any]
    embedding_vector: list[float]
    schema_name: str = "default"


class DocumentStoredPayload(BaseModel):
    """Published by Document DB Service after a JSON document is persisted."""

    document_id: str
    image_id: str
    model_name: str


class EmbeddingIndexedPayload(BaseModel):
    """Published by Embedding Service after a vector enters the index."""

    image_id: str
    schema_name: str
    dimensions: int


class SearchRequestedPayload(BaseModel):
    """Published by Web UI when a similarity search is requested."""

    query_id: str
    schema_name: str = "default"
    vector: list[float]
    top_k: int = 5


class SearchHit(BaseModel):
    image_id: str
    similarity: float


class SearchCompletedPayload(BaseModel):
    """Published by Embedding Service when a search query finishes."""

    query_id: str
    schema_name: str
    results: list[SearchHit]


# Map topic → payload schema, so test / generator code can validate messages
# without every caller having to know the mapping.
PAYLOAD_SCHEMAS: dict[str, type[BaseModel]] = {
    "image.uploaded": ImageUploadedPayload,
    "inference.completed": InferenceCompletedPayload,
    "document.stored": DocumentStoredPayload,
    "embedding.indexed": EmbeddingIndexedPayload,
    "search.requested": SearchRequestedPayload,
    "search.completed": SearchCompletedPayload,
}


def validate_payload(event_type: str, payload: dict[str, Any]) -> BaseModel:
    """Parse a raw payload dict into its typed schema — raises on mismatch."""
    schema = PAYLOAD_SCHEMAS.get(event_type)
    if schema is None:
        raise KeyError(f"Unknown event_type: {event_type!r}")
    return schema(**payload)

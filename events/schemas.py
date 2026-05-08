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

Request/reply convention
------------------------
Every read operation is a pair of topics: ``foo.requested`` /
``foo.completed``. The reply is matched to the request by the
envelope's ``correlation_id`` — which is already unique per event and
already propagated by every handler — so payloads carry no separate
correlation field. On error, a completed payload sets ``error``
(a short string) instead of the success fields.
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
# Pipeline (write path)
# ----------------------------------------------------------------------

class ImageUploadedPayload(BaseModel):
    """Published by the Web Service after a file lands on disk."""

    image_id: str
    file_path: str
    file_size_bytes: int
    mime_type: str


class InferenceCompletedPayload(BaseModel):
    """Published by Inference Service after a model returns.

    ``annotations`` is an unconstrained dict on purpose — different models
    produce different nested shapes, which is why the downstream store
    is a document DB.
    """

    image_id: str
    model_name: str
    annotations: dict[str, Any]
    schema_name: str = "semantic"


class DocumentStoredPayload(BaseModel):
    """Published by Document DB Service after a document is persisted."""

    document_id: str
    image_id: str
    model_name: str


class VectorComputedPayload(BaseModel):
    """Published by Embedding Service after deriving a vector from tags."""

    image_id: str
    schema_name: str
    vector: list[float]


class EmbeddingIndexedPayload(BaseModel):
    """Published by Vector Service after a vector enters the index."""

    image_id: str
    schema_name: str
    dimensions: int


# ----------------------------------------------------------------------
# Search path
# ----------------------------------------------------------------------

class VectorSearchRequestedPayload(BaseModel):
    """Published by Embedding Service after vectorizing a search query."""

    query_id: str
    schema_name: str
    vector: list[float]
    top_k: int = 5


class SearchRequestedPayload(BaseModel):
    """Published by Web UI when a similarity search is requested."""

    query_id: str
    schema_name: str = "semantic"
    vector: list[float] | None = None
    query_text: str | None = None
    top_k: int = 5


class SearchHit(BaseModel):
    image_id: str
    similarity: float


class SearchCompletedPayload(BaseModel):
    """Published by Vector Service when a search query finishes."""

    query_id: str
    schema_name: str
    results: list[SearchHit]


# ----------------------------------------------------------------------
# Read request/reply pairs
# ----------------------------------------------------------------------

class _ReplyPayload(BaseModel):
    """Base for reply payloads — surfaces an optional error field."""

    error: str | None = None


# Documents
class DocumentsListRequestedPayload(BaseModel):
    pass


class DocumentsListCompletedPayload(_ReplyPayload):
    document_ids: list[str] = []


class DocumentGetRequestedPayload(BaseModel):
    image_id: str


class DocumentGetCompletedPayload(_ReplyPayload):
    document: dict[str, Any] | None = None


# Embeddings (Vector)
class EmbeddingGetRequestedPayload(BaseModel):
    schema_name: str
    image_id: str


class EmbeddingGetCompletedPayload(_ReplyPayload):
    schema_name: str | None = None
    image_id: str | None = None
    vector: list[float] | None = None
    dimensions: int | None = None


# Schemas
class SchemasListRequestedPayload(BaseModel):
    pass


class SchemaSpec(BaseModel):
    name: str
    dimensions: int


class SchemasListCompletedPayload(_ReplyPayload):
    schemas: list[SchemaSpec] = []


class SchemaCreateRequestedPayload(BaseModel):
    name: str
    dimensions: int


class SchemaCreateCompletedPayload(_ReplyPayload):
    name: str | None = None
    dimensions: int | None = None


# Stats
class StatsRequestedPayload(BaseModel):
    pass


class StatsCompletedPayload(_ReplyPayload):
    schemas: int = 0
    total_embeddings: int = 0
    by_schema: dict[str, int] = {}


# Vocabulary
class VocabularyRequestedPayload(BaseModel):
    pass


class VocabularyCategory(BaseModel):
    index: int
    name: str
    synonyms: list[str]


class VocabularyCompletedPayload(_ReplyPayload):
    dimensions: int = 0
    schema_name: str | None = None
    categories: list[VocabularyCategory] = []


# Embedding (text/tags)
class EmbedTextRequestedPayload(BaseModel):
    text: str


class EmbedTagsRequestedPayload(BaseModel):
    tags: list[str]


class EmbedCompletedPayload(_ReplyPayload):
    schema_name: str | None = None
    vector: list[float] | None = None
    dimensions: int | None = None


# ----------------------------------------------------------------------
# Topic → schema map
# ----------------------------------------------------------------------

PAYLOAD_SCHEMAS: dict[str, type[BaseModel]] = {
    # pipeline
    "image.uploaded": ImageUploadedPayload,
    "inference.completed": InferenceCompletedPayload,
    "document.stored": DocumentStoredPayload,
    "vector.computed": VectorComputedPayload,
    "embedding.indexed": EmbeddingIndexedPayload,
    # search
    "search.requested": SearchRequestedPayload,
    "vector.search.requested": VectorSearchRequestedPayload,
    "search.completed": SearchCompletedPayload,
    # read request/reply
    "documents.list.requested": DocumentsListRequestedPayload,
    "documents.list.completed": DocumentsListCompletedPayload,
    "documents.get.requested": DocumentGetRequestedPayload,
    "documents.get.completed": DocumentGetCompletedPayload,
    "embeddings.get.requested": EmbeddingGetRequestedPayload,
    "embeddings.get.completed": EmbeddingGetCompletedPayload,
    "schemas.list.requested": SchemasListRequestedPayload,
    "schemas.list.completed": SchemasListCompletedPayload,
    "schemas.create.requested": SchemaCreateRequestedPayload,
    "schemas.create.completed": SchemaCreateCompletedPayload,
    "stats.requested": StatsRequestedPayload,
    "stats.completed": StatsCompletedPayload,
    "vocabulary.requested": VocabularyRequestedPayload,
    "vocabulary.completed": VocabularyCompletedPayload,
    "embed.text.requested": EmbedTextRequestedPayload,
    "embed.text.completed": EmbedCompletedPayload,
    "embed.tags.requested": EmbedTagsRequestedPayload,
    "embed.tags.completed": EmbedCompletedPayload,
}


def validate_payload(event_type: str, payload: dict[str, Any]) -> BaseModel:
    """Parse a raw payload dict into its typed schema — raises on mismatch."""
    schema = PAYLOAD_SCHEMAS.get(event_type)
    if schema is None:
        raise KeyError(f"Unknown event_type: {event_type!r}")
    return schema(**payload)

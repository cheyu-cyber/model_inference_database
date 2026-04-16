"""Event definitions — topic names and Pydantic message schemas."""

from .topics import (
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    DOCUMENT_STORED,
    EMBEDDING_INDEXED,
    SEARCH_REQUESTED,
    SEARCH_COMPLETED,
    ALL_TOPICS,
)
from .schemas import (
    EventEnvelope,
    ImageUploadedPayload,
    InferenceCompletedPayload,
    DocumentStoredPayload,
    EmbeddingIndexedPayload,
    SearchRequestedPayload,
    SearchCompletedPayload,
    SearchHit,
    make_event,
    validate_payload,
)

__all__ = [
    "IMAGE_UPLOADED",
    "INFERENCE_COMPLETED",
    "DOCUMENT_STORED",
    "EMBEDDING_INDEXED",
    "SEARCH_REQUESTED",
    "SEARCH_COMPLETED",
    "ALL_TOPICS",
    "EventEnvelope",
    "ImageUploadedPayload",
    "InferenceCompletedPayload",
    "DocumentStoredPayload",
    "EmbeddingIndexedPayload",
    "SearchRequestedPayload",
    "SearchCompletedPayload",
    "SearchHit",
    "make_event",
    "validate_payload",
]

"""Document DB Service — owns annotation documents in MongoDB.

Pure pub/sub: no HTTP. Writes are driven by ``inference.completed``;
reads are request/reply over the bus.

Subscribes
----------
* ``inference.completed``       → upsert document, publish ``document.stored``
* ``documents.list.requested``  → publish ``documents.list.completed``
* ``documents.get.requested``   → publish ``documents.get.completed``

Publishes
---------
* ``document.stored``
* ``documents.list.completed``
* ``documents.get.completed``
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DB.model_inference_database.events import (
    DOCUMENT_GET_COMPLETED,
    DOCUMENT_GET_REQUESTED,
    DOCUMENT_STORED,
    DOCUMENTS_LIST_COMPLETED,
    DOCUMENTS_LIST_REQUESTED,
    INFERENCE_COMPLETED,
    DocumentGetCompletedPayload,
    DocumentsListCompletedPayload,
    DocumentStoredPayload,
    make_event,
    validate_payload,
)
from DB.model_inference_database.messaging import MessageBus

log = logging.getLogger(__name__)

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "model_inference")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "documents")

_bus: MessageBus | None = None
_collection: Any = None


def set_bus(bus: MessageBus | None) -> None:
    global _bus
    _bus = bus


def set_collection(collection: Any) -> None:
    """Inject a collection (used by tests with mongomock)."""
    global _collection
    _collection = collection


def get_collection() -> Any:
    """Lazy-connect to MongoDB on first use; honors set_collection() override."""
    global _collection
    if _collection is None:
        from pymongo import MongoClient

        client = MongoClient(MONGO_URL)
        _collection = client[MONGO_DB][MONGO_COLLECTION]
        _collection.create_index("image_id", unique=True)
    return _collection


def _publish(topic: str, payload, correlation_id: str | None) -> None:
    if _bus is None:
        return
    _bus.publish(topic, make_event(topic, payload, correlation_id=correlation_id))


# ── Write path ───────────────────────────────────────────────────────

def handle_inference_completed(event: dict[str, Any]) -> None:
    payload = validate_payload(INFERENCE_COMPLETED, event["payload"])

    document = {
        "document_id": f"doc_{payload.image_id}",  # type: ignore[attr-defined]
        "image_id": payload.image_id,  # type: ignore[attr-defined]
        "model_name": payload.model_name,  # type: ignore[attr-defined]
        "annotations": payload.annotations,  # type: ignore[attr-defined]
    }
    get_collection().replace_one(
        {"image_id": document["image_id"]}, document, upsert=True
    )

    _publish(
        DOCUMENT_STORED,
        DocumentStoredPayload(
            document_id=document["document_id"],
            image_id=document["image_id"],
            model_name=document["model_name"],
        ),
        correlation_id=event.get("correlation_id"),
    )


# ── Read path (request/reply) ────────────────────────────────────────

def handle_documents_list_requested(event: dict[str, Any]) -> None:
    validate_payload(DOCUMENTS_LIST_REQUESTED, event["payload"])
    try:
        coll = get_collection()
        ids = [
            doc["document_id"]
            for doc in coll.find({}, {"document_id": 1, "_id": 0})
        ]
        reply = DocumentsListCompletedPayload(document_ids=ids)
    except Exception as exc:  # surface the error to the caller
        log.exception("documents.list failed")
        reply = DocumentsListCompletedPayload(error=str(exc))
    _publish(DOCUMENTS_LIST_COMPLETED, reply, event.get("correlation_id"))


def handle_document_get_requested(event: dict[str, Any]) -> None:
    payload = validate_payload(DOCUMENT_GET_REQUESTED, event["payload"])
    try:
        doc = get_collection().find_one(
            {"image_id": payload.image_id}, {"_id": 0}  # type: ignore[attr-defined]
        )
        reply = DocumentGetCompletedPayload(
            document=doc,
            error=None if doc is not None else "not_found",
        )
    except Exception as exc:
        log.exception("documents.get failed")
        reply = DocumentGetCompletedPayload(error=str(exc))
    _publish(DOCUMENT_GET_COMPLETED, reply, event.get("correlation_id"))


def register(bus: MessageBus) -> None:
    set_bus(bus)
    bus.subscribe(INFERENCE_COMPLETED, handle_inference_completed)
    bus.subscribe(DOCUMENTS_LIST_REQUESTED, handle_documents_list_requested)
    bus.subscribe(DOCUMENT_GET_REQUESTED, handle_document_get_requested)


if __name__ == "__main__":  # pragma: no cover
    from DB.model_inference_database.messaging import make_default_bus

    bus = make_default_bus()
    register(bus)
    print("[docdb] subscribed; waiting for events…")
    bus.run_forever()

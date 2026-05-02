"""Document DB Service — owns annotation documents in MongoDB.

Annotations are nested and vary between models ({box, contours, tags} for
a detector, flat labels for a classifier).  A document store accepts
whatever shape inference produces — no schema migrations per model.

Writes: event-driven (inference.completed → upsert → document.stored).
Reads:  HTTP (UI clients need request/response).

Subscribes: inference.completed
Publishes:  document.stored
HTTP:
    GET /documents              list document IDs
    GET /documents/{image_id}   retrieve one document
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException

from DB.model_inference_database.events import (
    DOCUMENT_STORED,
    INFERENCE_COMPLETED,
    DocumentStoredPayload,
    make_event,
    validate_payload,
)
from DB.model_inference_database.messaging import MessageBus

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "model_inference")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "documents")

app = FastAPI(title="Document DB Service")
_bus: MessageBus | None = None
_collection: Any = None


def set_bus(bus: MessageBus) -> None:
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


def handle_inference_completed(event: dict[str, Any]) -> None:
    payload = validate_payload(INFERENCE_COMPLETED, event["payload"])

    document = {
        "document_id": f"doc_{payload.image_id}",
        "image_id": payload.image_id,
        "model_name": payload.model_name,
        "annotations": payload.annotations,
    }
    get_collection().replace_one(
        {"image_id": payload.image_id}, document, upsert=True
    )

    if _bus is not None:
        stored = DocumentStoredPayload(
            document_id=document["document_id"],
            image_id=payload.image_id,
            model_name=payload.model_name,
        )
        _bus.publish(
            DOCUMENT_STORED,
            make_event(
                DOCUMENT_STORED,
                stored,
                correlation_id=event.get("correlation_id"),
            ),
        )


@app.get("/documents")
def list_documents() -> dict[str, list[str]]:
    coll = get_collection()
    doc_ids = [
        doc["document_id"]
        for doc in coll.find({}, {"document_id": 1, "_id": 0})
    ]
    return {"document_ids": doc_ids}


@app.get("/documents/{image_id}")
def get_document(image_id: str) -> dict[str, Any]:
    doc = get_collection().find_one({"image_id": image_id}, {"_id": 0})
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


def register(bus: MessageBus) -> None:
    set_bus(bus)
    bus.subscribe(INFERENCE_COMPLETED, handle_inference_completed)


if __name__ == "__main__":  # pragma: no cover
    import threading
    import uvicorn
    from DB.model_inference_database.messaging import make_default_bus

    bus = make_default_bus()
    register(bus)
    threading.Thread(target=bus.run_forever, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8003)

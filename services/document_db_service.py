"""Document DB Service — owns annotation documents (JSON, one per image).

Annotations are nested and vary between models ({box, contours, tags} for
a detector, flat labels for a classifier).  A document store accepts
whatever shape inference produces — no schema migrations per model.

Writes: event-driven (inference.completed → write → document.stored).
Reads:  HTTP (UI clients need request/response).

Subscribes: inference.completed
Publishes:  document.stored
HTTP:
    GET /documents              list document IDs
    GET /documents/{image_id}   retrieve one document
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException

from events import (
    DOCUMENT_STORED,
    INFERENCE_COMPLETED,
    DocumentStoredPayload,
    make_event,
    validate_payload,
)
from messaging import MessageBus

STORAGE_DIR = os.getenv("DOCDB_STORAGE_DIR", "./data/documents")

app = FastAPI(title="Document DB Service")
_bus: MessageBus | None = None


def set_bus(bus: MessageBus) -> None:
    global _bus
    _bus = bus


def _doc_path(image_id: str) -> str:
    return os.path.join(STORAGE_DIR, f"doc_{image_id}.json")


def handle_inference_completed(event: dict[str, Any]) -> None:
    payload = validate_payload(INFERENCE_COMPLETED, event["payload"])

    os.makedirs(STORAGE_DIR, exist_ok=True)
    document = {
        "document_id": f"doc_{payload.image_id}",
        "image_id": payload.image_id,
        "model_name": payload.model_name,
        "annotations": payload.annotations,
    }
    with open(_doc_path(payload.image_id), "w") as f:
        json.dump(document, f, indent=2)

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
    os.makedirs(STORAGE_DIR, exist_ok=True)
    doc_ids = [
        f.replace(".json", "")
        for f in os.listdir(STORAGE_DIR)
        if f.endswith(".json")
    ]
    return {"document_ids": doc_ids}


@app.get("/documents/{image_id}")
def get_document(image_id: str) -> dict[str, Any]:
    path = _doc_path(image_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Document not found")
    with open(path) as f:
        return json.load(f)


def register(bus: MessageBus) -> None:
    set_bus(bus)
    bus.subscribe(INFERENCE_COMPLETED, handle_inference_completed)


if __name__ == "__main__":  # pragma: no cover
    import threading
    import uvicorn
    from messaging import make_default_bus

    bus = make_default_bus()
    register(bus)
    threading.Thread(target=bus.run_forever, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8003)

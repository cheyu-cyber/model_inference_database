"""Web Service — the only HTTP-speaking service in the system.

Browsers can't speak Redis pub/sub, so this service serves as the edge:
HTTP in (from the browser), bus out (to every backend). Every former
inter-service HTTP call has been replaced by a request/reply pair on
the bus, mediated by :class:`RequestTracker`.

This service also owns image-file storage on disk — the bytes never go
on the bus. The browser POSTs to ``/api/upload``; this service writes
the file to ``UPLOAD_STORAGE_DIR`` and publishes ``image.uploaded``.

Browser endpoints
-----------------
    GET  /                              HTML page
    POST /api/upload                    save bytes, publish image.uploaded
    GET  /api/documents                 ↔ documents.list
    GET  /api/documents/{image_id}      ↔ documents.get
    GET  /api/schemas                   ↔ schemas.list
    POST /api/schemas                   ↔ schemas.create
    GET  /api/embeddings/{schema}/{id}  ↔ embeddings.get
    GET  /api/stats                     ↔ stats
    GET  /api/vocabulary                ↔ vocabulary
    POST /api/search                    ↔ search.requested → search.completed
                                          (text → embed.text, image_id → embeddings.get)
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from DB.model_inference_database.events import (
    DOCUMENT_GET_COMPLETED,
    DOCUMENT_GET_REQUESTED,
    DOCUMENTS_LIST_COMPLETED,
    DOCUMENTS_LIST_REQUESTED,
    EMBED_TEXT_COMPLETED,
    EMBED_TEXT_REQUESTED,
    EMBEDDING_GET_COMPLETED,
    EMBEDDING_GET_REQUESTED,
    IMAGE_UPLOADED,
    SCHEMA_CREATE_COMPLETED,
    SCHEMA_CREATE_REQUESTED,
    SCHEMAS_LIST_COMPLETED,
    SCHEMAS_LIST_REQUESTED,
    SEARCH_COMPLETED,
    SEARCH_REQUESTED,
    STATS_COMPLETED,
    STATS_REQUESTED,
    VOCABULARY_COMPLETED,
    VOCABULARY_REQUESTED,
    ImageUploadedPayload,
    make_event,
)
from DB.model_inference_database.messaging import (
    MessageBus,
    RequestTracker,
    RequestTimeoutError,
)

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.normpath(os.path.join(HERE, "..", "web", "index.html"))

UPLOAD_STORAGE_DIR = os.getenv("UPLOAD_STORAGE_DIR", "./data/uploads")
REQUEST_TIMEOUT = float(os.getenv("BUS_REQUEST_TIMEOUT", "5.0"))

app = FastAPI(title="Semantic Image DB — Web UI")


_bus: MessageBus | None = None
_tracker: RequestTracker | None = None


# Every reply topic Web listens for. Pre-subscribed at startup so the
# bus listener thread already has them registered before the first
# request — avoids a redis-py race when subscribing mid-listen.
_REPLY_TOPICS = (
    DOCUMENTS_LIST_COMPLETED,
    DOCUMENT_GET_COMPLETED,
    EMBEDDING_GET_COMPLETED,
    SCHEMAS_LIST_COMPLETED,
    SCHEMA_CREATE_COMPLETED,
    STATS_COMPLETED,
    VOCABULARY_COMPLETED,
    EMBED_TEXT_COMPLETED,
    SEARCH_COMPLETED,
)


def set_bus(bus: MessageBus | None) -> None:
    """Wire (or unwire) the bus + RequestTracker for this service."""
    global _bus, _tracker
    _bus = bus
    if bus is None:
        _tracker = None
        return
    _tracker = RequestTracker(bus)
    _tracker.subscribe_replies(*_REPLY_TOPICS)


def get_bus() -> MessageBus:
    if _bus is None:
        raise RuntimeError("Web service bus is not configured")
    return _bus


def _request(req_topic: str, reply_topic: str, payload: dict[str, Any]) -> dict[str, Any]:
    if _tracker is None:
        raise RuntimeError("Web service bus is not configured")
    try:
        reply = _tracker.request(req_topic, reply_topic, payload, timeout=REQUEST_TIMEOUT)
    except RequestTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    if reply.get("error"):
        if reply["error"] == "not_found":
            raise HTTPException(status_code=404, detail="Not found")
        raise HTTPException(status_code=400, detail=reply["error"])
    return reply


# ── HTML UI ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    with open(INDEX_HTML) as f:
        return HTMLResponse(f.read())


# ── Upload (HTTP in → bus out) ───────────────────────────────────────

@app.post("/api/upload")
def api_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Save the file to disk and publish ``image.uploaded``.

    Bytes never traverse the bus — only the saved file path does. The
    rest of the pipeline (Inference → DocDB / Embedding → Vector) reacts
    asynchronously.
    """
    os.makedirs(UPLOAD_STORAGE_DIR, exist_ok=True)

    image_id = str(uuid.uuid4())
    suffix = "jpg"
    if file.filename and "." in file.filename:
        suffix = file.filename.rsplit(".", 1)[-1]
    dest_path = os.path.join(UPLOAD_STORAGE_DIR, f"{image_id}.{suffix}")

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = os.path.getsize(dest_path)
    mime = file.content_type or "image/jpeg"

    payload = ImageUploadedPayload(
        image_id=image_id,
        file_path=dest_path,
        file_size_bytes=file_size,
        mime_type=mime,
    )
    get_bus().publish(IMAGE_UPLOADED, make_event(IMAGE_UPLOADED, payload))

    return {
        "image_id": image_id,
        "file_path": dest_path,
        "file_size_bytes": file_size,
        "mime_type": mime,
    }


# ── Documents ────────────────────────────────────────────────────────

@app.get("/api/documents")
def api_list_documents() -> dict[str, Any]:
    reply = _request(DOCUMENTS_LIST_REQUESTED, DOCUMENTS_LIST_COMPLETED, {})
    return {"document_ids": reply.get("document_ids", [])}


@app.get("/api/documents/{image_id}")
def api_get_document(image_id: str) -> dict[str, Any]:
    reply = _request(
        DOCUMENT_GET_REQUESTED, DOCUMENT_GET_COMPLETED, {"image_id": image_id}
    )
    doc = reply.get("document")
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ── Schemas / Vector index ───────────────────────────────────────────

@app.get("/api/schemas")
def api_list_schemas() -> dict[str, Any]:
    reply = _request(SCHEMAS_LIST_REQUESTED, SCHEMAS_LIST_COMPLETED, {})
    return {"schemas": reply.get("schemas", [])}


class SchemaBody(BaseModel):
    name: str
    dimensions: int


@app.post("/api/schemas")
def api_create_schema(body: SchemaBody) -> dict[str, Any]:
    reply = _request(
        SCHEMA_CREATE_REQUESTED,
        SCHEMA_CREATE_COMPLETED,
        body.model_dump(),
    )
    return {"name": reply["name"], "dimensions": reply["dimensions"]}


@app.get("/api/embeddings/{schema_name}/{image_id}")
def api_get_embedding(schema_name: str, image_id: str) -> dict[str, Any]:
    reply = _request(
        EMBEDDING_GET_REQUESTED,
        EMBEDDING_GET_COMPLETED,
        {"schema_name": schema_name, "image_id": image_id},
    )
    return {
        "image_id": reply["image_id"],
        "schema_name": reply["schema_name"],
        "vector": reply["vector"],
        "dimensions": reply["dimensions"],
    }


@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    reply = _request(STATS_REQUESTED, STATS_COMPLETED, {})
    return {
        "schemas": reply.get("schemas", 0),
        "total_embeddings": reply.get("total_embeddings", 0),
        "by_schema": reply.get("by_schema", {}),
    }


# ── Vocabulary ───────────────────────────────────────────────────────

@app.get("/api/vocabulary")
def api_vocabulary() -> dict[str, Any]:
    reply = _request(VOCABULARY_REQUESTED, VOCABULARY_COMPLETED, {})
    return {
        "dimensions": reply.get("dimensions"),
        "schema_name": reply.get("schema_name"),
        "categories": reply.get("categories", []),
    }


# ── Search ───────────────────────────────────────────────────────────

class SearchBody(BaseModel):
    vector: list[float] | None = None
    query_text: str | None = None
    image_id: str | None = None
    schema_name: str = "semantic"
    top_k: int = 5


@app.post("/api/search")
def api_search(body: SearchBody) -> dict[str, Any]:
    """Search by text, raw vector, or image_id.

    The search itself is a request/reply over ``search.requested`` /
    ``search.completed``: Embedding receives the query, vectorizes it
    (if needed) and forwards as ``vector.search.requested``; Vector
    runs FAISS and emits ``search.completed``. We resolve the query
    vector here only for the ``image_id`` and raw-``vector`` paths,
    since those don't need the embedding hop.
    """
    if body.query_text:
        # Free-form text — let the search pipeline do the vectorization.
        reply = _request(
            SEARCH_REQUESTED,
            SEARCH_COMPLETED,
            {
                "query_id": str(uuid.uuid4()),
                "schema_name": body.schema_name,
                "query_text": body.query_text,
                "top_k": body.top_k,
            },
        )
        # The query_vector field is for UI display only — recover it via
        # the embedding service so the response shape matches the old API.
        emb = _request(
            EMBED_TEXT_REQUESTED, EMBED_TEXT_COMPLETED, {"text": body.query_text}
        )
        return {"query_vector": emb["vector"], "results": reply.get("results", [])}

    if body.vector is not None:
        vec = body.vector
    elif body.image_id:
        emb = _request(
            EMBEDDING_GET_REQUESTED,
            EMBEDDING_GET_COMPLETED,
            {"schema_name": body.schema_name, "image_id": body.image_id},
        )
        vec = emb["vector"]
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide query_text, vector, or image_id",
        )

    reply = _request(
        SEARCH_REQUESTED,
        SEARCH_COMPLETED,
        {
            "query_id": str(uuid.uuid4()),
            "schema_name": body.schema_name,
            "vector": vec,
            "top_k": body.top_k,
        },
    )
    return {"query_vector": vec, "results": reply.get("results", [])}


if __name__ == "__main__":  # pragma: no cover
    import threading
    import uvicorn
    from DB.model_inference_database.messaging import make_default_bus

    bus = make_default_bus()
    set_bus(bus)
    threading.Thread(target=bus.run_forever, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)

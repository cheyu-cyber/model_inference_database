"""Web UI Service — all-in-one gateway.

This is the single entry point for running the full system.  It:

1. Creates a message bus (InMemory for standalone, Redis for distributed).
2. Registers every pipeline service (Upload, Inference, DocDB, Embedding).
3. Serves a static HTML UI at ``/``.
4. Exposes REST endpoints under ``/api/*`` that directly call into the
   service modules — no inter-service HTTP proxying.

The gateway owns no data itself.  It delegates writes through the bus
(Upload → image.uploaded → Inference → inference.completed → DocDB +
Embedding) and reads through the services' internal state.

Start the system
----------------
::

    python services/web_service.py          # InMemoryBus, port 8000
    BUS_BACKEND=redis python services/web_service.py   # RedisBus
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from events import (
    SEARCH_COMPLETED,
    SEARCH_REQUESTED,
    SearchRequestedPayload,
    make_event,
)
from messaging import InMemoryBus, MessageBus, make_default_bus
import services.upload_service as upload_mod
import services.inference_service as inference_mod
import services.document_db_service as docdb_mod
import services.embedding_service as embedding_mod

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.normpath(os.path.join(HERE, "..", "web", "index.html"))

_bus: MessageBus | None = None


@asynccontextmanager
async def lifespan(_app):
    if _bus is None:
        init_app()
    yield


app = FastAPI(title="Semantic Image DB", lifespan=lifespan)


def _get_bus() -> MessageBus:
    if _bus is None:
        raise RuntimeError("Web service bus not initialised")
    return _bus


def init_app(bus: MessageBus | None = None) -> None:
    """Wire every service to ``bus`` and store a reference for the API."""
    global _bus
    if bus is None:
        bus = make_default_bus()
    _bus = bus
    upload_mod.register(bus)
    inference_mod.register(bus)
    docdb_mod.register(bus)
    embedding_mod.register(bus)


# ── HTML UI ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    with open(INDEX_HTML) as f:
        return HTMLResponse(f.read())


# ── Upload ───────────────────────────────────────────────────────────

@app.post("/api/upload")
def api_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Save a file and publish image.uploaded (triggers full pipeline)."""
    from fastapi.testclient import TestClient

    client = TestClient(upload_mod.app)
    content = file.file.read()
    resp = client.post(
        "/upload",
        files={"file": (file.filename or "upload.jpg", content, file.content_type or "image/jpeg")},
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


# ── Documents (read from docdb module) ───────────────────────────────

@app.get("/api/documents")
def api_list_documents() -> dict[str, Any]:
    from fastapi.testclient import TestClient

    resp = TestClient(docdb_mod.app).get("/documents")
    return resp.json()


@app.get("/api/documents/{image_id}")
def api_get_document(image_id: str) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    resp = TestClient(docdb_mod.app).get(f"/documents/{image_id}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Document not found")
    return resp.json()


# ── Embeddings / Schemas ─────────────────────────────────────────────

@app.get("/api/schemas")
def api_list_schemas() -> dict[str, Any]:
    from fastapi.testclient import TestClient

    return TestClient(embedding_mod.app).get("/schemas").json()


class SchemaBody(BaseModel):
    name: str
    dimensions: int
    metric: str = "cosine"


@app.post("/api/schemas")
def api_create_schema(body: SchemaBody) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    resp = TestClient(embedding_mod.app).post("/schemas", json=body.model_dump())
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.get("/api/embeddings/{schema_name}/{image_id}")
def api_get_embedding(schema_name: str, image_id: str) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    resp = TestClient(embedding_mod.app).get(f"/embeddings/{schema_name}/{image_id}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Embedding not found")
    return resp.json()


@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    from fastapi.testclient import TestClient

    return TestClient(embedding_mod.app).get("/stats").json()


# ── Search ───────────────────────────────────────────────────────────

class SearchBody(BaseModel):
    vector: list[float] | None = None
    image_id: str | None = None
    schema_name: str = "default"
    top_k: int = 5


@app.post("/api/search")
def api_search(body: SearchBody) -> dict[str, Any]:
    """Search by raw vector or by image_id (look up its stored vector)."""
    from fastapi.testclient import TestClient

    query_vector = body.vector
    if query_vector is None and body.image_id:
        emb_resp = TestClient(embedding_mod.app).get(
            f"/embeddings/{body.schema_name}/{body.image_id}"
        )
        if emb_resp.status_code != 200:
            raise HTTPException(status_code=404, detail="Embedding not found for that image")
        query_vector = emb_resp.json()["vector"]

    if query_vector is None:
        raise HTTPException(status_code=400, detail="Provide vector or image_id")

    resp = TestClient(embedding_mod.app).post(
        "/search/similar",
        json={"vector": query_vector, "top_k": body.top_k, "schema_name": body.schema_name},
    )
    return resp.json()


# ── Startup ──────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup() -> None:
    if _bus is None:
        init_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    init_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

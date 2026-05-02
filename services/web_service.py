"""Web UI Service — HTTP gateway.

The browser talks to this service; this service talks to every other
service over HTTP (``httpx``).  No other service is imported as a Python
module — each is reached at its configured URL.  Inter-service write-path
coordination still happens over Redis; this service is a *read/upload*
gateway for the UI plus the publisher for ``search.requested``.

Service map
-----------
    Embedding Service  — vocabulary, text/tags → vector
    Vector Service     — FAISS index, schemas, similarity search
    Document DB        — annotation documents
    Upload Service     — image bytes

Endpoints
---------
    GET  /                              HTML page
    POST /api/upload                    → POST {UPLOAD_URL}/upload
    GET  /api/documents                 → GET  {DOCDB_URL}/documents
    GET  /api/documents/{image_id}      → GET  {DOCDB_URL}/documents/{id}
    GET  /api/schemas                   → GET  {VECTOR_URL}/schemas
    POST /api/schemas                   → POST {VECTOR_URL}/schemas
    GET  /api/embeddings/{schema}/{id}  → GET  {VECTOR_URL}/embeddings/…
    GET  /api/stats                     → GET  {VECTOR_URL}/stats
    GET  /api/vocabulary                → GET  {EMBEDDING_URL}/vocabulary
    POST /api/search                    → chain: embedding (text→vec) → vector (search)
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.normpath(os.path.join(HERE, "..", "web", "index.html"))

UPLOAD_URL = os.getenv("UPLOAD_SERVICE_URL", "http://localhost:8001")
DOCDB_URL = os.getenv("DOCDB_SERVICE_URL", "http://localhost:8003")
EMBEDDING_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8004")
VECTOR_URL = os.getenv("VECTOR_SERVICE_URL", "http://localhost:8005")

app = FastAPI(title="Semantic Image DB — Web UI")


# The HTTP client is injectable so tests can route calls to in-process
# ASGI apps without opening real sockets.
_client: httpx.Client | None = None


def set_client(client: httpx.Client | None) -> None:
    """Override the httpx client used to reach downstream services."""
    global _client
    _client = client


def _get_client() -> httpx.Client:
    return _client if _client is not None else httpx.Client(timeout=10.0)


def _forward(resp: httpx.Response) -> dict[str, Any]:
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


# ── HTML UI ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    with open(INDEX_HTML) as f:
        return HTMLResponse(f.read())


# ── Upload ───────────────────────────────────────────────────────────

@app.post("/api/upload")
def api_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    content = file.file.read()
    resp = _get_client().post(
        f"{UPLOAD_URL}/upload",
        files={"file": (file.filename or "upload.jpg", content, file.content_type or "image/jpeg")},
    )
    return _forward(resp)


# ── Documents ────────────────────────────────────────────────────────

@app.get("/api/documents")
def api_list_documents() -> dict[str, Any]:
    return _forward(_get_client().get(f"{DOCDB_URL}/documents"))


@app.get("/api/documents/{image_id}")
def api_get_document(image_id: str) -> dict[str, Any]:
    return _forward(_get_client().get(f"{DOCDB_URL}/documents/{image_id}"))


# ── Schemas / Vector index (Vector Service) ──────────────────────────

@app.get("/api/schemas")
def api_list_schemas() -> dict[str, Any]:
    return _forward(_get_client().get(f"{VECTOR_URL}/schemas"))


class SchemaBody(BaseModel):
    name: str
    dimensions: int


@app.post("/api/schemas")
def api_create_schema(body: SchemaBody) -> dict[str, Any]:
    return _forward(_get_client().post(
        f"{VECTOR_URL}/schemas", json=body.model_dump()
    ))


@app.get("/api/embeddings/{schema_name}/{image_id}")
def api_get_embedding(schema_name: str, image_id: str) -> dict[str, Any]:
    return _forward(_get_client().get(
        f"{VECTOR_URL}/embeddings/{schema_name}/{image_id}"
    ))


@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    return _forward(_get_client().get(f"{VECTOR_URL}/stats"))


# ── Vocabulary (Embedding Service) ───────────────────────────────────

@app.get("/api/vocabulary")
def api_vocabulary() -> dict[str, Any]:
    return _forward(_get_client().get(f"{EMBEDDING_URL}/vocabulary"))


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

    Text queries are vectorized by the Embedding Service; the resulting
    vector is then searched against the Vector Service's FAISS index.
    Vector / image_id queries skip the embedding hop.
    """
    if body.query_text:
        emb = _forward(_get_client().post(
            f"{EMBEDDING_URL}/embed/text",
            json={"text": body.query_text},
        ))
        vec = emb["vector"]
    elif body.vector is not None:
        vec = body.vector
    elif body.image_id:
        emb = _forward(_get_client().get(
            f"{VECTOR_URL}/embeddings/{body.schema_name}/{body.image_id}"
        ))
        vec = emb["vector"]
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide query_text, vector, or image_id",
        )

    return _forward(_get_client().post(
        f"{VECTOR_URL}/search/similar",
        json={"vector": vec, "top_k": body.top_k, "schema_name": body.schema_name},
    ))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

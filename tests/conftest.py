"""Shared test fixtures for the messaging-based services.

Bus wiring
----------
Every service accepts a ``MessageBus`` at ``register(bus)`` time.  Tests
use :class:`InMemoryBus` so publishes dispatch synchronously and
assertions run immediately after each ``publish``.

HTTP wiring
-----------
``web_service`` calls other services over HTTP with ``httpx``.  In tests
we inject an ``httpx.Client`` whose transport dispatches to each service's
in-process ``TestClient`` — no sockets are opened, but the web service's
code path is identical to production.
"""

from __future__ import annotations

import os
import sys

import httpx
import mongomock
import pytest

# Make the project root importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from DB.model_inference_database.messaging import InMemoryBus, EventGenerator
import DB.model_inference_database.services.upload_service as upload_mod
import DB.model_inference_database.services.inference_service as inference_mod
import DB.model_inference_database.services.document_db_service as docdb_mod
import DB.model_inference_database.services.embedding_service as embedding_mod
import DB.model_inference_database.services.vector_service as vector_mod
import DB.model_inference_database.services.web_service as web_mod


@pytest.fixture
def bus() -> InMemoryBus:
    """Fresh in-memory bus per test."""
    return InMemoryBus()


@pytest.fixture
def generator(bus: InMemoryBus) -> EventGenerator:
    return EventGenerator(bus, seed=1234)


@pytest.fixture(autouse=True)
def _reset_service_state(tmp_path, monkeypatch):
    """Reset all service module-level state and point at per-test dirs."""
    monkeypatch.setattr(upload_mod, "STORAGE_DIR", str(tmp_path / "uploads"))
    docdb_mod.set_collection(mongomock.MongoClient()["test"]["documents"])
    vector_mod.reset_state()
    upload_mod.set_bus(None)  # type: ignore[arg-type]
    inference_mod.set_bus(None)  # type: ignore[arg-type]
    docdb_mod.set_bus(None)  # type: ignore[arg-type]
    embedding_mod.set_bus(None)
    vector_mod.set_bus(None)
    web_mod.set_client(None)
    yield
    vector_mod.reset_state()
    web_mod.set_client(None)
    docdb_mod.set_collection(None)


@pytest.fixture
def wired_bus(bus: InMemoryBus) -> InMemoryBus:
    """Bus with every pipeline service subscribed — one-line test wiring."""
    upload_mod.register(bus)
    inference_mod.register(bus)
    docdb_mod.register(bus)
    embedding_mod.register(bus)
    vector_mod.register(bus)
    return bus


@pytest.fixture
def upload_client(bus: InMemoryBus) -> TestClient:
    upload_mod.register(bus)
    return TestClient(upload_mod.app)


@pytest.fixture
def docdb_client(bus: InMemoryBus) -> TestClient:
    docdb_mod.register(bus)
    return TestClient(docdb_mod.app)


@pytest.fixture
def embedding_client(bus: InMemoryBus) -> TestClient:
    """Embedding + Vector both subscribed — embedding alone produces no
    indexable side-effect, since vector ownership lives next door."""
    embedding_mod.register(bus)
    vector_mod.register(bus)
    return TestClient(embedding_mod.app)


@pytest.fixture
def vector_client(bus: InMemoryBus) -> TestClient:
    """Vector Service alone — useful for tests that drive `vector.computed`
    directly without going through the embedding hop."""
    vector_mod.register(bus)
    return TestClient(vector_mod.app)


@pytest.fixture
def http_gateway_client(wired_bus: InMemoryBus) -> TestClient:
    """TestClient for web_service with httpx routed to in-process sub-apps.

    Every call the web service makes to UPLOAD_URL / DOCDB_URL /
    EMBEDDING_URL / VECTOR_URL is dispatched to the matching FastAPI app
    via a ``httpx.MockTransport`` — so tests exercise the real HTTP code
    path without any actual sockets.
    """
    subclients = {
        "localhost:8001": TestClient(upload_mod.app),
        "localhost:8003": TestClient(docdb_mod.app),
        "localhost:8004": TestClient(embedding_mod.app),
        "localhost:8005": TestClient(vector_mod.app),
    }

    def _dispatch(request: httpx.Request) -> httpx.Response:
        key = f"{request.url.host}:{request.url.port}"
        sub = subclients.get(key)
        if sub is None:
            return httpx.Response(502, text=f"no route for {key}")
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "connection")
        }
        resp = sub.request(
            request.method,
            request.url.path,
            params=dict(request.url.params),
            content=request.content or None,
            headers=headers,
        )
        return httpx.Response(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            content=resp.content,
        )

    web_mod.set_client(httpx.Client(transport=httpx.MockTransport(_dispatch)))
    return TestClient(web_mod.app)

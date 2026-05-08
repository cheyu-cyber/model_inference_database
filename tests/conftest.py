"""Shared test fixtures for the pure-pub/sub services.

Bus wiring
----------
Every backend service is a pure subscriber daemon: ``register(bus)``
subscribes its handlers and that's the entire entry point. Tests use
:class:`InMemoryBus` so publishes dispatch synchronously and assertions
run immediately.

Web service
-----------
Web is the only service with HTTP. It still uses FastAPI's
``TestClient``, but every backend call now flows over the same
``InMemoryBus``: Web publishes a ``*.requested`` event, the matching
backend handles it inline (synchronous bus), publishes the
``*.completed`` event, and the :class:`RequestTracker`'s future
resolves before ``publish`` returns. No sockets, no MockTransport, no
parallel ASGI apps.
"""

from __future__ import annotations

import os
import sys

import mongomock
import pytest

# Make the project root importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from DB.model_inference_database.messaging import InMemoryBus, EventGenerator
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
    monkeypatch.setattr(web_mod, "UPLOAD_STORAGE_DIR", str(tmp_path / "uploads"))
    docdb_mod.set_collection(mongomock.MongoClient()["test"]["documents"])
    vector_mod.reset_state()
    inference_mod.set_bus(None)  # type: ignore[arg-type]
    docdb_mod.set_bus(None)
    embedding_mod.set_bus(None)
    vector_mod.set_bus(None)
    web_mod.set_bus(None)
    yield
    vector_mod.reset_state()
    web_mod.set_bus(None)
    docdb_mod.set_collection(None)


def _wire_pipeline(bus: InMemoryBus) -> None:
    """Subscribe every backend to the bus."""
    inference_mod.register(bus)
    docdb_mod.register(bus)
    embedding_mod.register(bus)
    vector_mod.register(bus)


@pytest.fixture
def wired_bus(bus: InMemoryBus) -> InMemoryBus:
    """Bus with every pipeline service subscribed — one-line test wiring."""
    _wire_pipeline(bus)
    return bus


@pytest.fixture
def web_client(wired_bus: InMemoryBus) -> TestClient:
    """TestClient for web_service with every backend subscribed to the bus.

    Web's HTTP handlers issue bus request/reply calls via RequestTracker;
    because InMemoryBus is synchronous, the round-trip completes inline
    and the HTTP response is ready by the time TestClient returns.
    """
    web_mod.set_bus(wired_bus)
    return TestClient(web_mod.app)

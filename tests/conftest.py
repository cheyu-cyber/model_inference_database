"""Shared test fixtures for the messaging-based services.

Every service accepts a ``MessageBus`` at ``register(bus)`` time.  The
tests use :class:`InMemoryBus` so publishes dispatch synchronously and
assertions can run immediately after each ``publish`` call.
"""

from __future__ import annotations

import os
import sys

import pytest

# Make the project root importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from messaging import InMemoryBus, EventGenerator
import services.upload_service as upload_mod
import services.inference_service as inference_mod
import services.document_db_service as docdb_mod
import services.embedding_service as embedding_mod


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
    monkeypatch.setattr(docdb_mod, "STORAGE_DIR", str(tmp_path / "documents"))
    embedding_mod.reset_state()
    upload_mod.set_bus(None)  # type: ignore[arg-type]
    inference_mod.set_bus(None)  # type: ignore[arg-type]
    docdb_mod.set_bus(None)  # type: ignore[arg-type]
    embedding_mod.set_bus(None)  # type: ignore[arg-type]
    yield
    embedding_mod.reset_state()


@pytest.fixture
def wired_bus(bus: InMemoryBus) -> InMemoryBus:
    """Bus with every pipeline service subscribed — one-line test wiring."""
    upload_mod.register(bus)
    inference_mod.register(bus)
    docdb_mod.register(bus)
    embedding_mod.register(bus)
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
    embedding_mod.register(bus)
    return TestClient(embedding_mod.app)

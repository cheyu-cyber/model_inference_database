"""Deterministic event generator for tests.

Tests need three related capabilities that bare ``publish()`` calls make
awkward:

1. **Contracts** — payloads must match the Pydantic schemas in
   ``events.schemas``.  ``EventGenerator`` builds payloads via the
   schemas so any test that uses the generator is implicitly checking
   that the schema still accepts what services expect.

2. **Deterministic replay** — ``EventGenerator(seed=n)`` seeds a private
   ``random.Random``.  Re-running a test produces byte-identical
   embedding vectors and annotation values, so failures are reproducible.

3. **Fault injection** — the generator exposes ``inject_fault`` which
   forwards to :meth:`InMemoryBus.inject_fault`; tests can simulate a
   broker error on a particular topic without touching private bus state.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from events import (
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    SEARCH_REQUESTED,
    ImageUploadedPayload,
    InferenceCompletedPayload,
    SearchRequestedPayload,
    make_event,
)
from .bus import InMemoryBus, MessageBus


class EventGenerator:
    """Create and emit well-formed events against a :class:`MessageBus`."""

    def __init__(self, bus: MessageBus, seed: int = 0) -> None:
        self.bus = bus
        self.random = random.Random(seed)

    # -- building blocks -------------------------------------------------
    def new_image_id(self) -> str:
        # Use the seeded RNG so tests are reproducible.
        return f"img-{self.random.randrange(10**9):09d}"

    def random_vector(self, dim: int = 128) -> list[float]:
        return [round(self.random.gauss(0, 1), 4) for _ in range(dim)]

    # -- event emitters --------------------------------------------------
    def emit_image_uploaded(
        self,
        image_id: str | None = None,
        file_path: str = "/tmp/fake.jpg",
        mime_type: str = "image/jpeg",
        file_size_bytes: int = 1024,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        payload = ImageUploadedPayload(
            image_id=image_id or self.new_image_id(),
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            mime_type=mime_type,
        )
        event = make_event(IMAGE_UPLOADED, payload, correlation_id)
        self.bus.publish(IMAGE_UPLOADED, event)
        return event

    def emit_inference_completed(
        self,
        image_id: str,
        model_name: str = "stub-classifier-v1",
        annotations: dict[str, Any] | None = None,
        embedding_vector: list[float] | None = None,
        schema_name: str = "default",
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        payload = InferenceCompletedPayload(
            image_id=image_id,
            model_name=model_name,
            annotations=annotations or self._default_annotations(),
            embedding_vector=embedding_vector or self.random_vector(),
            schema_name=schema_name,
        )
        event = make_event(INFERENCE_COMPLETED, payload, correlation_id)
        self.bus.publish(INFERENCE_COMPLETED, event)
        return event

    def emit_search_requested(
        self,
        vector: list[float] | None = None,
        top_k: int = 5,
        schema_name: str = "default",
        query_id: str | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        payload = SearchRequestedPayload(
            query_id=query_id or f"q-{uuid.uuid4().hex[:8]}",
            vector=vector or self.random_vector(),
            top_k=top_k,
            schema_name=schema_name,
        )
        event = make_event(SEARCH_REQUESTED, payload, correlation_id)
        self.bus.publish(SEARCH_REQUESTED, event)
        return event

    # -- replay / fault injection ---------------------------------------
    def replay(self, events: list[tuple[str, dict[str, Any]]]) -> None:
        """Re-publish a previously captured (topic, envelope) sequence."""
        for topic, envelope in events:
            self.bus.publish(topic, envelope)

    def inject_fault(self, topic: str, exc: Exception) -> None:
        if not isinstance(self.bus, InMemoryBus):
            raise TypeError("Fault injection is only supported on InMemoryBus")
        self.bus.inject_fault(topic, exc)

    # -- internal --------------------------------------------------------
    def _default_annotations(self) -> dict[str, Any]:
        x, y = self.random.randint(0, 80), self.random.randint(0, 80)
        return {
            "objects": [
                {
                    "box": [x, y, self.random.randint(10, 40), self.random.randint(10, 40)],
                    "contours": [[x + self.random.randint(-2, 2), y + self.random.randint(-2, 2)] for _ in range(5)],
                    "tags": self.random.sample(["cell", "mitotic", "healthy", "abnormal"], k=2),
                    "confidence": round(self.random.uniform(0.7, 1.0), 3),
                }
            ],
        }

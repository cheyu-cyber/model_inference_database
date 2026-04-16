"""Inference Service — stateless transform, no data of its own.

Reacts to ``image.uploaded``, runs a stub model that produces nested
{box, contours, tags} annotations and a fixed-length embedding vector,
then publishes ``inference.completed``.

Subscribes: image.uploaded
Publishes:  inference.completed
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events import (
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    ImageUploadedPayload,
    InferenceCompletedPayload,
    make_event,
    validate_payload,
)
from messaging import MessageBus

MODEL_NAME = os.getenv("MODEL_NAME", "stub-classifier-v1")
EMBEDDING_DIM = 128
SCHEMA_NAME = "default"

_bus: MessageBus | None = None
_rng = random.Random()  # module-level so tests can seed it


def set_bus(bus: MessageBus) -> None:
    global _bus
    _bus = bus


def _run_inference() -> dict[str, Any]:
    """Stub model: nested {box, contours, tags} per object + a vector."""
    num_objects = _rng.randint(1, 3)
    objects = []
    for _ in range(num_objects):
        x, y = _rng.randint(0, 80), _rng.randint(0, 80)
        w, h = _rng.randint(10, 40), _rng.randint(10, 40)
        objects.append({
            "box": [x, y, w, h],
            "contours": [[x + _rng.randint(-2, 2), y + _rng.randint(-2, 2)] for _ in range(5)],
            "tags": _rng.sample(["cell", "mitotic", "healthy", "abnormal", "artifact"], k=_rng.randint(1, 3)),
            "confidence": round(_rng.uniform(0.7, 1.0), 3),
        })
    annotations = {"objects": objects}
    vector = [round(_rng.gauss(0, 1), 4) for _ in range(EMBEDDING_DIM)]
    return {"annotations": annotations, "embedding_vector": vector}


def handle_image_uploaded(event: dict[str, Any]) -> None:
    """Bus handler: validate envelope, run model, publish result."""
    payload = validate_payload(IMAGE_UPLOADED, event["payload"])
    assert isinstance(payload, ImageUploadedPayload)
    result = _run_inference()

    completed = InferenceCompletedPayload(
        image_id=payload.image_id,
        model_name=MODEL_NAME,
        annotations=result["annotations"],
        embedding_vector=result["embedding_vector"],
        schema_name=SCHEMA_NAME,
    )
    if _bus is None:
        raise RuntimeError("Inference service bus is not configured")
    _bus.publish(
        INFERENCE_COMPLETED,
        make_event(
            INFERENCE_COMPLETED,
            completed,
            correlation_id=event.get("correlation_id"),
        ),
    )


def register(bus: MessageBus) -> None:
    set_bus(bus)
    bus.subscribe(IMAGE_UPLOADED, handle_image_uploaded)


if __name__ == "__main__":  # pragma: no cover
    from messaging import make_default_bus

    bus = make_default_bus()
    register(bus)
    print(f"[inference] subscribed to {IMAGE_UPLOADED}, waiting for events…")
    bus.run_forever()

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

from DB.model_inference_database.events import (
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
    ImageUploadedPayload,
    InferenceCompletedPayload,
    make_event,
    validate_payload,
)
from DB.model_inference_database.messaging import MessageBus

MODEL_NAME = os.getenv("MODEL_NAME", "stub-yolo-v1")
SCHEMA_NAME = "semantic"

_bus: MessageBus | None = None
_rng = random.Random()  # module-level so tests can seed it

# Candidate tags the stub picks from.  These overlap with the embedding
# service's vocabulary so semantic search works end-to-end.  When a real
# YOLO model is dropped in, its class labels replace this pool.
_TAG_POOL = (
    "person", "pedestrian", "man", "woman",
    "car", "bus", "truck",
    "bicycle", "motorcycle",
    "dog", "cat", "bird",
    "traffic light", "stop sign",
    "tree", "building",
)


def set_bus(bus: MessageBus) -> None:
    global _bus
    _bus = bus


def _run_inference() -> dict[str, Any]:
    """Stub YOLO-style detection: per-object {box, contours, tags}.

    The real model can replace this function wholesale — the only
    contract downstream services rely on is the annotations shape.
    """
    num_objects = _rng.randint(1, 3)
    objects = []
    for _ in range(num_objects):
        x, y = _rng.randint(0, 80), _rng.randint(0, 80)
        w, h = _rng.randint(10, 40), _rng.randint(10, 40)
        objects.append({
            "box": [x, y, w, h],
            "contours": [[x + _rng.randint(-2, 2), y + _rng.randint(-2, 2)] for _ in range(5)],
            "tags": _rng.sample(_TAG_POOL, k=_rng.randint(1, 3)),
            "confidence": round(_rng.uniform(0.7, 1.0), 3),
        })
    return {"annotations": {"objects": objects}}


def handle_image_uploaded(event: dict[str, Any]) -> None:
    """Bus handler: validate envelope, run model, publish result."""
    payload = validate_payload(IMAGE_UPLOADED, event["payload"])
    assert isinstance(payload, ImageUploadedPayload)
    result = _run_inference()

    completed = InferenceCompletedPayload(
        image_id=payload.image_id,
        model_name=MODEL_NAME,
        annotations=result["annotations"],
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
    from DB.model_inference_database.messaging import make_default_bus

    bus = make_default_bus()
    register(bus)
    print(f"[inference] subscribed to {IMAGE_UPLOADED}, waiting for events…")
    bus.run_forever()

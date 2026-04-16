"""Inference Service.

Owns: nothing — stateless transform.

Subscribes: image.uploaded
Publishes:  inference.completed

Workflow:
    image.uploaded  →  run_inference()  →  inference.completed

The model here is a deterministic stub; what matters for the assignment is
that each run produces a *nested, variable-shape* annotation dict (which
justifies a document store downstream) and a fixed-length embedding
vector (which the Embedding Service indexes).
"""

from __future__ import annotations

import os
import random
from typing import Any

from events import (
    IMAGE_UPLOADED,
    INFERENCE_COMPLETED,
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


def _run_inference(image_id: str) -> dict[str, Any]:
    """Stub model: returns nested annotations + a random vector."""
    annotations = {
        "objects": [
            {
                "label": "cell",
                "confidence": round(_rng.uniform(0.7, 1.0), 3),
                "bbox": [_rng.randint(0, 100) for _ in range(4)],
                "attributes": {
                    "mitotic": _rng.choice([True, False]),
                    "area_px": _rng.randint(200, 5000),
                },
            }
        ],
        "classification": {
            "top_label": _rng.choice(["healthy", "abnormal", "artifact"]),
            "scores": {"healthy": 0.6, "abnormal": 0.3, "artifact": 0.1},
        },
    }
    vector = [round(_rng.gauss(0, 1), 4) for _ in range(EMBEDDING_DIM)]
    return {"annotations": annotations, "embedding_vector": vector}


def handle_image_uploaded(event: dict[str, Any]) -> None:
    """Bus handler: validate envelope, run model, publish result."""
    payload = validate_payload(IMAGE_UPLOADED, event["payload"])
    result = _run_inference(payload.image_id)

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

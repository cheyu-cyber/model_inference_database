"""Tests for the deterministic event generator."""

import pytest

from DB.model_inference_database.events import IMAGE_UPLOADED
from DB.model_inference_database.events.schemas import validate_payload
from DB.model_inference_database.messaging import EventGenerator, InMemoryBus


def test_seeded_runs_are_reproducible():
    """Payloads are seeded; envelope metadata (event_id, timestamp) is not."""
    bus_a = InMemoryBus()
    bus_b = InMemoryBus()
    gen_a = EventGenerator(bus_a, seed=42)
    gen_b = EventGenerator(bus_b, seed=42)

    gen_a.emit_inference_completed(image_id="img-1")
    gen_b.emit_inference_completed(image_id="img-1")

    payloads_a = [msg["payload"] for _, msg in bus_a.published]
    payloads_b = [msg["payload"] for _, msg in bus_b.published]
    assert payloads_a == payloads_b


def test_different_seeds_produce_different_vectors():
    gen_a = EventGenerator(InMemoryBus(), seed=1)
    gen_b = EventGenerator(InMemoryBus(), seed=2)
    assert gen_a.random_vector(8) != gen_b.random_vector(8)


def test_generated_events_match_schemas():
    bus = InMemoryBus()
    gen = EventGenerator(bus, seed=7)

    gen.emit_image_uploaded()
    gen.emit_inference_completed(image_id="img-x")
    gen.emit_search_requested()

    for topic, envelope in bus.published:
        validate_payload(topic, envelope["payload"])


def test_replay_reapplies_captured_events():
    bus = InMemoryBus()
    gen = EventGenerator(bus, seed=3)

    gen.emit_image_uploaded(image_id="img-a")
    captured = list(bus.published)

    replay_bus = InMemoryBus()
    seen = []
    replay_bus.subscribe(IMAGE_UPLOADED, seen.append)
    EventGenerator(replay_bus).replay(captured)

    assert len(seen) == 1
    assert seen[0]["payload"]["image_id"] == "img-a"


def test_inject_fault_only_on_in_memory_bus():
    gen = EventGenerator(InMemoryBus())
    gen.inject_fault(IMAGE_UPLOADED, RuntimeError("nope"))
    with pytest.raises(RuntimeError, match="nope"):
        gen.emit_image_uploaded()


def test_correlation_id_is_preserved():
    bus = InMemoryBus()
    gen = EventGenerator(bus, seed=0)
    gen.emit_image_uploaded(correlation_id="trace-123")
    assert bus.published[0][1]["correlation_id"] == "trace-123"

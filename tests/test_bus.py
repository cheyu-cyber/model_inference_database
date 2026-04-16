"""Unit tests for the message bus abstraction.

RedisBus is not exercised here (it needs a live broker).  These tests
cover the in-memory bus's contract — which is the same contract
production code relies on — plus the fault-injection escape hatch.
"""

import pytest

from messaging import InMemoryBus


def test_publish_delivers_to_subscribers():
    bus = InMemoryBus()
    seen = []
    bus.subscribe("topic.a", seen.append)
    bus.publish("topic.a", {"n": 1})
    assert seen == [{"n": 1}]


def test_publish_to_topic_without_subscribers_is_noop():
    bus = InMemoryBus()
    bus.publish("topic.b", {"k": "v"})
    assert bus.published == [("topic.b", {"k": "v"})]


def test_multiple_subscribers_called_in_order():
    bus = InMemoryBus()
    order = []
    bus.subscribe("t", lambda m: order.append("first"))
    bus.subscribe("t", lambda m: order.append("second"))
    bus.publish("t", {})
    assert order == ["first", "second"]


def test_publish_records_all_messages():
    bus = InMemoryBus()
    bus.publish("t1", {"i": 1})
    bus.publish("t2", {"i": 2})
    bus.publish("t1", {"i": 3})
    assert bus.messages_on("t1") == [{"i": 1}, {"i": 3}]
    assert bus.messages_on("t2") == [{"i": 2}]


def test_fault_injection_raises_on_publish():
    bus = InMemoryBus()
    bus.inject_fault("bad", RuntimeError("broker down"))
    with pytest.raises(RuntimeError, match="broker down"):
        bus.publish("bad", {})


def test_fault_injection_does_not_affect_other_topics():
    bus = InMemoryBus()
    seen = []
    bus.subscribe("good", seen.append)
    bus.inject_fault("bad", RuntimeError("x"))
    bus.publish("good", {"ok": True})
    assert seen == [{"ok": True}]


def test_clear_resets_subscribers_and_history():
    bus = InMemoryBus()
    bus.subscribe("t", lambda m: None)
    bus.publish("t", {})
    bus.clear()
    assert bus.published == []
    assert bus.messages_on("t") == []


def test_handler_exception_propagates():
    """InMemoryBus is synchronous — a buggy handler surfaces to the test."""
    bus = InMemoryBus()
    bus.subscribe("t", lambda m: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        bus.publish("t", {})

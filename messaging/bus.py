"""Message bus abstraction.

Two implementations are provided:

* :class:`InMemoryBus` — dispatches subscribers synchronously on publish.
  Used by unit / integration tests; gives deterministic ordering and makes
  fault injection trivial.

* :class:`RedisBus` — thin wrapper over ``redis.Redis.pubsub``.  Used in
  production / when the services are run as separate processes.

Both implementations expose the same three methods: ``publish``,
``subscribe`` and ``run_forever``.  Services are written only against the
abstract :class:`MessageBus` interface and never import ``redis`` directly.

Envelope encoding
-----------------
Messages are JSON-encoded dicts.  Publishers pass a Python dict; the bus
serializes to JSON for Redis and deserializes on the receiving side so
subscribers always see a dict.  Using dicts (rather than raw Pydantic
models) keeps the bus library-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], None]


class MessageBus(ABC):
    """Abstract pub/sub message bus."""

    @abstractmethod
    def publish(self, topic: str, message: dict[str, Any]) -> None:
        """Send ``message`` on ``topic``."""

    @abstractmethod
    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register ``handler`` to be invoked for every message on ``topic``."""

    @abstractmethod
    def run_forever(self) -> None:
        """Block and dispatch incoming messages until stopped."""

    @abstractmethod
    def stop(self) -> None:
        """Stop a running ``run_forever`` loop."""


# ----------------------------------------------------------------------
# In-memory bus (tests)
# ----------------------------------------------------------------------

class InMemoryBus(MessageBus):
    """Synchronous, in-process bus used for tests.

    ``publish`` calls subscribers inline, so by the time ``publish``
    returns, every downstream side-effect has already run.  This lets
    tests write linear assertions without awaiting async plumbing.

    Every published message is also recorded in ``published`` for
    introspection — test code can assert "exactly one inference event was
    emitted with these fields" without wiring an extra spy subscriber.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self.published: list[tuple[str, dict[str, Any]]] = []
        self._fault_rules: dict[str, Exception] = {}
        self._running = False

    # -- MessageBus API --------------------------------------------------
    def publish(self, topic: str, message: dict[str, Any]) -> None:
        self.published.append((topic, message))
        # Fault injection: if a rule is set for this topic, raise instead
        # of dispatching — lets tests simulate a broker failure at publish.
        if topic in self._fault_rules:
            raise self._fault_rules[topic]
        for handler in list(self._handlers.get(topic, ())):
            handler(message)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers.setdefault(topic, []).append(handler)

    def run_forever(self) -> None:
        self._running = True
        # InMemoryBus dispatches synchronously during publish(), so the
        # "run loop" has nothing to block on.  We still honour stop().
        while self._running:
            threading.Event().wait(0.05)

    def stop(self) -> None:
        self._running = False

    # -- Test helpers ----------------------------------------------------
    def inject_fault(self, topic: str, exc: Exception) -> None:
        """Cause ``publish(topic, …)`` to raise ``exc``."""
        self._fault_rules[topic] = exc

    def clear(self) -> None:
        self._handlers.clear()
        self.published.clear()
        self._fault_rules.clear()

    def messages_on(self, topic: str) -> list[dict[str, Any]]:
        """Return every message published on ``topic`` so far."""
        return [m for (t, m) in self.published if t == topic]


# ----------------------------------------------------------------------
# Redis bus (production)
# ----------------------------------------------------------------------

class RedisBus(MessageBus):
    """Pub/sub bus backed by Redis.

    A single ``Redis`` connection is shared between publisher and
    subscriber; subscription delivery uses the blocking
    ``pubsub.listen()`` iterator in a dedicated thread.
    """

    def __init__(self, url: str | None = None) -> None:
        try:
            import redis  # lazy import — only needed at runtime
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "RedisBus requires the 'redis' package. "
                "Install with: pip install redis"
            ) from exc

        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis = redis.Redis.from_url(self._url, decode_responses=True)
        self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        self._handlers: dict[str, Handler] = {}
        self._thread: threading.Thread | None = None
        self._running = False

    # -- MessageBus API --------------------------------------------------
    def publish(self, topic: str, message: dict[str, Any]) -> None:
        self._redis.publish(topic, json.dumps(message))

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic] = handler
        self._pubsub.subscribe(topic)

    def run_forever(self) -> None:
        self._running = True
        for raw in self._pubsub.listen():
            if not self._running:
                break
            if raw.get("type") != "message":
                continue
            topic = raw["channel"]
            try:
                message = json.loads(raw["data"])
            except (ValueError, TypeError):
                log.warning("dropping malformed message on %s", topic)
                continue
            handler = self._handlers.get(topic)
            if handler is None:
                continue
            try:
                handler(message)
            except Exception:  # pragma: no cover - logged and continued
                log.exception("handler for %s raised", topic)

    def stop(self) -> None:
        self._running = False
        try:
            self._pubsub.close()
        except Exception:  # pragma: no cover
            pass


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def make_default_bus() -> MessageBus:
    """Return an ``InMemoryBus`` or ``RedisBus`` based on ``BUS_BACKEND``.

    Environment:
        BUS_BACKEND = "redis" (default) | "memory"
        REDIS_URL   = redis://host:port/db
    """
    backend = os.getenv("BUS_BACKEND", "redis").lower()
    if backend == "memory":
        return InMemoryBus()
    return RedisBus()

"""Request/reply over a pub/sub bus.

All read operations in the system are pairs of topics:
``foo.requested`` / ``foo.completed``. The :class:`RequestTracker` lets
a caller (the Web service) make a synchronous-looking ``request(...)``
call on top of that asynchronous contract.

How replies are matched
-----------------------
We match on the **envelope's ``correlation_id``**, not anything in the
payload. ``correlation_id`` is already part of every event, already
copied by every handler (each service threads it from the input event
into the events it publishes), and is unique per request — so it is
the natural reply-matching token. No extra payload field needed; no
service-side code changes needed when adding a new request/reply pair.

Mechanics
---------
- ``request(request_topic, reply_topic, payload, timeout)`` mints a
  fresh ``correlation_id``, registers a :class:`Future` against it,
  publishes the request, and blocks on the future until a reply with
  that ``correlation_id`` arrives or ``timeout`` elapses.
- A single subscription per reply topic is registered the first time
  it's used. Replies whose ``correlation_id`` doesn't match an
  in-flight call are silently discarded — they came from a different
  requester or a previous timeout.

Why ``concurrent.futures.Future``
---------------------------------
Both bus implementations call into ``_on_reply`` from a producer
thread (RedisBus: a dedicated subscriber thread; InMemoryBus: inline
on ``publish``). ``Future.set_result`` is thread-safe and
``Future.result(timeout)`` provides exactly the blocking wait the
caller needs.

For the InMemoryBus case, ``publish`` dispatches handlers synchronously
during the call, so by the time ``bus.publish(request_topic, …)``
returns, the reply has already been published and the future is already
resolved — ``.result()`` returns immediately. Same code path,
no special-casing.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from typing import Any

from DB.model_inference_database.events import make_event
from .bus import MessageBus


DEFAULT_TIMEOUT = 5.0


class RequestTimeoutError(Exception):
    """Raised when a request didn't receive a reply in time."""


class RequestTracker:
    """Send a request and wait for the matching reply on the bus."""

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._subscribed: set[str] = set()
        self._lock = threading.Lock()

    def subscribe_replies(self, *reply_topics: str) -> None:
        """Pre-subscribe to reply topics before the bus listener starts.

        Doing this up front avoids a race in :class:`RedisBus` where
        ``pubsub.subscribe`` from a request thread can interleave with
        ``pubsub.listen`` on the listener thread and miss messages.
        """
        with self._lock:
            for topic in reply_topics:
                if topic not in self._subscribed:
                    self._bus.subscribe(topic, self._on_reply)
                    self._subscribed.add(topic)

    def request(
        self,
        request_topic: str,
        reply_topic: str,
        payload: dict[str, Any],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """Publish a request and block on the matching reply.

        Returns the reply payload dict.
        """
        correlation_id = str(uuid.uuid4())
        future: Future[dict[str, Any]] = Future()

        with self._lock:
            self._futures[correlation_id] = future
            if reply_topic not in self._subscribed:
                self._bus.subscribe(reply_topic, self._on_reply)
                self._subscribed.add(reply_topic)

        self._bus.publish(
            request_topic,
            make_event(request_topic, payload, correlation_id=correlation_id),
        )

        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise RequestTimeoutError(
                f"No reply on {reply_topic} for correlation_id={correlation_id} "
                f"within {timeout}s"
            ) from exc
        finally:
            with self._lock:
                self._futures.pop(correlation_id, None)

    def _on_reply(self, event: dict[str, Any]) -> None:
        correlation_id = event.get("correlation_id")
        if not correlation_id:
            return
        with self._lock:
            future = self._futures.pop(correlation_id, None)
        if future is not None and not future.done():
            future.set_result(event.get("payload") or {})

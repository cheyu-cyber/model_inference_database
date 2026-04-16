"""Messaging layer — message bus abstraction plus an event generator.

The :class:`MessageBus` interface lets every service stay unaware of whether
it is talking to a real Redis broker or an in-process fake. Tests use
:class:`InMemoryBus` (synchronous, deterministic); production uses
:class:`RedisBus` (``redis.Redis.pubsub``).
"""

from .bus import MessageBus, InMemoryBus, RedisBus, make_default_bus
from .generator import EventGenerator

__all__ = [
    "MessageBus",
    "InMemoryBus",
    "RedisBus",
    "make_default_bus",
    "EventGenerator",
]

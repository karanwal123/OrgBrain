"""In-memory TTL cache for expensive AI responses."""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Generic, Hashable, Optional, TypeVar

T = TypeVar("T")


@dataclass
class _CacheItem(Generic[T]):
    """A cached value with its absolute expiration timestamp."""

    value: T
    expires_at: datetime


class TTLCache(Generic[T]):
    """Small thread-safe in-memory TTL cache with LRU-like eviction."""

    def __init__(self, ttl_seconds: int, max_entries: int = 1000):
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._max_entries = max(1, int(max_entries))
        self._items: OrderedDict[Hashable, _CacheItem[T]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: Hashable) -> Optional[T]:
        """Return cached value if present and not expired."""
        now = datetime.now(timezone.utc)
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if item.expires_at <= now:
                self._items.pop(key, None)
                return None

            # Refresh recency on hit.
            self._items.move_to_end(key)
            return item.value

    def set(self, key: Hashable, value: T) -> None:
        """Insert or update a cache entry and evict old items when needed."""
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._ttl_seconds)
        with self._lock:
            self._items[key] = _CacheItem(value=value, expires_at=expires_at)
            self._items.move_to_end(key)

            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)

    def delete(self, key: Hashable) -> None:
        """Remove a cache entry if present."""
        with self._lock:
            self._items.pop(key, None)

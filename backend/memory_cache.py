"""Redis-backed hot cache for channel memory state and search results."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from backend.cache import TTLCache
from backend.config import Settings, get_settings
from backend.schemas import ChannelMemoryState, MemoryRetrievalHit

logger = logging.getLogger(__name__)


def _query_cache_key(query: str) -> str:
    normalized = query.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


class MemoryCache:
    """
    Two-tier cache: Redis when configured, otherwise in-process TTL fallback.

    Invalidation strategy:
    - On channel state write: delete state key + all search keys for that channel.
    - Search entries expire via TTL (default 15 min).
    - State entries expire via TTL but are invalidated eagerly on writes.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._prefix = self.settings.redis_memory_prefix.rstrip(":")
        self._ttl = max(1, int(self.settings.redis_cache_ttl_seconds))
        self._local_state: TTLCache[ChannelMemoryState] = TTLCache(
            ttl_seconds=self._ttl,
            max_entries=500,
        )
        self._local_search: TTLCache[list[MemoryRetrievalHit]] = TTLCache(
            ttl_seconds=self._ttl,
            max_entries=2000,
        )
        self._redis = None
        if self.settings.redis_url:
            try:
                # pyrefly: ignore [missing-import]
                import redis

                # Cloud Redis connections benefit from socket timeouts, retries, and keepalive pings
                self._redis = redis.from_url(
                    self.settings.redis_url,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
                self._redis.ping()
                logger.info("Redis memory cache connected (Cloud mode ready)")
            except Exception as exc:
                logger.warning("Redis unavailable, using in-process memory cache: %s", exc)
                self._redis = None

    def _state_key(self, channel_id: str) -> str:
        return f"{self._prefix}:state:{channel_id}"

    def _search_key(self, channel_id: str, query: str) -> str:
        return f"{self._prefix}:search:{channel_id}:{_query_cache_key(query)}"

    def _search_pattern(self, channel_id: str) -> str:
        return f"{self._prefix}:search:{channel_id}:*"

    def get_state(self, channel_id: str) -> Optional[ChannelMemoryState]:
        """Return cached channel state when present."""
        if self._redis is not None:
            try:
                print(f"[Redis GET] Requesting state for channel: {channel_id}")
                raw = self._redis.get(self._state_key(channel_id))
                if raw:
                    print(f"   [Redis HIT] Found state for channel: {channel_id}")
                    state = ChannelMemoryState.model_validate(json.loads(raw))
                    self._local_state.set(channel_id, state)
                    return state
                print(f"   [Redis MISS] No state found for channel: {channel_id}")
            except Exception as exc:
                logger.warning("Redis state read failed for %s: %s", channel_id, exc)
        local_hit = self._local_state.get(channel_id)
        if local_hit is not None:
            print(f"   [Local Cache HIT] Found cached state for channel: {channel_id}")
        return local_hit

    def set_state(self, state: ChannelMemoryState) -> None:
        """Cache channel state."""
        self._local_state.set(state.channel_id, state)
        if self._redis is None:
            return
        try:
            print(f"[Redis SET] Caching state for channel: {state.channel_id} (TTL={self._ttl}s)")
            self._redis.setex(
                self._state_key(state.channel_id),
                self._ttl,
                state.model_dump_json(),
            )
        except Exception as exc:
            logger.warning("Redis state write failed for %s: %s", state.channel_id, exc)

    def invalidate_channel(self, channel_id: str) -> None:
        """Drop cached state and search results for a channel."""
        self._local_state.delete(channel_id)
        
        # Clear matching local search cache keys (both for this channel and global scope "*")
        with self._local_search._lock:
            keys_to_del = [
                k for k in self._local_search._items.keys() 
                if isinstance(k, tuple) and (k[0] == channel_id or k[0] == "*")
            ]
            for k in keys_to_del:
                self._local_search._items.pop(k, None)

        if self._redis is None:
            return
        try:
            print(f"[Redis DEL] Invalidating cache for channel: {channel_id}")
            self._redis.delete(self._state_key(channel_id))
            
            # 1. Clear channel-specific search keys
            cursor = 0
            pattern = self._search_pattern(channel_id)
            while True:
                cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    print(f"   [Redis DEL] Deleting query search keys: {keys}")
                    self._redis.delete(*keys)
                if int(cursor) == 0:
                    break
                    
            # 2. Clear global search keys (scope "*") since they are affected by channel changes
            cursor = 0
            global_pattern = self._search_pattern("*")
            while True:
                cursor, keys = self._redis.scan(cursor=cursor, match=global_pattern, count=100)
                if keys:
                    print(f"   [Redis DEL] Deleting global query search keys: {keys}")
                    self._redis.delete(*keys)
                if int(cursor) == 0:
                    break
        except Exception as exc:
            logger.warning("Redis invalidation failed for %s: %s", channel_id, exc)

    def get_search(
        self,
        channel_id: Optional[str],
        query: str,
    ) -> Optional[list[MemoryRetrievalHit]]:
        """Return cached search hits when present."""
        scope = channel_id or "*"
        if self._redis is not None:
            try:
                print(f"[Redis GET] Requesting search cache for query: '{query}' (Scope: {scope})")
                raw = self._redis.get(self._search_key(scope, query))
                if raw:
                    payload = json.loads(raw)
                    hits = [MemoryRetrievalHit.model_validate(item) for item in payload]
                    print(f"   [Redis HIT] Found {len(hits)} cached search results for query: '{query}'")
                    self._local_search.set((scope, query.strip().lower()), hits)
                    return hits
                print(f"   [Redis MISS] No cached search results for query: '{query}'")
            except Exception as exc:
                logger.warning("Redis search read failed: %s", exc)
        local_hit = self._local_search.get((scope, query.strip().lower()))
        if local_hit is not None:
            print(f"   [Local Cache HIT] Found {len(local_hit)} cached search results for query: '{query}' (Scope: {scope})")
        return local_hit

    def set_search(
        self,
        channel_id: Optional[str],
        query: str,
        hits: list[MemoryRetrievalHit],
    ) -> None:
        """Cache ranked search hits."""
        scope = channel_id or "*"
        self._local_search.set((scope, query.strip().lower()), hits)
        if self._redis is None:
            return
        try:
            print(f"[Redis SET] Caching {len(hits)} search hits for query: '{query}' (Scope: {scope}, TTL={self._ttl}s)")
            payload = json.dumps([hit.model_dump(mode="json") for hit in hits], ensure_ascii=False)
            self._redis.setex(self._search_key(scope, query), self._ttl, payload)
        except Exception as exc:
            logger.warning("Redis search write failed: %s", exc)

    # -------------------------------------------------------------------------
    # Incremental Pipeline: Message Tracking, Lock, Delta Check
    # -------------------------------------------------------------------------

    def _ts_key(self, channel_id: str) -> str:
        """Redis key for the latest message timestamp seen in a channel."""
        return f"{self._prefix}:last_msg_ts:{channel_id}"

    def _processed_ts_key(self, channel_id: str) -> str:
        """Redis key for the latest message timestamp already summarized."""
        return f"{self._prefix}:last_processed_ts:{channel_id}"

    def _pending_count_key(self, channel_id: str) -> str:
        """Redis key for the count of unsummarized messages."""
        return f"{self._prefix}:pending_count:{channel_id}"

    def _lock_key(self, channel_id: str) -> str:
        """Redis key for the distributed worker lock on a channel."""
        return f"{self._prefix}:lock:{channel_id}"

    def track_message_arrival(self, channel_id: str, ts: float) -> None:
        """
        Step 1 of incremental pipeline: record a new message arrival.

        Called on EVERY incoming Slack message — no Gemini, no summarization.
        Updates last_msg_ts if newer and increments pending_count atomically.
        """
        if self._redis is None:
            return
        try:
            pipe = self._redis.pipeline()
            # Only update if this ts is newer (using a Lua-style compare via GETSET fallback)
            current = self._redis.get(self._ts_key(channel_id))
            should_update = False
            if current is None:
                should_update = True
            else:
                try:
                    should_update = float(ts) > float(current)
                except (ValueError, TypeError):
                    should_update = True

            if should_update:
                pipe.set(self._ts_key(channel_id), str(ts))
                print(f"   [Redis TS] Updated last_msg_ts for {channel_id} → {ts}")
            pipe.incr(self._pending_count_key(channel_id))
            pipe.execute()
        except Exception as exc:
            logger.warning("Redis track_message_arrival failed for %s: %s", channel_id, exc)

    def should_process_channel(self, channel_id: str) -> bool:
        """
        Step 3 of incremental pipeline: O(1) check — does this channel have new messages?

        Compares last_msg_ts vs last_processed_ts in Redis.
        Returns True only when new unprocessed messages exist.
        """
        if self._redis is None:
            return True  # No Redis → always try (fall back to storage check)
        try:
            latest = self._redis.get(self._ts_key(channel_id))
            processed = self._redis.get(self._processed_ts_key(channel_id))
            if latest is None:
                print(f"   [Redis DELTA] {channel_id}: no messages tracked yet — default to True (always process when untracked)")
                return True
            try:
                latest_f = float(latest)
            except (ValueError, TypeError):
                latest_f = 0.0
            try:
                processed_f = float(processed) if processed else 0.0
            except (ValueError, TypeError):
                processed_f = 0.0
            has_new = latest_f > processed_f
            status = "HAS NEW" if has_new else "UP TO DATE"
            print(f"   [Redis DELTA] {channel_id}: last_msg={latest_f:.0f} last_processed={processed_f:.0f} → {status}")
            return has_new
        except Exception as exc:
            logger.warning("Redis should_process_channel failed for %s: %s", channel_id, exc)
            return True

    def acquire_lock(self, channel_id: str, ttl_seconds: int = 300) -> bool:
        """
        Step 4 of incremental pipeline: distributed SETNX lock.

        Returns True if lock acquired (this worker should proceed).
        Returns False if another worker already holds the lock — skip.
        """
        if self._redis is None:
            return True  # No Redis → no distributed lock needed
        try:
            acquired = self._redis.set(
                self._lock_key(channel_id),
                "1",
                nx=True,   # Only set if Not eXists (SETNX)
                ex=ttl_seconds,
            )
            if acquired:
                print(f"   [Redis LOCK] Acquired lock for channel {channel_id} (TTL={ttl_seconds}s)")
            else:
                print(f"   [Redis LOCK] Skipped {channel_id} — another worker holds the lock")
            return bool(acquired)
        except Exception as exc:
            logger.warning("Redis acquire_lock failed for %s: %s", channel_id, exc)
            return True

    def release_lock(self, channel_id: str) -> None:
        """
        Step 10 of incremental pipeline: release the distributed lock.

        Always call this in a finally block after flush completes or fails.
        """
        if self._redis is None:
            return
        try:
            self._redis.delete(self._lock_key(channel_id))
            print(f"   [Redis LOCK] Released lock for channel {channel_id}")
        except Exception as exc:
            logger.warning("Redis release_lock failed for %s: %s", channel_id, exc)

    def update_processed_ts(self, channel_id: str, ts: float) -> None:
        """
        Step 8 of incremental pipeline: advance the checkpoint after a successful flush.

        Sets last_processed_ts = latest message ts just summarized.
        """
        if self._redis is None:
            return
        try:
            self._redis.set(self._processed_ts_key(channel_id), str(ts))
            print(f"   [Redis CHECKPOINT] Updated last_processed_ts for {channel_id} → {ts}")
        except Exception as exc:
            logger.warning("Redis update_processed_ts failed for %s: %s", channel_id, exc)

    def reset_pending_count(self, channel_id: str) -> None:
        """Reset the pending message counter for a channel after successful flush."""
        if self._redis is None:
            return
        try:
            self._redis.set(self._pending_count_key(channel_id), "0")
            print(f"   [Redis COUNTER] Reset pending_count to 0 for channel {channel_id}")
        except Exception as exc:
            logger.warning("Redis reset_pending_count failed for %s: %s", channel_id, exc)

    def get_pending_count(self, channel_id: str) -> int:
        """Return the current pending (unsummarized) message count for a channel."""
        if self._redis is None:
            return 0
        try:
            val = self._redis.get(self._pending_count_key(channel_id))
            return int(val) if val else 0
        except Exception:
            return 0

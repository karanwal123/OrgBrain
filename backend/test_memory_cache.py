"""Tests for MemoryCache backend cache implementation."""

from unittest.mock import MagicMock, patch
import pytest
from backend.schemas import ChannelMemoryState, MemoryRetrievalHit, MemoryUnit, MemoryUnitType
from backend.memory_cache import MemoryCache
from backend.config import Settings


def test_local_fallback_when_no_redis():
    """Verify that MemoryCache falls back to in-process TTL cache when Redis is disabled."""
    settings = Settings(
        redis_url=None,
        redis_memory_prefix="test_prefix",
        redis_cache_ttl_seconds=100,
        google_cloud_project="test-project",
    )
    cache = MemoryCache(settings=settings)
    assert cache._redis is None

    # Test state cache
    state = ChannelMemoryState(
        channel_id="C_TEST",
        memory_store=[
            MemoryUnit(
                memory_id="m1",
                channel_id="C_TEST",
                unit_type=MemoryUnitType.decision,
                summary="Decision test",
            )
        ],
    )
    cache.set_state(state)
    assert cache.get_state("C_TEST") == state

    # Test search cache
    hits = [
        MemoryRetrievalHit(
            memory_id="m1",
            channel_id="C_TEST",
            summary="Decision test",
            unit_type=MemoryUnitType.decision,
            score=0.9,
            semantic_score=0.8,
            recency_score=0.7,
            importance_score=0.6,
            unresolved_score=0.5,
        )
    ]
    cache.set_search("C_TEST", "test query", hits)
    cache.set_search(None, "global query", hits)
    assert cache.get_search("C_TEST", "test query") == hits
    assert cache.get_search(None, "global query") == hits

    # Test invalidation
    cache.invalidate_channel("C_TEST")
    assert cache.get_state("C_TEST") is None
    assert cache.get_search("C_TEST", "test query") is None
    assert cache.get_search(None, "global query") is None


@patch("redis.from_url")
def test_redis_operations(mock_from_url):
    """Verify that MemoryCache operations write to and read from Redis when available."""
    mock_redis = MagicMock()
    mock_from_url.return_value = mock_redis

    settings = Settings(
        redis_url="redis://localhost:6379",
        redis_memory_prefix="test_prefix",
        redis_cache_ttl_seconds=100,
        google_cloud_project="test-project",
    )
    cache = MemoryCache(settings=settings)
    assert cache._redis == mock_redis

    # 1. State GET (hit)
    state = ChannelMemoryState(channel_id="C_TEST")
    mock_redis.get.return_value = state.model_dump_json()
    res = cache.get_state("C_TEST")
    assert res is not None
    assert res.channel_id == "C_TEST"
    # Ensure it populated the local cache
    assert cache._local_state.get("C_TEST") == res

    # 2. State SET
    mock_redis.reset_mock()
    cache.set_state(state)
    mock_redis.setex.assert_called_once()

    # 3. Search GET (hit)
    mock_redis.get.return_value = "[]"
    search_hits = cache.get_search("C_TEST", "test query")
    assert search_hits == []
    # Ensure it populated local search cache
    assert cache._local_search.get(("C_TEST", "test query")) == []

    # 4. Search SET
    mock_redis.reset_mock()
    cache.set_search("C_TEST", "test query", [])
    mock_redis.setex.assert_called_once()

    # 5. Invalidation
    mock_redis.scan.return_value = (0, ["key1", "key2"])
    cache.invalidate_channel("C_TEST")
    mock_redis.delete.assert_any_call("test_prefix:state:C_TEST")
    mock_redis.delete.assert_any_call("key1", "key2")

    # 6. Pipeline operations
    mock_redis.reset_mock()
    mock_pipe = MagicMock()
    mock_redis.pipeline.return_value = mock_pipe
    mock_redis.get.return_value = None
    cache.track_message_arrival("C_TEST", 12345.67)
    mock_redis.get.assert_called_with("test_prefix:last_msg_ts:C_TEST")
    mock_pipe.set.assert_called_once_with("test_prefix:last_msg_ts:C_TEST", "12345.67")
    mock_pipe.incr.assert_called_once_with("test_prefix:pending_count:C_TEST")
    mock_pipe.execute.assert_called_once()

    # 7. Delta checks
    mock_redis.reset_mock()
    mock_redis.get.side_dict = {"test_prefix:last_msg_ts:C_TEST": "12345.67", "test_prefix:last_processed_ts:C_TEST": "12300.00"}
    mock_redis.get.side_effect = lambda k: mock_redis.get.side_dict.get(k)
    assert cache.should_process_channel("C_TEST") is True

    # 8. Lock operations
    mock_redis.reset_mock()
    mock_redis.set.return_value = True
    assert cache.acquire_lock("C_TEST") is True
    mock_redis.set.assert_called_once_with("test_prefix:lock:C_TEST", "1", nx=True, ex=300)

    cache.release_lock("C_TEST")
    mock_redis.delete.assert_called_with("test_prefix:lock:C_TEST")

    # 9. Counters and checkpoints
    mock_redis.reset_mock()
    cache.update_processed_ts("C_TEST", 12345.67)
    mock_redis.set.assert_called_once_with("test_prefix:last_processed_ts:C_TEST", "12345.67")

    cache.reset_pending_count("C_TEST")
    mock_redis.set.assert_called_with("test_prefix:pending_count:C_TEST", "0")

    mock_redis.get.side_effect = None
    mock_redis.get.return_value = "5"
    assert cache.get_pending_count("C_TEST") == 5


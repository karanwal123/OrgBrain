"""Background worker that flushes pending channel memory on an interval."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from backend.config import Settings, get_settings

if TYPE_CHECKING:
    from backend.memory import ChannelMemoryService

logger = logging.getLogger(__name__)


class MemoryWorker:
    """Periodically runs delta summarization for channels with pending messages."""

    def __init__(
        self,
        service: ChannelMemoryService,
        *,
        settings: Optional[Settings] = None,
    ):
        self.service = service
        self.settings = settings or get_settings()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    @property
    def interval_seconds(self) -> int:
        return max(1, int(self.settings.memory_summary_interval_minutes)) * 60

    async def run_once(self) -> list[str]:
        """Flush channels that have buffered messages, using Redis delta check + lock."""
        cache = self.service._cache  # type: ignore[attr-defined]

        # Collect candidate channels from storage + in-memory states
        channel_ids: set[str] = set()
        storage = self.service._storage  # type: ignore[attr-defined]
        if storage is not None:
            channel_ids.update(storage.list_channels_with_pending())
        states = self.service._states  # type: ignore[attr-defined]
        channel_ids.update(
            cid for cid, state in states.items() if state.pending_message_payloads
        )

        flushed: list[str] = []
        for channel_id in sorted(channel_ids):
            # O(1) Redis delta check -- skip if no new messages since last flush
            if cache is not None and not cache.should_process_channel(channel_id):
                print(f"   [Memory Worker] Skipping {channel_id} -- no new messages since last flush")
                continue

            before = self.service.get_state(channel_id)
            if not before.pending_message_payloads:
                continue

            # Distributed lock -- prevents two workers summarizing same channel
            if cache is not None and not cache.acquire_lock(channel_id, ttl_seconds=300):
                print(f"   [Memory Worker] Skipping {channel_id} -- lock held by another worker")
                continue

            try:
                self.service.flush_channel(channel_id)
                flushed.append(channel_id)
            except Exception as exc:
                print(f"   [Memory Worker] FAILED flushing {channel_id}: {exc}")
                logger.exception("Memory worker flush failed for %s", channel_id)
            finally:
                if cache is not None:
                    cache.release_lock(channel_id)

        if flushed:
            logger.info("Memory worker flushed channels: %s", ", ".join(flushed))
        return flushed

    async def _loop(self) -> None:
        print(f"\n[Memory Worker] Background worker started. Checking every {self.settings.memory_summary_interval_minutes} minutes.")
        logger.info(
            "Memory worker started (interval=%s min)",
            self.settings.memory_summary_interval_minutes,
        )
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                break
            except asyncio.TimeoutError:
                pass
            try:
                print("\n[Memory Worker] Tick: Checking for channels with pending messages...")
                flushed = await self.run_once()
                if flushed:
                    print(f"[Memory Worker] SUCCESS: Flushed channels: {', '.join(flushed)}")
                else:
                    print("[Memory Worker] Acknowledged: No pending messages to flush.")
            except Exception as exc:
                print(f"[Memory Worker] FAILED: Worker cycle error: {exc}")
                logger.exception("Memory worker cycle failed")

    def start(self) -> None:
        """Start the background worker loop."""
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="org-brain-memory-worker")

    async def stop(self) -> None:
        """Stop the background worker loop."""
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

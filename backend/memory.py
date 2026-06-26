"""Channel-scoped memory helpers for delta summarization and retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Optional, TYPE_CHECKING

from backend.config import Settings, get_settings
from backend.schemas import (
    ChannelMemoryState,
    MemoryCheckpoint,
    MemoryDeltaBatch,
    MemoryRetrievalHit,
    MemoryUnit,
    MemoryUnitType,
)

if TYPE_CHECKING:
    from backend.embeddings import MemoryEmbeddingStore
    from backend.memory_cache import MemoryCache
    from backend.memory_embeddings import MemoryEmbedder
    from backend.storage import MemoryStorage

from vertexai.generative_models import GenerationConfig, GenerativeModel

logger = logging.getLogger(__name__)

BATCH_JUDGING_PROMPT = """You are an AI organizational memory classifier.
Analyze the list of Slack messages and determine if each message contains important organizational memory (decisions, deadlines, blockers, action items, critical updates, alerts, meetings, fire drills, inspections).

Rules:
- If a message has no lasting organizational value (social chit-chat, greetings, trivial replies like "ok", "cool", "thanks", "hello"), set its importance close to 0.0 or 0.1.
- If it has important information, assign an importance score between 0.5 and 1.0 (deadlines, blockers, meetings, drills, and decisions should be >= 0.85).
- Output JSON only with this shape:
{{
    "results": [
        {{
            "message_id": "corresponding id from input",
            "unit_type": "decision|problem|agreement|action_item|unresolved_issue|context",
            "importance": <float between 0.0 and 1.0>,
            "unresolved": <boolean, true if problem/action/issue is unresolved>,
            "summary": "one concise sentence summarizing the core information of the message"
        }}
    ]
}}

Messages to classify:
{messages_json}
"""


def _clean_json_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _judge_messages_batch(
    messages: list[dict[str, Any]],
    model: GenerativeModel,
) -> dict[str, dict[str, Any]]:
    """Use Gemini to classify and judge the importance of a batch of messages in a single call."""
    input_list = []
    for msg in messages:
        msg_id = str(msg.get("id") or msg.get("ts") or "")
        text = str(msg.get("text", "")).strip()
        if msg_id and text:
            input_list.append({"id": msg_id, "text": text})

    if not input_list:
        return {}

    prompt = BATCH_JUDGING_PROMPT.format(messages_json=json.dumps(input_list, ensure_ascii=False, indent=2))
    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text or ""
        if not raw_text.strip():
            return {}
        
        cleaned = _clean_json_output(raw_text)
        payload = json.loads(cleaned)
        
        results = {}
        if isinstance(payload, dict) and "results" in payload:
            for item in payload["results"]:
                msg_id = item.get("message_id")
                if msg_id:
                    results[msg_id] = item
        return results
    except Exception as exc:
        logger.warning("Batch judging via Gemini failed, falling back to heuristics: %s", exc)
        return {}


MEMORY_SUMMARIZATION_PROMPT = """You are OrgBrain's organizational memory compressor.

Your job is to convert a delta of Slack messages into durable memory for one channel only.

Channel: {channel_id}
Previous memory snapshot:
{previous_snapshot}

New messages since the last checkpoint:
{delta_messages}

Rules:
- Preserve intent, decisions, commitments, blockers, dependencies, owners, deadlines, and unresolved issues.
- Discard greetings, repeated chatter, social small talk, and verbatim phrasing.
- Never mix information from other channels.
- Never re-state memory that is already present in the previous snapshot unless the delta changes it.
- Prefer canonical statements like "Payment service has recurring Redis timeouts under load" instead of quote-like paraphrases.
- If an issue is unresolved, mark it unresolved.
- Output JSON only with this shape:
{{
    "channel_id": "...",
    "summary_state": "short updated channel memory snapshot",
    "memory_units": [
        {{
            "memory_id": "stable id",
            "unit_type": "decision|problem|agreement|action_item|unresolved_issue|context",
            "summary": "compressed memory statement",
            "owners": ["optional names"],
            "tags": ["optional tags"],
            "importance": 0.0,
            "unresolved": false,
            "source_message_ids": ["..."],
            "source_timestamps": ["ISO-8601 timestamps"]
        }}
    ]
}}
"""

_PROBLEM_RE = re.compile(r"(?i)\b(issue|problem|blocked|timeout|error|fail|failing|bug|incident|drill|inspection|fire|blocker)\b")
_DECISION_RE = re.compile(r"(?i)\b(decide|decision|decided|going with|went with|choose|chosen|approve|approved)\b")
_ACTION_RE = re.compile(r"(?i)\b(todo|action item|follow up|follow-up|need to|should|will|next step|must|deadline|due)\b")
_AGREEMENT_RE = re.compile(r"(?i)\b(agree|agreed|confirmed|aligned|sounds good|works for me)\b")
_OPEN_ISSUE_RE = re.compile(r"(?i)\b(open|unresolved|pending|waiting on|blocked on|still stuck)\b")
_URGENT_RE = re.compile(r"(?i)\b(deadline|drill|inspection|cancel|important|critical|urgent|meeting|fire|must|hard deadline|due)\b")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "will",
    "your",
    "you",
    "our",
    "are",
    "was",
    "were",
    "not",
    "but",
    "can",
    "all",
    "any",
    "has",
    "had",
    "into",
    "out",
    "need",
    "just",
    "now",
    "team",
    "today",
    "yesterday",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    text = str(value).strip()
    return text or None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_keywords(text: str, *, limit: int = 4) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z][a-z0-9+_-]{2,}", text.lower())
        if token not in _STOPWORDS
    ]
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _infer_unit_type(text: str) -> MemoryUnitType:
    if _OPEN_ISSUE_RE.search(text) and _PROBLEM_RE.search(text):
        return MemoryUnitType.unresolved_issue
    if _DECISION_RE.search(text):
        return MemoryUnitType.decision
    if _ACTION_RE.search(text):
        return MemoryUnitType.action_item
    if _AGREEMENT_RE.search(text):
        return MemoryUnitType.agreement
    if _PROBLEM_RE.search(text):
        return MemoryUnitType.problem
    return MemoryUnitType.context


def _canonical_summary(message: dict[str, Any], unit_type: MemoryUnitType) -> tuple[str, list[str], list[str], bool, float]:
    text = _normalize_text(str(message.get("text", "")))
    speaker = _normalize_text(
        str(message.get("user_real_name") or message.get("user_name") or message.get("user") or "")
    )
    keywords = _extract_keywords(text)
    first_sentence = re.split(r"[.!?]\s+", text, maxsplit=1)[0]

    if unit_type is MemoryUnitType.problem:
        summary = first_sentence
        unresolved = True
        importance = 0.9 if _PROBLEM_RE.search(text) else 0.7
    elif unit_type is MemoryUnitType.unresolved_issue:
        summary = first_sentence
        unresolved = True
        importance = 0.85
    elif unit_type is MemoryUnitType.action_item:
        summary = first_sentence
        unresolved = True
        importance = 0.8
    elif unit_type is MemoryUnitType.decision:
        summary = first_sentence
        unresolved = False
        importance = 0.85
    elif unit_type is MemoryUnitType.agreement:
        summary = first_sentence
        unresolved = False
        importance = 0.6
    else:
        summary = first_sentence or text
        unresolved = False
        importance = 0.4

    if _URGENT_RE.search(text):
        importance = max(importance, 0.85)

    if speaker and speaker.lower() not in summary.lower():
        summary = f"{speaker}: {summary}" if unit_type is MemoryUnitType.context else summary

    summary = _normalize_text(summary)
    if len(summary) > 220:
        summary = summary[:217].rstrip() + "..."
    tags = keywords[:]
    if unit_type is MemoryUnitType.problem and "blocker" not in tags:
        tags.insert(0, "blocker")
    return summary, tags, [speaker] if speaker else [], unresolved, importance


def memory_unit_fingerprint(channel_id: str, unit_type: MemoryUnitType, summary: str) -> str:
    """Build a stable fingerprint for deduplicating channel memory."""
    basis = f"{channel_id}:{unit_type.value}:{_normalize_text(summary).lower()}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def select_delta_messages(
    messages: list[dict[str, Any]],
    *,
    last_summary_ts: Optional[datetime] = None,
    last_processed_message_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return only messages that are newer than the stored checkpoint."""
    start_idx = 0
    if last_processed_message_id:
        for idx, msg in enumerate(messages):
            msg_id = str(msg.get("id") or msg.get("ts") or "").strip()
            if msg_id == last_processed_message_id:
                start_idx = idx + 1
                break

    selected: list[dict[str, Any]] = []
    for message in messages[start_idx:]:
        message_id = str(message.get("id") or message.get("ts") or "").strip()
        timestamp_value = message.get("ts") or message.get("timestamp")
        if timestamp_value is None:
            continue
        try:
            message_ts = float(timestamp_value)
        except (TypeError, ValueError):
            continue

        message_dt = datetime.fromtimestamp(message_ts, tz=timezone.utc)
        if last_summary_ts is not None and message_dt <= last_summary_ts:
            continue
        if last_processed_message_id and message_id and message_id == last_processed_message_id:
            continue
        selected.append(message)
    return selected


def build_memory_prompt(batch: MemoryDeltaBatch, previous_state: str) -> str:
    """Build the compression prompt for a channel delta."""
    messages_json = json.dumps(batch.messages, ensure_ascii=False, indent=2)
    return MEMORY_SUMMARIZATION_PROMPT.format(
        channel_id=batch.channel_id,
        previous_snapshot=previous_state.strip() or "(empty)",
        delta_messages=messages_json,
    )


def extract_memory_units(batch: MemoryDeltaBatch, model: Optional[GenerativeModel] = None) -> list[MemoryUnit]:
    """Compress new messages into channel-local memory units, using Gemini judge if available."""
    units: list[MemoryUnit] = []
    
    # Run batch judging if model is available
    judgments = {}
    if model is not None:
        print(f"   [Memory Service] Judging {len(batch.messages)} messages using Gemini Flash...")
        judgments = _judge_messages_batch(batch.messages, model)
        print(f"   [Memory Service] Acknowledged: Gemini returned judgments for {len(judgments)} messages")

    for message in batch.messages:
        text = _normalize_text(str(message.get("text", "")))
        if not text:
            continue
            
        message_id = str(message.get("id") or message.get("ts") or "")
        judgment = judgments.get(message_id) if message_id else None
        
        if judgment:
            try:
                unit_type = MemoryUnitType(judgment.get("unit_type", MemoryUnitType.context.value))
            except ValueError:
                unit_type = MemoryUnitType.context
                
            summary = judgment.get("summary") or text
            importance = float(judgment.get("importance") or 0.4)
            unresolved = bool(judgment.get("unresolved", False))
            
            # extract keywords and speaker normally
            speaker = _normalize_text(
                str(message.get("user_real_name") or message.get("user_name") or message.get("user") or "")
            )
            keywords = _extract_keywords(text)
            if speaker and speaker.lower() not in summary.lower():
                summary = f"{speaker}: {summary}" if unit_type is MemoryUnitType.context else summary
            summary = _normalize_text(summary)
            if len(summary) > 220:
                summary = summary[:217].rstrip() + "..."
            tags = keywords[:]
            if unit_type is MemoryUnitType.problem and "blocker" not in tags:
                tags.insert(0, "blocker")
        else:
            # Fallback to regex heuristics
            unit_type = _infer_unit_type(text)
            summary, tags, owners, unresolved, importance = _canonical_summary(message, unit_type)
            speaker = owners[0] if owners else ""

        memory_id = memory_unit_fingerprint(batch.channel_id, unit_type, summary)
        source_timestamp = message.get("ts") or message.get("timestamp")
        units.append(
            MemoryUnit(
                memory_id=memory_id,
                channel_id=batch.channel_id,
                unit_type=unit_type,
                summary=summary,
                source_message_ids=[message_id] if message_id else [],
                source_timestamps=[datetime.fromtimestamp(float(source_timestamp), tz=timezone.utc)]
                if source_timestamp is not None
                else [],
                owners=[speaker] if speaker else [],
                tags=tags,
                importance=importance,
                unresolved=unresolved,
            )
        )
    return units


def merge_memory_units(previous_units: list[MemoryUnit], delta_units: list[MemoryUnit]) -> list[MemoryUnit]:
    """Merge a previous channel memory snapshot with the newly extracted delta."""
    merged: dict[str, MemoryUnit] = {unit.memory_id: unit.model_copy(deep=True) for unit in previous_units}
    for unit in delta_units:
        existing = merged.get(unit.memory_id)
        if existing is None:
            merged[unit.memory_id] = unit.model_copy(deep=True)
            continue
        existing.source_message_ids = sorted(set(existing.source_message_ids) | set(unit.source_message_ids))
        existing.source_timestamps = sorted({*existing.source_timestamps, *unit.source_timestamps})
        existing.owners = sorted(set(existing.owners) | set(unit.owners))
        existing.tags = sorted(set(existing.tags) | set(unit.tags))
        existing.importance = max(existing.importance, unit.importance)
        existing.unresolved = existing.unresolved or unit.unresolved
        existing.updated_at = _utc_now()
    return sorted(merged.values(), key=lambda item: (item.unit_type.value, item.summary))


def update_channel_memory_state(
    previous_state: Optional[ChannelMemoryState],
    batch: MemoryDeltaBatch,
    model: Optional[GenerativeModel] = None,
) -> ChannelMemoryState:
    """Apply a delta batch to the prior state and advance the checkpoint."""
    previous_units = previous_state.memory_store if previous_state else []
    delta_units = extract_memory_units(batch, model=model)
    merged_units = merge_memory_units(previous_units, delta_units)

    compressed_context = "\n".join(f"- {unit.summary}" for unit in merged_units)
    pending_messages = [str(message.get("id") or message.get("ts") or "") for message in batch.messages]
    last_message = batch.messages[-1] if batch.messages else {}
    last_summary_ts = None
    if last_message:
        timestamp_value = last_message.get("ts") or last_message.get("timestamp")
        if timestamp_value is not None:
            last_summary_ts = datetime.fromtimestamp(float(timestamp_value), tz=timezone.utc)

    return ChannelMemoryState(
        channel_id=batch.channel_id,
        memory_store=merged_units,
        last_summary_ts=last_summary_ts,
        last_summary_timestamp=last_summary_ts,
        last_processed_message_id=pending_messages[-1] if pending_messages else (previous_state.last_processed_message_id if previous_state else None),
        compressed_context=compressed_context,
        cached_summary_state=compressed_context,
        pending_messages=[],
        cached_embeddings=dict(previous_state.cached_embeddings) if previous_state else {},
        updated_at=_utc_now(),
    )


def recency_score(updated_at: datetime, *, half_life_hours: float = 72.0) -> float:
    """Score recency using an exponential half-life decay."""
    now = _utc_now()
    age_hours = max(0.0, (now - updated_at.astimezone(timezone.utc)).total_seconds() / 3600.0)
    if half_life_hours <= 0:
        return 0.0
    return math.exp(-math.log(2) * age_hours / half_life_hours)


def rank_memory_hit(
    unit: MemoryUnit,
    *,
    semantic_score: float,
    query_channel_id: Optional[str] = None,
) -> MemoryRetrievalHit:
    """Combine semantic relevance with channel isolation and memory freshness."""
    semantic_score = max(0.0, min(1.0, semantic_score))
    recency = recency_score(unit.updated_at)
    importance = max(0.0, min(1.0, unit.importance))
    unresolved = 1.0 if unit.unresolved else 0.0
    channel_bonus = 1.0 if query_channel_id and unit.channel_id == query_channel_id else 0.0
    score = (
        semantic_score * 0.5
        + recency * 0.2
        + importance * 0.2
        + unresolved * 0.08
        + channel_bonus * 0.02
    )
    return MemoryRetrievalHit(
        memory_id=unit.memory_id,
        channel_id=unit.channel_id,
        summary=unit.summary,
        unit_type=unit.unit_type,
        score=round(score, 6),
        semantic_score=round(semantic_score, 6),
        recency_score=round(recency, 6),
        importance_score=round(importance, 6),
        unresolved_score=round(unresolved, 6),
        source_message_ids=list(unit.source_message_ids),
        owners=list(unit.owners),
        tags=list(unit.tags),
    )


def checkpoint_from_state(state: ChannelMemoryState) -> MemoryCheckpoint:
    """Convert a channel memory state into a durable checkpoint document."""
    return MemoryCheckpoint(
        channel_id=state.channel_id,
        last_summary_ts=state.last_summary_ts,
        last_processed_message_id=state.last_processed_message_id,
        cached_summary_state=state.cached_summary_state,
        updated_at=state.updated_at,
    )


def memory_checkpoint_from_batch(batch: MemoryDeltaBatch) -> MemoryCheckpoint:
    """Convenience helper for initial checkpoint documents."""
    return MemoryCheckpoint(
        channel_id=batch.channel_id,
        last_summary_ts=batch.last_summary_ts,
        last_processed_message_id=batch.last_processed_message_id,
        cached_summary_state=json.dumps(batch.messages, ensure_ascii=False),
    )


def _keyword_overlap_score(query: str, summary: str) -> float:
    query_terms = set(_extract_keywords(query, limit=12))
    summary_terms = set(_extract_keywords(summary, limit=12))
    if not query_terms or not summary_terms:
        return 0.0
    overlap = query_terms & summary_terms
    return min(1.0, len(overlap) / max(1, len(query_terms)))


class ChannelMemoryService:
    """Channel-local memory ingestion, delta summarization, and retrieval."""

    def __init__(
        self,
        *,
        storage: Optional[MemoryStorage] = None,
        cache: Optional[MemoryCache] = None,
        embedding_store: Optional[MemoryEmbeddingStore] = None,
        embedder: Optional[MemoryEmbedder] = None,
        settings: Optional[Settings] = None,
        model: Optional[GenerativeModel] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._storage = storage
        self._cache = cache
        self._embedding_store = embedding_store
        self._embedder = embedder
        self._states: dict[str, ChannelMemoryState] = {}
        self._model = model
        if self._model is None:
            try:
                from backend.config import init_vertex_ai
                self._model = init_vertex_ai(self.settings)
            except Exception as exc:
                logger.warning("Vertex AI model initialization failed in memory service: %s", exc)

    @property
    def storage(self) -> Optional[MemoryStorage]:
        return self._storage

    def track_arrival(self, channel_id: str, ts: float) -> None:
        """
        Record a new Slack message arrival in the Redis incremental pipeline.

        Called on EVERY incoming message — no Gemini, no DB hit.
        Updates last_msg_ts in Redis and increments pending_count so the
        worker can do an O(1) delta check before deciding to flush.
        """
        if self._cache is not None:
            self._cache.track_message_arrival(channel_id, ts)

    def get_state(self, channel_id: str) -> ChannelMemoryState:
        """Return the latest state for a channel, or an empty one."""
        if channel_id in self._states:
            return self._states[channel_id]

        if self._cache is not None:
            cached = self._cache.get_state(channel_id)
            if cached is not None:
                self._states[channel_id] = cached
                return cached

        if self._storage is not None:
            stored = self._storage.get_state(channel_id)
            if stored is not None:
                self._states[channel_id] = stored
                if self._cache is not None:
                    self._cache.set_state(stored)
                return stored

        return ChannelMemoryState(channel_id=channel_id)

    def _persist(self, state: ChannelMemoryState, *, invalidate_search: bool = False) -> None:
        self._states[state.channel_id] = state
        if self._storage is not None:
            self._storage.save_state(state)
        if self._cache is not None:
            if invalidate_search:
                self._cache.invalidate_channel(state.channel_id)
            self._cache.set_state(state)

    def ingest_messages(self, batch: MemoryDeltaBatch) -> ChannelMemoryState:
        """Buffer new messages for the next delta summarization cycle."""
        state = self.get_state(batch.channel_id)
        filtered_messages = select_delta_messages(
            batch.messages,
            last_summary_ts=state.last_summary_ts,
            last_processed_message_id=state.last_processed_message_id,
        )
        if not filtered_messages:
            return state

        print(f"[Memory Buffer] Ingesting {len(filtered_messages)} new messages for channel {batch.channel_id} into MongoDB buffer...")
        payloads = list(state.pending_message_payloads)
        payloads.extend(filtered_messages)
        updated = state.model_copy(
            update={
                "pending_message_payloads": payloads,
                "pending_messages": [
                    str(message.get("id") or message.get("ts") or "")
                    for message in payloads
                ],
                "updated_at": _utc_now(),
            }
        )
        self._persist(updated, invalidate_search=False)
        return updated

    def ingest_message(self, channel_id: str, message: dict[str, Any]) -> ChannelMemoryState:
        """Convenience wrapper for a single Slack message event."""
        batch = MemoryDeltaBatch(channel_id=channel_id, messages=[message])
        return self.ingest_messages(batch)

    def flush_channel(self, channel_id: str) -> ChannelMemoryState:
        """Run delta summarization for one channel and persist the merged state."""
        state = self.get_state(channel_id)
        if not state.pending_message_payloads:
            return state

        print(f"\n[Memory Delta] Summarizing and merging {len(state.pending_message_payloads)} new messages for channel {channel_id}...")
        batch = MemoryDeltaBatch(
            channel_id=channel_id,
            messages=state.pending_message_payloads,
            last_summary_ts=state.last_summary_ts,
            last_processed_message_id=state.last_processed_message_id,
        )
        merged = update_channel_memory_state(state, batch, model=self._model)
        merged = merged.model_copy(update={"pending_message_payloads": []})

        if self._embedding_store is not None and self._embedder is not None:
            print(f"   [Qdrant Synced] Embedding and upserting {len(merged.memory_store)} updated memory units...")
        self._sync_embeddings(merged, previous=state)

        print(f"   [MongoDB Persist] Saving updated memory state to MongoDB Atlas...")
        self._persist(merged, invalidate_search=True)
        print(f"   [Redis Invalidated] Cleared cache & search queries for channel {channel_id}")

        # Advance the Redis incremental pipeline checkpoint
        if self._cache is not None and merged.last_summary_ts is not None:
            latest_ts = merged.last_summary_ts.timestamp()
            self._cache.update_processed_ts(channel_id, latest_ts)
            self._cache.reset_pending_count(channel_id)

        return merged

    def flush_all(self) -> list[str]:
        """Flush every channel that has buffered pending messages."""
        channel_ids: set[str] = set()
        if self._storage is not None:
            channel_ids.update(self._storage.list_channels_with_pending())
        channel_ids.update(
            channel_id
            for channel_id, state in self._states.items()
            if state.pending_message_payloads
        )

        flushed: list[str] = []
        for channel_id in sorted(channel_ids):
            before = self.get_state(channel_id)
            if not before.pending_message_payloads:
                continue
            self.flush_channel(channel_id)
            flushed.append(channel_id)
        return flushed

    def _sync_embeddings(
        self,
        state: ChannelMemoryState,
        *,
        previous: Optional[ChannelMemoryState] = None,
    ) -> None:
        if self._embedding_store is None or self._embedder is None:
            return

        previous_units = {
            unit.memory_id: unit.summary
            for unit in (previous.memory_store if previous else [])
        }
        embeddings = dict(state.cached_embeddings)

        for unit in state.memory_store:
            if unit.memory_id in embeddings and previous_units.get(unit.memory_id) == unit.summary:
                continue
            vector = self._embedder.embed(unit.summary)
            embeddings[unit.memory_id] = vector
            self._embedding_store.upsert_memory_unit(unit, vector)

        state.cached_embeddings = embeddings

    def search(
        self,
        query: str,
        *,
        channel_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryRetrievalHit]:
        """Retrieve ranked memory hits via cache → vector search → keyword fallback."""
        query = query.strip()
        if not query:
            return []

        print(f"\n[Memory Recall] Query: '{query}' (Channel Scope: {channel_id or 'All'})")
        result_limit = limit or self.settings.memory_retrieval_limit
        if self._cache is not None:
            cached = self._cache.get_search(channel_id, query)
            if cached is not None:
                print(f"   [Redis HIT] Returned {len(cached)} cached results for: '{query}'")
                return cached[:result_limit]
            else:
                print(f"   [Redis MISS] No cached results found for: '{query}'. Querying backend stores...")

        hits: list[MemoryRetrievalHit] = []
        if self._embedder is not None and self._embedding_store is not None:
            print(f"   [Qdrant Vector Search] Querying semantic collection '{self.settings.qdrant_memory_collection}'")
            hits = self._search_vector(query, channel_id=channel_id, limit=result_limit)
        if not hits:
            print(f"   [Fallback Search] Querying local keyword matching...")
            hits = self._search_keyword(query, channel_id=channel_id, limit=result_limit)

        if self._cache is not None:
            print(f"   [Redis Cache Write] Caching {len(hits)} results for query: '{query}'")
            self._cache.set_search(channel_id, query, hits)
        return hits[:result_limit]

    def _search_vector(
        self,
        query: str,
        *,
        channel_id: Optional[str],
        limit: int,
    ) -> list[MemoryRetrievalHit]:
        vector = self._embedder.embed(query)  # type: ignore[union-attr]
        raw_hits = self._embedding_store.search_memory_units(  # type: ignore[union-attr]
            vector,
            limit=max(limit * 2, 10),
            channel_id=channel_id,
        )
        hits: list[MemoryRetrievalHit] = []
        for raw in raw_hits:
            current_channel_id = str(raw.get("channel_id") or "")
            memory_id = str(raw.get("memory_id") or "")
            state = self.get_state(current_channel_id)
            unit = next((item for item in state.memory_store if item.memory_id == memory_id), None)
            if unit is None:
                unit = MemoryUnit(
                    memory_id=memory_id,
                    channel_id=current_channel_id,
                    unit_type=MemoryUnitType(str(raw.get("unit_type") or MemoryUnitType.context.value)),
                    summary=str(raw.get("summary") or ""),
                    importance=float(raw.get("importance") or 0.5),
                    unresolved=bool(raw.get("unresolved")),
                    owners=list(raw.get("owners") or []),
                    tags=list(raw.get("tags") or []),
                    source_message_ids=list(raw.get("source_message_ids") or []),
                )
            hits.append(
                rank_memory_hit(
                    unit,
                    semantic_score=float(raw.get("score") or 0.0),
                    query_channel_id=channel_id,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _search_keyword(
        self,
        query: str,
        *,
        channel_id: Optional[str],
        limit: int,
    ) -> list[MemoryRetrievalHit]:
        hits: list[MemoryRetrievalHit] = []
        if channel_id:
            channel_ids = [channel_id]
        elif self._storage is not None:
            channel_ids = self._storage.list_channel_ids()
        else:
            channel_ids = list(self._states.keys())

        for current_channel_id in channel_ids:
            state = self.get_state(current_channel_id)
            for unit in state.memory_store:
                semantic_score = _keyword_overlap_score(query, unit.summary)
                if semantic_score <= 0.0:
                    continue
                hits.append(
                    rank_memory_hit(
                        unit,
                        semantic_score=semantic_score,
                        query_channel_id=channel_id,
                    )
                )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


def create_channel_memory_service(
    settings: Optional[Settings] = None,
    model: Optional[GenerativeModel] = None,
) -> ChannelMemoryService:
    """Build a fully wired memory service with MongoDB, Redis, Qdrant, and embeddings."""
    from backend.embeddings import MemoryEmbeddingStore
    from backend.memory_cache import MemoryCache
    from backend.memory_embeddings import MemoryEmbedder
    from backend.storage import MemoryStorage

    cfg = settings or get_settings()
    storage: Optional[MemoryStorage] = None
    cache: Optional[MemoryCache] = None
    embedding_store: Optional[MemoryEmbeddingStore] = None
    embedder: Optional[MemoryEmbedder] = None

    try:
        storage = MemoryStorage(cfg)
    except Exception as exc:
        logger.warning("MongoDB memory storage unavailable: %s", exc)

    try:
        cache = MemoryCache(cfg)
    except Exception as exc:
        logger.warning("Memory cache unavailable: %s", exc)

    try:
        embedding_store = MemoryEmbeddingStore(cfg)
        embedder = MemoryEmbedder(cfg)
    except Exception as exc:
        logger.warning("Memory embeddings unavailable: %s", exc)

    return ChannelMemoryService(
        storage=storage,
        cache=cache,
        embedding_store=embedding_store,
        embedder=embedder,
        settings=cfg,
        model=model,
    )

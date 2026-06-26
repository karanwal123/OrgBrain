"""Qdrant operations for skill embedding storage and similarity search."""

import hashlib
import logging
from typing import Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.config import Settings, get_settings
from backend.schemas import MemoryUnit

logger = logging.getLogger(__name__)


def _skill_point_id(person: str, skill: str) -> UUID:
    """Derive a deterministic UUID for a person/skill pair."""
    digest = hashlib.sha256(f"{person}:{skill}".encode()).hexdigest()
    return UUID(digest[:32])


def _memory_point_id(channel_id: str, memory_id: str) -> UUID:
    """Derive a deterministic UUID for a channel memory unit."""
    digest = hashlib.sha256(f"{channel_id}:{memory_id}".encode()).hexdigest()
    return UUID(digest[:32])


def _create_qdrant_client(settings: Settings) -> QdrantClient:
    """Connect to Qdrant Cloud (URL + API key) or a local/self-hosted instance."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


class SkillEmbeddingStore:
    """Qdrant-backed store for employee skill embeddings."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Qdrant client and ensure the collection exists.

        Args:
            settings: Optional settings override.
        """
        self.settings = settings or get_settings()
        self._client = _create_qdrant_client(self.settings)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the skill embeddings collection if it does not exist."""
        collections = {c.name for c in self._client.get_collections().collections}
        if self.settings.qdrant_collection not in collections:
            self._client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=qmodels.VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", self.settings.qdrant_collection)

    def upsert_skill(
        self,
        person: str,
        skill: str,
        vector: list[float],
        *,
        role: Optional[str] = None,
        team: Optional[str] = None,
    ) -> None:
        """
        Upsert a skill embedding for an employee.

        Args:
            person: Employee identifier.
            skill: Skill label.
            vector: Embedding vector (length must match embedding_dimension).
            role: Optional role metadata.
            team: Optional team metadata.
        """
        if len(vector) != self.settings.embedding_dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match "
                f"expected {self.settings.embedding_dimension}"
            )

        point_id = _skill_point_id(person, skill)
        self._client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=[
                qmodels.PointStruct(
                    id=str(point_id),
                    vector=vector,
                    payload={
                        "person": person,
                        "skill": skill,
                        "role": role,
                        "team": team,
                    },
                )
            ],
        )
        logger.debug("Upserted skill embedding person=%s skill=%s", person, skill)

    def search_similar_skills(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        person_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Find employees with skills similar to a query embedding.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            person_filter: Optional filter to restrict results to one person.

        Returns:
            List of scored hits with person, skill, and metadata payloads.
        """
        query_filter = None
        if person_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="person",
                        match=qmodels.MatchValue(value=person_filter),
                    )
                ]
            )

        response = self._client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
        results = response.points

        return [
            {
                "person": hit.payload.get("person") if hit.payload else None,
                "skill": hit.payload.get("skill") if hit.payload else None,
                "role": hit.payload.get("role") if hit.payload else None,
                "team": hit.payload.get("team") if hit.payload else None,
                "score": hit.score,
            }
            for hit in results
        ]

    def delete_person_skills(self, person: str) -> None:
        """
        Remove all skill embeddings for a given person.

        Args:
            person: Employee identifier.
        """
        self._client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="person",
                            match=qmodels.MatchValue(value=person),
                        )
                    ]
                )
            ),
        )
        logger.info("Deleted skill embeddings for person=%s", person)


class MemoryEmbeddingStore:
    """Qdrant-backed store for compressed channel memory embeddings."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client = _create_qdrant_client(self.settings)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = {c.name for c in self._client.get_collections().collections}
        if self.settings.qdrant_memory_collection not in collections:
            self._client.create_collection(
                collection_name=self.settings.qdrant_memory_collection,
                vectors_config=qmodels.VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant memory collection: %s",
                self.settings.qdrant_memory_collection,
            )

    def upsert_memory_unit(self, unit: MemoryUnit, vector: list[float]) -> None:
        """Upsert an embedded compressed memory unit."""
        if len(vector) != self.settings.embedding_dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match expected {self.settings.embedding_dimension}"
            )

        point_id = _memory_point_id(unit.channel_id, unit.memory_id)
        print(f"[Qdrant UPSERT] Upserting memory point {unit.memory_id} (Channel: {unit.channel_id})")
        self._client.upsert(
            collection_name=self.settings.qdrant_memory_collection,
            points=[
                qmodels.PointStruct(
                    id=str(point_id),
                    vector=vector,
                    payload={
                        "channel_id": unit.channel_id,
                        "memory_id": unit.memory_id,
                        "unit_type": unit.unit_type.value,
                        "summary": unit.summary,
                        "importance": unit.importance,
                        "unresolved": unit.unresolved,
                        "updated_at": unit.updated_at.isoformat(),
                        "owners": unit.owners,
                        "tags": unit.tags,
                        "source_message_ids": unit.source_message_ids,
                    },
                )
            ],
        )

    def search_memory_units(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        channel_id: Optional[str] = None,
    ) -> list[dict]:
        """Search semantic memory units with an optional channel filter."""
        query_filter = None
        if channel_id:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="channel_id",
                        match=qmodels.MatchValue(value=channel_id),
                    )
                ]
            )

        print(f"[Qdrant SEARCH] Querying collection '{self.settings.qdrant_memory_collection}' (limit={limit}, channel_filter={channel_id or 'All'})")
        response = self._client.query_points(
            collection_name=self.settings.qdrant_memory_collection,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
        results = response.points
        print(f"   [Qdrant HIT] Found {len(results)} matches in semantic space")
        return [
            {
                "channel_id": hit.payload.get("channel_id"),
                "memory_id": hit.payload.get("memory_id"),
                "summary": hit.payload.get("summary"),
                "unit_type": hit.payload.get("unit_type"),
                "importance": hit.payload.get("importance"),
                "unresolved": hit.payload.get("unresolved"),
                "score": hit.score,
                "owners": hit.payload.get("owners", []),
                "tags": hit.payload.get("tags", []),
                "source_message_ids": hit.payload.get("source_message_ids", []),
            }
            for hit in results
        ]

    def delete_channel_memory(self, channel_id: str) -> None:
        """Delete all embedded memories for a channel."""
        print(f"[Qdrant DELETE] Removing all memory embeddings for channel: {channel_id}")
        self._client.delete(
            collection_name=self.settings.qdrant_memory_collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="channel_id",
                            match=qmodels.MatchValue(value=channel_id),
                        )
                    ]
                )
            ),
        )

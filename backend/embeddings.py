"""Qdrant operations for skill embedding storage and similarity search."""

import hashlib
import logging
from typing import Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _skill_point_id(person: str, skill: str) -> UUID:
    """Derive a deterministic UUID for a person/skill pair."""
    digest = hashlib.sha256(f"{person}:{skill}".encode()).hexdigest()
    return UUID(digest[:32])


class SkillEmbeddingStore:
    """Qdrant-backed store for employee skill embeddings."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Qdrant client and ensure the collection exists.

        Args:
            settings: Optional settings override.
        """
        self.settings = settings or get_settings()
        self._client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )
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

        results = self._client.search(
            collection_name=self.settings.qdrant_collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            {
                "person": hit.payload.get("person"),
                "skill": hit.payload.get("skill"),
                "role": hit.payload.get("role"),
                "team": hit.payload.get("team"),
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

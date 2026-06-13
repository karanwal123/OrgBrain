"""MongoDB operations for employee profile storage."""

import logging
import re
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from backend.config import Settings, get_settings
from backend.schemas import EmployeeProfile, ExtractionResult

logger = logging.getLogger(__name__)


class ProfileStorage:
    """MongoDB-backed storage for living employee profiles."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize MongoDB client and profile collection.

        Args:
            settings: Optional settings override.
        """
        self.settings = settings or get_settings()
        self._client = MongoClient(self.settings.mongodb_uri)
        self._db: Database = self._client[self.settings.mongodb_database]
        self._collection: Collection = self._db[self.settings.mongodb_profiles_collection]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create indexes for fast person lookups."""
        self._collection.create_index("person", unique=True)
        self._collection.create_index("skills")
        self._collection.create_index("projects")

    def close(self) -> None:
        """Close the MongoDB client connection."""
        self._client.close()

    def get_profile(self, person: str) -> Optional[EmployeeProfile]:
        """
        Fetch an employee profile by person identifier.

        Args:
            person: Canonical person name or Slack identifier.

        Returns:
            EmployeeProfile if found, otherwise None.
        """
        doc = self._collection.find_one({"person": person})
        if not doc:
            return None
        doc.pop("_id", None)
        return EmployeeProfile.model_validate(doc)

    def find_profile_by_name(self, name: str) -> Optional[EmployeeProfile]:
        """
        Find a profile by exact or partial name match (case-insensitive).

        Args:
            name: Person name or substring to search for.

        Returns:
            First matching EmployeeProfile, or None.
        """
        name = name.strip()
        if not name:
            return None

        exact = self.get_profile(name)
        if exact:
            return exact

        doc = self._collection.find_one(
            {"person": {"$regex": f"^{re.escape(name)}$", "$options": "i"}}
        )
        if doc:
            doc.pop("_id", None)
            return EmployeeProfile.model_validate(doc)

        pattern = re.compile(re.escape(name), re.IGNORECASE)
        doc = self._collection.find_one({"person": pattern})
        if doc:
            doc.pop("_id", None)
            return EmployeeProfile.model_validate(doc)

        return None

    def upsert_from_extraction(
        self,
        extraction: ExtractionResult,
        source_message: str,
    ) -> EmployeeProfile:
        """
        Create or update an employee profile from an extraction result.

        Args:
            extraction: Structured extraction from a Slack message.
            source_message: Original message text for provenance tracking.

        Returns:
            The persisted EmployeeProfile.
        """
        existing = self.get_profile(extraction.person)
        if existing:
            profile = existing.merge_extraction(extraction, source_message)
        else:
            profile = EmployeeProfile(
                person=extraction.person,
                role=extraction.role,
                company=extraction.company,
                skills=extraction.skills,
                projects=extraction.projects,
                confidence=extraction.confidence,
                source_messages=[source_message],
            )

        payload = profile.model_dump()
        self._collection.update_one(
            {"person": profile.person},
            {"$set": payload},
            upsert=True,
        )
        logger.info("Upserted profile for person=%s", profile.person)
        return profile

    def list_profiles(self, limit: int = 100) -> list[EmployeeProfile]:
        """
        List employee profiles ordered by most recently updated.

        Args:
            limit: Maximum number of profiles to return.

        Returns:
            List of EmployeeProfile documents.
        """
        cursor = self._collection.find().sort("updated_at", -1).limit(limit)
        profiles: list[EmployeeProfile] = []
        for doc in cursor:
            doc.pop("_id", None)
            profiles.append(EmployeeProfile.model_validate(doc))
        return profiles

    def search_by_skill(self, skill: str, limit: int = 10) -> list[EmployeeProfile]:
        """
        Find employee profiles matching a skill, project, or expertise area.

        Args:
            skill: Search term (case-insensitive substring match).
            limit: Maximum number of profiles to return.

        Returns:
            Profiles sorted by confidence descending.
        """
        pattern = re.compile(re.escape(skill.strip()), re.IGNORECASE)
        query = {
            "$or": [
                {"skills": {"$regex": pattern}},
                {"areas_of_expertise": {"$regex": pattern}},
                {"projects": {"$regex": pattern}},
            ]
        }
        cursor = self._collection.find(query).sort("confidence", -1).limit(limit)
        profiles: list[EmployeeProfile] = []
        for doc in cursor:
            doc.pop("_id", None)
            profiles.append(EmployeeProfile.model_validate(doc))
        return profiles

    def search_by_topics(
        self,
        topics: list[str],
        *,
        limit_per_topic: int = 5,
    ) -> list[EmployeeProfile]:
        """
        Find profiles matching any of the given topics, deduplicated by person.

        Args:
            topics: Skill or domain search terms.
            limit_per_topic: Max results per topic before deduplication.

        Returns:
            Profiles sorted by confidence descending.
        """
        seen: set[str] = set()
        profiles: list[EmployeeProfile] = []
        for topic in topics:
            topic = topic.strip()
            if not topic:
                continue
            for profile in self.search_by_skill(topic, limit=limit_per_topic):
                if profile.person not in seen:
                    seen.add(profile.person)
                    profiles.append(profile)
        profiles.sort(key=lambda p: p.confidence, reverse=True)
        return profiles

    def ping(self) -> dict:
        """
        Check MongoDB connectivity and return basic collection stats.

        Returns:
            Dict with status, database name, collection name, and profile count.
        """
        self._client.admin.command("ping")
        count = self._collection.count_documents({})
        return {
            "status": "ok",
            "database": self.settings.mongodb_database,
            "collection": self.settings.mongodb_profiles_collection,
            "profile_count": count,
        }

    def delete_profile(self, person: str) -> bool:
        """
        Delete an employee profile by person identifier.

        Args:
            person: Canonical person name or Slack identifier.

        Returns:
            True if a document was deleted, False otherwise.
        """
        result = self._collection.delete_one({"person": person})
        return result.deleted_count > 0

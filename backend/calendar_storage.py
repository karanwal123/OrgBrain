"""MongoDB storage for employee availability (calendar feature)."""

import logging
from datetime import datetime
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from backend.config import Settings, get_settings
from backend.schemas import AvailabilityEntry

logger = logging.getLogger(__name__)


class AvailabilityStorage:
    """MongoDB-backed storage for employee availability entries."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize MongoDB client and availability collection.

        Args:
            settings: Optional settings override.
        """
        self.settings = settings or get_settings()
        self._client = MongoClient(self.settings.mongodb_uri)
        self._db: Database = self._client[self.settings.mongodb_database]
        self._collection: Collection = self._db[
            self.settings.mongodb_availability_collection
        ]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create indexes for fast availability lookups."""
        self._collection.create_index([("date_start", 1), ("date_end", 1)])
        self._collection.create_index([("user_id", 1), ("date_start", 1)])
        self._collection.create_index([("status", 1), ("date_start", 1)])

    def close(self) -> None:
        """Close the MongoDB client connection."""
        self._client.close()

    def save_entry(self, entry: AvailabilityEntry) -> AvailabilityEntry:
        """
        Save an availability entry, overwriting any existing entries for the
        same user on the same date range.

        This implements upsert-overwrite: deletes old entries that overlap the
        same user + date_start + date_end first, then inserts the new one.

        Args:
            entry: The availability entry to save.

        Returns:
            The saved AvailabilityEntry.
        """
        # Remove existing entries for this user on these exact dates
        self._collection.delete_many({
            "user_id": entry.user_id,
            "date_start": entry.date_start,
            "date_end": entry.date_end,
        })
        entry.updated_at = datetime.utcnow()
        self._collection.insert_one(entry.model_dump())
        logger.info(
            "Saved availability: user=%s status=%s date=%s-%s",
            entry.user_id,
            entry.status,
            entry.date_start,
            entry.date_end,
        )
        return entry

    def get_entries_for_date(self, date_str: str) -> list[AvailabilityEntry]:
        """
        Get all availability entries that cover a specific date.

        Args:
            date_str: Date in YYYY-MM-DD format.

        Returns:
            List of AvailabilityEntry documents.
        """
        cursor = self._collection.find({
            "date_start": {"$lte": date_str},
            "date_end": {"$gte": date_str},
        })
        entries: list[AvailabilityEntry] = []
        for doc in cursor:
            doc.pop("_id", None)
            entries.append(AvailabilityEntry.model_validate(doc))
        return entries

    def get_entries_for_user(
        self,
        user_id: str,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
    ) -> list[AvailabilityEntry]:
        """
        Get all availability entries for a specific user, optionally filtered
        by date range.

        Args:
            user_id: Slack user ID.
            date_start: Optional start date filter (inclusive).
            date_end: Optional end date filter (inclusive).

        Returns:
            List of AvailabilityEntry documents sorted by date.
        """
        query: dict = {"user_id": user_id}
        if date_start:
            query["date_end"] = {"$gte": date_start}
        if date_end:
            query["date_start"] = {"$lte": date_end}

        cursor = self._collection.find(query).sort("date_start", 1)
        entries: list[AvailabilityEntry] = []
        for doc in cursor:
            doc.pop("_id", None)
            entries.append(AvailabilityEntry.model_validate(doc))
        return entries

    def clear_user_date(self, user_id: str, date_str: str) -> int:
        """
        Remove all entries for a user on a specific date.

        Args:
            user_id: Slack user ID.
            date_str: Date in YYYY-MM-DD format.

        Returns:
            Number of deleted documents.
        """
        result = self._collection.delete_many({
            "user_id": user_id,
            "date_start": {"$lte": date_str},
            "date_end": {"$gte": date_str},
        })
        logger.info(
            "Cleared %d entries for user=%s date=%s",
            result.deleted_count,
            user_id,
            date_str,
        )
        return result.deleted_count

    def clear_user_range(
        self, user_id: str, date_start: str, date_end: str
    ) -> int:
        """
        Remove all entries for a user across a date range.

        Args:
            user_id: Slack user ID.
            date_start: Start date YYYY-MM-DD.
            date_end: End date YYYY-MM-DD.

        Returns:
            Number of deleted documents.
        """
        result = self._collection.delete_many({
            "user_id": user_id,
            "date_start": {"$lte": date_end},
            "date_end": {"$gte": date_start},
        })
        return result.deleted_count

    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a single entry by MongoDB ObjectId string.

        Args:
            entry_id: String representation of the ObjectId.

        Returns:
            True if deleted, False otherwise.
        """
        from bson import ObjectId

        result = self._collection.delete_one({"_id": ObjectId(entry_id)})
        return result.deleted_count > 0

    def get_upcoming_team_entries(self, date_start: str, limit: int = 50) -> list[AvailabilityEntry]:
        """
        Get upcoming availability entries for the entire team, sorted by date.

        Args:
            date_start: The ISO date string (YYYY-MM-DD) from which to fetch.
            limit: Maximum number of entries to return.

        Returns:
            List of AvailabilityEntry documents.
        """
        cursor = self._collection.find({
            "date_end": {"$gte": date_start}
        }).sort("date_start", 1).limit(limit)
        entries: list[AvailabilityEntry] = []
        for doc in cursor:
            doc.pop("_id", None)
            entries.append(AvailabilityEntry.model_validate(doc))
        return entries

    def ping(self) -> dict:
        """Check MongoDB connectivity and return collection stats."""
        self._client.admin.command("ping")
        count = self._collection.count_documents({})
        return {
            "status": "ok",
            "database": self.settings.mongodb_database,
            "collection": self.settings.mongodb_availability_collection,
            "entry_count": count,
        }

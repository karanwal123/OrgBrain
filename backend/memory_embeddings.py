"""Text embedding helpers for semantic channel memory retrieval."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Optional

import vertexai
from vertexai.language_models import TextEmbeddingModel

from backend.config import Settings, configure_google_credentials, get_settings

logger = logging.getLogger(__name__)


def _hash_embedding(text: str, dimension: int) -> list[float]:
    """Deterministic pseudo-embedding for local/dev fallback."""
    digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
    values: list[float] = []
    seed = int.from_bytes(digest[:8], "big")
    while len(values) < dimension:
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        values.append((seed / 0x7FFFFFFF) * 2.0 - 1.0)
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class MemoryEmbedder:
    """Vertex AI text embeddings with hash fallback when unavailable."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._model: Optional[TextEmbeddingModel] = None
        try:
            configure_google_credentials(self.settings)
            vertexai.init(
                project=self.settings.google_cloud_project,
                location=self.settings.google_cloud_location,
            )
            self._model = TextEmbeddingModel.from_pretrained(self.settings.embedding_model_name)
            logger.info("Memory embedder ready: %s", self.settings.embedding_model_name)
        except Exception as exc:
            logger.warning("Memory embedder using hash fallback: %s", exc)

    def embed(self, text: str) -> list[float]:
        """Return a normalized embedding vector for the given text."""
        clean = text.strip()
        if not clean:
            return [0.0] * self.settings.embedding_dimension

        if self._model is not None:
            try:
                vectors = self._model.get_embeddings([clean])
                values = list(vectors[0].values)
                if len(values) == self.settings.embedding_dimension:
                    return values
            except Exception as exc:
                logger.warning("Embedding request failed, using hash fallback: %s", exc)

        return _hash_embedding(clean, self.settings.embedding_dimension)

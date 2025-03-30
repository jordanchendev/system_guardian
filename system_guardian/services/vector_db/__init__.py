"""Vector database services module."""

# Export types first to avoid circular imports
from system_guardian.services.vector_db.types import VectorRecord

# Export the client implementations
from system_guardian.services.vector_db.qdrant_client import (
    QdrantClient,
    get_qdrant_client,
)

__all__ = ["QdrantClient", "get_qdrant_client", "VectorRecord"]

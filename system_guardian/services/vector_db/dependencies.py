"""Qdrant dependencies module."""

from typing import AsyncGenerator, Callable, Any
import asyncio
from system_guardian.settings import settings
from fastapi import Depends
from loguru import logger


# Forward to the correct function
def get_qdrant_client():
    """Get the Qdrant client instance."""
    from system_guardian.services.vector_db.qdrant_client import (
        get_qdrant_client as _get_client,
    )

    return _get_client()


def get_client_dependency() -> Callable[[], Any]:
    """Get the Qdrant client factory function."""
    from system_guardian.services.vector_db.qdrant_client import get_qdrant_client

    return get_qdrant_client


async def get_qdrant_dependency(
    qdrant_client=Depends(get_client_dependency()),
) -> AsyncGenerator[Any, None]:
    """
    Get Qdrant client for FastAPI dependency injection.

    :param qdrant_client: Qdrant client instance
    :yields: QdrantClient instance
    """
    try:
        yield qdrant_client
    finally:
        # No cleanup needed, as client is a singleton managed by the lru_cache
        pass


async def initialize_vector_collections():
    """
    Initialize all required Qdrant collections during application startup.
    This ensures collections are checked once during startup rather than
    checking each time when uploading documents.
    """
    from system_guardian.services.vector_db.qdrant_client import get_qdrant_client

    # Standard collection names used in the application
    COLLECTIONS = {
        settings.qdrant_knowledge_collection_name: 1536,  # OpenAI embedding dimension
        settings.qdrant_incidents_collection_name: 1536,  # OpenAI embedding dimension
    }

    logger.info("Pre-initializing Qdrant collections...")
    client = get_qdrant_client()

    for collection_name, vector_size in COLLECTIONS.items():
        try:
            await client.ensure_collection_exists(
                collection_name=collection_name,
                vector_size=vector_size,
                distance="Cosine",
            )
            logger.info(f"Collection {collection_name} initialized")
        except Exception as e:
            logger.error(f"Failed to initialize collection {collection_name}: {e}")

    logger.info("Vector collections initialization completed")

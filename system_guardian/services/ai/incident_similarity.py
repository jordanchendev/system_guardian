"""Incident similarity service using vector embeddings."""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

# Use forward imports to avoid circular imports
from system_guardian.services.vector_db import types
from system_guardian.settings import settings
from system_guardian.services.ai.severity_classifier import SeverityClassifier


class IncidentEmbedding(BaseModel):
    """Incident embedding model for vector search."""

    incident_id: str
    title: str
    description: str
    severity: Optional[str] = None
    status: str
    source: str
    created_at: str
    embedding: Optional[List[float]] = None


class IncidentSimilarityService:
    """Service for finding similar incidents using vector embeddings."""

    # Collection name for incident vectors
    COLLECTION_NAME = "incident_vectors"
    # Vector size for OpenAI embeddings (text-embedding-3-small)
    VECTOR_SIZE = 1536

    def __init__(
        self,
        qdrant_client,  # Remove type annotation to avoid circular imports
        openai_client: Optional[AsyncOpenAI] = None,
        embedding_model: str = "text-embedding-3-small",
        ai_engine=None,
    ):
        """
        Initialize the incident similarity service.

        :param qdrant_client: Qdrant client for vector storage and search
        :param openai_client: OpenAI client for generating embeddings
        :param embedding_model: Model to use for embeddings
        :param ai_engine: Optional AIEngine instance for enhanced embeddings
        """
        self.qdrant = qdrant_client
        self.openai = openai_client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.embedding_model = embedding_model
        self.ai_engine = ai_engine

        # Lazy import to avoid circular imports
        self.severity_classifier = SeverityClassifier(
            openai_client=self.openai, ai_engine=self.ai_engine
        )

    async def ensure_collection_exists(self) -> bool:
        """
        Ensure that the incident vectors collection exists.

        :returns: True if collection exists or was created successfully
        """
        exists = await self.qdrant.collection_exists(self.COLLECTION_NAME)
        if not exists:
            return await self.qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vector_size=self.VECTOR_SIZE,
                distance="Cosine",
            )
        return True

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.

        :param text: Text to generate embedding for
        :returns: Vector embedding
        """
        try:
            # If we have an AIEngine, use it for generating embeddings
            if self.ai_engine:
                logger.debug(
                    f"Using AIEngine to generate embedding with model: {self.ai_engine.embedding_model}"
                )
                return await self.ai_engine.generate_embedding(text)

            # Fall back to standard OpenAI client
            logger.debug(
                f"Using standard OpenAI client to generate embedding with model: {self.embedding_model}"
            )
            response = await self.openai.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.VECTOR_SIZE

    def _generate_incident_text(self, incident: IncidentEmbedding) -> str:
        """
        Generate a text representation of an incident for embedding.

        :param incident: Incident data
        :returns: Text representation
        """
        return f"""Incident: {incident.title}
Description: {incident.description}
Severity: {incident.severity}
Source: {incident.source}
Status: {incident.status}"""

    def _generate_id(self, incident_id: str) -> str:
        """
        Generate a deterministic ID for an incident.

        :param incident_id: Original incident ID
        :returns: Deterministic ID
        """
        return hashlib.md5(incident_id.encode()).hexdigest()

    async def index_incident(self, incident: IncidentEmbedding) -> bool:
        """
        Index an incident in the vector database.

        :param incident: Incident data
        :returns: True if indexing was successful
        """
        # Ensure collection exists
        if not await self.ensure_collection_exists():
            logger.error("Failed to ensure collection exists")
            return False

        # Auto-classify severity if it's not provided or empty
        if (
            not incident.severity
            or incident.severity.lower() == "none"
            or incident.severity.strip() == ""
        ):
            try:
                logger.info(
                    f"Auto-classifying severity for incident {incident.incident_id}"
                )
                incident.severity = await self.severity_classifier.classify_severity(
                    incident_title=incident.title,
                    incident_description=incident.description,
                    source=incident.source,
                )
                logger.info(
                    f"Classified incident {incident.incident_id} with severity {incident.severity}"
                )
            except Exception as e:
                logger.error(f"Failed to auto-classify severity: {e}")
                # Default to medium if classification fails
                incident.severity = "medium"

        # Generate text for embedding
        incident_text = self._generate_incident_text(incident)

        # Generate embedding
        embedding = await self.generate_embedding(incident_text)

        # Create vector record
        vector_record = types.VectorRecord(
            id=self._generate_id(incident.incident_id),
            vector=embedding,
            metadata={
                "incident_id": incident.incident_id,
                "title": incident.title,
                "description": incident.description,
                "severity": incident.severity,
                "status": incident.status,
                "source": incident.source,
                "created_at": incident.created_at,
            },
        )

        # Upsert vector
        return await self.qdrant.upsert_vectors(
            collection_name=self.COLLECTION_NAME,
            vectors=[vector_record],
        )

    async def find_similar_incidents(
        self,
        query_text: str,
        limit: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find incidents similar to the given query text.

        :param query_text: Text to find similar incidents for
        :param limit: Maximum number of results to return
        :param filter_condition: Additional filter conditions
        :returns: List of similar incidents with similarity scores
        """
        # Ensure collection exists
        if not await self.ensure_collection_exists():
            logger.error("Failed to ensure collection exists")
            return []

        # If we have an AIEngine, use it directly for finding similar incidents
        if self.ai_engine:
            try:
                logger.debug(f"Using AIEngine to find similar incidents")
                similar_incidents = await self.ai_engine.find_similar_incidents(
                    incident_text=query_text,
                    limit=limit,
                    filter_condition=filter_condition,
                    min_similarity_score=0.5,
                )
                return similar_incidents
            except Exception as e:
                logger.error(f"Error using AIEngine for similarity search: {e}")
                # Fall back to standard search below

        # Standard embedding-based search
        # Generate embedding
        embedding = await self.generate_embedding(query_text)

        # Search for similar vectors
        results = await self.qdrant.search_vectors(
            collection_name=self.COLLECTION_NAME,
            query_vector=embedding,
            limit=limit,
            filter_condition=filter_condition,
        )

        # Convert to result format
        return [
            {
                "incident_id": r.metadata["incident_id"],
                "title": r.metadata["title"],
                "description": r.metadata["description"],
                "severity": r.metadata["severity"],
                "status": r.metadata["status"],
                "source": r.metadata["source"],
                "created_at": r.metadata["created_at"],
                "similarity_score": r.score,
            }
            for r in results
        ]

    async def delete_incident(self, incident_id: str) -> bool:
        """
        Delete an incident from the vector database.

        :param incident_id: ID of the incident to delete
        :returns: True if deletion was successful
        """
        return await self.qdrant.delete_vectors(
            collection_name=self.COLLECTION_NAME,
            vector_ids=[self._generate_id(incident_id)],
        )

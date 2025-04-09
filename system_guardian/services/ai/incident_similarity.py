"""Incident similarity service using vector embeddings."""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
import time
from datetime import datetime

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy import select

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
    COLLECTION_NAME = settings.qdrant_incidents_collection_name
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

        # Generate embedding using AIEngine
        embedding = await self.ai_engine.generate_embedding(incident_text)

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
        # Generate embedding using AIEngine
        embedding = await self.ai_engine.generate_embedding(query_text)

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

    async def find_related_incidents(
        self,
        db_session,
        incident_id: Optional[int] = None,
        query_text: Optional[str] = None,
        limit: int = 5,
        include_resolved: bool = True,
        min_similarity_score: float = 0.5,
    ) -> Dict:
        """
        Find similar past incidents and provide insights based on them.

        :param db_session: Database session
        :param incident_id: Optional ID of the incident to find related incidents for
        :param query_text: Optional text to search for related incidents
        :param limit: Maximum number of related incidents to return
        :param include_resolved: Whether to include resolved incidents
        :param min_similarity_score: Minimum similarity score for related incidents
        :return: Dict with related incidents, insights, and current incident
        """
        start_time = time.time()
        logger.info(
            f"Finding related incidents for incident_id={incident_id}, query_text_provided={bool(query_text)}"
        )

        try:
            incident_data = None
            final_query_text = query_text

            # If incident ID is provided, get incident details
            if incident_id:
                incident_data = await self._get_incident_details(
                    db_session, incident_id
                )

                if not incident_data:
                    raise ValueError(f"Incident with ID {incident_id} not found")

                # Create query text if not provided
                if not final_query_text:
                    final_query_text = await self._generate_query_text_from_incident(
                        incident_data
                    )

            if not final_query_text:
                raise ValueError("Either incident_id or query_text must be provided")

            # Define filter condition based on parameters
            filter_condition = {}
            if not include_resolved:
                filter_condition = {
                    "must": [
                        {"key": "status", "match": {"any": ["open", "investigating"]}}
                    ]
                }
                logger.debug("Applied filter to exclude resolved incidents")

            # Find similar incidents
            logger.debug(f"Searching for similar incidents with limit={limit}")
            similar_incidents_data = await self.find_similar_incidents(
                query_text=final_query_text,
                limit=limit,
                filter_condition=filter_condition,
            )

            # Filter by minimum similarity score
            similar_incidents_data = [
                incident
                for incident in similar_incidents_data
                if incident.get("similarity_score", 0) >= min_similarity_score
            ]
            logger.info(
                f"Found {len(similar_incidents_data)} similar incidents with score >= {min_similarity_score}"
            )

            # Convert to standardized format
            related_incidents = self._standardize_incident_results(
                similar_incidents_data,
                current_incident_id=incident_data["id"] if incident_data else None,
            )

            # Generate insights based on related incidents
            logger.debug("Generating insights based on related incidents")
            insights = await self.ai_engine.generate_insights(
                current_incident=incident_data, related_incidents=related_incidents
            )

            return {
                "incidents": related_incidents,
                "insights": insights,
                "current_incident": incident_data,
            }
        except Exception as e:
            logger.error(f"Error finding related incidents: {str(e)}", exc_info=True)
            raise
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"find_related_incidents completed in {processing_time:.2f}s")
            self.ai_engine._track_metric("total_processing_time", processing_time)

    async def _get_incident_details(self, db_session, incident_id: int) -> Dict:
        """
        Get incident details from the database.

        :param db_session: Database session
        :param incident_id: ID of the incident
        :return: Dictionary with incident details or None if not found
        """
        from system_guardian.db.models.incidents import Incident, Event, Resolution

        try:
            # Query the incident
            incident_query = select(Incident).where(Incident.id == incident_id)
            result = await db_session.execute(incident_query)
            incident = result.scalars().first()

            if not incident:
                logger.warning(f"Incident with ID {incident_id} not found")
                return None

            # Get all related events
            events_query = select(Event).where(Event.related_incident_id == incident_id)
            events_result = await db_session.execute(events_query)
            related_events = events_result.scalars().all()
            logger.debug(
                f"Found {len(related_events)} events for incident {incident_id}"
            )

            # Get resolution if any
            resolution_query = select(Resolution).where(
                Resolution.incident_id == incident_id
            )
            resolution_result = await db_session.execute(resolution_query)
            resolution = resolution_result.scalars().first()

            # Format incident data
            incident_data = {
                "id": incident.id,
                "title": incident.title,
                "description": incident.description,
                "severity": incident.severity,
                "status": incident.status,
                "source": incident.source,
                "created_at": incident.created_at.isoformat(),
                "resolved_at": (
                    incident.resolved_at.isoformat() if incident.resolved_at else None
                ),
                "resolution": resolution.suggestion if resolution else None,
                "similarity_score": 1.0,  # Perfect match with itself
                "events": [
                    {
                        "id": event.id,
                        "source": event.source,
                        "event_type": event.event_type,
                        "content": event.content,
                        "created_at": event.created_at.isoformat(),
                    }
                    for event in related_events
                ],
            }

            return incident_data
        except Exception as e:
            logger.error(f"Error getting incident details: {str(e)}", exc_info=True)
            return None

    async def _generate_query_text_from_incident(self, incident_data: Dict) -> str:
        """
        Generate query text from incident data for vector search.

        :param incident_data: Incident data
        :return: Query text
        """
        # Create a text representation from the incident title and description
        query_text = f"{incident_data.get('title', '')}"
        if incident_data.get("description"):
            query_text += f" {incident_data.get('description', '')}"

        # Add event summaries if available
        for event in incident_data.get("events", [])[:3]:  # Just use first 3 events
            if "summary" in event:
                query_text += f" {event['summary']}"

        # Truncate if too long
        if len(query_text) > 1000:
            query_text = query_text[:1000]

        return query_text

    def _standardize_incident_results(
        self, incidents: List[Dict], current_incident_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Standardize incident results from the vector database.

        :param incidents: List of incidents from the vector database
        :param current_incident_id: ID of the current incident to exclude
        :return: List of standardized incident dictionaries
        """
        standardized_incidents = []

        for incident in incidents:
            # Skip the current incident if it's in the results
            if current_incident_id and str(current_incident_id) == str(
                incident.get("incident_id")
            ):
                continue

            standardized_incidents.append(
                {
                    "id": (
                        int(incident.get("incident_id"))
                        if incident.get("incident_id")
                        else 0
                    ),
                    "title": incident.get("title", ""),
                    "description": incident.get("description", ""),
                    "severity": incident.get("severity", "medium"),
                    "status": incident.get("status", "unknown"),
                    "source": incident.get("source", "unknown"),
                    "created_at": incident.get(
                        "created_at", datetime.utcnow().isoformat()
                    ),
                    "resolved_at": incident.get("resolved_at"),
                    "resolution": incident.get("resolution"),
                    "similarity_score": incident.get("similarity_score", 0),
                }
            )

        logger.debug(f"Standardized {len(standardized_incidents)} incident results")
        return standardized_incidents

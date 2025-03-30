"""Incidents API views."""
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Body

from system_guardian.services.vector_db.qdrant_client import QdrantClient
from system_guardian.services.vector_db.dependencies import get_qdrant_dependency
from system_guardian.services.ai.incident_similarity import (
    IncidentSimilarityService,
    IncidentEmbedding,
)
from system_guardian.web.api.incidents.schema import (
    SimilarIncidentQuery,
    IncidentResponse,
    IndexResponse,
)


router = APIRouter()


@router.post("/similar", response_model=List[IncidentResponse])
async def find_similar_incidents(
    query: SimilarIncidentQuery = Body(...),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
) -> List[IncidentResponse]:
    """
    Find incidents similar to the provided query text.
    
    :param query: Parameters for similarity search
    :param qdrant_client: Qdrant client dependency
    :returns: List of similar incidents with similarity scores
    """
    # Initialize similarity service
    similarity_service = IncidentSimilarityService(qdrant_client=qdrant_client)
    
    # Build filter condition
    filter_condition = {}
    if query.filter_by_severity:
        filter_condition["severity"] = query.filter_by_severity
    if query.filter_by_source:
        filter_condition["source"] = query.filter_by_source
    if query.filter_by_status:
        filter_condition["status"] = query.filter_by_status
    
    # Use empty filter if no conditions specified
    filter_to_use = filter_condition if filter_condition else None
    
    # Find similar incidents
    similar_incidents = await similarity_service.find_similar_incidents(
        query_text=query.query_text,
        limit=query.limit,
        filter_condition=filter_to_use,
    )
    
    # Convert to response model
    return [IncidentResponse(**incident) for incident in similar_incidents]


@router.post("/index", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def index_incident(
    incident: IncidentEmbedding = Body(...),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
) -> IndexResponse:
    """
    Index an incident for similarity search.
    
    :param incident: Incident data to index
    :param qdrant_client: Qdrant client dependency
    :returns: Status of indexing operation
    :raises HTTPException: If indexing fails
    """
    # Initialize similarity service
    similarity_service = IncidentSimilarityService(qdrant_client=qdrant_client)
    
    # Index the incident (severity will be automatically classified if needed)
    success = await similarity_service.index_incident(incident)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index incident",
        )
    
    return IndexResponse(success=True) 
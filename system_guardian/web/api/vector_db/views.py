"""Vector database API views."""

from typing import Dict, List, Any
import os
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from system_guardian.services.vector_db.qdrant_client import QdrantClient
from system_guardian.services.vector_db.dependencies import get_qdrant_dependency
from system_guardian.db.dependencies import get_db_session
from system_guardian.db.models.incidents import Incident
from system_guardian.services.ai.engine import AIEngine
from system_guardian.services.vector_db.types import VectorRecord
from system_guardian.web.api.vector_db.schema import (
    CollectionResponse,
    CollectionsListResponse,
    CollectionInfoResponse,
    KnowledgeUploadResponse,
    SyncIncidentsResponse,
)
from system_guardian.settings import settings

router = APIRouter()

# Constants for knowledge base
KNOWLEDGE_COLLECTION_NAME = settings.qdrant_knowledge_collection_name
INCIDENTS_COLLECTION_NAME = settings.qdrant_incidents_collection_name
VECTOR_SIZE = 1536  # OpenAI embedding dimension
DISTANCE_METRIC = "Cosine"


@router.post(
    "/collections/{collection_name}",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_collection(
    collection_name: str,
    vector_size: int = Query(..., description="Dimension of the vector space"),
    distance: str = Query(
        "Cosine", description="Distance metric to use (Cosine, Euclid, Dot)"
    ),
    recreate: bool = Query(
        False, description="Whether to recreate the collection if it exists"
    ),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
) -> CollectionResponse:
    """
    Create a new vector collection.

    :param collection_name: Name of the collection
    :param vector_size: Dimension of the vector space
    :param distance: Distance metric to use (Cosine, Euclid, Dot)
    :param recreate: Whether to recreate the collection if it exists
    :param qdrant_client: Qdrant client instance
    :returns: Collection response indicating success or failure
    :raises HTTPException: If collection creation fails
    """
    result = await qdrant_client.create_collection(
        collection_name=collection_name,
        vector_size=vector_size,
        distance=distance,
        recreate=recreate,
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection {collection_name}",
        )

    return CollectionResponse(success=True)


@router.get(
    "/collections",
    response_model=CollectionsListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_collections(
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
) -> CollectionsListResponse:
    """
    List all vector collections.

    :param qdrant_client: Qdrant client instance
    :returns: List of collection names
    :raises HTTPException: If listing collections fails
    """
    loop = __import__("asyncio").get_event_loop()

    try:
        collections_info = await loop.run_in_executor(
            None, lambda: qdrant_client.client.get_collections().collections
        )
        return CollectionsListResponse(collections=[c.name for c in collections_info])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}",
        )


@router.get(
    "/collections/{collection_name}",
    response_model=CollectionInfoResponse,
    status_code=status.HTTP_200_OK,
)
async def get_collection_info(
    collection_name: str,
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
) -> CollectionInfoResponse:
    """
    Get information about a vector collection.

    :param collection_name: Name of the collection
    :param qdrant_client: Qdrant client instance
    :returns: Collection information
    :raises HTTPException: If collection does not exist
    """
    collection_info = await qdrant_client.get_collection_info(collection_name)

    if not collection_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_name} not found",
        )

    return CollectionInfoResponse(**collection_info)


@router.post(
    "/knowledge/upload",
    response_model=KnowledgeUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_knowledge(
    file: UploadFile = File(...),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
) -> KnowledgeUploadResponse:
    """
    Upload knowledge base file (markdown or txt) to Qdrant.

    :param file: The knowledge base file to upload
    :param qdrant_client: Qdrant client instance
    :returns: Upload response with collection info
    :raises HTTPException: If file processing or upload fails
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".md", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .md and .txt files are supported",
        )

    # Read and process file content
    content = await file.read()
    text_content = content.decode("utf-8").strip()

    # Add metadata with file name and creation time
    metadata = {"file_name": file.filename, "created_at": datetime.now().isoformat()}
    # Upload the entire content as one piece to Qdrant with metadata
    try:
        points_count = await qdrant_client.upsert_texts(
            collection_name=KNOWLEDGE_COLLECTION_NAME,
            texts=[text_content],
            metadata=metadata,
        )

        return KnowledgeUploadResponse(
            success=True,
            collection_name=KNOWLEDGE_COLLECTION_NAME,
            points_count=points_count,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload knowledge: {str(e)}",
        )


@router.post(
    "/incidents/sync",
    response_model=SyncIncidentsResponse,
    status_code=status.HTTP_200_OK,
)
async def sync_incidents(
    force_update: bool = Query(
        False, description="Force update all incidents even if they exist in Qdrant"
    ),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
    db_session: AsyncSession = Depends(get_db_session),
) -> SyncIncidentsResponse:
    """
    Sync all incidents from PostgreSQL to Qdrant vector database.
    This ensures all incidents are searchable via vector similarity.

    :param force_update: Whether to force update all incidents even if they exist
    :param qdrant_client: Qdrant client instance
    :param db_session: Database session
    :returns: Sync response with success status and counts
    :raises HTTPException: If sync operation fails
    """
    try:
        # Initialize AI engine for generating embeddings
        ai_engine = AIEngine(
            vector_db_client=qdrant_client,
            enable_metrics=True,
        )

        # Ensure incidents collection exists
        collection_exists = await qdrant_client.collection_exists(
            INCIDENTS_COLLECTION_NAME
        )
        if not collection_exists:
            success = await qdrant_client.create_collection(
                collection_name=INCIDENTS_COLLECTION_NAME,
                vector_size=VECTOR_SIZE,
                distance=DISTANCE_METRIC,
            )
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create collection {INCIDENTS_COLLECTION_NAME}",
                )

        # Get all incidents from PostgreSQL
        stmt = select(Incident)
        result = await db_session.execute(stmt)
        incidents = result.scalars().all()

        if not incidents:
            return SyncIncidentsResponse(
                success=True,
                total_incidents=0,
                synced_count=0,
                message="No incidents found to sync",
            )

        # Get existing incident IDs from Qdrant if not forcing update
        existing_ids = set()
        if not force_update:
            existing_ids = await qdrant_client.get_existing_records(
                collection_name=INCIDENTS_COLLECTION_NAME,
                vector_size=VECTOR_SIZE,
            )

        # Process each incident
        synced_count = 0
        skipped_count = 0
        for incident in incidents:
            try:
                incident_id_str = str(incident.id)

                # Skip if incident already exists and not forcing update
                if not force_update and incident_id_str in existing_ids:
                    skipped_count += 1
                    logger.debug(
                        f"Skipping incident {incident.id} as it already exists in Qdrant"
                    )
                    continue

                # Combine incident information for embedding
                incident_text = f"{incident.title}\n{incident.description or ''}"

                # Generate embedding
                embedding = await ai_engine.generate_embedding(incident_text)

                # Create metadata
                metadata = {
                    "incident_id": incident_id_str,
                    "title": incident.title,
                    "description": incident.description,
                    "severity": incident.severity,
                    "status": incident.status,
                    "source": incident.source,
                    "created_at": (
                        incident.created_at.isoformat() if incident.created_at else None
                    ),
                    "resolved_at": (
                        incident.resolved_at.isoformat()
                        if incident.resolved_at
                        else None
                    ),
                }

                # Create VectorRecord
                vector_record = VectorRecord(
                    id=incident_id_str,
                    vector=embedding,
                    metadata=metadata,
                )

                # Upsert to Qdrant
                success = await qdrant_client.upsert_vectors(
                    collection_name=INCIDENTS_COLLECTION_NAME,
                    vectors=[vector_record],
                )

                if success:
                    synced_count += 1
                    logger.info(f"Successfully synced incident {incident.id} to Qdrant")
                else:
                    logger.error(f"Failed to sync incident {incident.id} to Qdrant")

            except Exception as e:
                logger.error(f"Error processing incident {incident.id}: {str(e)}")
                continue

        message = (
            f"Successfully synced {synced_count} incidents"
            + (
                f", skipped {skipped_count} existing incidents"
                if skipped_count > 0
                else ""
            )
            + f" out of {len(incidents)} total incidents"
        )

        return SyncIncidentsResponse(
            success=True,
            total_incidents=len(incidents),
            synced_count=synced_count,
            message=message,
        )

    except Exception as e:
        logger.error(f"Error during incident sync: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync incidents: {str(e)}",
        )

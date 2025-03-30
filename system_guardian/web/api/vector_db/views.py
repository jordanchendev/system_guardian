"""Vector database API views."""

from typing import Dict, List, Any
import os
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from fastapi.responses import JSONResponse
from loguru import logger

from system_guardian.services.vector_db.qdrant_client import QdrantClient
from system_guardian.services.vector_db.dependencies import get_qdrant_dependency
from system_guardian.web.api.vector_db.schema import (
    CollectionResponse,
    CollectionsListResponse,
    CollectionInfoResponse,
    KnowledgeUploadResponse,
)

router = APIRouter()

# Constants for knowledge base
KNOWLEDGE_COLLECTION_NAME = "system_knowledge"
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

    # Upload the entire content as one piece to Qdrant
    try:
        points_count = await qdrant_client.upsert_texts(
            collection_name=KNOWLEDGE_COLLECTION_NAME, texts=[text_content]
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

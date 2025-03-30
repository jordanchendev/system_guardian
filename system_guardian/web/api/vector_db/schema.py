"""Schema for vector database API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CollectionResponse(BaseModel):
    """Response model for collection operations."""

    success: bool = Field(..., description="Indicates if the operation was successful")


class CollectionsListResponse(BaseModel):
    """Response model for listing collections."""

    collections: List[str] = Field(..., description="List of collection names")


class CollectionInfoResponse(BaseModel):
    """Response model for collection information."""

    name: str = Field(..., description="Collection name")
    vector_size: int = Field(..., description="Dimension of the vector space")
    distance: str = Field(..., description="Distance metric used (Cosine, Euclid, Dot)")
    points_count: int = Field(..., description="Number of vectors in the collection")


class KnowledgeUploadResponse(BaseModel):
    """Response model for knowledge upload operation."""

    success: bool = Field(..., description="Indicates if the operation was successful")
    collection_name: str = Field(
        ..., description="Name of the collection where knowledge was uploaded"
    )
    points_count: int = Field(..., description="Number of knowledge points uploaded")

"""Shared types for vector database module."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """Vector record with metadata."""

    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

"""Incidents API schemas."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SimilarIncidentQuery(BaseModel):
    """Parameters for similar incident search."""
    
    query_text: str
    limit: int = 5
    filter_by_severity: Optional[str] = None
    filter_by_source: Optional[str] = None
    filter_by_status: Optional[str] = None


class IncidentResponse(BaseModel):
    """Incident response model."""
    
    id: str
    title: str
    description: Optional[str] = None
    severity: Optional[str] = None
    similarity_score: float
    source: Optional[str] = None


class IndexResponse(BaseModel):
    """Response for index operation."""
    
    success: bool


class SeverityClassificationRequest(BaseModel):
    """Request for severity classification."""
    
    incident_id: Optional[int] = None
    title: str
    description: Optional[str] = None
    source: Optional[str] = None
    events_data: Optional[List[Dict[str, Any]]] = None


class SeverityClassificationResponse(BaseModel):
    """Response for severity classification."""
    
    incident_id: Optional[int] = None
    predicted_severity: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)  # Confidence score between 0-1 
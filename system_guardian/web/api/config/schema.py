"""Schema for configuration API endpoints."""
from pydantic import BaseModel
from typing import Dict, Any, Optional


class ConfigResponse(BaseModel):
    """Response for configuration endpoints."""
    
    success: bool
    message: str
    config_path: Optional[str] = None
    
    
class StatusResponse(BaseModel):
    """Status response for configuration operations."""
    
    enabled: bool
    source_configs: Dict[str, Any] 
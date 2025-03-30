"""Schema for Datadog webhook events."""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


from pydantic import validator


class DatadogAlert(BaseModel):
    """Datadog alert data model."""

    title: str = Field(..., description="Alert title")
    text: str = Field(..., description="Alert description")
    alert_id: str = Field(..., description="Alert ID")
    alert_status: str = Field(
        ..., description="Alert status (triggered, recovered, etc.)"
    )
    alert_metric: str = Field(..., description="Alert metric name")
    alert_tags: List[str] = Field(default_factory=list, description="Alert tags")
    alert_created_at: int = Field(..., description="Alert creation timestamp")
    alert_updated_at: int = Field(..., description="Alert last update timestamp")
    org_id: str = Field(..., description="Organization ID")
    org_name: str = Field(..., description="Organization name")
    host: Optional[str] = Field(None, description="Host name")

    @validator("alert_tags", pre=True)
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        elif isinstance(v, list):
            return v
        return []


class DatadogWebhookRequest(BaseModel):
    """Datadog webhook request model."""

    alerts: List[DatadogAlert] = Field(..., description="List of alerts")


class DatadogWebhookResponse(BaseModel):
    """Datadog webhook response model."""

    success: bool = Field(
        ..., description="Whether the webhook was processed successfully"
    )
    message: str = Field(..., description="Response message")
    processed_alerts: int = Field(..., description="Number of alerts processed")

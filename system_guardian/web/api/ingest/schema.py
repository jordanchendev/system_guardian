"""Schema for ingest API."""

from typing import Any, Dict, Optional
import json
from pydantic import BaseModel, Field, deprecated
from datetime import datetime


class StandardEventMessage(BaseModel):
    """
    Standard model for all event messages in the system.

    This is the recommended format for all event messages throughout the system.
    All webhook handlers and event producers should use this format.
    """

    source: str = Field(
        ..., description="Source of the event (e.g., 'github', 'jira', 'datadog')"
    )
    event_type: str = Field(
        ..., description="Type of the event (e.g., 'push', 'issue', 'alert')"
    )
    event_id: str = Field(..., description="Unique identifier for the event")
    timestamp: Any = Field(
        ..., description="Event timestamp (string ISO format or integer unix timestamp)"
    )
    raw_payload: Dict[str, Any] = Field(
        ..., description="Raw event payload with all original data"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary."""
        data = {
            "source": self.source,
            "event_type": self.event_type,
            "event_id": self.event_id,
            "raw_payload": self.raw_payload,
        }

        # Handle datetime timestamp
        if isinstance(self.timestamp, datetime):
            data["timestamp"] = self.timestamp.isoformat()
        else:
            data["timestamp"] = self.timestamp

        return data

    def to_json(self) -> str:
        """Convert the event to a JSON string."""
        return json.dumps(self.to_dict())

    def model_dump(self) -> Dict[str, Any]:
        """Override model_dump to handle datetime objects."""
        data = super().model_dump()

        # Handle datetime timestamp
        if isinstance(data["timestamp"], datetime):
            data["timestamp"] = data["timestamp"].isoformat()

        return data

from typing import Any, Dict, Union

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Simple message model."""

    message: Union[str, Dict[str, Any], Any] = Field(
        description="Can be a simple string message or a GitHub webhook payload"
    )

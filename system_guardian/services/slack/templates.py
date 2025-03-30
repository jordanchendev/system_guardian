"""Slack message templates for notifications."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SlackMessageTemplate(BaseModel):
    """Base class for Slack message templates."""

    fallback_text: str = Field(
        ..., description="Fallback text for clients that don't support blocks"
    )
    blocks: Optional[List[Dict[str, Any]]] = Field(
        None, description="Slack blocks for formatting"
    )
    attachments: Optional[List[Dict[str, Any]]] = Field(
        None, description="Slack attachments"
    )

    @classmethod
    def create_simple_message(cls, message: str) -> "SlackMessageTemplate":
        """
        Create a simple message template.

        Args:
            message: Message text

        Returns:
            SlackMessageTemplate instance
        """
        return cls(
            fallback_text=message,
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message,
                    },
                }
            ],
        )

    @classmethod
    def create_alert(
        cls,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        details: Optional[str] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
    ) -> "SlackMessageTemplate":
        """
        Create an alert message template.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            details: Additional details
            actions: Action buttons

        Returns:
            SlackMessageTemplate instance
        """
        # Map severity to color
        color_map = {
            AlertSeverity.INFO: "#36C5F0",  # Blue
            AlertSeverity.WARNING: "#ECB22E",  # Yellow
            AlertSeverity.ERROR: "#E01E5A",  # Red
            AlertSeverity.CRITICAL: "#7B0000",  # Dark Red
        }
        color = color_map.get(severity, "#36C5F0")  # Default to blue

        # Map severity to emoji
        emoji_map = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.ERROR: ":x:",
            AlertSeverity.CRITICAL: ":rotating_light:",
        }
        emoji = emoji_map.get(severity, ":information_source:")

        # Create blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message,
                },
            },
            {
                "type": "divider",
            },
        ]

        # Add details if provided
        if details:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Details:*\n{details}",
                    },
                }
            )

        # Add actions if provided
        if actions:
            action_block = {
                "type": "actions",
                "elements": actions,
            }
            blocks.append(action_block)

        # Create attachments for color
        attachments = [{"color": color}]

        return cls(
            fallback_text=f"{title}: {message}",
            blocks=blocks,
            attachments=attachments,
        )

    @classmethod
    def create_system_status(
        cls,
        status: str,
        metrics: Dict[str, Any],
        details: Optional[str] = None,
    ) -> "SlackMessageTemplate":
        """
        Create a system status message template.

        Args:
            status: System status (e.g., "Healthy", "Degraded", "Down")
            metrics: Key metrics to display
            details: Additional details

        Returns:
            SlackMessageTemplate instance
        """
        # Determine status color and emoji
        status_lower = status.lower()
        if "healthy" in status_lower or "normal" in status_lower:
            color = "#36a64f"  # Green
            emoji = ":white_check_mark:"
        elif "degraded" in status_lower or "warning" in status_lower:
            color = "#ECB22E"  # Yellow
            emoji = ":warning:"
        else:
            color = "#E01E5A"  # Red
            emoji = ":x:"

        # Format metrics
        metrics_text = "\n".join(
            [f"• *{key}:* {value}" for key, value in metrics.items()]
        )

        # Create blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} System Status: {status}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Key Metrics:*\n" + metrics_text,
                },
            },
        ]

        # Add details if provided
        if details:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Additional Information:*\n{details}",
                    },
                }
            )

        # Add timestamp
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Updated: <!date^{int(__import__('time').time())}^{{date_num}} {{time_secs}}|now>",
                    }
                ],
            }
        )

        # Create attachments for color
        attachments = [{"color": color}]

        return cls(
            fallback_text=f"System Status: {status} - {metrics_text.replace('• *', '').replace(':*', ':')}",
            blocks=blocks,
            attachments=attachments,
        )

    @classmethod
    def create_incident_notification(
        cls,
        incident_id: str,
        title: str,
        severity: AlertSeverity,
        description: str,
        timestamp: Optional[str] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
    ) -> "SlackMessageTemplate":
        """
        Create an incident notification template.

        Args:
            incident_id: Incident ID
            title: Incident title
            severity: Incident severity
            description: Incident description
            timestamp: Incident timestamp
            actions: Action buttons

        Returns:
            SlackMessageTemplate instance
        """
        # Map severity to color
        color_map = {
            AlertSeverity.INFO: "#36C5F0",  # Blue
            AlertSeverity.WARNING: "#ECB22E",  # Yellow
            AlertSeverity.ERROR: "#E01E5A",  # Red
            AlertSeverity.CRITICAL: "#7B0000",  # Dark Red
        }
        color = color_map.get(severity, "#36C5F0")  # Default to blue

        # Map severity to emoji and text
        emoji_map = {
            AlertSeverity.INFO: ":information_source: Low",
            AlertSeverity.WARNING: ":warning: Medium",
            AlertSeverity.ERROR: ":x: High",
            AlertSeverity.CRITICAL: ":rotating_light: Critical",
        }
        severity_display = emoji_map.get(severity, ":information_source: Info")

        # Create blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Incident Alert: {title}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*ID:*\n{incident_id}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{severity_display}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{description}",
                },
            },
        ]

        # Add timestamp if provided
        if timestamp:
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:* {timestamp}",
                        }
                    ],
                }
            )
        else:
            # Use current time
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:* <!date^{int(__import__('time').time())}^{{date_num}} {{time_secs}}|now>",
                        }
                    ],
                }
            )

        # Add actions if provided
        if actions:
            action_block = {
                "type": "actions",
                "elements": actions,
            }
            blocks.append(action_block)

        # Create attachments for color
        attachments = [{"color": color}]

        return cls(
            fallback_text=f"Incident {incident_id}: {title} ({severity.value}) - {description}",
            blocks=blocks,
            attachments=attachments,
        )

"""Example usage of the Slack client."""

import asyncio
import logging
from typing import Dict, Any

from system_guardian.services.slack.client import SlackClient
from system_guardian.services.slack.templates import AlertSeverity, SlackMessageTemplate

logger = logging.getLogger(__name__)


async def send_simple_message() -> Dict[str, Any]:
    """Send a simple message to Slack."""
    client = SlackClient()
    result = await client.send_message(
        text="Hello from System Guardian!",
    )
    logger.info("Sent simple message: %s", result)
    return result


async def send_alert() -> Dict[str, Any]:
    """Send an alert to Slack."""
    client = SlackClient()
    template = SlackMessageTemplate.create_alert(
        title="System Alert",
        message="High CPU usage detected in production environment.",
        severity=AlertSeverity.WARNING,
        details="CPU usage at 85% for over 5 minutes on web server cluster.",
    )
    result = await client.send_template(template)
    logger.info("Sent alert message: %s", result)
    return result


async def send_system_status() -> Dict[str, Any]:
    """Send a system status message to Slack."""
    client = SlackClient()
    template = SlackMessageTemplate.create_system_status(
        status="Degraded",
        metrics={
            "CPU": "85%",
            "Memory": "70%",
            "Disk": "60%",
            "Response Time": "250ms",
            "Error Rate": "2.5%",
        },
        details="Database connection pool showing intermittent failures. Engineering team is investigating.",
    )
    result = await client.send_template(template)
    logger.info("Sent system status message: %s", result)
    return result


async def send_incident_notification() -> Dict[str, Any]:
    """Send an incident notification to Slack."""
    client = SlackClient()
    template = SlackMessageTemplate.create_incident_notification(
        incident_id="INC-2023-001",
        title="API Gateway Outage",
        severity=AlertSeverity.ERROR,
        description="API Gateway is currently experiencing intermittent 503 errors affecting 15% of requests.",
        actions=[
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View Dashboard",
                    "emoji": True,
                },
                "value": "view_dashboard",
                "url": "https://monitoring.example.com/dashboard/api-gateway",
            },
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Acknowledge",
                    "emoji": True,
                },
                "value": "acknowledge",
                "style": "primary",
            },
        ],
    )
    result = await client.send_template(template)
    logger.info("Sent incident notification: %s", result)
    return result


async def main() -> None:
    """Run example functions."""
    # Check if Slack is configured
    client = SlackClient()
    if not client.is_configured:
        logger.warning(
            "Slack client is not properly configured. Please set SYSTEM_GUARDIAN_SLACK_ENABLED=True and "
            "SYSTEM_GUARDIAN_SLACK_BOT_TOKEN in your .env file."
        )
        return

    # Send different types of messages
    await send_simple_message()
    await asyncio.sleep(1)
    await send_alert()
    await asyncio.sleep(1)
    await send_system_status()
    await asyncio.sleep(1)
    await send_incident_notification()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the async functions
    asyncio.run(main())

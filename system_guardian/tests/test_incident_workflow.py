"""Test script for simulating the complete incident workflow."""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from loguru import logger

from system_guardian.services.jira.client import JiraClient
from system_guardian.services.slack.client import SlackClient
from system_guardian.services.slack.templates import AlertSeverity
from system_guardian.services.ingest.message_publisher import MessagePublisher
from system_guardian.web.api.ingest.schema import StandardEventMessage


async def simulate_event_triggering_incident():
    """Simulate an event that triggers an incident detection."""
    logger.info("Simulating an event that will trigger incident detection")

    # Create a Slack client to check if it's configured
    slack_client = SlackClient()
    if not slack_client.is_configured:
        logger.warning("Slack notifications are disabled - check your .env file")

    # Create a JIRA client to check if it's configured
    jira_client = JiraClient()
    if not jira_client.is_configured:
        logger.warning("JIRA integration is disabled - check your .env file")

    # Create a mock event that would trigger an incident
    event = StandardEventMessage(
        source="test",
        event_type="critical_error",
        event_id="TEST-1234",
        timestamp=datetime.utcnow(),
        raw_payload={
            "alert_id": "ALERT-5678",
            "severity": "critical",
            "status": "firing",
            "title": "Database Connection Failure",
            "message": "The primary database connection has failed. Multiple services affected.",
            "host": "db-primary-01",
            "affected_services": ["api", "web", "auth"],
            "error_count": 15,
            "first_occurrence": datetime.utcnow().isoformat(),
        },
    )

    # Create mock incident info
    incident_id = 12324
    incident_info = {
        "created_at": datetime.utcnow().isoformat(),
        "severity": "critical",
        "title": "Database Connection Failure",
        "description": "The primary database connection has failed. Multiple services are affected including API, web, and authentication services. Errors have been occurring for the last 5 minutes.",
    }

    # 創建RMQ通道池（實際場景中會是真實連接）
    rmq_channel_pool = None

    logger.info(f"Triggering incident notification flow for incident #{incident_id}")

    # 調用發布事件通知方法
    await MessagePublisher.publish_incident_detection(
        rmq_channel_pool=rmq_channel_pool,
        event_message=event,
        incident_id=incident_id,
        incident_info=incident_info,
    )

    logger.info("Incident notification flow completed")
    logger.info("Check JIRA for a new ticket and Slack for a new notification")


async def run_simulation():
    """Run the incident workflow simulation."""
    logger.info("Starting incident workflow simulation...")

    await simulate_event_triggering_incident()

    logger.info("Simulation completed!")


if __name__ == "__main__":
    # Configure loguru logger
    import sys
    from system_guardian.logging_config import configure_logging

    # Configure logging with default settings
    configure_logging()

    # Run the simulation
    asyncio.run(run_simulation())

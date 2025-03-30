"""Test script for creating JIRA tickets for incidents."""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from loguru import logger

from system_guardian.services.jira.client import JiraClient
from system_guardian.services.consumers.event_consumer import EventConsumer


async def test_direct_jira_ticket_creation():
    """Test creating a JIRA ticket directly via JiraClient."""
    jira_client = JiraClient()

    # Check if JIRA is properly configured
    if not jira_client.is_configured:
        logger.error(
            "JIRA is not configured. Please check your .env file and set SYSTEM_GUARDIAN_JIRA_ENABLED=True."
        )
        return

    # Create a test ticket
    result = await jira_client.create_incident_ticket(
        incident_id="TEST-001",
        title="Test Incident Ticket",
        description="This is a test incident ticket created from the test script.",
        severity="medium",
        source="test",
        event_type="jira_test",
    )

    if "key" in result:
        logger.info(f"Successfully created JIRA ticket: {result['key']}")
    else:
        logger.error(
            f"Failed to create JIRA ticket: {result.get('error', 'unknown error')}"
        )


async def test_event_consumer_ticket_creation():
    """Test the event consumer incident notification processing for JIRA ticket creation."""
    # Create a mock incident data
    incident_data = {
        "incident_id": "TEST-002",
        "title": "Test Event Consumer Incident",
        "description": "This is a test incident notification processed by EventConsumer for JIRA ticket creation.",
        "severity": "high",
        "detection_time": datetime.utcnow().isoformat(),
        "original_event": {
            "source": "test",
            "event_type": "incident_test",
            "timestamp": datetime.utcnow().isoformat(),
        },
    }

    # Convert to JSON string
    message_body = json.dumps(incident_data)

    # Create a minimal EventConsumer instance (without DB and message broker connections)
    event_consumer = EventConsumer(
        db_session_factory=None,  # No DB needed for this test
        auto_incident_creation=False,
    )

    # Process the mock incident notification
    logger.info("Processing mock incident notification through EventConsumer...")
    await event_consumer.process_incident_notification(message_body)
    logger.info("Completed processing mock incident notification.")


async def run_tests():
    """Run all tests."""
    logger.info("Starting JIRA ticket creation tests...")

    # Run direct ticket creation test
    logger.info("Test 1: Direct JIRA ticket creation")
    await test_direct_jira_ticket_creation()

    # Wait a bit between tests
    await asyncio.sleep(2)

    # Run event consumer ticket creation test
    logger.info("Test 2: EventConsumer JIRA ticket creation")
    await test_event_consumer_ticket_creation()

    logger.info("All tests completed!")


if __name__ == "__main__":
    # Configure loguru logger
    import sys
    from system_guardian.logging_config import configure_logging

    # Configure logging with default settings
    configure_logging()

    # Run the tests
    asyncio.run(run_tests())

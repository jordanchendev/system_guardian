from system_guardian.services.slack.client import SlackClient
from system_guardian.services.slack.templates import SlackMessageTemplate, AlertSeverity
import asyncio
from loguru import logger

slack_client = SlackClient()


async def notify_simple():
    await slack_client.send_message("System Guardian is running")


async def notify_alert(title, message, severity=AlertSeverity.INFO):
    template = SlackMessageTemplate.create_alert(
        title=title, message=message, severity=severity
    )
    await slack_client.send_template(template)


async def notify_incident(incident_id, title, description, severity):
    template = SlackMessageTemplate.create_incident_notification(
        incident_id=incident_id, title=title, description=description, severity=severity
    )
    await slack_client.send_template(template)


async def run_all_tests():
    """Run all Slack message tests."""
    # 1. Send simple message
    logger.info("Sending simple message...")
    await notify_simple()
    await asyncio.sleep(1)

    # 2. Send alerts with different severity levels
    logger.info("Sending INFO level alert...")
    await notify_alert(
        "Information Notice",
        "System has successfully started, all services are running normally.",
        AlertSeverity.INFO,
    )
    await asyncio.sleep(1)

    logger.info("Sending WARNING level alert...")
    await notify_alert(
        "System Warning",
        "CPU usage has exceeded 80%, please monitor the system.",
        AlertSeverity.WARNING,
    )
    await asyncio.sleep(1)

    logger.info("Sending ERROR level alert...")
    await notify_alert(
        "System Error",
        "Main database connection failed, switched to backup database.",
        AlertSeverity.ERROR,
    )
    await asyncio.sleep(1)

    logger.info("Sending CRITICAL level alert...")
    await notify_alert(
        "Critical Alert",
        "Severe system resource shortage may cause service interruption.",
        AlertSeverity.CRITICAL,
    )
    await asyncio.sleep(1)

    # 3. Send incident notification
    logger.info("Sending incident notification...")
    await notify_incident(
        "INC-2024-001",
        "API Service Outage",
        "Core API service is experiencing 500 errors affecting 30% of user requests. Technical team is addressing the issue.",
        AlertSeverity.ERROR,
    )

    logger.info("All test messages have been sent successfully!")


if __name__ == "__main__":
    # Configure loguru logger
    import sys
    from system_guardian.logging_config import configure_logging

    # Configure logging with default settings
    configure_logging()

    # Check if Slack is properly configured
    if not slack_client.is_configured:
        logger.error(
            "Error: Slack client is not properly configured. Please ensure the following variables are set in your .env file:"
        )
        logger.error("  - SYSTEM_GUARDIAN_SLACK_ENABLED=True")
        logger.error(
            "  - SYSTEM_GUARDIAN_SLACK_BOT_TOKEN is set with a valid Bot Token"
        )
        logger.error(
            "  - SYSTEM_GUARDIAN_SLACK_CHANNEL_ID is set with a valid channel ID"
        )
    else:
        # Run the tests
        asyncio.run(run_all_tests())

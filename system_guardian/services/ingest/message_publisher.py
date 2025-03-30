"""Services for processing and forwarding webhook events."""

import asyncio
import re
from typing import Optional, Dict, List, Any
import json
from datetime import datetime

from loguru import logger
from aio_pika import Channel, Message, ExchangeType, DeliveryMode
from aio_pika.pool import Pool

# Import Slack notification components
from system_guardian.services.slack.client import SlackClient
from system_guardian.services.slack.templates import SlackMessageTemplate, AlertSeverity
from system_guardian.services.jira.client import JiraClient


class MessagePublisher:
    """Service for publishing messages to different message queues."""

    # Default topics mapping
    DEFAULT_TOPICS: Dict[str, str] = {
        "github": "github_events",
        "jira": "jira_events",
        "datadog": "datadog_events",
        # Add more sources here as needed
    }

    # Critical event types that should be routed to RabbitMQ for real-time processing
    CRITICAL_EVENT_TYPES: List[str] = [
        "error",
        "failure",
        "alert",
        "security",
        "outage",
        "incident",
        # Add more critical event types here
    ]

    # Incident-related topics and routing
    INCIDENT_TOPIC = "system_incidents"
    INCIDENT_ROUTING_KEY = "incidents.detected"
    INCIDENT_EXCHANGE = "system_incidents"

    @staticmethod
    def sanitize_topic_name(topic: str) -> str:
        """
        Sanitize the topic name.

        This method replaces any invalid characters with underscores.

        :param topic: The raw topic name
        :returns: A sanitized topic name
        """
        # Replace all non-alphanumeric characters except dots, underscores, and hyphens with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9\._-]", "_", topic)

        # Ensure the topic name doesn't start with a dot or underscore
        if sanitized and sanitized[0] in [".", "_"]:
            sanitized = "topic" + sanitized

        return sanitized

    @staticmethod
    def is_critical_event(event_message) -> bool:
        """
        Determine if an event should be considered critical and processed in real-time.

        :param event_message: The standardized event message
        :returns: True if the event is critical, False otherwise
        """
        # Check if event type contains any critical keywords
        event_type_lower = event_message.event_type.lower()
        if any(
            critical_type in event_type_lower
            for critical_type in MessagePublisher.CRITICAL_EVENT_TYPES
        ):
            return True

        # Check if the payload contains urgent/critical flags
        payload = event_message.raw_payload
        if isinstance(payload, dict):
            # Check for priority or severity indicators in common fields
            priority = payload.get("priority", "").lower()
            severity = payload.get("severity", "").lower()

            if any(word in priority for word in ["high", "urgent", "critical"]):
                return True

            if any(
                word in severity for word in ["high", "urgent", "critical", "fatal"]
            ):
                return True

        return False

    @staticmethod
    async def send_to_rabbitmq(
        channel_pool: Pool[Channel],
        event_message,
        exchange_name: str = "webhook_events",
        routing_key: Optional[str] = None,
    ) -> None:
        """
        Send a message to RabbitMQ.

        :param channel_pool: RabbitMQ channel pool
        :param event_message: The standardized event message (object or dict)
        :param exchange_name: The exchange to publish to
        :param routing_key: Optional routing key, if not provided uses source.eventtype format
        """
        if not routing_key:
            # Check if event_message is a dictionary
            if isinstance(event_message, dict):
                source = event_message.get("source", "unknown")
                event_type = event_message.get("event_type", "unknown")
                # Generate routing key based on source and event type
                routing_key = f"{source}.{event_type}"
            else:
                # Original object format
                # Generate routing key based on source and event type
                routing_key = f"{event_message.source}.{event_message.event_type}"

        try:
            logger.info(f"Sending message to RabbitMQ: {exchange_name}/{routing_key}")
            async with channel_pool.acquire() as conn:
                exchange = await conn.declare_exchange(
                    name=exchange_name,
                    type=ExchangeType.TOPIC,
                    durable=False,
                    auto_delete=False,
                )

                # Prepare message content
                message_body = event_message
                if not isinstance(event_message, str):
                    if isinstance(event_message, dict):
                        message_body = json.dumps(event_message).encode("utf-8")
                    else:
                        # If it's an object with to_json method
                        if hasattr(event_message, "to_json") and callable(
                            event_message.to_json
                        ):
                            message_body = event_message.to_json().encode("utf-8")
                        else:
                            # Try to convert directly to JSON
                            message_body = json.dumps(event_message.__dict__).encode(
                                "utf-8"
                            )

                await exchange.publish(
                    message=Message(
                        body=message_body,
                        content_encoding="utf-8",
                        content_type="application/json",
                        delivery_mode=DeliveryMode.PERSISTENT,
                    ),
                    routing_key=routing_key,
                )
            logger.info(
                f"Successfully published message to RabbitMQ with routing key: {routing_key}"
            )
        except Exception as e:
            logger.error(f"Failed to send message to RabbitMQ: {str(e)}")
            # Consider retrying or storing for later processing

    @staticmethod
    async def publish_incident_detection(
        rmq_channel_pool: Optional[Pool[Channel]],
        event_message,
        incident_id: int,
        incident_info: Dict[str, Any],
    ) -> None:
        """
        Publish a notification that an incident was detected from an event.

        :param rmq_channel_pool: RabbitMQ channel pool for real-time notifications
        :param event_message: The original event message that triggered the incident
        :param incident_id: The ID of the newly created incident
        :param incident_info: Additional information about the incident
        """
        # Create an enhanced message with incident information
        # Convert timestamp to ISO format string if it's a datetime object
        timestamp = event_message.timestamp
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()

        incident_payload = {
            "original_event": {
                "source": event_message.source,
                "event_type": event_message.event_type,
                "timestamp": timestamp,
            },
            "incident_id": incident_id,
            "detection_time": incident_info.get("created_at", ""),
            "severity": incident_info.get("severity", "unknown"),
            "title": incident_info.get("title", ""),
            "description": incident_info.get("description", ""),
        }

        # Always send to RabbitMQ for real-time notifications
        if rmq_channel_pool:
            try:
                # Simplified logging
                logger.info(
                    f"Publishing incident #{incident_id} notification to RabbitMQ"
                )
                async with rmq_channel_pool.acquire() as conn:
                    exchange = await conn.declare_exchange(
                        name=MessagePublisher.INCIDENT_EXCHANGE,
                        type=ExchangeType.DIRECT,
                        durable=False,
                        auto_delete=False,
                    )
                    await exchange.publish(
                        message=Message(
                            body=json.dumps(incident_payload).encode("utf-8"),
                            content_encoding="utf-8",
                            content_type="application/json",
                            delivery_mode=DeliveryMode.PERSISTENT,
                        ),
                        routing_key=MessagePublisher.INCIDENT_ROUTING_KEY,
                    )
            except Exception as e:
                logger.error(
                    f"Failed to publish incident notification to RabbitMQ: {str(e)}"
                )

        # Send Slack notification directly
        try:
            # Create SlackClient
            slack_client = SlackClient()

            # Skip if Slack is not configured
            if not slack_client.is_configured:
                logger.warning(
                    "Slack notifications disabled, skipping incident notification"
                )
                return

            # Map severity from incident to AlertSeverity
            severity_map = {
                "low": AlertSeverity.INFO,
                "medium": AlertSeverity.WARNING,
                "high": AlertSeverity.ERROR,
                "critical": AlertSeverity.CRITICAL,
                # Default mappings if the severity doesn't match exactly
                "info": AlertSeverity.INFO,
                "warning": AlertSeverity.WARNING,
                "error": AlertSeverity.ERROR,
            }

            # Get incident details
            severity_str = incident_info.get("severity", "medium").lower()
            detection_time = incident_info.get(
                "created_at", datetime.utcnow().isoformat()
            )

            # Map severity to AlertSeverity enum
            severity = severity_map.get(severity_str, AlertSeverity.WARNING)

            # Send notification using incident template
            logger.info(
                f"Sending direct Slack notification for incident #{incident_id}"
            )
            template = SlackMessageTemplate.create_incident_notification(
                incident_id=str(incident_id),
                title=incident_info.get("title", "Untitled Incident"),
                severity=severity,
                description=incident_info.get(
                    "description", "No description available"
                ),
                timestamp=detection_time,
            )

            await slack_client.send_template(template)
            logger.info(f"Direct Slack notification sent for incident #{incident_id}")

        except Exception as e:
            logger.error(
                f"Error sending direct Slack notification for incident: {str(e)}"
            )

        # Create JIRA ticket directly
        try:
            # Create JIRA client
            jira_client = JiraClient()

            # Skip if JIRA is not configured
            if not jira_client.is_configured:
                logger.warning(
                    "JIRA is not properly configured, incident ticket will not be created"
                )
                return

            # Get incident details
            title = incident_info.get("title", "Untitled Incident")
            description = incident_info.get("description", "No description available")
            severity_str = incident_info.get("severity", "medium")

            # Get source information if available
            source = event_message.source
            event_type = event_message.event_type

            # Log JIRA configuration
            logger.info(
                f"JIRA configuration: project_key={jira_client.project_key}, issue_type={jira_client.issue_type}"
            )

            # Create JIRA ticket
            logger.info(f"Creating direct JIRA ticket for incident #{incident_id}")
            result = await jira_client.create_incident_ticket(
                incident_id=str(incident_id),
                title=title,
                description=description,
                severity=severity_str,
                source=source,
                event_type=event_type,
            )

            if "key" in result:
                logger.info(
                    f"Direct JIRA ticket created successfully: {result['key']} for incident #{incident_id}"
                )
            else:
                error_code = result.get("error", "unknown error")
                error_details = result.get("details", "no details available")
                logger.error(
                    f"Failed to create direct JIRA ticket for incident #{incident_id}: {error_code}"
                )
                logger.error(f"Error details: {error_details}")

        except Exception as e:
            logger.error(f"Error creating direct JIRA ticket for incident: {str(e)}")

    @staticmethod
    async def publish_event(
        event_message,
        rmq_channel_pool: Optional[Pool[Channel]] = None,
        auto_detect_incident: bool = True,
    ) -> None:
        """
        Publish an event to RabbitMQ message queue.

        :param event_message: The standardized event message
        :param rmq_channel_pool: Optional RabbitMQ channel pool
        :param auto_detect_incident: Whether to automatically detect incidents from this event
        """
        tasks = []

        # Add auto_detect_incident flag to the event
        event_message.auto_detect_incident = auto_detect_incident

        logger.info(
            f"Publishing event from {event_message.source} with auto_detect_incident={auto_detect_incident}"
        )

        # Process with RabbitMQ
        if rmq_channel_pool:
            # Send event to RabbitMQ
            tasks.append(
                MessagePublisher.send_to_rabbitmq(
                    channel_pool=rmq_channel_pool,
                    event_message=event_message,
                )
            )

        if tasks:
            # Run all publishing tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.warning("No message queue configured for event publishing")

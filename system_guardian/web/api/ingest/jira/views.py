from typing import Optional, Dict, Any
import json
from urllib.parse import parse_qs

from fastapi import APIRouter, Header, Request, Depends, BackgroundTasks
from loguru import logger
from aio_pika import Channel
from aio_pika.pool import Pool

from system_guardian.web.api.ingest.jira.schema import Message
from system_guardian.web.api.ingest.schema import StandardEventMessage
from system_guardian.services.ingest import MessagePublisher
from system_guardian.services.rabbit.dependencies import get_rmq_channel_pool
from system_guardian.settings import settings

router = APIRouter()


def is_system_generated_ticket(body: Dict[str, Any]) -> bool:
    """
    Check if the JIRA event comes from a system auto-created ticket to avoid processing loops.

    :param body: JIRA webhook message content
    :return: True if the ticket is system-generated, False otherwise
    """
    try:
        # Check if the issue field exists
        if "issue" in body and "fields" in body["issue"]:
            fields = body["issue"]["fields"]

            # Check if labels contain system-generated identifiers
            if "labels" in fields and isinstance(fields["labels"], list):
                if any(
                    label in ["auto-created", "system-guardian"]
                    for label in fields["labels"]
                ):
                    logger.debug(
                        "Detected system labels: auto-created or system-guardian"
                    )
                    logger.debug(fields["labels"])
                    return True

            # Check if the title contains a specific system pattern
            if "summary" in fields and fields["summary"]:
                summary = fields["summary"]
                # Check if format is "[Incident #number]"
                if summary.startswith("[Incident #"):
                    logger.debug(f"Detected system-generated title format: {summary}")
                    return True

            # Check if description contains specific patterns
            if "description" in fields and fields["description"]:
                description = fields["description"]
                # Check if contains "Incident ID:" or "Automatically generated" text
                if (
                    "Incident ID:" in description
                    or "Automatically generated" in description
                ):
                    logger.debug(
                        "Detected description containing Incident ID or automatic generation markers"
                    )
                    return True

                # Check for multiple nested event history, indicating cascaded ticket creation
                incident_count = description.count("Incident ID:")
                if incident_count > 1:
                    logger.debug(
                        f"Detected description with multiple ({incident_count}) incident ID references, possible circular creation"
                    )
                    return True
    except Exception as e:
        logger.error(f"Error while checking system-generated ticket: {str(e)}")

    return False


@router.post("/", response_model=Message)
async def process_jira_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_jira_event: Optional[str] = Header(None, alias="X-Jira-Event"),
    rmq_channel_pool: Optional[Pool[Channel]] = Depends(get_rmq_channel_pool),
) -> Message:
    """
    Process Jira webhook events.

    This endpoint accepts Jira webhook payloads for various events
    and processes them according to the event type specified in the X-Jira-Event header.
    It also forwards the event to configured message queues.

    :param request: The incoming request object
    :param background_tasks: FastAPI background tasks object for async processing
    :param x_jira_event: Jira event type from X-Jira-Event header
    :param rmq_channel_pool: RabbitMQ channel pool dependency
    :returns: message indicating successful processing
    """
    logger.info(f"Received Jira event: {x_jira_event}")

    try:
        # Parse the request body as JSON
        body = await request.json()
        logger.debug(f"Received Jira webhook body: {body}")

        # Check if this is a system auto-generated ticket
        is_system_ticket = is_system_generated_ticket(body)
        if is_system_ticket:
            logger.warning(
                "Detected system auto-generated JIRA ticket, skipping processing to avoid infinite loops"
            )
            return Message(
                message={"status": "skipped", "reason": "system_generated_ticket"}
            )

        # Get default value for auto-creating incidents from JIRA events from settings
        auto_detect_incident = settings.jira_auto_detect_incident

        # Emphasize this point because JIRA's default setting is False
        logger.debug(
            f"JIRA auto_detect_incident set to: {auto_detect_incident} (default is typically False)"
        )

        event_type = x_jira_event or extract_event_type_from_body(body) or "unknown"

        # Create standardized event message
        event_message = StandardEventMessage(
            source="jira",
            event_type=event_type.split(":")[-1],
            event_id=str(body.get("id", "unknown")),
            timestamp=body.get("timestamp", ""),
            raw_payload=body,
        )

        # Forward to message queues in the background
        # This allows us to respond to the webhook quickly without waiting for message queue processing
        background_tasks.add_task(
            MessagePublisher.publish_event,
            event_message=event_message,
            rmq_channel_pool=rmq_channel_pool,
            auto_detect_incident=auto_detect_incident,
        )

        return Message(message=body)
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors
        error_msg = f"Invalid JSON in Jira webhook payload: {str(e)}"
        logger.error(error_msg)
        return Message(message={"error": error_msg})
    except Exception as e:
        # Handle any other unexpected errors
        error_msg = f"Error processing Jira webhook: {str(e)}"
        logger.exception(error_msg)
        return Message(message={"error": error_msg})


def extract_event_type_from_body(body: Dict[str, Any]) -> Optional[str]:
    """
    Extract the event type from the Jira webhook body if not provided in the header.

    :param body: The parsed JSON body of the Jira webhook
    :returns: The extracted event type or None if not found
    """
    # Different ways to extract event type based on Jira webhook format
    if "webhookEvent" in body:
        return body["webhookEvent"]
    if "issue_event_type_name" in body:
        return body["issue_event_type_name"]

    # Add more extraction logic as needed for different Jira webhook formats

    return None

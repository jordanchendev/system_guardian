"""Views for Datadog webhook events."""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from loguru import logger
from aio_pika import Channel
from aio_pika.pool import Pool
from datetime import datetime

from system_guardian.web.api.ingest.datadog.schema import (
    DatadogWebhookRequest,
    DatadogWebhookResponse,
)
from system_guardian.web.api.ingest.schema import StandardEventMessage
from system_guardian.services.ingest import MessagePublisher
from system_guardian.services.rabbit.dependencies import get_rmq_channel_pool
from system_guardian.services.ai.severity_classifier import SeverityClassifier
from system_guardian.settings import settings

router = APIRouter()


@router.post(
    "/",
    response_model=DatadogWebhookResponse,
    status_code=status.HTTP_200_OK,
)
async def handle_datadog_webhook(
    webhook_data: DatadogWebhookRequest,
    background_tasks: BackgroundTasks,
    rmq_channel_pool: Pool[Channel] = Depends(get_rmq_channel_pool),
) -> DatadogWebhookResponse:
    """
    Handle incoming Datadog webhook alerts.

    :param webhook_data: The webhook data from Datadog
    :param background_tasks: FastAPI background tasks object for async processing
    :param rmq_channel_pool: RabbitMQ channel pool dependency
    :returns: Response indicating success or failure
    """
    logger.info(f"Received Datadog webhook data: {webhook_data}")
    try:
        # Create a SeverityClassifier instance
        severity_classifier = SeverityClassifier()
        processed_count = 0

        # 從設置中獲取Datadog事件是否自動創建incident的默認值
        auto_detect_incident = settings.datadog_auto_detect_incident
        logger.info(f"Datadog auto_detect_incident set to: {auto_detect_incident}")

        for alert in webhook_data.alerts:
            # Convert alerts to dict
            alert_dict = alert.model_dump()

            # Determine severity using SeverityClassifier
            severity = await severity_classifier.classify_severity(
                incident_title=alert.title,
                incident_description=alert.text,
                source="datadog",
                events_data=[alert_dict],
            )

            alert_dict["severity"] = severity

            # Create standardized event message
            event_message = StandardEventMessage(
                source="datadog",
                event_type="alert",
                event_id=alert.alert_id,
                timestamp=alert.alert_created_at,
                raw_payload=alert_dict,
            )

            # 只使用RabbitMQ處理事件
            # Forward to RabbitMQ in the background
            background_tasks.add_task(
                MessagePublisher.publish_event,
                event_message=event_message,
                rmq_channel_pool=rmq_channel_pool,
                auto_detect_incident=auto_detect_incident,
            )
            processed_count += 1

        logger.info(f"Successfully processed {processed_count} Datadog alerts")
        return DatadogWebhookResponse(
            success=True,
            message=f"Successfully processed {processed_count} alerts",
            processed_alerts=processed_count,
        )

    except Exception as e:
        logger.error(f"Error processing Datadog webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process webhook: {str(e)}",
        )

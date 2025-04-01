"""Event consumer for processing events from message queues."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import hashlib

from loguru import logger
from aio_pika import connect_robust, IncomingMessage, ExchangeType
from aio_pika.pool import Pool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from system_guardian.db.models.incidents import Event, Incident
from system_guardian.settings import settings
from system_guardian.services.ai.incident_detector import IncidentDetector
from system_guardian.services.config import ConfigManager
from system_guardian.services.ai.severity_classifier import SeverityClassifier
from system_guardian.services.ingest.message_publisher import MessagePublisher
from system_guardian.services.slack.client import SlackClient
from system_guardian.services.slack.templates import SlackMessageTemplate, AlertSeverity
from system_guardian.services.jira.client import JiraClient


class EventConsumer:
    """Consumer for processing events from RabbitMQ."""

    # Critical event types that should be processed with higher priority
    CRITICAL_EVENT_TYPES = [
        "error",
        "failure",
        "alert",
        "security",
        "outage",
        "incident",
        "critical",
        "emergency",
        "urgent",
    ]

    def __init__(
        self,
        db_session_factory,
        rmq_exchange: str = "webhook_events",
        rmq_queue: str = "webhook_events_queue",
        rmq_routing_keys: Optional[List[str]] = None,
        auto_incident_creation: bool = True,
        ai_engine=None,
        rmq_channel_pool=None,
    ):
        """Initialize the event consumer."""
        self.db_session_factory = db_session_factory
        self.rmq_exchange = rmq_exchange
        self.rmq_queue = rmq_queue
        self.rmq_routing_keys = rmq_routing_keys or [
            "github.*",
            "jira.*",
            "datadog.*",
            "incidents.*",
        ]
        self.should_exit = False
        self.auto_incident_creation = auto_incident_creation
        self.ai_engine = ai_engine
        self.rmq_channel_pool = rmq_channel_pool

        # Initialize components
        self.config_manager = ConfigManager()
        self.severity_classifier = SeverityClassifier()
        self.incident_detector = IncidentDetector(
            config_manager=self.config_manager,
            llm_client=self.ai_engine.llm if self.ai_engine else None,
            llm_model=(
                settings.ai_incident_detection_model
                if settings.ai_allow_advanced_models
                else settings.openai_completion_model
            ),
            severity_classifier=self.severity_classifier,
        )

    async def start(self) -> None:
        """Start consuming messages from RabbitMQ."""
        await self.test_database_connection()
        await self.start_rabbitmq_consumer()

    async def stop(self) -> None:
        """Stop all consumers."""
        self.should_exit = True
        logger.info("Event consumer stopping")

    async def test_database_connection(self) -> None:
        """Test database connection."""
        logger.info("Testing database connection")
        try:
            async with self.db_session_factory() as session:
                result = await session.execute(text("SELECT 1 as test"))
                if result.scalar() == 1:
                    logger.info("Database connection successful")

                # Verify events table exists
                table_check = await session.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'events')"
                    )
                )
                if not table_check.scalar():
                    logger.error("Events table does not exist")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")

    async def start_rabbitmq_consumer(self) -> None:
        """Start consuming messages from RabbitMQ."""
        logger.info(f"Starting RabbitMQ consumer for exchange: {self.rmq_exchange}")

        max_retries = 3
        retry_count = 0
        retry_delay = 5
        connection = None

        while retry_count < max_retries:
            try:
                # Connect to RabbitMQ
                connection = await connect_robust(
                    host=settings.rabbit_host,
                    port=settings.rabbit_port,
                    login=settings.rabbit_user,
                    password=settings.rabbit_pass,
                    virtualhost=settings.rabbit_vhost,
                    timeout=10,
                    heartbeat=60,
                )

                # Create channel
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=10)

                # Create channel pool if not provided
                if self.rmq_channel_pool is None:
                    self.rmq_channel_pool = await self._create_channel_pool()

                # Setup exchanges and queues
                webhook_exchange = await channel.declare_exchange(
                    name=self.rmq_exchange,
                    type=ExchangeType.TOPIC,
                    auto_delete=False,
                    durable=False,
                )

                webhook_queue = await channel.declare_queue(
                    name=self.rmq_queue,
                    durable=True,
                    auto_delete=False,
                )

                # Bind routing keys
                await self._bind_routing_keys(webhook_queue, webhook_exchange)

                # Setup webhook message consumer
                async def process_webhook_message(message: IncomingMessage) -> None:
                    """Process a webhook message."""
                    try:
                        async with message.process():
                            try:
                                body = message.body.decode("utf-8")
                                if len(body) < 5:  # Skip empty messages
                                    return

                                # Check if this is a priority message
                                is_priority = any(
                                    critical_type in message.routing_key.lower()
                                    for critical_type in self.CRITICAL_EVENT_TYPES
                                )

                                # Process the message
                                logger.info(
                                    f"Processing message: {message.routing_key}"
                                )
                                await self.process_message(
                                    body, is_priority=is_priority
                                )

                            except UnicodeDecodeError:
                                logger.error("Error decoding message body")
                            except Exception as e:
                                logger.error(f"Error processing message: {str(e)}")
                    except Exception as e:
                        logger.error(f"Message processing error: {str(e)}")

                await webhook_queue.consume(process_webhook_message)

                # Setup incidents exchange and queue
                incidents_exchange = await channel.declare_exchange(
                    name="system_incidents",
                    auto_delete=False,
                )

                incidents_queue = await channel.declare_queue(
                    name="system_incidents_queue",
                    durable=True,
                    auto_delete=False,
                )

                await incidents_queue.bind(
                    exchange=incidents_exchange,
                    routing_key="incidents.*",
                )

                # Setup incident notification consumer
                async def process_incident_notification(
                    message: IncomingMessage,
                ) -> None:
                    """Process an incident notification message."""
                    async with message.process():
                        try:
                            body = message.body.decode("utf-8")
                            await self.process_incident_notification(body)
                        except Exception as e:
                            logger.error(
                                f"Error processing incident notification: {str(e)}"
                            )

                await incidents_queue.consume(process_incident_notification)

                # Keep alive
                logger.info("RabbitMQ consumer started")
                while not self.should_exit:
                    await asyncio.sleep(1)

                # Clean exit
                await connection.close()
                return

            except Exception as e:
                retry_count += 1
                logger.error(f"RabbitMQ connection error: {str(e)}")

                if connection and not connection.is_closed:
                    await connection.close()

                if retry_count < max_retries:
                    wait_time = retry_delay * retry_count
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to start RabbitMQ consumer after {max_retries} retries"
                    )
                    break

    async def _create_channel_pool(self):
        """Create a RabbitMQ channel pool."""
        try:

            async def get_connection():
                return await connect_robust(
                    host=settings.rabbit_host,
                    port=settings.rabbit_port,
                    login=settings.rabbit_user,
                    password=settings.rabbit_pass,
                    virtualhost=settings.rabbit_vhost,
                )

            connection_pool = Pool(get_connection, max_size=settings.rabbit_pool_size)

            async def get_channel():
                async with connection_pool.acquire() as connection:
                    return await connection.channel()

            return Pool(get_channel, max_size=settings.rabbit_channel_pool_size)
        except Exception as e:
            logger.error(f"Failed to create channel pool: {str(e)}")
            return None

    async def _bind_routing_keys(self, queue, exchange):
        """Bind routing keys to queue."""
        routing_key_map = {
            "github.*": ["push", "pull_request", "issues", "commit_comment", "release"],
            "jira.*": [
                "issue_created",
                "issue_updated",
                "comment_added",
                "issue_deleted",
            ],
            "datadog.*": ["alert", "metric", "event", "monitor"],
        }

        for routing_key in self.rmq_routing_keys:
            if "*" in routing_key:
                base_key = routing_key.replace("*", "")
                if base_key in routing_key_map:
                    for event in routing_key_map[base_key]:
                        specific_key = f"{base_key}{event}"
                        await queue.bind(exchange=exchange, routing_key=specific_key)
            else:
                await queue.bind(exchange=exchange, routing_key=routing_key)

        # Bind fallback routing keys
        await queue.bind(exchange=exchange, routing_key="test.event")
        await queue.bind(exchange=exchange, routing_key="#")  # Capture all messages

    async def process_incident_notification(self, message_body: str) -> None:
        """Process an incident notification message."""
        try:
            # Parse incident data
            incident_data = json.loads(message_body)

            # Validate incident data
            if not isinstance(incident_data, dict):
                logger.error("Invalid incident data format")
                return

            incident_id = incident_data.get("incident_id")
            if not incident_id:
                logger.warning("Missing incident_id in notification")
                return

            # Add defaults for missing fields
            required_fields = ["incident_id", "title", "severity"]
            for field in required_fields:
                if field not in incident_data:
                    if field == "title":
                        incident_data["title"] = f"Incident #{incident_id}"
                    elif field == "severity":
                        incident_data["severity"] = "medium"

            # Send notifications to configured services
            if settings.slack_enabled and settings.slack_bot_token:
                await self.send_slack_notification(incident_data)

            if settings.jira_enabled and settings.jira_url:
                await self.create_jira_ticket(incident_data)

        except json.JSONDecodeError:
            logger.error("Invalid JSON in incident notification")
        except Exception as e:
            logger.error(f"Error processing incident notification: {str(e)}")

    async def process_message(
        self, message_body: str, is_priority: bool = False
    ) -> None:
        """Process a message from the queue."""
        trace_id = hashlib.md5(message_body.encode()).hexdigest()[:8]

        try:
            # Parse message
            data = json.loads(message_body)

            # Extract metadata
            source = data.get("source", "unknown")
            event_type = data.get("event_type", "unknown")
            raw_payload = data.get("raw_payload", {})

            # Determine if auto incident detection should be used
            auto_detect_incident = data.get("auto_detect_incident")
            if auto_detect_incident is None:
                auto_detect_incident = False if source.lower() == "jira" else True

            # Process the event
            async with self.db_session_factory() as session:
                # Store event
                event = await self.store_event(session, source, event_type, raw_payload)
                if not event:
                    logger.error(f"Failed to store event: {source}/{event_type}")
                    return

                # Check if we should create an incident
                if (
                    auto_detect_incident
                    and not event.related_incident_id
                    and self.auto_incident_creation
                ):

                    # Run incident detection
                    detector = self.incident_detector
                    conditions_met = await detector.check_event_conditions(
                        event.content, event.source, event.event_type
                    )

                    if conditions_met:
                        # Check thresholds and keywords
                        threshold_breach = await detector.check_event_thresholds(
                            session, event.source, event.event_type
                        )

                        has_keywords = await detector.analyze_content_for_keywords(
                            event.content, event.source, event.event_type
                        )

                        if threshold_breach or has_keywords:
                            # Create incident
                            incident = await detector.create_incident_from_event(
                                session,
                                event.source,
                                event.event_type,
                                event.content,
                                event.id,
                            )

                            if incident:
                                # Publish incident notification
                                from system_guardian.web.api.ingest.schema import (
                                    StandardEventMessage,
                                )

                                event_message = StandardEventMessage(
                                    source=event.source,
                                    event_type=event.event_type,
                                    event_id=str(event.id),
                                    timestamp=event.created_at,
                                    raw_payload=event.content,
                                )

                                incident_info = {
                                    "created_at": (
                                        incident.created_at.isoformat()
                                        if incident.created_at
                                        else datetime.utcnow().isoformat()
                                    ),
                                    "severity": incident.severity,
                                    "title": incident.title,
                                    "description": incident.description,
                                }

                                await MessagePublisher.publish_incident_detection(
                                    rmq_channel_pool=self.rmq_channel_pool,
                                    event_message=event_message,
                                    incident_id=incident.id,
                                    incident_info=incident_info,
                                )

        except json.JSONDecodeError:
            logger.error("Invalid JSON in message")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

    async def store_event(
        self,
        session: AsyncSession,
        source: str,
        event_type: str,
        raw_payload: Dict[str, Any],
        related_incident_id: Optional[int] = None,
    ) -> Optional[Event]:
        """Store an event to the database."""
        try:
            # Create event using ORM
            event = Event(
                source=source,
                event_type=event_type,
                content=raw_payload,
                created_at=datetime.utcnow(),
                related_incident_id=related_incident_id,
            )

            # Add to session and commit
            session.add(event)
            await session.commit()
            await session.refresh(event)

            return event

        except Exception as e:
            logger.error(f"Error storing event: {str(e)}")

            # Fallback to direct SQL if ORM fails
            try:
                event_content = json.dumps(raw_payload)
                current_time = datetime.utcnow()

                # Build SQL query
                sql = "INSERT INTO events (source, event_type, content, created_at{0}) VALUES (:source, :event_type, :content, :created_at{1}) RETURNING id"
                related_col = ", related_incident_id" if related_incident_id else ""
                related_val = ", :related_incident_id" if related_incident_id else ""

                params = {
                    "source": source,
                    "event_type": event_type,
                    "content": event_content,
                    "created_at": current_time,
                }

                if related_incident_id:
                    params["related_incident_id"] = related_incident_id

                formatted_sql = sql.format(related_col, related_val)
                result = await session.execute(text(formatted_sql), params)
                new_id = result.scalar()

                if new_id:
                    await session.commit()
                    return Event(
                        id=new_id,
                        source=source,
                        event_type=event_type,
                        content=raw_payload,
                        created_at=current_time,
                        related_incident_id=related_incident_id,
                    )
            except Exception as sql_err:
                logger.error(f"SQL insert failed: {str(sql_err)}")

            return None

    async def find_relevant_incident(
        self,
        session: AsyncSession,
        source: str,
        event_type: str,
        raw_payload: Dict[str, Any],
    ) -> Optional[int]:
        """Find a relevant incident for the event."""
        try:
            # Get recent open incidents from the same source
            query = (
                select(Incident)
                .where(Incident.source == source)
                .where(Incident.status.in_(["open", "investigating"]))
                .order_by(Incident.created_at.desc())
            )

            result = await session.execute(query)
            recent_incidents = result.scalars().fetchmany(5)

            if not recent_incidents:
                return None

            # Extract content from payload based on source
            event_title, event_description = self._extract_event_content(
                source, event_type, raw_payload
            )

            # If no content extracted, return most recent incident
            if not event_title and not event_description:
                return recent_incidents[0].id if recent_incidents else None

            # Try similarity service if available
            try:
                incident_id = await self._find_similar_incident(
                    f"{event_title}\n{event_description}",
                    [incident.id for incident in recent_incidents],
                )
                if incident_id:
                    return incident_id
            except Exception:
                pass  # Continue to fallback method

            # Fallback: Basic keyword matching
            return self._keyword_match_incident(
                event_title, event_description, recent_incidents
            )

        except Exception as e:
            logger.error(f"Error finding relevant incident: {str(e)}")
            return None

    def _extract_event_content(self, source, event_type, payload):
        """Extract title and description from event payload."""
        title = ""
        description = ""

        try:
            if source == "github":
                if "issue" in event_type:
                    title = payload.get("issue", {}).get("title", "")
                    description = payload.get("issue", {}).get("body", "")
                elif "pull_request" in event_type:
                    title = payload.get("pull_request", {}).get("title", "")
                    description = payload.get("pull_request", {}).get("body", "")
            elif source == "jira":
                title = payload.get("issue", {}).get("fields", {}).get("summary", "")
                description = (
                    payload.get("issue", {}).get("fields", {}).get("description", "")
                )
            elif source == "datadog":
                title = payload.get("title", "")
                description = payload.get("message", "") or payload.get("text", "")
            elif source == "slack":
                description = payload.get("text", "") or payload.get("message", {}).get(
                    "text", ""
                )
        except Exception:
            pass

        return title, description

    async def _find_similar_incident(self, event_text, incident_ids):
        """
        Find similar incidents using vector search.

        :param event_text: Text to find similar incidents for
        :param incident_ids: List of incident IDs to search within
        :returns: ID of the most similar incident if found, None otherwise
        """
        from system_guardian.services.ai.incident_similarity import (
            IncidentSimilarityService,
        )
        from system_guardian.services.vector_db.qdrant_client import QdrantClient
        from openai import AsyncOpenAI

        # Create clients
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )

        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        # Create similarity service
        similarity_service = IncidentSimilarityService(
            qdrant_client=qdrant_client,
            openai_client=openai_client,
        )

        # Convert incident IDs to strings for filter
        str_incident_ids = [str(id) for id in incident_ids]

        # Create filter condition
        filter_condition = {
            "must": [{"key": "incident_id", "match": {"any": str_incident_ids}}]
        }

        # Find similar incidents
        similar_incidents = await similarity_service.find_similar_incidents(
            query_text=event_text,
            limit=3,
            filter_condition=filter_condition,
        )

        # Return incident ID if similarity is high enough
        for similar in similar_incidents:
            if similar.get("similarity_score", 0) > 0.75:
                incident_id = similar.get("incident_id")
                if incident_id:
                    return int(incident_id)

        return None

    def _keyword_match_incident(self, event_title, event_description, incidents):
        """Match incident using keyword similarity."""
        best_match = None
        best_score = 0

        for incident in incidents:
            # Split titles into tokens
            incident_tokens = set(incident.title.lower().split())
            event_tokens = set(
                event_title.lower().split()
                if event_title
                else event_description.lower().split()
            )

            # Calculate Jaccard similarity
            if incident_tokens and event_tokens:
                intersection = len(incident_tokens.intersection(event_tokens))
                union = len(incident_tokens.union(event_tokens))
                if union > 0:
                    score = intersection / union
                    if score > best_score and score > 0.3:
                        best_score = score
                        best_match = incident

        return best_match.id if best_match else None

    async def check_for_auto_incident_creation(
        self,
        session: AsyncSession,
        event_id: int,
        source: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> Optional[Incident]:
        """Check if an incident should be created for this event."""
        try:
            # Get the event
            stmt = select(Event).where(Event.id == event_id)
            result = await session.execute(stmt)
            event = result.scalar_one_or_none()

            if not event:
                return None

            # Run incident detection
            should_create, reason, severity, title, description = (
                await self._run_incident_detection(source, event_type, payload)
            )

            if should_create:
                # Create incident
                incident = Incident(
                    title=title or f"Incident from {source} {event_type}",
                    description=description
                    or f"Automatically generated incident from {source} {event_type} event.",
                    status="open",
                    severity=severity or "medium",
                    source=source,
                    created_at=datetime.utcnow(),
                )

                # Save incident and associate event
                session.add(incident)
                await session.commit()
                await session.refresh(incident)

                event.related_incident_id = incident.id
                await session.commit()

                return incident

            return None

        except Exception as e:
            logger.error(f"Error in auto incident detection: {str(e)}")
            return None

    async def _run_incident_detection(
        self, source: str, event_type: str, payload: Dict[str, Any]
    ) -> Tuple[bool, str, str, str, str]:
        """Run the incident detection logic."""
        # Check if this is a critical event type
        is_critical = any(
            critical_type in event_type.lower()
            for critical_type in [
                "error",
                "failure",
                "alert",
                "security",
                "outage",
                "incident",
            ]
        )

        if is_critical:
            return (
                True,
                "Critical event type detected",
                "high",
                f"Critical {event_type} from {source}",
                f"Automatic incident created from critical {source} {event_type} event",
            )

        return False, "No incident detection rules matched", None, None, None

    async def _associate_related_events(
        self,
        session: AsyncSession,
        incident: Incident,
        event_source: str,
        event_type: str,
    ) -> int:
        """Associate related events with an incident."""
        # Find events of the same type not associated with an incident
        stmt = select(Event).where(
            (Event.source == event_source)
            & (Event.event_type == event_type)
            & (Event.related_incident_id.is_(None))
            & (Event.created_at >= datetime.utcnow() - timedelta(days=1))
        )

        result = await session.execute(stmt)
        events = result.scalars().all()

        # Associate events with this incident
        for event in events:
            event.related_incident_id = incident.id

        # Commit changes
        if events:
            await session.commit()

        return len(events)

    async def send_slack_notification(self, incident_data: Dict[str, Any]) -> None:
        """Send a Slack notification for an incident."""
        try:
            slack_client = SlackClient()

            if not slack_client.is_configured:
                return

            # Map severity
            severity_map = {
                "low": AlertSeverity.INFO,
                "medium": AlertSeverity.WARNING,
                "high": AlertSeverity.ERROR,
                "critical": AlertSeverity.CRITICAL,
                "info": AlertSeverity.INFO,
                "warning": AlertSeverity.WARNING,
                "error": AlertSeverity.ERROR,
            }

            # Get incident details
            incident_id = incident_data.get("incident_id", "unknown")
            title = incident_data.get("title", "Unnamed Event")
            description = incident_data.get("description", "No description provided")
            severity_str = incident_data.get("severity", "medium").lower()
            detection_time = incident_data.get(
                "detection_time", datetime.utcnow().isoformat()
            )

            # Map severity
            severity = severity_map.get(severity_str, AlertSeverity.WARNING)

            # Get source info
            original_event = incident_data.get("original_event", {})
            source = original_event.get("source", "unknown")
            event_type = original_event.get("event_type", "unknown")

            # Add source info to description
            if source != "unknown" and event_type != "unknown":
                description += f"\n\nSource: {source}\nEvent Type: {event_type}"

            # Send notification
            template = SlackMessageTemplate.create_incident_notification(
                incident_id=str(incident_id),
                title=title,
                severity=severity,
                description=description,
                timestamp=detection_time,
            )

            await slack_client.send_template(template)

        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")

    async def create_jira_ticket(self, incident_data: Dict[str, Any]) -> None:
        """Create a JIRA ticket for an incident."""
        try:
            jira_client = JiraClient()

            if not jira_client.is_configured:
                return

            # Get incident details
            incident_id = incident_data.get("incident_id", "unknown")
            title = incident_data.get("title", "Unnamed Event")
            description = incident_data.get("description", "No description provided")
            severity = incident_data.get("severity", "medium")

            # Get source info
            original_event = incident_data.get("original_event", {})
            source = original_event.get("source", "unknown")
            event_type = original_event.get("event_type", "unknown")

            # Create ticket
            result = await jira_client.create_incident_ticket(
                incident_id=str(incident_id),
                title=title,
                description=description,
                severity=severity,
                source=source,
                event_type=event_type,
            )

            if "key" not in result:
                logger.error(
                    f"Failed to create JIRA ticket: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"Error creating JIRA ticket: {str(e)}")

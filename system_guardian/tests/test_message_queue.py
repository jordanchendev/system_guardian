#!/usr/bin/env python3
"""
Test message queue connection and message sending functionality
"""
import sys
import json
import asyncio
import random
import uuid
import time
from datetime import datetime, timezone, timedelta

import aio_pika
from loguru import logger
import asyncpg
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker

# Import configuration from settings
from system_guardian.settings import settings
from system_guardian.db.models.incidents import Event
from system_guardian.db.models import load_all_models

# Define default queue names
DEFAULT_EVENTS_QUEUE_NAME = "webhook_events_queue"


# Get queue names with fallbacks to defaults
def get_events_queue_name():
    try:
        return settings.events_queue_name
    except (AttributeError, ValueError):
        logger.warning(
            f"settings.events_queue_name not found, using default: '{DEFAULT_EVENTS_QUEUE_NAME}'"
        )
        return DEFAULT_EVENTS_QUEUE_NAME


# Test data generation
def generate_test_event():
    """Generate test event data"""
    event_types = [
        "server_startup",
        "server_shutdown",
        "cpu_usage_high",
        "memory_usage_high",
        "disk_space_low",
        "network_traffic_spike",
        "application_error",
        "database_connection_failure",
    ]

    sources = [
        "web_server_01",
        "app_server_02",
        "database_01",
        "load_balancer",
        "cache_server",
        "auth_service",
    ]

    # Generate a unique test ID for tracking this specific test event
    test_id = f"test_{uuid.uuid4()}"

    # Use the exact format expected by the event consumer:
    # {
    #   "source": str,
    #   "event_type": str,
    #   "timestamp": str (ISO format),
    #   "raw_payload": dict
    # }
    timestamp = datetime.now(timezone.utc).isoformat()

    # Generate random event data with a consistent format
    # This should match the StandardEventMessage format expected by the consumer
    return {
        "source": random.choice(sources),
        "event_type": random.choice(event_types),
        "timestamp": timestamp,
        "raw_payload": {
            "test_id": test_id,
            "severity": random.choice(["info", "warning", "error", "critical"]),
            "message": f"Test message - {datetime.now().isoformat()}",
            "details": {
                "cpu": random.randint(0, 100),
                "memory": random.randint(0, 100),
                "disk": random.randint(0, 100),
                "test_value": random.random(),
            },
        },
    }, test_id


async def test_rabbitmq_connection():
    """Test RabbitMQ connection and message sending"""
    logger.info("Testing RabbitMQ connection...")

    # Display connection settings
    logger.info(f"RabbitMQ URL: {settings.rabbit_url}")

    try:
        # Establish connection
        connection = await aio_pika.connect_robust(str(settings.rabbit_url))

        async with connection:
            logger.info("RabbitMQ connection successful!")

            # Create channel
            channel = await connection.channel()

            # Declare queue
            queue_name = "test_events"
            queue = await channel.declare_queue(
                queue_name, durable=True, auto_delete=False
            )

            logger.info(f"Successfully declared queue: {queue_name}")

            # Send test message
            test_data, test_id = generate_test_event()
            message_body = json.dumps(test_data).encode()

            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=message_body,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    content_type="application/json",
                ),
                routing_key=queue_name,
            )

            logger.info(f"Successfully sent test message to RabbitMQ: {test_data}")

            # Attempt to receive message to confirm
            message_count = 0
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        received_data = json.loads(message.body.decode())
                        logger.info(f"Received message from RabbitMQ: {received_data}")
                        message_count += 1
                        if message_count >= 1:  # Only read one message and exit
                            break

            return True, test_id, test_data
    except Exception as e:
        logger.error(f"RabbitMQ test failed: {e}")
        return False, None, None


async def send_test_events_to_system():
    """Send test events to the system"""
    logger.info("Sending test events to system...")

    test_ids = []
    test_events = []

    try:
        # Use RabbitMQ to send
        connection = await aio_pika.connect_robust(str(settings.rabbit_url))

        async with connection:
            channel = await connection.channel()

            # Use the actual queue name used by the system
            queue_name = get_events_queue_name()
            queue = await channel.declare_queue(
                queue_name, durable=True, auto_delete=False
            )

            # Send test events
            num_events = 5  # Reduced number for easier tracking
            for i in range(num_events):
                test_data, test_id = generate_test_event()
                message_body = json.dumps(test_data).encode()

                # Keep track of test IDs for later verification
                test_ids.append(test_id)
                test_events.append(test_data)

                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=message_body,
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        content_type="application/json",
                    ),
                    routing_key=queue_name,
                )

                logger.info(
                    f"RabbitMQ - Successfully sent test event {i+1}/{num_events} with ID {test_id}"
                )

                # Small delay to avoid flooding
                await asyncio.sleep(0.1)

            logger.info(
                f"Successfully sent {num_events} test events to RabbitMQ queue: {queue_name}"
            )

    except Exception as e:
        logger.error(f"Error sending test events: {e}")
        return [], []

    return test_ids, test_events


async def verify_events_in_database(test_ids, wait_time=5):
    """Verify that sent events appear in the database"""
    logger.info(f"Verifying events in database after waiting {wait_time} seconds...")

    # Wait for events to be processed
    logger.info(f"Waiting {wait_time} seconds for events to be processed...")
    await asyncio.sleep(wait_time)

    # Use proper session management with AsyncSession
    try:
        # Create engine with async driver
        engine = create_async_engine(str(settings.db_url), echo=False)

        # Make sure all models are loaded
        load_all_models()

        # Create session maker
        async_session = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

        # Verify events in database
        found_events = []

        async with async_session() as session:
            for test_id in test_ids:
                # Build a more efficient SQL query
                stmt = select(Event).where(
                    Event.raw_payload["test_id"].as_string() == test_id
                )

                result = await session.execute(stmt)
                event = result.scalar_one_or_none()

                if event:
                    logger.info(
                        f"Found event with test_id {test_id} in database: {event}"
                    )
                    found_events.append(test_id)
                else:
                    logger.warning(
                        f"Event with test_id {test_id} not found in database!"
                    )

        logger.info(f"Found {len(found_events)}/{len(test_ids)} events in database")

        # Return success if all or majority of events were found
        return len(found_events) >= len(test_ids) * 0.8  # 80% success rate

    except Exception as e:
        logger.error(f"Error verifying events in database: {e}")
        return False
    finally:
        if "engine" in locals():
            await engine.dispose()


async def test_end_to_end_flow():
    """Test end-to-end flow of sending events to message queue and verifying in database"""
    logger.info("Testing end-to-end flow of event processing...")

    # Send test events to system through RabbitMQ
    test_ids, test_events = await send_test_events_to_system()

    if not test_ids:
        logger.error("Failed to send test events, aborting end-to-end test")
        return False

    # Verify events appear in database
    result = await verify_events_in_database(test_ids, wait_time=10)

    if result:
        logger.info("🎉 End-to-end test successful! Events properly processed")
    else:
        logger.error("❌ End-to-end test failed! Events not properly processed")

    return result


async def main():
    """Main entry point for the test script"""
    # Configure logging
    from system_guardian.logging_config import configure_logging

    configure_logging(log_level="INFO")

    logger.info("Starting message queue test...")
    logger.info(f"System Guardian Version: 0.1.0")
    logger.info("==== Environment Check ====")
    logger.info(f"Database URL: {settings.db_url}")
    logger.info(f"RabbitMQ URL: {settings.rabbit_url}")
    logger.info("============================")

    # Step 1: Test connections to message queue systems
    logger.info("1. Testing messaging systems connections")
    logger.info("----------------------------------------")

    # Test RabbitMQ
    rmq_test, rmq_test_id, rmq_test_data = await test_rabbitmq_connection()

    # Report results
    logger.info("==== Connection Tests Results ====")
    logger.info(
        f"RabbitMQ Connection Test: {'✅ Success' if rmq_test else '❌ Failed'}"
    )
    logger.info("====================================")

    # Step 4: Test end-to-end flow
    # Only run if at least one message queue system is working
    e2e_test = False
    if rmq_test:
        logger.info("4. Testing end-to-end flow")
        logger.info("-------------------------")
        e2e_test = await test_end_to_end_flow()
    else:
        logger.warning("Skipping end-to-end test as RabbitMQ connection failed")

    # If basic tests pass but end-to-end fails, investigate why
    if rmq_test and not e2e_test:
        logger.error(
            "SYSTEM CHECK: Basic message queue connections work but end-to-end processing failed"
        )
        logger.error(
            "This suggests a problem with the event consumer service or database connections"
        )
        logger.info("3. Ensure the event_consumer_service is running")
        logger.info("4. Ensure message queue (RabbitMQ) is properly configured")
        logger.info("5. Check database connection and permissions")
        return 1

    # Overall result
    if rmq_test:
        logger.info("🎉 Message queue systems are properly configured")
        logger.info("The system should be able to process events correctly")
        return 0
    else:
        logger.error("❌ Message queue connectivity check failed")
        logger.error("The system will not be able to process events correctly")
        logger.info("Please check your RabbitMQ configuration")
        return 1


if __name__ == "__main__":
    # Run the main test function
    result_code = asyncio.run(main())
    sys.exit(result_code)

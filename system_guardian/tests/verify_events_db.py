#!/usr/bin/env python3
"""
Script to verify events in the database
Use this to check if events have been successfully processed and stored
"""
import sys
import json
import asyncio
from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker

from system_guardian.settings import settings
from system_guardian.db.models import load_all_models


async def verify_recent_events(minutes=10, limit=20):
    """
    Display recent events from the database

    Args:
        minutes: How many minutes back to look
        limit: Maximum number of events to display
    """
    logger.info(
        f"Checking for events in the database from the last {minutes} minutes..."
    )

    # Load models
    load_all_models()

    # Create database engine
    engine = create_async_engine(str(settings.db_url), echo=False, pool_pre_ping=True)

    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    try:
        async with session_factory() as session:
            # Get count of recent events
            count_query = text(
                """
                SELECT COUNT(*) FROM events
                WHERE created_at > NOW() - (INTERVAL '1 minute' * :minutes)
            """
            )
            count_result = await session.execute(count_query, {"minutes": minutes})
            recent_count = count_result.scalar_one()

            if recent_count == 0:
                logger.warning(f"No events found in the last {minutes} minutes")

                # Check if there are any events at all
                total_count_query = text("SELECT COUNT(*) FROM events")
                total_count_result = await session.execute(total_count_query)
                total_count = total_count_result.scalar_one()

                if total_count == 0:
                    logger.warning(
                        "The events table is empty. No events have been stored."
                    )
                else:
                    logger.info(f"Found {total_count} total events in the database.")

                    # Get the most recent event
                    most_recent_query = text(
                        """
                        SELECT id, source, event_type, created_at, content::text
                        FROM events
                        ORDER BY created_at DESC
                        LIMIT 1
                    """
                    )
                    most_recent_result = await session.execute(most_recent_query)
                    most_recent = most_recent_result.first()

                    if most_recent:
                        logger.info(
                            f"Most recent event: ID={most_recent[0]}, Source={most_recent[1]}, Type={most_recent[2]}"
                        )
                        logger.info(
                            f"Created at: {most_recent[3]} ({datetime.now() - most_recent[3]} ago)"
                        )

                return

            logger.info(f"Found {recent_count} events in the last {minutes} minutes")

            # Get the recent events
            recent_query = text(
                """
                SELECT id, source, event_type, created_at, content::text
                FROM events
                WHERE created_at > NOW() - (INTERVAL '1 minute' * :minutes)
                ORDER BY created_at DESC
                LIMIT :limit
            """
            )

            recent_result = await session.execute(
                recent_query, {"minutes": minutes, "limit": limit}
            )
            recent_events = recent_result.fetchall()

            logger.info(f"Displaying {len(recent_events)} most recent events:")

            for i, event in enumerate(recent_events):
                event_id = event[0]
                source = event[1]
                event_type = event[2]
                created_at = event[3]
                content_str = event[4]

                logger.info(f"\n--- Event {i+1}/{len(recent_events)} ---")
                logger.info(f"ID: {event_id}")
                logger.info(f"Source: {source}")
                logger.info(f"Type: {event_type}")
                logger.info(
                    f"Created: {created_at} ({datetime.now() - created_at} ago)"
                )

                # Try to parse and pretty-print the content
                try:
                    content = json.loads(content_str)
                    # Check if the content has a test_id field
                    test_id = None
                    if isinstance(content, dict) and "raw_payload" in content:
                        if (
                            isinstance(content["raw_payload"], dict)
                            and "test_id" in content["raw_payload"]
                        ):
                            test_id = content["raw_payload"]["test_id"]
                            logger.info(f"Test ID: {test_id}")

                    # Print abbreviated content
                    if len(content_str) > 500:
                        logger.info(f"Content (abbreviated): {content_str[:500]}...")
                    else:
                        logger.info(f"Content: {content_str}")
                except json.JSONDecodeError:
                    logger.warning(f"Content is not valid JSON: {content_str[:200]}...")

            # Also display any test events that might be older
            test_query = text(
                """
                SELECT id, source, event_type, created_at, content::text
                FROM events
                WHERE content::text LIKE '%test_%'
                ORDER BY created_at DESC
                LIMIT 5
            """
            )

            test_result = await session.execute(test_query)
            test_events = test_result.fetchall()

            if test_events:
                logger.info("\n=== Recent Test Events ===")
                for event in test_events:
                    logger.info(
                        f"ID={event[0]}, Source={event[1]}, Type={event[2]}, Created={event[3]}"
                    )
                    # Extract test_id if possible
                    try:
                        content = json.loads(event[4])
                        if isinstance(content, dict) and "raw_payload" in content:
                            if (
                                isinstance(content["raw_payload"], dict)
                                and "test_id" in content["raw_payload"]
                            ):
                                logger.info(
                                    f"Test ID: {content['raw_payload']['test_id']}"
                                )
                    except:
                        pass

    except Exception as e:
        logger.error(f"Error checking recent events: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        await engine.dispose()


async def main():
    """Run the verification"""
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("Verifying events in the database...")

    # Try to find events from the last hour
    await verify_recent_events(minutes=60, limit=20)

    logger.info("\nVerification complete.")


if __name__ == "__main__":
    asyncio.run(main())

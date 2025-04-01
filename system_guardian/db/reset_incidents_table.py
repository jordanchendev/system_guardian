#!/usr/bin/env python3
"""
Script to rebuild all tables (events, incidents, resolutions).
This script will create all tables based on the current model definitions.
"""

import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
import asyncpg
from loguru import logger

from system_guardian.settings import settings
from system_guardian.db.models import load_all_models
from system_guardian.db.base import Base


async def check_relations():
    """
    Check the relationships between events and incidents directly in the database.

    This function verifies the trigger_event_id relationships in the incidents table
    and ensures they point to valid events. If no relationships exist, it will
    attempt to create a test relationship between the latest incident and event.

    :returns: None
    """
    # Fix connection string
    connection_string = str(settings.db_url).replace("postgresql+asyncpg", "postgresql")
    logger.info(f"Using connection string: {connection_string}")

    try:
        # Connect directly to database
        connection = await asyncpg.connect(connection_string)

        # Check trigger_for relationships in incidents table
        incidents_with_trigger = await connection.fetch(
            "SELECT id, title, trigger_event_id FROM incidents WHERE trigger_event_id IS NOT NULL"
        )

        logger.info(
            f"Number of incidents with trigger_event_id: {len(incidents_with_trigger)}"
        )
        for incident in incidents_with_trigger:
            logger.info(
                f"Incident #{incident['id']}: '{incident['title']}', trigger_event_id={incident['trigger_event_id']}"
            )

            # Check if related event exists
            event = await connection.fetchrow(
                "SELECT id, source, event_type FROM events WHERE id = $1",
                incident["trigger_event_id"],
            )

            if event:
                logger.info(
                    f"  Related event exists: Event #{event['id']}, {event['source']}/{event['event_type']}"
                )
            else:
                logger.warning(f"  Related event does not exist!")

        # If no incident has trigger_event_id, manually set a test relationship
        if len(incidents_with_trigger) == 0:
            # Get latest incident and event
            latest_incident = await connection.fetchrow(
                "SELECT id, title FROM incidents ORDER BY id DESC LIMIT 1"
            )

            latest_event = await connection.fetchrow(
                "SELECT id, source, event_type FROM events ORDER BY id DESC LIMIT 1"
            )

            if latest_incident and latest_event:
                logger.info(
                    f"\nNo incident has trigger_event_id, attempting to set a test relationship:"
                )
                logger.info(
                    f"Latest incident: #{latest_incident['id']} '{latest_incident['title']}'"
                )
                logger.info(
                    f"Latest event: #{latest_event['id']} ({latest_event['source']}/{latest_event['event_type']})"
                )

                # Update incident's trigger_event_id
                await connection.execute(
                    "UPDATE incidents SET trigger_event_id = $1 WHERE id = $2",
                    latest_event["id"],
                    latest_incident["id"],
                )

                logger.info(
                    f"Set incident #{latest_incident['id']} trigger_event_id to {latest_event['id']}"
                )

                # Verify update
                updated = await connection.fetchrow(
                    "SELECT trigger_event_id FROM incidents WHERE id = $1",
                    latest_incident["id"],
                )

                logger.info(
                    f"Verification: trigger_event_id = {updated['trigger_event_id']}"
                )

        await connection.close()

    except Exception as e:
        logger.error(f"Error: {str(e)}")


async def rebuild_tables():
    """
    Rebuild all database tables based on current model definitions.

    This function will:
    1. Load all models
    2. Create a new database engine
    3. Create all tables with current schema
    4. Verify table creation and column existence

    :returns: None
    """
    # Load all models
    load_all_models()

    # Create database engine
    engine = create_async_engine(str(settings.db_url), echo=True)

    try:
        # Create the tables with the current schema
        logger.info("Creating all tables based on current model definitions...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Tables created successfully with updated schema")

        # Verify that tables were created
        async_session = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

        async with async_session() as session:
            # Verify events table
            events_result = await session.execute(
                text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'events')"
                )
            )
            events_exists = events_result.scalar()

            # Verify incidents table
            incidents_result = await session.execute(
                text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'incidents')"
                )
            )
            incidents_exists = incidents_result.scalar()

            # Verify resolutions table
            resolutions_result = await session.execute(
                text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'resolutions')"
                )
            )
            resolutions_exists = resolutions_result.scalar()

            logger.info(f"Verification: events table exists: {events_exists}")
            logger.info(f"Verification: incidents table exists: {incidents_exists}")
            logger.info(f"Verification: resolutions table exists: {resolutions_exists}")

            # Check if trigger_event_id column exists in incidents table
            if incidents_exists:
                trigger_col_result = await session.execute(
                    text(
                        """
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'incidents' AND column_name = 'trigger_event_id'
                    )
                    """
                    )
                )
                trigger_col_exists = trigger_col_result.scalar()
                logger.info(
                    f"Verification: trigger_event_id column exists in incidents table: {trigger_col_exists}"
                )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(rebuild_tables())
    asyncio.run(check_relations())

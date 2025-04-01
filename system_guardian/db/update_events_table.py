#!/usr/bin/env python3
"""
Update the events table by renaming incident_id column to related_incident_id.

This script performs database schema changes to rename the incident_id column
to related_incident_id in the events table. It handles various edge cases and
provides detailed logging of the process.

:returns: None
"""

import asyncio
import asyncpg
from loguru import logger
from system_guardian.settings import settings


async def update_events_table_schema():
    """
    Rename the incident_id column to related_incident_id in the events table.

    This function performs the following steps:
    1. Checks if related_incident_id column already exists
    2. Checks if incident_id column exists
    3. Attempts to rename the column or creates a new one if needed
    4. Validates the changes

    :returns: None
    """
    connection_string = str(settings.db_url).replace("postgresql+asyncpg", "postgresql")
    logger.info(f"Using connection string: {connection_string}")

    try:
        # Connect to database
        connection = await asyncpg.connect(connection_string)

        # 1. Check if related_incident_id column exists
        has_related = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'events' AND column_name = 'related_incident_id'
            )
            """
        )

        if has_related:
            logger.info("related_incident_id column already exists, no update needed")
            await connection.close()
            return

        # 2. Check if incident_id column exists
        has_incident = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'events' AND column_name = 'incident_id'
            )
            """
        )

        if not has_incident:
            logger.warning("incident_id column does not exist in events table")

            # Create new column directly
            logger.info("Attempting to create related_incident_id column")
            await connection.execute(
                """
                ALTER TABLE events 
                ADD COLUMN related_incident_id INTEGER 
                REFERENCES incidents(id) ON DELETE SET NULL
                """
            )
            logger.success("Successfully created related_incident_id column")
            await connection.close()
            return

        # 3. Rename column
        logger.info(
            "Renaming incident_id column to related_incident_id in events table"
        )
        try:
            # Backup data first
            logger.info("Backing up incident_id values from events table")
            events_data = await connection.fetch(
                "SELECT id, incident_id FROM events WHERE incident_id IS NOT NULL"
            )
            logger.info(f"Found {len(events_data)} events with incident_id values")

            # Attempt to rename column
            await connection.execute(
                """
                BEGIN;
                -- Drop existing foreign key constraint
                ALTER TABLE events DROP CONSTRAINT IF EXISTS events_incident_id_fkey;
                
                -- Rename column
                ALTER TABLE events RENAME COLUMN incident_id TO related_incident_id;
                
                -- Add new foreign key constraint
                ALTER TABLE events 
                ADD CONSTRAINT events_related_incident_id_fkey 
                FOREIGN KEY (related_incident_id) 
                REFERENCES incidents(id) ON DELETE SET NULL;
                
                COMMIT;
                """
            )
            logger.success(
                "Successfully renamed incident_id column to related_incident_id"
            )

        except Exception as rename_err:
            logger.error(f"Error during column rename: {str(rename_err)}")

            # Try creating new column and copying data
            if len(events_data) > 0:
                logger.info("Attempting to create new column and copy data")
                try:
                    await connection.execute(
                        """
                        BEGIN;
                        -- Add new column
                        ALTER TABLE events ADD COLUMN related_incident_id INTEGER;
                        
                        -- Add foreign key constraint
                        ALTER TABLE events 
                        ADD CONSTRAINT events_related_incident_id_fkey 
                        FOREIGN KEY (related_incident_id) 
                        REFERENCES incidents(id) ON DELETE SET NULL;
                        
                        COMMIT;
                        """
                    )

                    # Copy data
                    for event in events_data:
                        event_id = event["id"]
                        incident_id = event["incident_id"]
                        if incident_id is not None:
                            await connection.execute(
                                "UPDATE events SET related_incident_id = $1 WHERE id = $2",
                                incident_id,
                                event_id,
                            )

                    logger.success(
                        f"Successfully set related_incident_id for {len(events_data)} events"
                    )

                    # Create successful, now drop old column
                    await connection.execute(
                        "ALTER TABLE events DROP COLUMN incident_id"
                    )
                    logger.success("Successfully dropped old incident_id column")

                except Exception as create_err:
                    logger.error(
                        f"Error during new column creation and data copy: {str(create_err)}"
                    )

        # 4. Validate results
        has_related_after = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'events' AND column_name = 'related_incident_id'
            )
            """
        )

        if has_related_after:
            logger.success(
                "Validation successful: related_incident_id column now exists"
            )
        else:
            logger.error("Validation failed: related_incident_id column does not exist")

        # Check if incident_id has been removed
        has_incident_after = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'events' AND column_name = 'incident_id'
            )
            """
        )

        if not has_incident_after:
            logger.success("Validation successful: incident_id column has been removed")
        else:
            logger.warning("Warning: incident_id column still exists")

        # Close connection
        await connection.close()

    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(update_events_table_schema())

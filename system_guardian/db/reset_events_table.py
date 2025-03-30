#!/usr/bin/env python3
"""
Reset events table to resolve event insertion issues
This script will completely rebuild the events table structure
"""

import asyncio
import asyncpg
from loguru import logger
from system_guardian.settings import settings


async def reset_events_table():
    """Completely rebuild the events table to fix potential structure issues"""
    connection_string = str(settings.db_url).replace("postgresql+asyncpg", "postgresql")
    print(f"Connecting to database: {connection_string}")

    try:
        # Connect to database
        connection = await asyncpg.connect(connection_string)

        # Check if events table exists
        exists = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'events'
            )
            """
        )

        if exists:
            print("Found existing events table, preparing to backup and rebuild")

            # Backup table structure
            schema = await connection.fetch(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'events'
                ORDER BY ordinal_position
                """
            )

            print(
                f"Current table structure: {', '.join([col['column_name'] for col in schema])}"
            )

            # Try to backup data (if any)
            try:
                backup_count = await connection.fetchval("SELECT COUNT(*) FROM events")
                if backup_count > 0:
                    print(f"Found {backup_count} event records, preparing backup")

                    # Create temporary backup table
                    await connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS events_backup AS 
                        SELECT * FROM events
                        """
                    )

                    backup_verify = await connection.fetchval(
                        "SELECT COUNT(*) FROM events_backup"
                    )
                    print(
                        f"Backed up {backup_verify} event records to events_backup table"
                    )
                else:
                    print("No data in events table, no backup needed")
            except Exception as backup_err:
                print(f"Error backing up data: {str(backup_err)}")
                print("Continuing with table rebuild")

            # Remove foreign key constraints
            print("Removing foreign key constraints...")
            await connection.execute(
                """
                DO $$
                BEGIN
                    -- Try to remove existing constraints (if they exist)
                    BEGIN
                        ALTER TABLE events DROP CONSTRAINT IF EXISTS events_related_incident_id_fkey;
                    EXCEPTION WHEN OTHERS THEN
                        RAISE NOTICE 'Error removing related_incident_id constraint: %', SQLERRM;
                    END;
                    
                    BEGIN
                        ALTER TABLE events DROP CONSTRAINT IF EXISTS events_incident_id_fkey;
                    EXCEPTION WHEN OTHERS THEN
                        RAISE NOTICE 'Error removing incident_id constraint: %', SQLERRM;
                    END;
                    
                    -- Try to remove any constraints that might reference this table
                    BEGIN
                        ALTER TABLE incidents DROP CONSTRAINT IF EXISTS incidents_trigger_event_id_fkey;
                    EXCEPTION WHEN OTHERS THEN
                        RAISE NOTICE 'Error removing trigger_event_id constraint: %', SQLERRM;
                    END;
                END $$;
                """
            )

            # Delete table
            print("Deleting events table...")
            await connection.execute("DROP TABLE IF EXISTS events CASCADE")
            print("Events table deleted")

        # Create brand new events table
        print("Creating new events table...")
        await connection.execute(
            """
            CREATE TABLE events (
                id SERIAL PRIMARY KEY,
                related_incident_id INTEGER NULL,
                source VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                content JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """
        )
        print("New events table created successfully")

        # Add foreign key constraints
        print("Adding foreign key constraints...")
        await connection.execute(
            """
            ALTER TABLE events 
            ADD CONSTRAINT events_related_incident_id_fkey 
            FOREIGN KEY (related_incident_id) 
            REFERENCES incidents(id) ON DELETE SET NULL
            """
        )
        print("Foreign key constraints added successfully")

        # Restore backup data (if any)
        try:
            backup_exists = await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'events_backup'
                )
                """
            )

            if backup_exists:
                backup_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM events_backup"
                )

                if backup_count > 0:
                    print(f"Restoring {backup_count} records from backup table...")

                    # Restore data from backup using new column names
                    await connection.execute(
                        """
                        INSERT INTO events (id, related_incident_id, source, event_type, content, created_at)
                        SELECT 
                            id, 
                            CASE 
                                WHEN related_incident_id IS NOT NULL THEN related_incident_id
                                WHEN incident_id IS NOT NULL THEN incident_id
                                ELSE NULL
                            END as related_incident_id,
                            source, 
                            event_type, 
                            content, 
                            created_at
                        FROM events_backup
                        """
                    )

                    restored = await connection.fetchval("SELECT COUNT(*) FROM events")
                    print(f"Successfully restored {restored} records")

                # Delete backup table
                await connection.execute("DROP TABLE events_backup")
                print("Backup table deleted")
        except Exception as restore_err:
            print(f"Error restoring data: {str(restore_err)}")

        # Fix sequence numbers
        print("Fixing auto-increment sequence...")
        await connection.execute(
            """
            SELECT setval('events_id_seq', COALESCE((SELECT MAX(id) FROM events), 0) + 1, false)
            """
        )

        # Verify table structure
        schema = await connection.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'events'
            ORDER BY ordinal_position
            """
        )

        print("Table structure after rebuild:")
        for col in schema:
            print(
                f"- {col['column_name']} ({col['data_type']}, {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'})"
            )

        # Close connection
        await connection.close()
        print("Table rebuild complete!")

    except Exception as e:
        print(f"Error rebuilding table: {str(e)}")


if __name__ == "__main__":
    asyncio.run(reset_events_table())

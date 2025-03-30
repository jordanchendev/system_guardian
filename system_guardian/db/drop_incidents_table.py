#!/usr/bin/env python3
"""
Script to drop the incidents table.
The table will be recreated with the updated schema when the application starts.
"""

import asyncio
import asyncpg
from system_guardian.settings import settings


async def drop_incidents_table():
    """Drop the incidents table so it can be recreated with the updated schema."""
    try:
        # Connect directly to the database
        # Fix connection string by replacing postgresql+asyncpg with postgresql
        connection_string = str(settings.db_url).replace(
            "postgresql+asyncpg", "postgresql"
        )
        print(f"Using connection string: {connection_string}")
        connection = await asyncpg.connect(connection_string)

        # Check if the table exists
        exists = await connection.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'incidents')"
        )

        if exists:
            print("Incidents table exists, dropping it...")
            # Drop the incidents and resolutions tables which depend on the structure
            await connection.execute("DROP TABLE IF EXISTS resolutions CASCADE")
            await connection.execute("DROP TABLE IF EXISTS incidents CASCADE")
            print("Incidents and related tables dropped successfully")
        else:
            print("Incidents table does not exist, nothing to drop")

        await connection.close()

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(drop_incidents_table())

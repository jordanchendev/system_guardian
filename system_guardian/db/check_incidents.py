#!/usr/bin/env python3
"""
Check the relationship between incidents and events tables
"""

import asyncio
import asyncpg
from system_guardian.settings import settings


async def check_incidents_and_events():
    """Check the relationship between incidents and events tables"""
    connection_string = str(settings.db_url).replace("postgresql+asyncpg", "postgresql")
    print(f"Using connection string: {connection_string}")

    try:
        # Connect to database
        connection = await asyncpg.connect(connection_string)

        # Check incidents table
        print("\n=== Incidents Table Status ===")
        incidents = await connection.fetch(
            "SELECT id, title, trigger_event_id FROM incidents ORDER BY id"
        )
        print(f"Total {len(incidents)} incidents")

        for incident in incidents:
            trigger_id = incident["trigger_event_id"]
            status = "Set" if trigger_id is not None else "Not set"
            title = incident["title"]
            if len(title) > 40:
                title = title[:40] + "..."
            print(f"Incident #{incident['id']}: '{title}'")
            print(f"  trigger_event_id: {trigger_id} ({status})")

            # If trigger_event_id exists, check if the corresponding event exists
            if trigger_id is not None:
                event = await connection.fetchrow(
                    "SELECT id, source, event_type FROM events WHERE id = $1",
                    trigger_id,
                )
                if event:
                    print(
                        f"  Corresponding event exists: Event #{event['id']} - {event['source']}/{event['event_type']}"
                    )
                else:
                    print(f"  ⚠️ Corresponding event does not exist!")

        # Check relationships between events and incidents
        print("\n=== Events Table Status ===")
        events = await connection.fetch(
            """
            SELECT e.id, e.source, e.event_type, e.related_incident_id,
                   i.id as related_incident_id, i.trigger_event_id
            FROM events e
            LEFT JOIN incidents i ON e.id = i.trigger_event_id
            ORDER BY e.id
            LIMIT 20
            """
        )
        total_events = await connection.fetchval("SELECT COUNT(*) FROM events")
        print(f"Total {total_events} events (showing first 20 only)")

        trigger_count = 0
        for event in events:
            event_id = event["id"]
            related_incident_id = event["related_incident_id"]
            is_trigger = event["related_incident_id"]

            # Basic information
            print(f"Event #{event_id} - {event['source']}/{event['event_type']}")

            # Check if related to incident
            if related_incident_id:
                print(f"  Associated with Incident #{related_incident_id}")
            else:
                print(f"  Not associated with any incident")

            # Check if trigger event
            if is_trigger:
                trigger_count += 1
                print(f"  Is trigger event for Incident #{is_trigger} ✓")
            else:
                print(f"  Not a trigger event for any incident")

        # Get total trigger event count
        total_trigger_count = await connection.fetchval(
            """
            SELECT COUNT(*) FROM events e
            JOIN incidents i ON e.id = i.trigger_event_id
            """
        )
        print(
            f"\nSummary: {total_trigger_count}/{total_events} events are trigger events"
        )

        # Check for inconsistencies
        print("\n=== Checking Inconsistencies ===")
        inconsistencies = await connection.fetch(
            """
            SELECT i.id, i.title, i.trigger_event_id, e.id AS event_id
            FROM incidents i
            LEFT JOIN events e ON i.trigger_event_id = e.id
            WHERE i.trigger_event_id IS NOT NULL AND e.id IS NULL
            """
        )

        if inconsistencies:
            print(
                f"Found {len(inconsistencies)} inconsistencies (trigger_event_id pointing to non-existent events)"
            )
            for row in inconsistencies:
                title = row["title"]
                if len(title) > 40:
                    title = title[:40] + "..."
                print(f"Incident #{row['id']} '{title}'")
                print(
                    f"  trigger_event_id={row['trigger_event_id']} points to non-existent event"
                )
        else:
            print("No inconsistencies found")

        # Close connection
        await connection.close()

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(check_incidents_and_events())

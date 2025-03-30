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

from system_guardian.settings import settings
from system_guardian.db.models import load_all_models
from system_guardian.db.base import Base


async def check_relations():
    """直接檢查事件和事件之間的關係"""
    # 連接字符串修正
    connection_string = str(settings.db_url).replace("postgresql+asyncpg", "postgresql")
    print(f"使用連接字符串: {connection_string}")

    try:
        # 直接連接數據庫
        connection = await asyncpg.connect(connection_string)

        # 檢查事件表中的trigger_for關係是否正確
        incidents_with_trigger = await connection.fetch(
            "SELECT id, title, trigger_event_id FROM incidents WHERE trigger_event_id IS NOT NULL"
        )

        print(f"有trigger_event_id的incidents數量: {len(incidents_with_trigger)}")
        for incident in incidents_with_trigger:
            print(
                f"Incident #{incident['id']}: '{incident['title']}', trigger_event_id={incident['trigger_event_id']}"
            )

            # 檢查對應事件是否存在
            event = await connection.fetchrow(
                "SELECT id, source, event_type FROM events WHERE id = $1",
                incident["trigger_event_id"],
            )

            if event:
                print(
                    f"  關聯的事件存在: Event #{event['id']}, {event['source']}/{event['event_type']}"
                )
            else:
                print(f"  關聯的事件不存在!")

        # 如果沒有incident有trigger_event_id，則手動設置一個測試關係
        if len(incidents_with_trigger) == 0:
            # 獲取最新的一個incident和最新的一個event
            latest_incident = await connection.fetchrow(
                "SELECT id, title FROM incidents ORDER BY id DESC LIMIT 1"
            )

            latest_event = await connection.fetchrow(
                "SELECT id, source, event_type FROM events ORDER BY id DESC LIMIT 1"
            )

            if latest_incident and latest_event:
                print(f"\n沒有incident設置trigger_event_id，嘗試手動設置一個測試關係:")
                print(
                    f"最新incident: #{latest_incident['id']} '{latest_incident['title']}'"
                )
                print(
                    f"最新event: #{latest_event['id']} ({latest_event['source']}/{latest_event['event_type']})"
                )

                # 更新incident的trigger_event_id
                await connection.execute(
                    "UPDATE incidents SET trigger_event_id = $1 WHERE id = $2",
                    latest_event["id"],
                    latest_incident["id"],
                )

                print(
                    f"已設置incident #{latest_incident['id']}的trigger_event_id為{latest_event['id']}"
                )

                # 驗證更新
                updated = await connection.fetchrow(
                    "SELECT trigger_event_id FROM incidents WHERE id = $1",
                    latest_incident["id"],
                )

                print(f"驗證更新: trigger_event_id = {updated['trigger_event_id']}")

        await connection.close()

    except Exception as e:
        print(f"錯誤: {str(e)}")


async def rebuild_tables():
    """Rebuild all database tables based on current model definitions."""
    # Load all models
    load_all_models()

    # Create database engine
    engine = create_async_engine(str(settings.db_url), echo=True)

    try:
        # Create the tables with the current schema
        print("Creating all tables based on current model definitions...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Tables created successfully with updated schema")

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

            print(f"Verification: events table exists: {events_exists}")
            print(f"Verification: incidents table exists: {incidents_exists}")
            print(f"Verification: resolutions table exists: {resolutions_exists}")

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
                print(
                    f"Verification: trigger_event_id column exists in incidents table: {trigger_col_exists}"
                )

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(rebuild_tables())
    asyncio.run(check_relations())

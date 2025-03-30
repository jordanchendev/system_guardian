#!/usr/bin/env python3
"""
修復incidents表中缺失的trigger_event_id關係
"""

import asyncio
import asyncpg
from system_guardian.settings import settings


async def fix_trigger_event_relations():
    """
    為所有缺少trigger_event_id的incidents尋找並設置關聯的事件
    """
    connection_string = str(settings.db_url).replace("postgresql+asyncpg", "postgresql")
    print(f"使用連接字符串: {connection_string}")

    try:
        # 連接數據庫
        connection = await asyncpg.connect(connection_string)

        # 1. 先取得所有缺少trigger_event_id的incidents
        incidents = await connection.fetch(
            """
            SELECT id, title, source
            FROM incidents
            WHERE trigger_event_id IS NULL
            ORDER BY id
            """
        )

        print(f"找到 {len(incidents)} 個缺少trigger_event_id的incidents")

        fixed_count = 0
        for incident in incidents:
            incident_id = incident["id"]
            title = incident["title"]
            source = incident["source"]
            print(
                f"\n處理 Incident #{incident_id}: '{title[:40]}...' (source: {source})"
            )

            # 2. 尋找關聯到這個incident的最早事件作為trigger
            related_events = await connection.fetch(
                """
                SELECT id, source, event_type, created_at
                FROM events
                WHERE related_incident_id = $1
                ORDER BY created_at ASC
                LIMIT 5
                """,
                incident_id,
            )

            if related_events:
                # 使用最早的事件作為trigger
                trigger_event = related_events[0]
                event_id = trigger_event["id"]

                print(
                    f"找到 {len(related_events)} 個關聯事件，使用最早的Event #{event_id} 作為trigger"
                )

                # 3. 更新incident的trigger_event_id
                await connection.execute(
                    """
                    UPDATE incidents
                    SET trigger_event_id = $1
                    WHERE id = $2
                    """,
                    event_id,
                    incident_id,
                )

                # 同時設置event的related_incident_id
                await connection.execute(
                    """
                    UPDATE events
                    SET related_incident_id = $1
                    WHERE id = $2
                    """,
                    incident_id,
                    event_id,
                )

                print(f"已將 Incident #{incident_id} 的trigger_event_id設為 {event_id}")
                print(
                    f"同時將 Event #{event_id} 的related_incident_id設為 {incident_id}"
                )
                fixed_count += 1
            else:
                # 如果沒有關聯事件，嘗試找同source的最早事件
                print(f"Incident #{incident_id} 沒有關聯事件，嘗試尋找同source的事件")

                if source:
                    source_events = await connection.fetch(
                        """
                        SELECT id, source, event_type
                        FROM events
                        WHERE source = $1 AND related_incident_id IS NULL
                        ORDER BY created_at ASC
                        LIMIT 1
                        """,
                        source,
                    )

                    if source_events:
                        event_id = source_events[0]["id"]
                        print(f"找到同source的Event #{event_id}，將其設為trigger")

                        # 更新incident的trigger_event_id
                        await connection.execute(
                            """
                            UPDATE incidents
                            SET trigger_event_id = $1
                            WHERE id = $2
                            """,
                            event_id,
                            incident_id,
                        )

                        # 同時設置event的related_incident_id
                        await connection.execute(
                            """
                            UPDATE events
                            SET related_incident_id = $1
                            WHERE id = $2
                            """,
                            incident_id,
                            event_id,
                        )

                        print(
                            f"已將 Incident #{incident_id} 的trigger_event_id設為 {event_id}"
                        )
                        print(
                            f"同時將 Event #{event_id} 的related_incident_id設為 {incident_id}"
                        )
                        fixed_count += 1
                    else:
                        print(f"未找到同source的未關聯事件，無法設置trigger_event_id")
                else:
                    print(f"Incident沒有source資訊，無法尋找相關事件")

        # 4. 最終確認修復結果
        if fixed_count > 0:
            print(
                f"\n成功修復 {fixed_count}/{len(incidents)} 個incidents的trigger_event_id"
            )

            # 確認修復後的情況
            print("\n修復後的incidents表情況:")
            fixed = await connection.fetchval(
                """
                SELECT COUNT(*)
                FROM incidents
                WHERE trigger_event_id IS NOT NULL
                """
            )
            total = await connection.fetchval("SELECT COUNT(*) FROM incidents")

            print(f"- 總計 {total} 個incidents")
            print(f"- 其中 {fixed} 個incidents有設置trigger_event_id")
            print(f"- 比例: {(fixed/total)*100:.1f}%")
        else:
            print("\n未能修復任何incidents的trigger_event_id")

        # 關閉連接
        await connection.close()

    except Exception as e:
        print(f"錯誤: {str(e)}")


if __name__ == "__main__":
    asyncio.run(fix_trigger_event_relations())

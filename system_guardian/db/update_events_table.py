#!/usr/bin/env python3
"""
更新events表中的incident_id列為related_incident_id
此腳本將執行資料庫結構變更
"""

import asyncio
import asyncpg
from system_guardian.settings import settings


async def update_events_table_schema():
    """將events表中的incident_id列改名為related_incident_id"""
    connection_string = str(settings.db_url).replace("postgresql+asyncpg", "postgresql")
    print(f"使用連接字符串: {connection_string}")

    try:
        # 連接數據庫
        connection = await asyncpg.connect(connection_string)

        # 1. 檢查是否已經有related_incident_id列
        has_related = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'events' AND column_name = 'related_incident_id'
            )
            """
        )

        if has_related:
            print("related_incident_id列已存在，無需更新")
            await connection.close()
            return

        # 2. 檢查是否有incident_id列
        has_incident = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'events' AND column_name = 'incident_id'
            )
            """
        )

        if not has_incident:
            print("events表中不存在incident_id列，無法重命名")

            # 直接創建新列
            print("嘗試直接創建related_incident_id列")
            await connection.execute(
                """
                ALTER TABLE events 
                ADD COLUMN related_incident_id INTEGER 
                REFERENCES incidents(id) ON DELETE SET NULL
                """
            )
            print("成功創建related_incident_id列")
            await connection.close()
            return

        # 3. 重命名列
        print("將events表中的incident_id列重命名為related_incident_id")
        try:
            # 先備份數據
            print("備份events表中的incident_id列值")
            events_data = await connection.fetch(
                "SELECT id, incident_id FROM events WHERE incident_id IS NOT NULL"
            )
            print(f"找到 {len(events_data)} 個有incident_id值的事件")

            # 嘗試重命名列
            await connection.execute(
                """
                BEGIN;
                -- 刪除原有的外鍵約束
                ALTER TABLE events DROP CONSTRAINT IF EXISTS events_incident_id_fkey;
                
                -- 重命名列
                ALTER TABLE events RENAME COLUMN incident_id TO related_incident_id;
                
                -- 添加新的外鍵約束
                ALTER TABLE events 
                ADD CONSTRAINT events_related_incident_id_fkey 
                FOREIGN KEY (related_incident_id) 
                REFERENCES incidents(id) ON DELETE SET NULL;
                
                COMMIT;
                """
            )
            print("成功將incident_id列重命名為related_incident_id")

        except Exception as rename_err:
            print(f"重命名列時出錯: {str(rename_err)}")

            # 嘗試創建新列並複製數據
            if len(events_data) > 0:
                print("嘗試創建新列並複製數據")
                try:
                    await connection.execute(
                        """
                        BEGIN;
                        -- 添加新列
                        ALTER TABLE events ADD COLUMN related_incident_id INTEGER;
                        
                        -- 添加外鍵約束
                        ALTER TABLE events 
                        ADD CONSTRAINT events_related_incident_id_fkey 
                        FOREIGN KEY (related_incident_id) 
                        REFERENCES incidents(id) ON DELETE SET NULL;
                        
                        COMMIT;
                        """
                    )

                    # 複製數據
                    for event in events_data:
                        event_id = event["id"]
                        incident_id = event["incident_id"]
                        if incident_id is not None:
                            await connection.execute(
                                "UPDATE events SET related_incident_id = $1 WHERE id = $2",
                                incident_id,
                                event_id,
                            )

                    print(f"已成功為 {len(events_data)} 個事件設置related_incident_id")

                    # 創建成功，現在刪除舊列
                    await connection.execute(
                        "ALTER TABLE events DROP COLUMN incident_id"
                    )
                    print("成功刪除舊的incident_id列")

                except Exception as create_err:
                    print(f"創建新列和複製數據時出錯: {str(create_err)}")

        # 4. 驗證結果
        has_related_after = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'events' AND column_name = 'related_incident_id'
            )
            """
        )

        if has_related_after:
            print("驗證成功：related_incident_id列現在存在")
        else:
            print("驗證失敗：related_incident_id列不存在")

        # 檢查incident_id是否已被刪除
        has_incident_after = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'events' AND column_name = 'incident_id'
            )
            """
        )

        if not has_incident_after:
            print("驗證成功：incident_id列已被刪除")
        else:
            print("警告：incident_id列仍然存在")

        # 關閉連接
        await connection.close()

    except Exception as e:
        print(f"錯誤: {str(e)}")


if __name__ == "__main__":
    asyncio.run(update_events_table_schema())

"""Test script to create JIRA tickets without priority field."""

import asyncio
import sys
from loguru import logger

from system_guardian.services.jira.client import JiraClient


async def test_create_jira_ticket_without_priority():
    """Test creating a JIRA ticket without setting priority field."""
    logger.info("Testing JIRA ticket creation without priority...")
    client = JiraClient()

    if not client.is_configured:
        logger.error("JIRA client is not properly configured")
        return

    # Get project key
    project_key = client.project_key
    logger.info(f"Using project key: {project_key}")

    # Create a ticket without priority
    logger.info("Creating test ticket without priority field...")
    result = await client.create_ticket(
        summary="Test Ticket Without Priority - Please Ignore",
        description="This is a test ticket created to verify that tickets can be created without a priority field.",
        issue_type="Task",  # 使用已知有效的問題類型
        project_key=project_key,
        priority=None,  # 明確設置為 None
    )

    if "key" in result:
        logger.info(f"Successfully created test ticket: {result.get('key')}")
        return True
    else:
        logger.error(
            f"Failed to create test ticket: {result.get('error', 'unknown error')}"
        )
        logger.error(f"Error details: {result.get('details', 'no details')}")
        return False


async def test_create_incident_ticket():
    """Test creating an incident ticket with severity that would normally map to priority."""
    logger.info("Testing incident ticket creation...")
    client = JiraClient()

    if not client.is_configured:
        logger.error("JIRA client is not properly configured")
        return

    # Create an incident ticket
    logger.info("Creating test incident ticket...")
    result = await client.create_incident_ticket(
        incident_id="TEST-123",
        title="Test Incident Ticket",
        description="This is a test incident ticket to verify our fix for the priority field issue.",
        severity="high",  # 通常會映射到優先級
        source="test",
        event_type="test",
    )

    if "key" in result:
        logger.info(f"Successfully created incident test ticket: {result.get('key')}")
        return True
    else:
        logger.error(
            f"Failed to create incident test ticket: {result.get('error', 'unknown error')}"
        )
        logger.error(f"Error details: {result.get('details', 'no details')}")
        return False


async def main():
    """Main function."""
    try:
        # 初始化問題類型映射
        client = JiraClient()
        await client.initialize_issue_type_map()

        # 測試直接創建票證（不帶優先級）
        regular_result = await test_create_jira_ticket_without_priority()

        # 測試創建事件票證（帶嚴重性，通常會映射到優先級）
        incident_result = await test_create_incident_ticket()

        # 輸出總結
        if regular_result and incident_result:
            logger.info("所有測試都成功，問題已解決!")
        elif regular_result:
            logger.warning("正常票證創建成功，但事件票證創建失敗。")
        elif incident_result:
            logger.warning("事件票證創建成功，但正常票證創建失敗。")
        else:
            logger.error("所有測試都失敗，問題仍然存在。")

    except Exception as e:
        logger.exception(f"Error in test script: {str(e)}")
    finally:
        # Clean up any resources if needed
        pass


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    )

    asyncio.run(main())

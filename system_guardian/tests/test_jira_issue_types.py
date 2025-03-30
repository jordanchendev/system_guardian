"""Test script to check JIRA issue types."""

import asyncio
import sys
from loguru import logger

from system_guardian.services.jira.client import JiraClient


async def get_jira_issue_types():
    """Get JIRA issue types for the configured project."""
    logger.info("Retrieving JIRA issue types...")
    client = JiraClient()

    if not client.is_configured:
        logger.error("JIRA client is not properly configured")
        return

    # Get project key
    project_key = client.project_key
    logger.info(f"Using project key: {project_key}")

    # Get issue types
    result = await client.get_issue_types(project_key)

    if "issue_types" in result:
        issue_types = result["issue_types"]
        logger.info(f"Found {len(issue_types)} issue types:")

        # Print issue types table
        logger.info("ID\t\tName")
        logger.info("-" * 30)

        for issue_type in issue_types:
            logger.info(f"{issue_type.get('id')}\t{issue_type.get('name')}")

        # Update issue type map
        await client.initialize_issue_type_map()

        # Try creating a test ticket with the first found issue type
        if issue_types:
            first_type = issue_types[0]
            logger.info(
                f"Testing ticket creation with issue type: {first_type.get('name')} (ID: {first_type.get('id')})"
            )

            result = await client.create_ticket(
                summary="Test Ticket - Please Ignore",
                description="This is a test ticket created to verify issue type configuration.",
                issue_type=first_type.get("name"),
                project_key=project_key,
            )

            if "key" in result:
                logger.info(f"Successfully created test ticket: {result.get('key')}")
            else:
                logger.error(
                    f"Failed to create test ticket: {result.get('error', 'unknown error')}"
                )
                logger.error(f"Error details: {result.get('details', 'no details')}")
    else:
        logger.error(
            f"Failed to get issue types: {result.get('error', 'unknown error')}"
        )
        logger.error(f"Error details: {result.get('details', 'no details')}")


async def main():
    """Main function."""
    try:
        await get_jira_issue_types()
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

#!/usr/bin/env python
"""
Test script for AI report generation functionality.

This script uses the llm_generate_fake_events.py to generate events,
which are then automatically detected as incidents by the system.
These system-generated incidents are then used to test report generation,
root cause analysis, and recommendation generation features.
"""
import json
import random
import asyncio
import uuid
import datetime
import argparse
import aiohttp
import importlib
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from openai import AsyncOpenAI
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from system_guardian.db.models.incidents import Incident
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker

from system_guardian.settings import settings

# Base URL of your API
BASE_URL = "http://localhost:5566"  # Change this to your API base URL

# Import the llm_generate_fake_events module
try:
    from system_guardian.tests.llm_generate_fake_events import (
        init_openai_client,
        generate_and_send_events,
        generate_random_incident,
        generate_similar_incidents,
        test_severity_classification,
    )

    logger.info("Successfully imported llm_generate_fake_events module")
except ImportError:
    logger.error(
        "Could not import llm_generate_fake_events module. Make sure it's available in the system_guardian.tests package."
    )
    sys.exit(1)

# Report formats
REPORT_FORMATS = ["json", "markdown", "html"]


async def wait_for_incidents(
    session: aiohttp.ClientSession, min_count: int = 5, max_wait_seconds: int = 120
) -> List[Dict[str, Any]]:
    """
    Wait for the system to generate incidents after sending events.
    Directly queries the incidents from the database instead of using the API endpoint.

    Args:
        session: aiohttp client session
        min_count: Minimum number of incidents to wait for
        max_wait_seconds: Maximum time to wait in seconds

    Returns:
        List of incidents
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.ext.asyncio import async_sessionmaker

    logger.info(
        f"Waiting for at least {min_count} incidents to be generated (max wait: {max_wait_seconds}s)..."
    )

    try:
        # Using the proper way to create database connection as in conftest.py
        engine = create_async_engine(str(settings.db_url))
        async_session_factory = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
    except Exception as e:
        logger.error(f"Failed to create database connection: {e}")
        logger.info("Falling back to API method...")
        return await wait_for_incidents_via_api(session, min_count, max_wait_seconds)

    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=max_wait_seconds)

    while datetime.datetime.now() < end_time:
        # Query the incidents table directly
        try:
            async with async_session_factory() as db_session:
                # Use a transaction
                async with db_session.begin():
                    # Execute query within transaction
                    result = await db_session.execute(select(Incident))
                    incidents_objs = result.scalars().all()

                    incidents = []
                    for incident in incidents_objs:
                        # Convert SQLAlchemy object to dict
                        incident_dict = {
                            "id": incident.id,
                            "title": incident.title,
                            "description": incident.description,
                            "severity": incident.severity,
                            "status": incident.status,
                            "created_at": (
                                incident.created_at.isoformat()
                                if incident.created_at
                                else None
                            ),
                            "resolved_at": (
                                incident.resolved_at.isoformat()
                                if incident.resolved_at
                                else None
                            ),
                            "source": incident.source,
                        }
                        incidents.append(incident_dict)

                if len(incidents) >= min_count:
                    logger.info(
                        f"Found {len(incidents)} incidents in database, proceeding with tests"
                    )
                    await engine.dispose()  # Properly dispose the engine
                    return incidents

                logger.info(
                    f"Found {len(incidents)} incidents in database, waiting for more..."
                )

        except Exception as e:
            logger.error(f"Error querying incidents from database: {e}")
            logger.info("Falling back to API method...")
            try:
                await engine.dispose()  # Try to clean up engine before falling back
            except:
                pass
            return await wait_for_incidents_via_api(
                session, min_count, max_wait_seconds
            )

        # Wait before checking again
        await asyncio.sleep(5)

    # If we get here, we've timed out - return whatever incidents we have
    try:
        async with async_session_factory() as db_session:
            async with db_session.begin():
                result = await db_session.execute(select(Incident))
                incidents_objs = result.scalars().all()

                incidents = []
                for incident in incidents_objs:
                    # Convert SQLAlchemy object to dict
                    incident_dict = {
                        "id": incident.id,
                        "title": incident.title,
                        "description": incident.description,
                        "severity": incident.severity,
                        "status": incident.status,
                        "created_at": (
                            incident.created_at.isoformat()
                            if incident.created_at
                            else None
                        ),
                        "resolved_at": (
                            incident.resolved_at.isoformat()
                            if incident.resolved_at
                            else None
                        ),
                        "source": incident.source,
                    }
                    incidents.append(incident_dict)

        logger.warning(
            f"Timed out waiting for incidents. Proceeding with {len(incidents)} incidents from database."
        )
        await engine.dispose()  # Properly dispose the engine
        return incidents
    except Exception as e:
        logger.error(f"Error querying incidents from database: {e}")
        try:
            await engine.dispose()  # Clean up engine
        except:
            pass
        return await wait_for_incidents_via_api(session, min_count, max_wait_seconds)

    logger.error(f"Timed out waiting for incidents and could not fetch any incidents")
    try:
        await engine.dispose()  # Final cleanup
    except:
        pass
    return []


async def wait_for_incidents_via_api(
    session: aiohttp.ClientSession, min_count: int = 5, max_wait_seconds: int = 120
) -> List[Dict[str, Any]]:
    """
    Fall back to using the API endpoint to wait for incidents if database access fails.

    Args:
        session: aiohttp client session
        min_count: Minimum number of incidents to wait for
        max_wait_seconds: Maximum time to wait in seconds

    Returns:
        List of incidents
    """
    logger.info(
        f"Using API to wait for at least {min_count} incidents (max wait: {max_wait_seconds}s)..."
    )

    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=max_wait_seconds)

    while datetime.datetime.now() < end_time:
        # Query the incidents endpoint
        try:
            async with session.get(f"{BASE_URL}/api/incidents") as response:
                if response.status == 200:
                    incidents = await response.json()

                    if len(incidents) >= min_count:
                        logger.info(
                            f"Found {len(incidents)} incidents via API, proceeding with tests"
                        )
                        return incidents

                    logger.info(
                        f"Found {len(incidents)} incidents via API, waiting for more..."
                    )
                else:
                    logger.warning(f"Failed to get incidents: {await response.text()}")
        except Exception as e:
            logger.error(f"Error querying incidents via API: {e}")

        # Wait before checking again
        await asyncio.sleep(5)

    # If we get here, we've timed out - return whatever incidents we have
    try:
        async with session.get(f"{BASE_URL}/api/incidents") as response:
            if response.status == 200:
                incidents = await response.json()
                logger.warning(
                    f"Timed out waiting for incidents. Proceeding with {len(incidents)} incidents from API."
                )
                return incidents
    except Exception as e:
        logger.error(f"Error querying incidents via API: {e}")

    logger.error(f"Timed out waiting for incidents and could not fetch any incidents")
    return []


async def generate_fake_resolution(
    session: aiohttp.ClientSession, incident_id: int
) -> bool:
    """
    Generate a resolution for the specified incident

    Args:
        session: aiohttp client session
        incident_id: Incident ID

    Returns:
        Success status
    """
    payload = {
        "incident_id": incident_id,
        "force_regenerate": True,
        "model": None,
        "temperature": 0.3,
    }

    try:
        async with session.post(
            f"{BASE_URL}/api/ai/generate-resolution", json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(
                    f"Successfully generated resolution for incident {incident_id}, ID: {result.get('id')}"
                )
                return True
            else:
                logger.error(f"Failed to generate resolution: {await response.text()}")
                return False
    except Exception as e:
        logger.error(f"Error generating resolution: {e}")
        return False


async def prepare_incidents_with_resolutions(
    session: aiohttp.ClientSession,
    incidents: List[Dict[str, Any]],
    resolution_percentage: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Prepare incidents by generating resolutions for some of them

    Args:
        session: aiohttp client session
        incidents: List of incidents
        resolution_percentage: Percentage of incidents to add resolutions to

    Returns:
        List of prepared incidents
    """
    # Decide which incidents should have resolutions
    num_resolutions = max(1, int(len(incidents) * resolution_percentage))
    resolution_incidents = random.sample(incidents, num_resolutions)

    # Generate resolutions
    tasks = []
    for incident in resolution_incidents:
        tasks.append(generate_fake_resolution(session, incident["id"]))

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    logger.info(f"Generated resolutions for {num_resolutions} incidents")
    return incidents


async def test_incident_report(
    session: aiohttp.ClientSession,
    incident_id: int,
    format: str = "markdown",
) -> Dict[str, Any]:
    """
    Test incident report generation functionality

    Args:
        session: aiohttp client session
        incident_id: Incident ID to generate report for
        format: Report format (json, markdown, html)

    Returns:
        Generated report data
    """
    try:
        # Always use markdown format to avoid model compatibility issues
        safe_format = "markdown"
        if format != "markdown":
            logger.info(
                f"Switching from {format} to markdown format for better model compatibility"
            )

        # Generate report request
        payload = {
            "incident_id": incident_id,
            "include_events": True,
            "include_resolution": True,
            "format": safe_format,
        }

        logger.info(
            f"Requesting incident report for incident {incident_id}, format: {safe_format}"
        )
        async with session.post(
            f"{BASE_URL}/api/ai/generate-incident-report", json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(
                    f"Successfully generated incident report, incident ID: {incident_id}"
                )
                return result
            else:
                logger.error(f"Failed to generate report: {await response.text()}")
                return {}
    except Exception as e:
        logger.error(f"Error testing incident report: {e}")
        return {}


async def test_summary_report(
    session: aiohttp.ClientSession, time_range_days: int = 7, format: str = "markdown"
) -> Dict[str, Any]:
    """
    Test summary report generation functionality

    Args:
        session: aiohttp client session
        time_range_days: Time range in days
        format: Report format (json, markdown, html)

    Returns:
        Generated summary report data
    """
    try:
        # Always use markdown format to avoid model compatibility issues
        safe_format = "markdown"
        if format != "markdown":
            logger.info(
                f"Switching from {format} to markdown format for better model compatibility"
            )

        # Generate report request
        payload = {
            "time_range_days": time_range_days,
            "format": safe_format,
            "include_severity_distribution": True,
            "include_resolution_times": True,
            "include_recommendations": True,
        }

        logger.info(
            f"Requesting summary report for the past {time_range_days} days, format: {safe_format}"
        )
        async with session.post(
            f"{BASE_URL}/api/ai/generate-summary-report", json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info("Successfully generated summary report")
                return result
            else:
                logger.error(
                    f"Failed to generate summary report: {await response.text()}"
                )
                return {}
    except Exception as e:
        logger.error(f"Error testing summary report: {e}")
        return {}


async def test_operational_recommendations(
    session: aiohttp.ClientSession, time_range_days: int = 30, format: str = "markdown"
) -> Dict[str, Any]:
    """
    Test operational recommendations generation functionality

    Args:
        session: aiohttp client session
        time_range_days: Time range in days
        format: Report format (json, markdown, html)

    Returns:
        Generated recommendations data
    """
    try:
        # Always use markdown format to avoid model compatibility issues
        safe_format = "markdown"
        if format != "markdown":
            logger.info(
                f"Switching from {format} to markdown format for better model compatibility"
            )

        # Generate recommendations request
        payload = {"time_range_days": time_range_days, "format": safe_format}

        logger.info(
            f"Requesting operational recommendations for the past {time_range_days} days, format: {safe_format}"
        )
        async with session.post(
            f"{BASE_URL}/api/ai/generate-recommendations", json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info("Successfully generated operational recommendations")
                return result
            else:
                logger.error(
                    f"Failed to generate operational recommendations: {await response.text()}"
                )
                return {}
    except Exception as e:
        logger.error(f"Error testing operational recommendations: {e}")
        return {}


async def test_root_cause_analysis(
    session: aiohttp.ClientSession, incident_id: int
) -> Dict[str, Any]:
    """
    Test root cause analysis functionality

    Args:
        session: aiohttp client session
        incident_id: Incident ID to analyze

    Returns:
        Root cause analysis results
    """
    try:
        # Generate root cause analysis request
        payload = {"incident_id": incident_id}

        logger.info(f"Requesting root cause analysis for incident {incident_id}")
        async with session.post(
            f"{BASE_URL}/api/ai/analytics/root-cause-analysis", json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(
                    f"Successfully generated root cause analysis, incident ID: {incident_id}"
                )
                return result
            else:
                logger.error(
                    f"Failed to generate root cause analysis: {await response.text()}"
                )
                return {}
    except Exception as e:
        logger.error(f"Error testing root cause analysis: {e}")
        return {}


async def setup_test_events(event_count: int = 10) -> None:
    """
    Generate fake events using llm_generate_fake_events

    Args:
        event_count: Number of events to generate
    """
    logger.info(
        f"Generating {event_count} test events using llm_generate_fake_events..."
    )

    # Initialize OpenAI client
    client = await init_openai_client()
    if not client:
        logger.error("Failed to initialize OpenAI client. Exiting.")
        return

    # Generate events
    await generate_and_send_events(num_events=event_count)

    logger.info(f"Event generation completed. Events should be detected as incidents.")


async def test_all_reports(
    session: aiohttp.ClientSession, num_events: int = 10
) -> None:
    """
    Test all report functionalities using system-generated incidents

    Args:
        session: aiohttp client session
        num_events: Number of events to generate
    """
    logger.info("Starting testing of all report functionalities...")

    # Generate events which will be detected as incidents
    await setup_test_events(num_events)

    # Wait for incidents to be generated
    incidents = await wait_for_incidents(session, min_count=3)
    if not incidents:
        logger.error("No incidents were detected, aborting test")
        return

    # Prepare incidents with resolutions
    await prepare_incidents_with_resolutions(session, incidents)

    # Select an incident for report testing
    test_incident = random.choice(incidents)
    incident_id = test_incident["id"]

    # Test only with markdown format to avoid compatibility issues
    format = "markdown"
    logger.info(f"Testing incident report with {format} format...")
    await test_incident_report(session, incident_id, format)

    # Test summary report
    logger.info("Testing summary report...")
    await test_summary_report(session, 7, format)

    # Test operational recommendations
    logger.info("Testing operational recommendations...")
    await test_operational_recommendations(session, 30, format)

    # Test root cause analysis
    logger.info("Testing root cause analysis...")
    await test_root_cause_analysis(session, incident_id)

    logger.info("All report functionality testing completed")


async def run_tests(
    specific_test: Optional[str] = None,
    event_count: int = 10,
    format: str = "markdown",
) -> None:
    """
    Run specified tests

    Args:
        specific_test: Specified test (incident_report, summary_report, recommendations, root_cause_analysis, all)
        event_count: Number of events to generate
        format: Report format
    """
    # Always use markdown format to ensure compatibility with all models
    safe_format = "markdown"
    if format != "markdown":
        logger.info(
            f"Overriding format parameter from {format} to {safe_format} for better model compatibility"
        )

    async with aiohttp.ClientSession() as session:
        # Generate events which will be detected as incidents
        await setup_test_events(event_count)

        # Wait for incidents to be generated
        incidents = await wait_for_incidents(session, min_count=3)
        if not incidents:
            logger.error("No incidents were detected, aborting test")
            return

        # Prepare incidents with resolutions
        incidents = await prepare_incidents_with_resolutions(session, incidents)

        # Select a test incident
        test_incident = random.choice(incidents)
        incident_id = test_incident["id"]

        if specific_test == "incident_report":
            # Test incident report
            await test_incident_report(session, incident_id, safe_format)
        elif specific_test == "summary_report":
            # Test summary report
            await test_summary_report(session, 7, safe_format)
        elif specific_test == "recommendations":
            # Test operational recommendations
            await test_operational_recommendations(session, 30, safe_format)
        elif specific_test == "root_cause_analysis":
            # Test root cause analysis
            await test_root_cause_analysis(session, incident_id)
        else:
            # Test all functionalities
            await test_all_reports(session, event_count)


def main():
    """Command line interface for testing AI report functionality"""
    global BASE_URL

    parser = argparse.ArgumentParser(
        description="Test System Guardian's AI report generation functionality"
    )
    parser.add_argument("--url", type=str, help=f"API base URL (default: {BASE_URL})")
    parser.add_argument(
        "--events", type=int, default=10, help="Number of events to generate"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "incident_report",
            "summary_report",
            "recommendations",
            "root_cause_analysis",
            "all",
        ],
        default="all",
        help="Specify which test to run",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=REPORT_FORMATS,
        default="markdown",
        help="Report format (Note: markdown format will always be used for compatibility reasons)",
    )

    args = parser.parse_args()

    # Update BASE_URL if provided
    if args.url:
        BASE_URL = args.url

    logger.info(f"Using API URL: {BASE_URL}")
    logger.info(f"Will generate {args.events} events and test report functionality")

    # Inform user about format override
    if args.format != "markdown":
        logger.warning(
            f"Note: Requested format '{args.format}' will be overridden with 'markdown' for model compatibility"
        )

    # Run async tests
    asyncio.run(
        run_tests(
            specific_test=args.test if args.test != "all" else None,
            event_count=args.events,
            format=args.format,
        )
    )

    logger.info("AI report functionality testing completed")


if __name__ == "__main__":
    main()

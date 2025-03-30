#!/usr/bin/env python
"""
LLM-powered fake event generator for System Guardian.

This script uses OpenAI to generate realistic fake events for testing
the System Guardian platform. It creates more organic and diverse
incidents compared to template-based approaches.
"""
import json
import random
import asyncio
import uuid
import datetime
import argparse
import aiohttp
from typing import List, Dict, Any, Optional
from loguru import logger
from openai import AsyncOpenAI

from system_guardian.settings import settings

# Base URL of your API
BASE_URL = "http://localhost:5566"  # Change this to your API's base URL

# OpenAI client
openai_client = None

# Event sources and their event types
EVENT_SOURCES = {
    "github": ["issue", "pull_request", "push", "workflow_run", "release"],
    "jira": [
        "issue_created",
        "issue_updated",
        "issue_resolved",
        "issue_commented",
        "sprint_started",
    ],
    "datadog": ["alert", "metric_alert", "service_check", "event", "monitor"],
}

# Severity levels and their weights
SEVERITY_LEVELS = {
    "critical": 5,
    "high": 15,
    "medium": 30,
    "low": 50,
}

# Incident themes to guide LLM generation
INCIDENT_THEMES = [
    "database performance issues",
    "network connectivity problems",
    "kubernetes cluster issues",
    "API integration failures",
    "memory leaks",
    "security breaches",
    "authentication service issues",
    "data consistency problems",
    "microservice communication failures",
    "load balancing issues",
    "cache invalidation problems",
    "deployment failures",
    "storage capacity issues",
    "rate limiting problems",
    "data pipeline failures",
    "logging system overload",
    "DNS resolution issues",
    "SSL certificate expiration",
    "backup system failures",
    "software license issues",
]

# System components to include in generated incidents
SYSTEM_COMPONENTS = [
    "database",
    "web server",
    "application server",
    "load balancer",
    "cache",
    "message queue",
    "API gateway",
    "authentication service",
    "storage system",
    "notification service",
    "payment processor",
    "search service",
    "recommendation engine",
    "analytics pipeline",
    "kubernetes cluster",
    "container registry",
    "CI/CD pipeline",
    "monitoring system",
    "logging infrastructure",
    "backup system",
]

# Instructions for generating incidents
INCIDENT_GENERATION_PROMPT = """
Generate a realistic system incident with the following characteristics:
- Theme: {theme}
- Severity: {severity}
- Component: {component}

For each incident, provide:
1. A concise, specific title (under 15 words)
2. A detailed description (2-4 sentences) that includes:
   - What happened
   - Visible symptoms
   - Potential impact
   - Any error codes or specific technical details
3. A list of 5 relevant technical keywords
4. A realistic resolution for this incident (1-2 sentences)

Format the response as a JSON object with these fields: title, description, keywords (array), resolution
"""

# Instructions for generating similar incidents
SIMILAR_INCIDENTS_PROMPT = """
Generate {count} similar but distinct system incidents related to:
"{original_incident}"

Each incident should have variations in:
- Specific symptoms
- Error messages
- Affected subsystems
- Severity levels

For each incident, provide:
1. A concise, specific title (under 15 words)
2. A detailed description (2-4 sentences) with technical details
3. A list of 5 relevant technical keywords
4. A severity level (one of: critical, high, medium, low)

Format the response as a JSON array with {count} objects, each containing: title, description, keywords (array), severity
"""


async def init_openai_client():
    """Initialize the OpenAI client"""
    global openai_client

    if not openai_client:
        try:
            if not settings.openai_api_key:
                logger.error(
                    "No OpenAI API key found in settings. This script requires an OpenAI API key."
                )
                return None

            openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI client initialized successfully")
            return openai_client
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            return None

    return openai_client


async def generate_incident_with_llm(
    theme: Optional[str] = None,
    severity: Optional[str] = None,
    component: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use OpenAI to generate a realistic incident.

    Args:
        theme: Optional incident theme to guide generation
        severity: Optional severity level
        component: Optional system component

    Returns:
        Dict containing the generated incident data
    """
    client = await init_openai_client()
    if not client:
        logger.error("Failed to initialize OpenAI client")
        return {}

    # Use provided values or select random ones
    theme = theme or random.choice(INCIDENT_THEMES)
    severity = (
        severity
        or random.choices(
            list(SEVERITY_LEVELS.keys()), weights=list(SEVERITY_LEVELS.values())
        )[0]
    )
    component = component or random.choice(SYSTEM_COMPONENTS)

    # Generate the prompt with our parameters
    prompt = INCIDENT_GENERATION_PROMPT.format(
        theme=theme, severity=severity, component=component
    )

    try:
        # Call OpenAI API - use different configurations based on model capability
        messages = [
            {
                "role": "system",
                "content": "You are a technical incident generator that creates realistic system incidents for testing.",
            },
            {"role": "user", "content": prompt},
        ]

        # Try to create with JSON response format first
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # You can change this to a more suitable model
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            incident_data = json.loads(response.choices[0].message.content)
        except Exception as e:
            # If response_format is not supported, try without it and parse manually
            if "response_format" in str(e):
                logger.warning(
                    "Model doesn't support response_format. Using text format instead."
                )
                # Request JSON in the prompt instead
                modified_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Return the response as a valid JSON object."
                )
                messages[1]["content"] = modified_prompt

                response = await client.chat.completions.create(
                    model="gpt-4",  # You can change this to a more suitable model
                    messages=messages,
                    temperature=0.7,
                )
                # Try to extract JSON from the response
                content = response.choices[0].message.content
                try:
                    # Find JSON object in the response if there's surrounding text
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        incident_data = json.loads(json_str)
                    else:
                        raise ValueError("No JSON object found in response")
                except:
                    # If JSON parsing fails, create a structured fallback
                    logger.warning(
                        "Failed to parse JSON from response, using fallback incident"
                    )
                    raise
            else:
                # If it's another error, re-raise
                raise

        # Add metadata about how it was generated
        incident_data["theme"] = theme
        incident_data["component"] = component
        incident_data["severity"] = severity

        logger.info(f"Generated incident: {incident_data['title']}")
        return incident_data

    except Exception as e:
        logger.error(f"Error generating incident with LLM: {e}")
        # Return a simple fallback incident
        return {
            "title": f"Error with {component} related to {theme}",
            "description": f"The system experienced an issue with the {component}. This needs investigation.",
            "keywords": [component, theme, severity, "error", "system"],
            "resolution": "Restart the affected service and investigate logs for root cause.",
            "severity": severity,
            "theme": theme,
            "component": component,
        }


async def generate_similar_incidents_with_llm(
    original_incident: Dict[str, Any], count: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate a group of similar incidents using LLM.

    Args:
        original_incident: The original incident to base similarities on
        count: Number of similar incidents to generate

    Returns:
        List of similar incidents
    """
    client = await init_openai_client()
    if not client:
        logger.error("Failed to initialize OpenAI client")
        return []

    # Create prompt with the original incident details
    original_text = f"{original_incident['title']}: {original_incident['description']}"

    prompt = SIMILAR_INCIDENTS_PROMPT.format(
        count=count, original_incident=original_text
    )

    try:
        # Create messages for the API call
        messages = [
            {
                "role": "system",
                "content": "You are a technical incident generator that creates realistic system incidents for testing.",
            },
            {"role": "user", "content": prompt},
        ]

        # Try to create with JSON response format first
        try:
            response = await client.chat.completions.create(
                model="gpt-4",  # You can change this to a more suitable model
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
        except Exception as e:
            # If response_format is not supported, try without it and parse manually
            if "response_format" in str(e):
                logger.warning(
                    "Model doesn't support response_format. Using text format instead."
                )
                # Request JSON in the prompt
                modified_prompt = (
                    prompt + "\n\nIMPORTANT: Return the response as a valid JSON array."
                )
                messages[1]["content"] = modified_prompt

                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                )
                content = response.choices[0].message.content
            else:
                # If it's another error, re-raise
                raise

        # Try to extract JSON from the response
        try:
            # Find JSON array in the response if there's surrounding text
            json_start = content.find("[")
            json_end = content.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                similar_incidents = json.loads(json_str)
            else:
                # Try as an object with an incidents array property
                similar_incidents = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from response: {content[:100]}...")
            raise

        # The response might come as {"incidents": [...]} or directly as an array
        if isinstance(similar_incidents, dict) and "incidents" in similar_incidents:
            similar_incidents = similar_incidents["incidents"]

        logger.info(f"Generated {len(similar_incidents)} similar incidents")
        return similar_incidents

    except Exception as e:
        logger.error(f"Error generating similar incidents with LLM: {e}")
        # Return a simple fallback
        return [
            {
                "title": f"Similar issue to {original_incident['title']} (variation {i})",
                "description": f"A similar issue to the original incident but with slightly different characteristics.",
                "keywords": original_incident.get(
                    "keywords", ["error", "system", "issue"]
                ),
                "severity": original_incident.get("severity", "medium"),
            }
            for i in range(count)
        ]


async def create_event_from_incident(
    incident: Dict[str, Any], source: str, event_type: str, days_ago: int = 0
) -> Dict[str, Any]:
    """
    Convert an incident into an event for a specific source and event type.

    Args:
        incident: The incident data
        source: Event source (github, jira, etc.)
        event_type: Event type
        days_ago: Days ago the event was created

    Returns:
        Dict containing the event data
    """
    # Generate timestamp
    timestamp = datetime.datetime.utcnow() - datetime.timedelta(days=days_ago)
    created_at = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Generate a unique test ID for tracking this specific test
    test_id = f"llm-test_{uuid.uuid4()}"

    # Get data from the incident
    title = incident.get("title", "Unknown incident")
    description = incident.get("description", "No description provided")
    keywords = incident.get("keywords", ["error", "system", "incident"])
    severity = incident.get("severity", "medium")
    resolution = incident.get("resolution", "")

    # Add test metadata
    test_metadata = {
        "test_id": test_id,
        "test_theme": incident.get("theme", "unknown"),
        "test_component": incident.get("component", "unknown"),
        "test_severity": severity,
        "test_keywords": keywords,
    }

    if resolution:
        test_metadata["test_resolution"] = resolution

    # Create raw payload based on the source
    raw_payload = {}

    # Create source-specific payload
    if source == "github":
        if event_type == "issue":
            raw_payload = {
                "action": "opened",
                "issue": {
                    "title": title,
                    "body": description,
                    "state": "open",
                    "labels": keywords[:3],  # Use first 3 keywords as labels
                    "created_at": created_at,
                    "updated_at": created_at,
                    "user": {
                        "login": "llm-test-user",
                        "id": random.randint(10000, 99999),
                    },
                    "number": random.randint(100, 999),
                },
                "repository": {
                    "name": "system-services",
                    "full_name": "org/system-services",
                    "owner": {"login": "org"},
                },
                "sender": {"login": "llm-test-user"},
                **test_metadata,
            }
        elif event_type == "pull_request":
            raw_payload = {
                "action": "opened",
                "pull_request": {
                    "title": title,
                    "body": description,
                    "state": "open",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "user": {
                        "login": "llm-test-user",
                        "id": random.randint(10000, 99999),
                    },
                    "number": random.randint(100, 999),
                },
                "repository": {
                    "name": "system-services",
                    "full_name": "org/system-services",
                    "owner": {"login": "org"},
                },
                "sender": {"login": "llm-test-user"},
                **test_metadata,
            }
        elif event_type == "push":
            raw_payload = {
                "action": "push",
                "repository": {
                    "name": "system-services",
                    "full_name": "org/system-services",
                    "owner": {"login": "org"},
                },
                "sender": {"login": "llm-test-user"},
                **test_metadata,
            }
        elif event_type == "workflow_run":
            raw_payload = {
                "action": "completed",
                "workflow_run": {
                    "name": title,
                    "conclusion": "success",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "author": {"login": "llm-test-user"},
                },
                "repository": {
                    "name": "system-services",
                    "full_name": "org/system-services",
                    "owner": {"login": "org"},
                },
                "sender": {"login": "llm-test-user"},
                **test_metadata,
            }
        elif event_type == "release":
            raw_payload = {
                "action": "published",
                "release": {
                    "name": title,
                    "body": description,
                    "created_at": created_at,
                    "published_at": created_at,
                    "author": {"login": "llm-test-user"},
                },
                "repository": {
                    "name": "system-services",
                    "full_name": "org/system-services",
                    "owner": {"login": "org"},
                },
                "sender": {"login": "llm-test-user"},
                **test_metadata,
            }

    elif source == "jira":
        issue_id = f"SYS-{random.randint(1000, 9999)}"
        priority = severity_to_jira_priority(severity)
        status = "Open" if event_type == "issue_created" else "In Progress"

        raw_payload = {
            "webhookEvent": event_type,
            "issue": {
                "id": random.randint(10000, 99999),
                "key": issue_id,
                "fields": {
                    "summary": title,
                    "description": description,
                    "issuetype": {"name": "Bug"},
                    "priority": {"name": priority},
                    "status": {"name": status},
                    "created": created_at,
                    "updated": created_at,
                    "reporter": {
                        "displayName": "LLM Test User",
                        "emailAddress": "llm-test@example.com",
                    },
                },
            },
            **test_metadata,
        }
    elif source == "datadog":
        # Create a Datadog alert that matches the expected schema
        alert = {
            "title": title,
            "text": description,
            "alert_id": f"alert-{random.randint(10000, 99999)}",
            "alert_status": "triggered",
            "alert_metric": f"system.{incident.get('component', 'service')}.error",
            "alert_tags": keywords,
            "alert_created_at": int(timestamp.timestamp()),
            "alert_updated_at": int(timestamp.timestamp()),
            "org_id": f"org-{random.randint(10000, 99999)}",
            "org_name": "Test Org",
            "host": f"host-{random.randint(1, 100)}",
        }

        # Wrap the alert in the expected format with alerts array
        raw_payload = {
            "alerts": [alert],
            **test_metadata,
        }
    # Create the standardized event format for sending to API
    return {
        "source": source,
        "event_type": event_type,
        "timestamp": created_at,
        "raw_payload": raw_payload,
    }


def severity_to_jira_priority(severity: str) -> str:
    """Convert severity to JIRA priority"""
    mapping = {"critical": "Highest", "high": "High", "medium": "Medium", "low": "Low"}
    return mapping.get(severity, "Medium")


async def send_event_to_api(
    session: aiohttp.ClientSession, event: Dict[str, Any]
) -> bool:
    """
    Send the event to the API for processing.

    Args:
        session: The aiohttp client session
        event: The event to send

    Returns:
        True if successful, False otherwise
    """
    source = event["source"]
    raw_event_type = event["event_type"]

    # Determine actual event type format to send
    if source == "jira":
        # Keep full format, ensure it starts with "jira:"
        if not raw_event_type.startswith("jira:"):
            event_type = f"jira:{raw_event_type.split(':')[-1]}"
        else:
            event_type = raw_event_type
    else:
        # For other sources, use only the part after the colon
        event_type = raw_event_type.split(":")[-1]

    # Generate API request body based on source
    if source == "jira":
        # For Jira events, send the entire webhook payload
        api_payload = event["raw_payload"]
        # Ensure webhookEvent and issue_event_type_name are correctly set
        api_payload["webhookEvent"] = event_type
        api_payload["issue_event_type_name"] = event_type.replace("jira:", "")
    elif source == "github":
        # For GitHub events, send the entire webhook payload
        api_payload = event["raw_payload"]
        # GitHub events don't need additional adjustments
    elif source == "datadog":
        # For Datadog events, send the entire raw_payload directly
        api_payload = event["raw_payload"]
    else:
        # Handle other event sources as before
        api_payload = {
            "source": source,
            "event_type": event_type,
            "timestamp": event["timestamp"],
            "raw_payload": event["raw_payload"],
        }

    logger.debug(f"Sending payload to API: {api_payload}")

    # Ingest endpoint depends on the source
    endpoint = f"{BASE_URL}/api/ingest/{source}/"

    # Set appropriate headers for API to correctly identify event type
    headers = {"Content-Type": "application/json"}

    # Add source-specific event type headers
    if source == "github":
        headers["X-GitHub-Event"] = event_type
    elif source == "jira":
        # For Jira, use event type without prefix as header
        headers["X-Jira-Event"] = event_type.split(":")[-1]
    elif source == "datadog":
        headers["X-Datadog-Event"] = event_type

    try:
        logger.info(f"Sending {source}:{event_type} event to {endpoint}")
        async with session.post(
            endpoint, json=api_payload, headers=headers
        ) as response:
            if response.status == 200:
                logger.info(f"Successfully sent {source}:{event_type} event to API")
                return True
            else:
                logger.error(f"Failed to send event: {await response.text()}")
                return False
    except Exception as e:
        logger.error(f"Error sending event: {e}")
        return False


async def generate_random_incident(
    session: aiohttp.ClientSession,
    theme: Optional[str] = None,
    severity: Optional[str] = None,
    component: Optional[str] = None,
    days_ago: int = 0,
) -> Dict[str, Any]:
    """
    Generate a random incident and send it to the API.

    Args:
        session: The aiohttp client session
        theme: Optional theme to guide generation
        severity: Optional severity level
        component: Optional system component
        days_ago: Days ago to create the event

    Returns:
        The generated incident data
    """
    # Generate the incident using LLM
    incident = await generate_incident_with_llm(theme, severity, component)
    if not incident:
        logger.error("Failed to generate incident with LLM")
        return {}

    # Choose a random source and event type
    source = random.choice(list(EVENT_SOURCES.keys()))
    event_type = random.choice(EVENT_SOURCES[source])

    # Create the event from the incident
    event = await create_event_from_incident(
        incident=incident, source=source, event_type=event_type, days_ago=days_ago
    )

    # Send to API
    success = await send_event_to_api(session, event)

    if success:
        logger.info(f"Successfully sent incident: {incident['title']}")
        # Add the source and event_type to the incident data for reference
        incident["source"] = source
        incident["event_type"] = event_type

        # Safe access to test_id, handling case where it might not exist
        try:
            incident["test_id"] = event["raw_payload"]["test_id"]
        except (KeyError, TypeError):
            # Generate a fallback ID if missing
            incident["test_id"] = f"fallback-test-id-{uuid.uuid4()}"
            logger.warning(f"Created fallback test_id: {incident['test_id']}")

        return incident
    else:
        logger.error(f"Failed to send incident: {incident['title']}")
        return {}


async def generate_similar_incidents(
    session: aiohttp.ClientSession,
    count: int = 3,
    base_theme: Optional[str] = None,
    base_component: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a group of similar incidents and send them to the API.

    Args:
        session: The aiohttp client session
        count: Number of similar incidents to generate
        base_theme: Optional theme for the base incident
        base_component: Optional component for the base incident

    Returns:
        List of generated incident data
    """
    # First, generate a base incident
    base_incident = await generate_incident_with_llm(base_theme, None, base_component)
    if not base_incident:
        logger.error("Failed to generate base incident")
        return []

    # Send the base incident
    source = random.choice(list(EVENT_SOURCES.keys()))
    event_type = random.choice(EVENT_SOURCES[source])

    base_event = await create_event_from_incident(
        incident=base_incident,
        source=source,
        event_type=event_type,
        days_ago=random.randint(10, 30),  # Base incident is older
    )

    await send_event_to_api(session, base_event)

    # Now generate similar incidents
    similar_incidents = await generate_similar_incidents_with_llm(base_incident, count)
    result = [base_incident]

    # Send each similar incident
    for i, incident in enumerate(similar_incidents):
        # Use the same source and event type for consistency
        days_ago = max(0, 10 - i)  # More recent than base incident

        event = await create_event_from_incident(
            incident=incident, source=source, event_type=event_type, days_ago=days_ago
        )

        success = await send_event_to_api(session, event)

        if success:
            incident["source"] = source
            incident["event_type"] = event_type
            incident["test_id"] = event["raw_payload"]["test_id"]
            result.append(incident)

        # Sleep to avoid overwhelming the API
        await asyncio.sleep(0.5)

    logger.info(f"Generated and sent {len(result)} incidents in a similar group")
    return result


async def test_incident_detection_task(session: aiohttp.ClientSession, count: int = 5):
    """
    Task for testing incident detection by creating varied incidents from different sources.

    Args:
        session: The aiohttp client session
        count: Number of incidents to generate
    """
    logger.info(f"Starting incident detection task for {count} incidents...")

    # Create multiple tasks to run concurrently
    tasks = []
    for _ in range(count):
        theme = random.choice(INCIDENT_THEMES)
        component = random.choice(SYSTEM_COMPONENTS)
        tasks.append(generate_random_incident(session, theme, None, component))

    # Wait for all tasks to complete
    incidents = await asyncio.gather(*tasks)

    # Calculate the number of successfully generated events
    successful = sum(1 for incident in incidents if incident)
    logger.info(
        f"Incident detection task complete: {successful}/{count} incidents created"
    )


async def test_incident_similarity_task(
    session: aiohttp.ClientSession, groups: int = 2, per_group: int = 3
):
    """
    Task for testing incident similarity by creating groups of similar incidents.

    Args:
        session: The aiohttp client session
        groups: Number of similar incident groups to create
        per_group: Number of incidents per group
    """
    logger.info(
        f"Starting incident similarity task with {groups} groups of {per_group} incidents..."
    )

    # Create multiple tasks to run multiple groups concurrently
    tasks = []
    for _ in range(groups):
        theme = random.choice(INCIDENT_THEMES)
        component = random.choice(SYSTEM_COMPONENTS)
        tasks.append(generate_similar_incidents(session, per_group, theme, component))

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)

    # Calculate the total number of successfully generated events
    total_incidents = sum(len(group) for group in results)
    logger.info(
        f"Incident similarity task complete: {total_incidents} incidents created in {groups} groups"
    )


async def generate_random_incidents_in_parallel(
    session: aiohttp.ClientSession,
    num_events: int = 20,
    historical_days: int = 60,
    batch_size: int = 10,
):
    """
    Generate random incidents in parallel batches for better performance.

    Args:
        session: The aiohttp client session
        num_events: Total number of events to generate
        historical_days: Maximum days in the past to generate events
        batch_size: Number of events to generate in parallel per batch
    """
    logger.info(
        f"Generating {num_events} LLM-powered incidents over the past {historical_days} days in parallel..."
    )

    total_generated = 0

    # Calculate the number of batches needed
    num_batches = (num_events + batch_size - 1) // batch_size

    for batch in range(num_batches):
        # Calculate the number of events to generate in this batch
        current_batch_size = min(batch_size, num_events - total_generated)

        # Create tasks for this batch
        tasks = []
        for _ in range(current_batch_size):
            days_ago = random.randint(0, historical_days)
            tasks.append(generate_random_incident(session, days_ago=days_ago))

        # Wait for all tasks in this batch to complete
        batch_results = await asyncio.gather(*tasks)

        # Calculate the number of successfully generated events
        successful = sum(1 for incident in batch_results if incident)
        total_generated += successful

        logger.info(
            f"Batch {batch + 1}/{num_batches} complete: {successful}/{current_batch_size} incidents created"
        )

    logger.info(f"Successfully generated {total_generated}/{num_events} incidents")


async def test_severity_classification(session: aiohttp.ClientSession):
    """
    Test severity classification by creating incidents with different severity levels.

    Args:
        session: The aiohttp client session
    """
    logger.info("Testing severity classification with LLM-generated incidents...")

    # Create asynchronous tasks for each severity level
    tasks = []
    for severity in SEVERITY_LEVELS.keys():
        # Choose random theme and component for each severity level
        theme = random.choice(INCIDENT_THEMES)
        component = random.choice(SYSTEM_COMPONENTS)

        logger.info(f"Scheduling {severity} severity incident generation")
        tasks.append(generate_random_incident(session, theme, severity, component))

    # Wait for all tasks to complete
    incidents = await asyncio.gather(*tasks)

    # Calculate the number of successfully generated events
    successful = sum(1 for incident in incidents if incident)
    logger.info(
        f"Severity classification test complete: {successful}/{len(SEVERITY_LEVELS)} incidents created"
    )


async def test_incident_detection(session: aiohttp.ClientSession, count: int = 5):
    """
    Test incident detection by creating varied incidents from different sources.
    This function is kept for backwards compatibility.

    Args:
        session: The aiohttp client session
        count: Number of incidents to generate
    """
    return await test_incident_detection_task(session, count)


async def test_incident_similarity(
    session: aiohttp.ClientSession, groups: int = 2, per_group: int = 3
):
    """
    Test incident similarity by creating groups of similar incidents.
    This function is kept for backwards compatibility.

    Args:
        session: The aiohttp client session
        groups: Number of similar incident groups to create
        per_group: Number of incidents per group
    """
    return await test_incident_similarity_task(session, groups, per_group)


async def generate_and_send_events(
    num_events: int = 20,
    specific_test: Optional[str] = None,
    historical_days: int = 60,
    similar_groups: int = 2,
) -> None:
    """
    Generate and send LLM-enhanced events to test various system features.
    All event generation is handled asynchronously for better performance.

    Args:
        num_events: Number of events to generate if running random tests
        specific_test: Specific test to run (detection, severity, similarity)
        historical_days: Maximum days in the past to generate events
        similar_groups: Number of similar incident groups to create for similarity testing
    """
    # Initialize OpenAI client
    client = await init_openai_client()
    if not client:
        logger.error("Failed to initialize OpenAI client. Exiting.")
        return

    async with aiohttp.ClientSession() as session:
        if specific_test == "detection":
            # Create multiple tasks to run concurrently
            tasks = [
                test_incident_detection_task(
                    session, num_events // 5 + (1 if i < num_events % 5 else 0)
                )
                for i in range(5)
            ]
            await asyncio.gather(*tasks)
        elif specific_test == "severity":
            await test_severity_classification(session)
        elif specific_test == "similarity":
            # Create multiple similar incident groups concurrently
            tasks = [
                generate_similar_incidents(
                    session,
                    3,
                    random.choice(INCIDENT_THEMES),
                    random.choice(SYSTEM_COMPONENTS),
                )
                for _ in range(similar_groups)
            ]
            await asyncio.gather(*tasks)
        elif specific_test == "all":
            # Run all tests concurrently
            tasks = [
                test_incident_detection_task(session, 5),
                test_severity_classification(session),
                test_incident_similarity_task(session, 2, 3),
            ]
            await asyncio.gather(*tasks)
        else:
            # Generate random incidents in parallel batches
            await generate_random_incidents_in_parallel(
                session, num_events, historical_days
            )


def main():
    """Command line interface for generating LLM-enhanced test events"""
    global BASE_URL

    parser = argparse.ArgumentParser(
        description="Generate LLM-enhanced test events for System Guardian"
    )
    parser.add_argument(
        "--url", type=str, help=f"Base URL of the API (default: {BASE_URL})"
    )
    parser.add_argument(
        "--events", type=int, default=20, help="Number of events to generate"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Maximum days in the past to generate events",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["detection", "severity", "similarity", "all"],
        help="Specific test to run",
    )
    parser.add_argument(
        "--similar-groups",
        type=int,
        default=2,
        help="Number of similar incident groups to create",
    )
    parser.add_argument(
        "--theme",
        type=str,
        help="Specific theme for incidents (if not provided, random themes will be used)",
    )

    args = parser.parse_args()

    # Update base URL if provided
    if args.url:
        BASE_URL = args.url

    logger.info(f"Using API URL: {BASE_URL}")

    if args.theme:
        logger.info(f"Using specific theme: {args.theme}")
        # If theme provided but not in our list, add it
        if args.theme not in INCIDENT_THEMES:
            INCIDENT_THEMES.append(args.theme)

    if args.test:
        logger.info(f"Running specific test: {args.test}")
    else:
        logger.info(
            f"Generating {args.events} LLM-enhanced incidents spanning {args.days} days"
        )

    # Run the async function
    asyncio.run(
        generate_and_send_events(
            num_events=args.events,
            specific_test=args.test,
            historical_days=args.days,
            similar_groups=args.similar_groups,
        )
    )

    logger.info("LLM-enhanced event generation complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Enhanced fake event generator for System Guardian.

This script generates fake events specifically designed to test:
- Incident detection
- Incident similarity
- Resolution generation
- Severity classification
- Related incidents & insights
- Recommend fix steps
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

# Enhanced templates for testing specific features
ENHANCED_TEMPLATES = {
    "database_issues": [
        {
            "title": "Database connection pool exhaustion causing application timeouts",
            "description": "Our application is experiencing timeout errors when connecting to the database. Logs show 'connection limit exceeded' and users are reporting slow response times. This is CRITICAL and requires immediate action.",
            "keywords": ["database", "connection", "timeout", "exceeded", "pool"],
            "severity": "critical",
            "resolution": "Increased max_connections parameter in postgresql.conf and implemented connection pooling with PgBouncer.",
        },
        {
            "title": "Database replication lag affecting data consistency",
            "description": "Read replicas are showing outdated data compared to the primary. Monitoring shows replication lag exceeding 30 seconds. This is causing data inconsistency issues in the application.",
            "keywords": ["database", "replication", "lag", "consistency", "replica"],
            "severity": "high",
            "resolution": "Optimized write operations on primary database and increased network bandwidth between primary and replicas.",
        },
        {
            "title": "Slow database queries degrading application performance",
            "description": "Several API endpoints are responding slowly due to inefficient database queries. Query times have increased from milliseconds to seconds, affecting user experience.",
            "keywords": ["database", "slow", "query", "performance", "api"],
            "severity": "medium",
            "resolution": "Added missing indexes and optimized JOIN operations in the problematic queries.",
        },
    ],
    "network_issues": [
        {
            "title": "Network connectivity issues between microservices",
            "description": "Intermittent connectivity issues between frontend and backend services. Users are experiencing timeouts and error messages. This OUTAGE is affecting all operations.",
            "keywords": [
                "network",
                "connectivity",
                "timeout",
                "microservice",
                "outage",
            ],
            "severity": "critical",
            "resolution": "Fixed misconfigured firewall rules that were blocking traffic between service subnets.",
        },
        {
            "title": "DNS resolution failures causing service unavailability",
            "description": "Services are unable to resolve hostnames correctly. Error logs show 'unknown host' messages. This is causing intermittent service disruptions.",
            "keywords": ["dns", "resolution", "hostname", "service", "unavailable"],
            "severity": "high",
            "resolution": "Repaired corrupted DNS cache and added redundant DNS servers for failover.",
        },
    ],
    "application_errors": [
        {
            "title": "Memory leak in authentication service",
            "description": "The authentication service is experiencing memory leaks, causing it to crash every few hours. Users are being logged out unexpectedly. This is CRITICAL as it affects all users.",
            "keywords": ["memory", "leak", "crash", "authentication", "service"],
            "severity": "critical",
            "resolution": "Fixed memory leak by properly closing database connections in exception handlers.",
        },
        {
            "title": "API rate limiting errors in third-party integration",
            "description": "Our integration with the payment processor is hitting rate limits during peak hours. This is causing payment failures and affecting customer purchases.",
            "keywords": ["api", "rate", "limit", "integration", "payment"],
            "severity": "high",
            "resolution": "Implemented request batching and exponential backoff retry strategy for the payment API calls.",
        },
    ],
    "security_incidents": [
        {
            "title": "Authentication service breach attempt detected",
            "description": "Multiple failed login attempts from unusual IP ranges detected on the authentication service. Pattern suggests a brute force attack. This is a CRITICAL security incident.",
            "keywords": ["security", "authentication", "breach", "login", "attack"],
            "severity": "critical",
            "resolution": "Blocked suspicious IP ranges and implemented CAPTCHA for repeated failed login attempts.",
        },
        {
            "title": "Suspicious activity in admin accounts",
            "description": "Unusual login patterns detected for admin accounts outside of normal business hours. Possible unauthorized access to admin portal.",
            "keywords": ["security", "admin", "suspicious", "login", "unauthorized"],
            "severity": "high",
            "resolution": "Reset admin credentials and enabled multi-factor authentication for all administrative accounts.",
        },
    ],
    "kubernetes_issues": [
        {
            "title": "Multiple Kubernetes nodes showing NotReady status",
            "description": "Several nodes in the Kubernetes cluster are showing NotReady status. Pods are being evicted and new deployments are stuck in Pending state. This is causing service disruption.",
            "keywords": ["kubernetes", "node", "notready", "pod", "eviction"],
            "severity": "critical",
            "resolution": "Fixed corrupted kubelet certificates and restarted kubelet services on affected nodes.",
        },
        {
            "title": "Pods stuck in CrashLoopBackOff due to configuration issues",
            "description": "Several critical service pods are stuck in CrashLoopBackOff state. Logs show configuration errors when starting. Services are partially unavailable.",
            "keywords": ["kubernetes", "pod", "crash", "configuration", "service"],
            "severity": "high",
            "resolution": "Fixed incorrect environment variables in the deployment configuration and redeployed affected services.",
        },
    ],
}

# Similar incident groups for testing incident similarity
SIMILAR_INCIDENTS = [
    # Database connection issues group
    [
        {
            "title": "Database connection pool exhausted in production",
            "description": "Connection pool limit reached in the production database, causing timeouts and errors for users. Application logs show 'connection limit exceeded'.",
            "keywords": ["database", "connection", "pool", "exhausted", "production"],
            "severity": "critical",
        },
        {
            "title": "Database connections maxed out during peak hours",
            "description": "During high traffic periods, we're seeing database connection errors. The pool is exhausted and new connections are being rejected.",
            "keywords": ["database", "connection", "maxed", "pool", "rejected"],
            "severity": "high",
        },
        {
            "title": "Connection timeout errors when accessing database",
            "description": "Users report timeout errors when performing data-intensive operations. Logs show database connection pool is being depleted.",
            "keywords": ["connection", "timeout", "database", "pool", "depleted"],
            "severity": "high",
        },
    ],
    # Memory leak group
    [
        {
            "title": "Memory leak in authentication service",
            "description": "Auth service is consuming more memory over time until it crashes. Requires restart every few hours to maintain service.",
            "keywords": ["memory", "leak", "authentication", "crash", "restart"],
            "severity": "critical",
        },
        {
            "title": "Authentication module showing memory growth pattern",
            "description": "Monitoring shows steady memory increase in the auth module until OOM errors occur. Service becomes unresponsive after running for several hours.",
            "keywords": ["authentication", "memory", "growth", "OOM", "unresponsive"],
            "severity": "high",
        },
        {
            "title": "Out of memory errors in identity service",
            "description": "Identity management service experiences out of memory errors after prolonged use. Memory usage graph shows continuous upward trend without plateaus.",
            "keywords": ["memory", "errors", "identity", "service", "usage"],
            "severity": "high",
        },
    ],
    # Kubernetes node issues
    [
        {
            "title": "Kubernetes nodes showing NotReady status",
            "description": "Multiple nodes in production Kubernetes cluster are in NotReady state. Pods are being evicted causing service disruption.",
            "keywords": ["kubernetes", "nodes", "notready", "pods", "evicted"],
            "severity": "critical",
        },
        {
            "title": "K8s cluster node failures affecting deployments",
            "description": "Several nodes in the Kubernetes cluster are failing health checks. New pods are stuck in pending state and cannot be scheduled.",
            "keywords": ["kubernetes", "nodes", "failures", "pods", "pending"],
            "severity": "critical",
        },
        {
            "title": "Worker nodes disconnected from Kubernetes master",
            "description": "Worker nodes are disconnected from the control plane. Kubelet service is failing to maintain connection to the API server.",
            "keywords": ["kubernetes", "worker", "nodes", "disconnected", "kubelet"],
            "severity": "high",
        },
    ],
]


async def init_openai_client():
    """Initialize the OpenAI client"""
    global openai_client

    if not openai_client:
        try:
            if not settings.openai_api_key:
                logger.warning(
                    "No OpenAI API key found in settings. Using template-based generation only."
                )
                return None

            openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI client initialized successfully")
            return openai_client
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            return None

    return openai_client


async def generate_enhanced_event(
    source: str,
    event_type: str,
    category: Optional[str] = None,
    template_index: Optional[int] = None,
    days_ago: int = 0,
    similar_group_index: Optional[int] = None,
    similar_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate an enhanced event with specific testing characteristics.

    Args:
        source: Event source (github, jira, etc.)
        event_type: Event type
        category: Category of incident to test
        template_index: Index of template to use from the category
        days_ago: Days ago the event was created
        similar_group_index: Index of similar incident group
        similar_index: Index within the similar incident group

    Returns:
        Dict containing the event data
    """
    # Generate timestamp
    timestamp = datetime.datetime.utcnow() - datetime.timedelta(days=days_ago)
    created_at = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Generate a unique test ID for tracking this specific test
    test_id = f"test_{uuid.uuid4()}"

    # Choose template based on parameters
    if similar_group_index is not None and similar_group_index < len(SIMILAR_INCIDENTS):
        # Use a template from the similar incidents group
        similar_group = SIMILAR_INCIDENTS[similar_group_index]

        # Choose the specific incident in the group or random if not specified
        if similar_index is not None and similar_index < len(similar_group):
            template = similar_group[similar_index]
        else:
            template = random.choice(similar_group)

        template["category"] = "similar_incidents"
    elif category and category in ENHANCED_TEMPLATES:
        # Use a template from the specified category
        templates = ENHANCED_TEMPLATES[category]

        if template_index is not None and template_index < len(templates):
            template = templates[template_index]
        else:
            template = random.choice(templates)

        template["category"] = category
    else:
        # Choose a random category and template
        category = random.choice(list(ENHANCED_TEMPLATES.keys()))
        template = random.choice(ENHANCED_TEMPLATES[category])
        template["category"] = category

    # Create raw payload based on the source
    raw_payload = {}

    # Add test metadata
    test_metadata = {
        "test_id": test_id,
        "test_category": template["category"],
        "test_severity": template["severity"],
        "test_keywords": template["keywords"],
    }

    if "resolution" in template:
        test_metadata["test_resolution"] = template["resolution"]

    # Create source-specific payload
    if source == "github":
        if event_type == "issue":
            raw_payload = {
                "action": "opened",
                "issue": {
                    "title": template["title"],
                    "body": template["description"],
                    "state": "open",
                    "labels": ["bug"],
                    "created_at": created_at,
                    "updated_at": created_at,
                    "user": {"login": "test-user", "id": random.randint(10000, 99999)},
                    "number": random.randint(100, 999),
                },
                "repository": {
                    "name": "test-repo",
                    "full_name": "test-org/test-repo",
                    "owner": {"login": "test-org"},
                },
                "sender": {"login": "test-user"},
                **test_metadata,
            }
        elif event_type == "pull_request":
            raw_payload = {
                "action": "opened",
                "pull_request": {
                    "title": template["title"],
                    "body": template["description"],
                    "state": "open",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "user": {"login": "test-user", "id": random.randint(10000, 99999)},
                    "number": random.randint(100, 999),
                },
                "repository": {
                    "name": "test-repo",
                    "full_name": "test-org/test-repo",
                    "owner": {"login": "test-org"},
                },
                "sender": {"login": "test-user"},
                **test_metadata,
            }
        elif event_type == "push":
            # Generate a random commit SHA
            commit_sha = "".join(random.choice("0123456789abcdef") for _ in range(40))
            previous_sha = "".join(random.choice("0123456789abcdef") for _ in range(40))

            # Create author information
            author = {
                "name": "test-user",
                "email": "test-user@example.com",
                "username": "test-user",
            }

            # Create commit with the template info
            commit = {
                "id": commit_sha,
                "url": f"https://github.com/test-org/test-repo/commit/{commit_sha}",
                "added": [f"path/to/file-{random.randint(1, 100)}.txt"],
                "author": author,
                "message": template["title"],
                "distinct": True,
                "removed": [],
                "modified": [f"path/to/existing-file-{random.randint(1, 100)}.txt"],
                "committer": author,
                "timestamp": created_at,
            }

            # Create repository information
            repository = {
                "id": random.randint(100000, 999999),
                "name": "test-repo",
                "full_name": "test-org/test-repo",
                "owner": {"login": "test-org", "id": random.randint(1000, 9999)},
                "private": False,
                "html_url": "https://github.com/test-org/test-repo",
                "description": template["description"],
            }

            # Create push event payload
            raw_payload = {
                "ref": "refs/heads/main",
                "before": previous_sha,
                "after": commit_sha,
                "repository": repository,
                "pusher": author,
                "sender": {"login": "test-user", "id": random.randint(10000, 99999)},
                "commits": [commit],
                "head_commit": commit,
                "created": False,
                "deleted": False,
                "forced": False,
                "base_ref": None,
                "compare": f"https://github.com/test-org/test-repo/compare/{previous_sha[0:7]}...{commit_sha[0:7]}",
                **test_metadata,
            }
    elif source == "jira":
        issue_id = f"TEST-{random.randint(1000, 9999)}"
        priority = severity_to_jira_priority(template["severity"])
        status = "Open" if event_type == "issue_created" else "In Progress"

        raw_payload = {
            "webhookEvent": event_type,
            "issue": {
                "id": random.randint(10000, 99999),
                "key": issue_id,
                "fields": {
                    "summary": template["title"],
                    "description": template["description"],
                    "issuetype": {"name": "Bug"},
                    "priority": {"name": priority},
                    "status": {"name": status},
                    "created": created_at,
                    "updated": created_at,
                    "reporter": {
                        "displayName": "Test User",
                        "emailAddress": "test@example.com",
                    },
                },
            },
            **test_metadata,
        }
    elif source == "datadog":
        # Create a Datadog alert that matches the expected schema
        alert = {
            "title": template["title"],
            "text": template["description"],
            "alert_id": f"alert-{random.randint(10000, 99999)}",
            "alert_status": "triggered",
            "alert_metric": f"test.metric.{event_type}",
            "alert_tags": template["keywords"],
            "alert_created_at": int(timestamp.timestamp()),
            "alert_updated_at": int(timestamp.timestamp()),
            "org_id": f"org-{random.randint(10000, 99999)}",
            "org_name": "Test Org",
            "host": f"test-host-{random.randint(1, 100)}",
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


async def create_similar_incident_group(
    session: aiohttp.ClientSession, group_index: int
) -> List[str]:
    """
    Create a group of similar incidents to test similarity detection.

    Args:
        session: The aiohttp client session
        group_index: Index of the similar incidents group to use

    Returns:
        List of test_ids for the created incidents
    """
    if group_index >= len(SIMILAR_INCIDENTS):
        logger.error(
            f"Invalid group index: {group_index}. Only {len(SIMILAR_INCIDENTS)} groups available."
        )
        return []

    logger.info(f"Creating similar incident group {group_index}")

    test_ids = []

    # Choose a random source and event type for this group
    source = random.choice(list(EVENT_SOURCES.keys()))
    event_type = random.choice(EVENT_SOURCES[source])

    # Create each incident in the group with different timestamps
    for i, _ in enumerate(SIMILAR_INCIDENTS[group_index]):
        # Create with different timestamps (older to newer)
        days_ago = max(0, 10 - i)  # 10, 9, 8, ... days ago

        incident = await generate_enhanced_event(
            source=source,
            event_type=event_type,
            days_ago=days_ago,
            similar_group_index=group_index,
            similar_index=i,
        )

        success = await send_event_to_api(session, incident)

        if success:
            test_ids.append(incident["raw_payload"]["test_id"])
            logger.info(
                f"Created similar incident {i+1}/{len(SIMILAR_INCIDENTS[group_index])}"
            )

        # Sleep to avoid overwhelming the API
        await asyncio.sleep(0.5)

    return test_ids


async def test_incident_detection(session: aiohttp.ClientSession):
    """
    Test incident detection by creating incidents from different sources and categories.

    Args:
        session: The aiohttp client session
    """
    logger.info("Testing incident detection...")

    # Try each source and multiple categories
    for source in EVENT_SOURCES:
        for category in random.sample(
            list(ENHANCED_TEMPLATES.keys()), min(2, len(ENHANCED_TEMPLATES))
        ):
            event_type = random.choice(EVENT_SOURCES[source])

            logger.info(f"Creating incident from {source} {event_type} for {category}")

            incident = await generate_enhanced_event(
                source=source, event_type=event_type, category=category
            )

            await send_event_to_api(session, incident)
            await asyncio.sleep(0.5)

    logger.info("Incident detection test complete")


async def test_severity_classification(session: aiohttp.ClientSession):
    """
    Test severity classification by creating incidents with different severity levels.

    Args:
        session: The aiohttp client session
    """
    logger.info("Testing severity classification...")

    # Choose a random source and event type
    source = random.choice(list(EVENT_SOURCES.keys()))
    event_type = random.choice(EVENT_SOURCES[source])

    # Go through each severity level
    for severity in ["critical", "high", "medium", "low"]:
        # Find categories and templates with matching severity
        matching_templates = []
        for category, templates in ENHANCED_TEMPLATES.items():
            for i, template in enumerate(templates):
                if template["severity"] == severity:
                    matching_templates.append((category, i))

        if matching_templates:
            category, template_index = random.choice(matching_templates)

            logger.info(
                f"Creating {severity} severity incident from {source} {event_type}"
            )

            incident = await generate_enhanced_event(
                source=source,
                event_type=event_type,
                category=category,
                template_index=template_index,
            )

            await send_event_to_api(session, incident)
            await asyncio.sleep(0.5)

    logger.info("Severity classification test complete")


async def test_incident_similarity(session: aiohttp.ClientSession):
    """
    Test incident similarity by creating groups of similar incidents.

    Args:
        session: The aiohttp client session
    """
    logger.info("Testing incident similarity...")

    # Create each similar incident group
    for group_index in range(len(SIMILAR_INCIDENTS)):
        test_ids = await create_similar_incident_group(session, group_index)
        logger.info(
            f"Created similar incident group {group_index} with {len(test_ids)} incidents"
        )
        await asyncio.sleep(1)

    logger.info("Incident similarity test complete")


async def test_resolution_generation(session: aiohttp.ClientSession):
    """
    Test resolution generation by creating incidents that should match knowledge base entries.

    Args:
        session: The aiohttp client session
    """
    logger.info("Testing resolution generation...")

    # Go through each category to ensure coverage of different types of incidents
    for category in ENHANCED_TEMPLATES:
        # Choose a random source and event type
        source = random.choice(list(EVENT_SOURCES.keys()))
        event_type = random.choice(EVENT_SOURCES[source])

        logger.info(f"Creating incident for {category} to test resolution generation")

        incident = await generate_enhanced_event(
            source=source, event_type=event_type, category=category
        )

        await send_event_to_api(session, incident)
        await asyncio.sleep(0.5)

    logger.info("Resolution generation test complete")


async def generate_and_send_events(
    num_events: int = 20,
    specific_test: Optional[str] = None,
    similar_group: Optional[int] = None,
    historical_days: int = 60,
) -> None:
    """
    Generate and send enhanced events to test various system features.

    Args:
        num_events: Number of events to generate if running random tests
        specific_test: Specific test to run (detection, severity, similarity, resolution)
        similar_group: Specific similar incident group to create
        historical_days: Maximum days in the past to generate events
    """
    # Initialize OpenAI client (for potential future use)
    await init_openai_client()

    async with aiohttp.ClientSession() as session:
        if specific_test == "detection":
            await test_incident_detection(session)
        elif specific_test == "severity":
            await test_severity_classification(session)
        elif specific_test == "similarity":
            if similar_group is not None:
                await create_similar_incident_group(session, similar_group)
            else:
                await test_incident_similarity(session)
        elif specific_test == "resolution":
            await test_resolution_generation(session)
        elif specific_test == "all":
            # Run all tests in sequence
            await test_incident_detection(session)
            await asyncio.sleep(1)

            await test_severity_classification(session)
            await asyncio.sleep(1)

            await test_incident_similarity(session)
            await asyncio.sleep(1)

            await test_resolution_generation(session)
        else:
            # Generate random incidents to test general functionality
            logger.info(
                f"Generating {num_events} random incidents over the past {historical_days} days..."
            )

            for _ in range(num_events):
                # Randomly select source, event type and category
                source = random.choice(list(EVENT_SOURCES.keys()))
                event_type = random.choice(EVENT_SOURCES[source])
                category = random.choice(list(ENHANCED_TEMPLATES.keys()))

                # Generate days ago (weighted towards more recent events)
                days_ago = random.randint(0, historical_days)

                # Generate the incident
                incident = await generate_enhanced_event(
                    source=source,
                    event_type=event_type,
                    category=category,
                    days_ago=days_ago,
                )

                # Send to API
                await send_event_to_api(session, incident)

                # Sleep briefly to avoid overwhelming the API
                await asyncio.sleep(0.2)


def main():
    """Command line interface for generating enhanced test events"""
    global BASE_URL

    parser = argparse.ArgumentParser(
        description="Generate enhanced test events for System Guardian"
    )
    parser.add_argument(
        "--url", type=str, help=f"Base URL of the API (default: {BASE_URL})"
    )
    parser.add_argument(
        "--events", type=int, default=20, help="Number of random events to generate"
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
        choices=["detection", "severity", "similarity", "resolution", "all"],
        help="Specific test to run",
    )
    parser.add_argument(
        "--similar-group",
        type=int,
        help="Specific similar incident group to create (0-2)",
    )

    args = parser.parse_args()

    # Update base URL if provided
    if args.url:
        BASE_URL = args.url

    logger.info(f"Using API URL: {BASE_URL}")

    if args.test:
        logger.info(f"Running specific test: {args.test}")
    elif args.similar_group is not None:
        logger.info(f"Creating similar incident group: {args.similar_group}")
    else:
        logger.info(
            f"Generating {args.events} random incidents spanning {args.days} days"
        )

    # Run the async function
    asyncio.run(
        generate_and_send_events(
            num_events=args.events,
            specific_test=args.test,
            similar_group=args.similar_group,
            historical_days=args.days,
        )
    )

    logger.info("Event generation complete")


if __name__ == "__main__":
    main()

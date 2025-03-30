#!/usr/bin/env python
"""
Fake event generator for System Guardian.

This script generates fake events from various sources and sends them to the system
through the API. It's useful for testing and populating the system with historical data.
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
    # "datadog": ["alert", "metric_alert", "service_check", "event", "monitor"],
}

# Severity levels and their weights
SEVERITY_LEVELS = {
    "critical": 5,
    "high": 15,
    "medium": 30,
    "low": 50,
}

# Common issues and their templates by source
ISSUE_TEMPLATES = {
    "github": {
        "issue": [
            {
                "title": "CRITICAL: API endpoint returning 500 error",
                "description": "The /api/users endpoint is consistently returning 500 errors when accessed with specific parameters. This is BROKEN and causing service failures.",
                "keywords": [
                    "error",
                    "api",
                    "500",
                    "endpoint",
                    "critical",
                    "broken",
                    "failure",
                ],
                "resolution": "Fixed the error by adding proper error handling for null user objects in the response processor.",
            },
            {
                "title": "URGENT: Memory leak in background worker causing CRASH",
                "description": "The background worker process is consuming increasing amounts of memory over time, eventually crashing. This is an URGENT issue affecting production.",
                "keywords": ["memory", "leak", "crash", "worker", "urgent"],
                "resolution": "Identified a circular reference in the object cache. Implemented weak references to break the cycle.",
            },
            {
                "title": "Build FAILURE with Node 18",
                "description": "Our build process is failing when using Node 18 but works fine on Node 16. The BROKEN process is preventing deployments.",
                "keywords": ["build", "failure", "node", "failing", "broken"],
                "resolution": "Updated webpack configuration to be compatible with the latest Node version.",
            },
            {
                "title": "CRITICAL BUG: Authentication service down",
                "description": "Users are unable to log in due to authentication service failures. This is causing an OUTAGE in the platform.",
                "keywords": ["bug", "critical", "authentication", "down", "outage"],
                "resolution": "Fixed connection pooling issues in the authentication service and added circuit breakers.",
            },
            {
                "title": "EMERGENCY: Database connection failures",
                "description": "The application is experiencing intermittent database connection failures. URGENT attention required.",
                "keywords": [
                    "database",
                    "failure",
                    "connection",
                    "emergency",
                    "urgent",
                ],
                "resolution": "Increased connection timeout settings and implemented retry mechanism with exponential backoff.",
            },
        ],
        "pull_request": [
            {
                "title": "HOTFIX: Database connection pooling",
                "description": "URGENT fix for connection pool settings to prevent exhaustion during peak load.",
                "keywords": [
                    "database",
                    "connection",
                    "pool",
                    "fix",
                    "hotfix",
                    "urgent",
                ],
                "resolution": "Increased max connections and implemented better release logic. Added monitoring.",
            },
            {
                "title": "EMERGENCY FIX: Optimize image processing pipeline",
                "description": "Refactored the image processing to use worker threads and reduce memory usage. Addresses CRITICAL performance issues.",
                "keywords": [
                    "optimize",
                    "image",
                    "processing",
                    "memory",
                    "emergency",
                    "fix",
                    "critical",
                ],
                "resolution": "Implemented streaming processing and reduced memory copies. 30% performance improvement.",
            },
            {
                "title": "URGENT HOTFIX: API rate limiting bypass",
                "description": "Fixed a security issue allowing API rate limiting to be bypassed. This is an EMERGENCY fix for a CRITICAL vulnerability.",
                "keywords": ["security", "api", "fix", "hotfix", "urgent", "emergency"],
                "resolution": "Implemented proper validation of rate limiting tokens and added additional security checks.",
            },
        ],
    },
    "jira": {
        "issue_created": [
            {
                "title": "BLOCKER: Production database latency spikes",
                "description": "Users reporting intermittent latency spikes in API responses. Database queries showing occasional 5+ second execution times. This is a CRITICAL issue affecting production.",
                "keywords": [
                    "database",
                    "latency",
                    "production",
                    "blocker",
                    "critical",
                ],
                "resolution": "Added missing index on frequently queried column and optimized slow query.",
            },
            {
                "title": "CRITICAL INCIDENT: Authentication service intermittently rejecting valid credentials",
                "description": "About 5% of login attempts are being rejected despite correct credentials being provided. This is a BLOCKER for affected users.",
                "keywords": [
                    "authentication",
                    "login",
                    "intermittent",
                    "credentials",
                    "critical",
                    "blocker",
                    "incident",
                ],
                "resolution": "Fixed race condition in token validation process. Added retry mechanism with exponential backoff.",
            },
            {
                "title": "Service OUTAGE: API Gateway DOWN",
                "description": "The API gateway is experiencing intermittent downtime, causing BROKEN service connections. This is a CRITICAL PROBLEM affecting all client applications.",
                "keywords": ["outage", "down", "api", "broken", "critical", "problem"],
                "resolution": "Identified network partition issue between availability zones. Implemented automated failover mechanism.",
            },
            {
                "title": "INCIDENT: Memory leak causing application crashes",
                "description": "The web application server is experiencing memory leaks leading to crashes. This is a BLOCKER BUG affecting production environments.",
                "keywords": ["memory", "leak", "crash", "incident", "blocker", "bug"],
                "resolution": "Fixed memory leak in connection handling code and added memory monitoring alerts.",
            },
            {
                "title": "Monitor system has critical error",
                "description": "The monitoring system is reporting critical errors and is unable to collect metrics from production services. This is causing BLIND SPOTS in our observability.",
                "keywords": ["monitor", "system", "critical", "error", "production"],
                "resolution": "Restarted monitoring agents and fixed configuration issues in the central collector.",
            },
            {
                "title": "URGENT: Database replication lag exceeding thresholds",
                "description": "Database replication lag has increased beyond acceptable thresholds. This is causing data inconsistency between primary and replica databases.",
                "keywords": [
                    "database",
                    "replication",
                    "lag",
                    "urgent",
                    "inconsistency",
                ],
                "resolution": "Optimized replication process and added additional replica capacity to handle the load.",
            },
            {
                "title": "CRITICAL: Kubernetes nodes DOWN in production",
                "description": "Multiple Kubernetes nodes in the production cluster are showing as NotReady. This is causing service OUTAGES and pod evictions.",
                "keywords": [
                    "kubernetes",
                    "nodes",
                    "down",
                    "production",
                    "critical",
                    "outage",
                ],
                "resolution": "Identified underlying storage issue causing node failures. Applied emergency patch and rebalanced the cluster.",
            },
            {
                "title": "EMERGENCY: Security breach detected in auth service",
                "description": "Unusual login patterns detected suggesting a potential security BREACH in the authentication service. This is a CRITICAL security issue.",
                "keywords": ["security", "breach", "auth", "emergency", "critical"],
                "resolution": "Blocked suspicious IP ranges, reset affected credentials, and patched the vulnerability in the authentication flow.",
            },
        ],
        "issue_resolved": [
            {
                "title": "RESOLVED: CDN cache not being invalidated properly",
                "description": "Updated assets are not showing up for users because the CDN cache is not being properly invalidated. This BROKEN functionality was causing outdated content to be served.",
                "keywords": ["cdn", "cache", "invalidation", "assets", "broken"],
                "resolution": "Fixed incorrect cache key generation and implemented purge-all functionality for critical updates.",
            },
            {
                "title": "FIXED: CRITICAL database performance PROBLEM",
                "description": "Slow query performance was causing an OUTAGE in the reporting system. This INCIDENT has been resolved.",
                "keywords": [
                    "database",
                    "performance",
                    "critical",
                    "problem",
                    "outage",
                    "incident",
                ],
                "resolution": "Optimized database queries and added appropriate indexes. Implemented query caching where appropriate.",
            },
        ],
    },
    "datadog": {
        "alert": [
            {
                "title": "CRITICAL: High CPU usage on API servers",
                "description": "CPU usage above 90% on multiple API server instances for over 15 minutes. URGENT attention required as this is causing service OUTAGE.",
                "keywords": ["cpu", "api", "server", "critical", "urgent", "outage"],
                "resolution": "Identified runaway query causing excessive CPU load. Added query timeout and optimized the problematic endpoint.",
            },
            {
                "title": "INCIDENT: Elevated error rate in payment processing",
                "description": "Error rate in payment processing service increased to 15% (threshold: 5%). This is a CRITICAL FAILURE affecting customer transactions.",
                "keywords": [
                    "error",
                    "payment",
                    "processing",
                    "rate",
                    "incident",
                    "critical",
                    "failure",
                ],
                "resolution": "External payment gateway was experiencing downtime. Implemented circuit breaker and fallback to secondary provider.",
            },
            {
                "title": "OUTAGE: Database connection pool exhausted",
                "description": "All available database connections are in use, causing application FAILURES. This is a CRITICAL EMERGENCY situation.",
                "keywords": [
                    "database",
                    "connection",
                    "outage",
                    "failures",
                    "critical",
                    "emergency",
                ],
                "resolution": "Increased connection pool size and implemented better connection lifecycle management.",
            },
        ],
        "metric_alert": [
            {
                "title": "URGENT: Memory usage threshold exceeded",
                "description": "Memory usage on database servers exceeded 85% for more than 10 minutes. This is a potential CRITICAL issue that could lead to OUTAGE.",
                "keywords": [
                    "memory",
                    "database",
                    "threshold",
                    "exceeded",
                    "urgent",
                    "critical",
                    "outage",
                ],
                "resolution": "Identified memory leak in query cache. Fixed cache eviction policy and implemented better memory monitoring.",
            },
            {
                "title": "BROKEN: API response time above threshold",
                "description": "API response times exceeding 2 seconds for more than 5 minutes. This CRITICAL performance PROBLEM is affecting user experience.",
                "keywords": ["api", "response", "broken", "critical", "problem"],
                "resolution": "Identified slow database queries and added missing indexes. Implemented application-level caching for frequently accessed data.",
            },
        ],
    },
}

# Status options for different event types
STATUS_OPTIONS = {
    "github": {
        "issue": ["open", "closed"],
        "pull_request": ["open", "closed", "merged"],
    },
    "jira": {
        "issue_created": ["open", "in progress"],
        "issue_updated": ["in progress", "blocked", "in review"],
        "issue_resolved": ["resolved", "closed", "done"],
    },
}


# Added a function to get keywords from config/incident_rules.json
def get_source_keywords(source: str) -> List[str]:
    """
    Get source-specific keywords from configuration.

    :param source: The source to get keywords for (github, jira, etc.)
    :type source: str

    :return: A list of keywords for the specified source
    :rtype: List[str]
    """
    # Build dictionary containing all possible keywords
    # These keywords come from config/incident_rules.json
    keywords = {
        "github": {
            "issue": ["critical", "urgent", "broken", "crash", "failure"],
            "pull_request": ["fix", "hotfix", "urgent", "emergency"],
            "global": ["crash", "bug", "error", "broken"],
        },
        "jira": {
            "issue": ["blocker", "critical", "outage", "down", "broken"],
            "global": ["bug", "incident", "problem"],
        },
        "global": ["critical", "urgent", "emergency", "outage", "down"],
    }

    # Get keywords for the specified source
    source_keywords = []

    # Add global keywords
    source_keywords.extend(keywords.get("global", []))

    # Add source-specific global keywords
    if source in keywords:
        source_keywords.extend(keywords[source].get("global", []))

        # Add all event type keywords for the source
        for event_type, type_keywords in keywords[source].items():
            if event_type != "global":
                source_keywords.extend(type_keywords)

    # Return deduplicated keyword list
    return list(set(source_keywords))


async def init_openai_client():
    """Initialize the OpenAI client."""
    global openai_client
    import os

    api_key = settings.openai_api_key
    if not api_key:
        logger.warning(
            "OpenAI API key not found. Falling back to template-based generation."
        )
        return False

    try:
        openai_client = AsyncOpenAI(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return False


async def generate_event_with_llm(
    source: str, event_type: str, severity: str = None
) -> Dict[str, Any]:
    """
    Generate event content using OpenAI LLM.

    :param source: Event source (github, jira, etc.)
    :param event_type: Event type
    :param severity: Optional severity level
    :return: Event template with title, description, keywords and resolution
    """
    if not openai_client:
        logger.info("OpenAI client not initialized. Using templates instead.")
        # Fall back to template-based generation
        return random.choice(
            ISSUE_TEMPLATES.get(source, {}).get(
                event_type,
                [
                    {
                        "title": f"Default {source} {event_type} event",
                        "description": "Default description",
                        "keywords": ["default"],
                    }
                ],
            )
        )

    # Determine severity if not provided
    if not severity:
        severity = random.choices(
            list(SEVERITY_LEVELS.keys()),
            weights=list(SEVERITY_LEVELS.values()),
        )[0]

    try:
        system_prompt = """
        You are a generator of realistic IT incident and event data. 
        Generate a plausible incident/event based on the source system, event type, and severity provided.
        Your response should be in JSON format with the following fields:
        - title: A descriptive title for the event
        - description: A detailed description of what happened
        - keywords: A list of relevant keywords for this event
        - resolution: Possible resolution steps if this were a real incident
        
        Make the event realistic and relevant to the source system and event type.
        For high or critical severity, include appropriate urgency in the language.
        """

        user_prompt = f"""
        Create a realistic {severity} severity {source} {event_type} event.
        
        For reference:
        - Source: {source} (like GitHub, Jira, Datadog, Slack)
        - Event Type: {event_type} (like issue, alert, message)
        - Severity: {severity} (low, medium, high, critical)
        
        For critical or high severity events, use words like CRITICAL, URGENT, OUTAGE, BROKEN, EMERGENCY, etc.
        For medium or low severity, use more moderate language.
        """

        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        content = json.loads(response.choices[0].message.content)

        # Ensure all required fields are present
        content.setdefault("title", f"Default {source} {event_type} title")
        content.setdefault("description", f"Default {source} {event_type} description")
        content.setdefault("keywords", ["default"])
        content.setdefault("resolution", "No resolution provided")

        logger.info(f"Generated event with LLM: {content['title']}")
        return content

    except Exception as e:
        logger.error(f"Error generating event with LLM: {e}")
        # Fall back to template-based generation
        return random.choice(
            ISSUE_TEMPLATES.get(source, {}).get(
                event_type,
                [
                    {
                        "title": f"Default {source} {event_type} event",
                        "description": "Default description",
                        "keywords": ["default"],
                    }
                ],
            )
        )


async def generate_fake_event(
    source: str,
    event_type: str,
    created_days_ago: int = None,
    severity: str = None,
    status: str = None,
    resolved: bool = None,
) -> Dict[str, Any]:
    """
    Generate a fake event.

    :param source: Event source (github, jira, etc.)
    :param event_type: Event type
    :param created_days_ago: Days ago the event was created
    :param severity: Severity level
    :param status: Status of the event
    :param resolved: Whether the event is resolved

    :return: Event data
    """
    # Generate timestamps
    if created_days_ago is None:
        created_days_ago = random.randint(0, 60)  # Up to 60 days ago

    timestamp = datetime.datetime.utcnow() - datetime.timedelta(days=created_days_ago)
    created_at = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Determine severity if not provided
    if severity is None:
        severity = random.choices(
            list(SEVERITY_LEVELS.keys()),
            weights=list(SEVERITY_LEVELS.values()),
        )[0]

    # Determine resolved status if not provided (more likely for older events)
    if resolved is None:
        resolution_probability = min(created_days_ago / 30.0, 0.9)  # 90% max chance
        resolved = random.random() < resolution_probability

    # Get a template using LLM or fallback to predefined templates
    template = await generate_event_with_llm(source, event_type, severity)

    # Determine status if not provided
    if status is None:
        if source == "github" and event_type in STATUS_OPTIONS.get("github", {}):
            status = random.choice(STATUS_OPTIONS["github"][event_type])
        elif source == "jira" and event_type in STATUS_OPTIONS.get("jira", {}):
            status = random.choice(STATUS_OPTIONS["jira"][event_type])
        elif resolved:
            status = "closed" if source == "github" else "resolved"
        else:
            status = "open" if source == "github" else "in progress"

    # Create raw payload
    raw_payload = generate_raw_payload(source, event_type, template, status, resolved)

    # Create the standardized event format
    return {
        "source": source,
        "event_type": event_type,
        "timestamp": created_at,
        "raw_payload": raw_payload,
    }


def generate_raw_payload(
    source: str, event_type: str, template: Dict[str, Any], status: str, resolved: bool
) -> Dict[str, Any]:
    """
    Generate a raw event payload similar to what would be received from the event source.

    :param source: The source of the event
    :type source: str
    :param event_type: The type of event
    :type event_type: str
    :param template: Template containing title, description, etc.
    :type template: Dict[str, Any]
    :param status: Current status of the event
    :type status: str
    :param resolved: Whether the event is resolved
    :type resolved: bool

    :return: A dictionary containing the raw event data
    :rtype: Dict[str, Any]
    """
    # Basic information
    title = template["title"]
    description = template["description"]
    keywords = template.get("keywords", [])
    severity = template.get("severity", "medium")

    # Get source-specific keywords and ensure some are included in the description
    source_keywords = get_source_keywords(source)

    # Basic payload
    payload = {}

    if source == "github":
        action = "opened"
        if status == "closed":
            action = "closed"
        elif "edited" in event_type:
            action = "edited"

        # Add more specific actions for GitHub event types
        if "issue" in event_type:
            # Consider create_incident_on keywords from config to increase trigger likelihood
            possible_actions = ["opened", "edited"]
            if (
                random.random() < 0.7
            ):  # 70% chance to use actions that may trigger events
                action = random.choice(possible_actions)

        # Ensure description has enough keyword information
        if (
            source_keywords and random.random() < 0.8
        ):  # 80% chance to strengthen keywords
            # Add 1-2 random keywords to the description (if not already present)
            for _ in range(random.randint(1, 2)):
                keyword = random.choice(source_keywords)
                if keyword.upper() not in description:
                    description += f" This issue is {keyword.upper()} and requires immediate attention."

        # According to event type, generate different payload structures
        if event_type == "push":
            # Generate a random commit SHA
            commit_sha = "".join(random.choice("0123456789abcdef") for _ in range(40))
            previous_sha = "".join(random.choice("0123456789abcdef") for _ in range(40))

            # Create submitter and author information
            author = {
                "name": random.choice(
                    ["jordanchenml", "dev_user", "admin_user", "test_user"]
                ),
                "email": f"{random.choice(['jordan', 'dev', 'admin', 'test'])}@example.com",
                "username": random.choice(
                    ["jordanchenml", "dev_user", "admin_user", "test_user"]
                ),
            }

            # Create commit list
            commits = [
                {
                    "id": commit_sha,
                    "url": f"https://github.com/jordanchenml/system_guardian/commit/{commit_sha}",
                    "added": [
                        f"system_guardian/{random.choice(['services', 'web', 'db'])}/{random.choice(['file1.py', 'file2.py', 'file3.py'])}"
                    ],
                    "author": author,
                    "message": title,
                    "removed": [],
                    "tree_id": "".join(
                        random.choice("0123456789abcdef") for _ in range(40)
                    ),
                    "distinct": True,
                    "modified": [
                        f"system_guardian/{random.choice(['services', 'web', 'db'])}/{random.choice(['existing1.py', 'existing2.py', 'existing3.py'])}"
                    ],
                    "committer": author,
                    "timestamp": datetime.datetime.now().strftime(
                        "%Y-%m-%dT%H:%M:%S%z"
                    ),
                }
            ]

            # Create repository information
            repository = {
                "id": random.randint(900000000, 999999999),
                "url": "https://github.com/jordanchenml/system_guardian",
                "fork": False,
                "name": "system_guardian",
                "size": random.randint(100, 500),
                "forks": random.randint(0, 10),
                "owner": {
                    "id": random.randint(10000000, 99999999),
                    "url": "https://api.github.com/users/jordanchenml",
                    "name": "jordanchenml",
                    "type": "User",
                    "email": "jordanchenml@example.com",
                    "login": "jordanchenml",
                    "node_id": "MDQ6VXNlcjQ5NzAzMDcx",
                    "html_url": "https://github.com/jordanchenml",
                    "gists_url": "https://api.github.com/users/jordanchenml/gists{/gist_id}",
                    "repos_url": "https://api.github.com/users/jordanchenml/repos",
                    "avatar_url": "https://avatars.githubusercontent.com/u/49703071?v=4",
                    "events_url": "https://api.github.com/users/jordanchenml/events{/privacy}",
                    "site_admin": False,
                    "gravatar_id": "",
                    "starred_url": "https://api.github.com/users/jordanchenml/starred{/owner}{/repo}",
                    "followers_url": "https://api.github.com/users/jordanchenml/followers",
                    "following_url": "https://api.github.com/users/jordanchenml/following{/other_user}",
                    "organizations_url": "https://api.github.com/users/jordanchenml/orgs",
                    "subscriptions_url": "https://api.github.com/users/jordanchenml/subscriptions",
                    "received_events_url": "https://api.github.com/users/jordanchenml/received_events",
                },
                "topics": [],
                "private": False,
                "html_url": "https://github.com/jordanchenml/system_guardian",
                "language": "Python",
                "watchers": random.randint(0, 10),
                "full_name": "jordanchenml/system_guardian",
                "description": "An AI-powered incident management platform designed to autonomously monitor, analyze, and suggest resolutions for on-call incidents.",
            }

            # Complete push event payload
            payload = {
                "ref": f"refs/heads/{random.choice(['main', 'dev', 'feature', 'hotfix'])}",
                "after": commit_sha,
                "before": previous_sha,
                "forced": random.choice([True, False]),
                "pusher": {"name": author["name"], "email": author["email"]},
                "sender": {
                    "id": repository["owner"]["id"],
                    "url": repository["owner"]["url"],
                    "type": repository["owner"]["type"],
                    "login": repository["owner"]["login"],
                    "node_id": repository["owner"]["node_id"],
                    "html_url": repository["owner"]["html_url"],
                    "gists_url": repository["owner"]["gists_url"],
                    "repos_url": repository["owner"]["repos_url"],
                    "avatar_url": repository["owner"]["avatar_url"],
                    "events_url": repository["owner"]["events_url"],
                    "site_admin": repository["owner"]["site_admin"],
                    "gravatar_id": repository["owner"]["gravatar_id"],
                    "starred_url": repository["owner"]["starred_url"],
                    "followers_url": repository["owner"]["followers_url"],
                    "following_url": repository["owner"]["following_url"],
                },
                "commits": commits,
                "compare": f"https://github.com/jordanchenml/system_guardian/compare/{previous_sha[0:7]}...{commit_sha[0:7]}",
                "created": False,
                "deleted": False,
                "base_ref": None,
                "repository": repository,
                "head_commit": commits[0],
            }
        elif "issue" in event_type:
            # Generate issue event payload
            issue_number = random.randint(1, 100)
            issue_id = random.randint(100000, 999999)
            repository_owner = {
                "id": random.randint(10000000, 99999999),
                "url": "https://api.github.com/users/jordanchenml",
                "type": "User",
                "login": "jordanchenml",
                "node_id": "MDQ6VXNlcjQ5NzAzMDcx",
                "html_url": "https://github.com/jordanchenml",
                "gists_url": "https://api.github.com/users/jordanchenml/gists{/gist_id}",
                "repos_url": "https://api.github.com/users/jordanchenml/repos",
                "avatar_url": "https://avatars.githubusercontent.com/u/49703071?v=4",
                "events_url": "https://api.github.com/users/jordanchenml/events{/privacy}",
                "site_admin": False,
                "gravatar_id": "",
                "starred_url": "https://api.github.com/users/jordanchenml/starred{/owner}{/repo}",
                "followers_url": "https://api.github.com/users/jordanchenml/followers",
                "following_url": "https://api.github.com/users/jordanchenml/following{/other_user}",
            }

            # Create issue information
            issue = {
                "id": issue_id,
                "url": f"https://api.github.com/repos/jordanchenml/system_guardian/issues/{issue_number}",
                "node_id": f"MDExOlB1bGxSZXF1ZXN0{random.randint(100000, 999999)}",
                "number": issue_number,
                "title": title,
                "user": repository_owner,
                "labels": [],
                "state": status,
                "locked": False,
                "assignee": None,
                "milestone": None,
                "comments": random.randint(0, 10),
                "created_at": (
                    datetime.datetime.now()
                    - datetime.timedelta(days=random.randint(1, 30))
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "updated_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "closed_at": (
                    datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                    if status == "closed"
                    else None
                ),
                "body": description,
                "timeline_url": f"https://api.github.com/repos/jordanchenml/system_guardian/issues/{issue_number}/timeline",
                "repository_url": "https://api.github.com/repos/jordanchenml/system_guardian",
                "html_url": f"https://github.com/jordanchenml/system_guardian/issues/{issue_number}",
            }

            # Add labels to issue
            label_names = [severity]
            for keyword in keywords:
                if random.random() < 0.6:  # 60% chance to add keyword label
                    label_names.append(keyword)

            for label_name in label_names:
                label_color = "".join(
                    random.choice("0123456789abcdef") for _ in range(6)
                )
                issue["labels"].append(
                    {
                        "id": random.randint(1000000, 9999999),
                        "url": f"https://api.github.com/repos/jordanchenml/system_guardian/labels/{label_name}",
                        "name": label_name,
                        "color": label_color,
                        "default": False,
                        "node_id": f"MDU6TGFiZWx{random.randint(1000000, 9999999)}",
                    }
                )

            # Complete issue event payload
            payload = {
                "action": action,
                "issue": issue,
                "repository": {
                    "id": random.randint(900000000, 999999999),
                    "url": "https://api.github.com/repos/jordanchenml/system_guardian",
                    "name": "system_guardian",
                    "full_name": "jordanchenml/system_guardian",
                    "owner": repository_owner,
                    "private": False,
                    "html_url": "https://github.com/jordanchenml/system_guardian",
                    "description": "An AI-powered incident management platform designed to autonomously monitor, analyze, and suggest resolutions for on-call incidents.",
                },
                "sender": repository_owner,
            }
        elif "pull_request" in event_type:
            # Generate pull request event payload
            pr_number = random.randint(1, 100)
            pr_id = random.randint(100000, 999999)
            repository_owner = {
                "id": random.randint(10000000, 99999999),
                "url": "https://api.github.com/users/jordanchenml",
                "type": "User",
                "login": "jordanchenml",
                "node_id": "MDQ6VXNlcjQ5NzAzMDcx",
                "html_url": "https://github.com/jordanchenml",
                "gists_url": "https://api.github.com/users/jordanchenml/gists{/gist_id}",
                "repos_url": "https://api.github.com/users/jordanchenml/repos",
                "avatar_url": "https://avatars.githubusercontent.com/u/49703071?v=4",
                "events_url": "https://api.github.com/users/jordanchenml/events{/privacy}",
                "site_admin": False,
                "gravatar_id": "",
                "starred_url": "https://api.github.com/users/jordanchenml/starred{/owner}{/repo}",
                "followers_url": "https://api.github.com/users/jordanchenml/followers",
                "following_url": "https://api.github.com/users/jordanchenml/following{/other_user}",
            }

            # Create PR information
            merged = status == "merged" or (
                status == "closed" and random.random() < 0.8
            )
            pull_request = {
                "id": pr_id,
                "url": f"https://api.github.com/repos/jordanchenml/system_guardian/pulls/{pr_number}",
                "node_id": f"MDExOlB1bGxSZXF1ZXN0{random.randint(100000, 999999)}",
                "number": pr_number,
                "title": title,
                "user": repository_owner,
                "body": description,
                "labels": [],
                "state": "closed" if merged else status,
                "locked": False,
                "assignee": None,
                "milestone": None,
                "comments": random.randint(0, 10),
                "created_at": (
                    datetime.datetime.now()
                    - datetime.timedelta(days=random.randint(1, 30))
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "updated_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "closed_at": (
                    datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                    if status in ["closed", "merged"]
                    else None
                ),
                "merged_at": (
                    datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                    if merged
                    else None
                ),
                "merge_commit_sha": (
                    "".join(random.choice("0123456789abcdef") for _ in range(40))
                    if merged
                    else None
                ),
                "head": {
                    "ref": random.choice(
                        ["feature-branch", "fix-branch", "improvement-branch"]
                    ),
                    "sha": "".join(
                        random.choice("0123456789abcdef") for _ in range(40)
                    ),
                    "repo": {
                        "id": random.randint(900000000, 999999999),
                        "url": "https://api.github.com/repos/jordanchenml/system_guardian",
                        "name": "system_guardian",
                        "full_name": "jordanchenml/system_guardian",
                    },
                },
                "base": {
                    "ref": "main",
                    "sha": "".join(
                        random.choice("0123456789abcdef") for _ in range(40)
                    ),
                    "repo": {
                        "id": random.randint(900000000, 999999999),
                        "url": "https://api.github.com/repos/jordanchenml/system_guardian",
                        "name": "system_guardian",
                        "full_name": "jordanchenml/system_guardian",
                    },
                },
                "merged": merged,
                "mergeable": not merged and random.choice([True, False, None]),
                "html_url": f"https://github.com/jordanchenml/system_guardian/pull/{pr_number}",
            }

            # Add labels to PR
            label_names = [severity]
            for keyword in keywords:
                if (
                    keyword in ["fix", "hotfix", "urgent", "emergency"]
                    and random.random() < 0.7
                ):  # 70% chance to add keyword label
                    label_names.append(keyword)

            for label_name in label_names:
                label_color = "".join(
                    random.choice("0123456789abcdef") for _ in range(6)
                )
                pull_request["labels"].append(
                    {
                        "id": random.randint(1000000, 9999999),
                        "url": f"https://api.github.com/repos/jordanchenml/system_guardian/labels/{label_name}",
                        "name": label_name,
                        "color": label_color,
                        "default": False,
                        "node_id": f"MDU6TGFiZWx{random.randint(1000000, 9999999)}",
                    }
                )

            # Complete pull request event payload
            payload = {
                "action": action,
                "pull_request": pull_request,
                "repository": {
                    "id": random.randint(900000000, 999999999),
                    "url": "https://api.github.com/repos/jordanchenml/system_guardian",
                    "name": "system_guardian",
                    "full_name": "jordanchenml/system_guardian",
                    "owner": repository_owner,
                    "private": False,
                    "html_url": "https://github.com/jordanchenml/system_guardian",
                    "description": "An AI-powered incident management platform designed to autonomously monitor, analyze, and suggest resolutions for on-call incidents.",
                },
                "sender": repository_owner,
                "number": pr_number,
            }
        else:
            # Generic payload structure for other GitHub events
            payload = {
                "action": action,
                "repository": {
                    "id": random.randint(900000000, 999999999),
                    "url": "https://api.github.com/repos/jordanchenml/system_guardian",
                    "name": "system_guardian",
                    "full_name": "jordanchenml/system_guardian",
                    "owner": {
                        "id": random.randint(10000000, 99999999),
                        "login": "jordanchenml",
                        "type": "User",
                    },
                    "private": False,
                    "html_url": "https://github.com/jordanchenml/system_guardian",
                },
                "sender": {
                    "id": random.randint(10000000, 99999999),
                    "login": "jordanchenml",
                    "type": "User",
                },
            }

            # Add specific fields based on event type
            if "workflow_run" in event_type:
                payload["workflow_run"] = {
                    "id": random.randint(10000000, 99999999),
                    "name": "CI/CD Pipeline",
                    "status": random.choice(["completed", "in_progress", "queued"]),
                    "conclusion": random.choice(
                        ["success", "failure", "cancelled", None]
                    ),
                }
            elif "release" in event_type:
                payload["release"] = {
                    "id": random.randint(10000000, 99999999),
                    "tag_name": f"v{random.randint(0, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                    "name": f"Release {random.randint(0, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                    "body": description,
                    "draft": False,
                    "prerelease": False,
                }

        return payload

    elif source == "jira":
        # According to create_incident_on in config, increase likelihood of triggering event actions
        jira_event_suffix = event_type.split("_")[-1]

        # Get Jira issue fields
        jira_description = description

        # Strengthen keyword presence
        if source_keywords and random.random() < 0.8:
            # Prioritize Jira priority-related keywords
            priority_keywords = ["blocker", "critical", "outage"]
            selected_keywords = [k for k in priority_keywords if k in source_keywords]

            if not selected_keywords:
                selected_keywords = random.sample(
                    source_keywords, min(2, len(source_keywords))
                )

            for keyword in selected_keywords:
                if keyword.upper() not in jira_description:
                    jira_description += (
                        f" This is marked as {keyword.upper()} priority."
                    )

        # Generate a unique issue ID and key
        issue_id = f"{random.randint(10000, 99999)}"
        issue_key = f"CCS-{random.randint(1, 100)}"

        # Create Jira webhook payload
        payload = {
            "user": {
                "self": f"https://example.atlassian.net/rest/api/2/user?accountId=user123456",
                "active": True,
                "timeZone": "Asia/Taipei",
                "accountId": f"user{random.randint(10000, 99999)}",
                "avatarUrls": {
                    "16x16": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                    "24x24": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                    "32x32": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                    "48x48": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                },
                "accountType": "atlassian",
                "displayName": random.choice(
                    [
                        "John Doe",
                        "Jane Smith",
                        "Alex Johnson",
                        "Sam Taylor",
                        "Terry Kim",
                    ]
                ),
            },
            "issue": {
                "id": issue_id,
                "key": issue_key,
                "self": f"https://example.atlassian.net/rest/api/2/{issue_id}",
                "fields": {
                    "votes": {
                        "self": f"https://example.atlassian.net/rest/api/2/issue/{issue_key}/votes",
                        "votes": 0,
                        "hasVoted": False,
                    },
                    "labels": [],
                    "status": {
                        "id": "10000",
                        "name": status.capitalize(),
                        "self": "https://example.atlassian.net/rest/api/2/status/10000",
                        "iconUrl": "https://example.atlassian.net/",
                        "description": "",
                        "statusCategory": {
                            "id": 2,
                            "key": "new",
                            "name": "New",
                            "self": "https://example.atlassian.net/rest/api/2/statuscategory/2",
                            "colorName": "blue-gray",
                        },
                    },
                    "created": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[
                        :-3
                    ]
                    + "+0800",
                    "creator": {
                        "self": f"https://example.atlassian.net/rest/api/2/user?accountId=user123456",
                        "active": True,
                        "timeZone": "Asia/Taipei",
                        "accountId": f"user{random.randint(10000, 99999)}",
                        "avatarUrls": {
                            "16x16": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                            "24x24": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                            "32x32": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                            "48x48": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                        },
                        "accountType": "atlassian",
                        "displayName": random.choice(
                            [
                                "John Doe",
                                "Jane Smith",
                                "Alex Johnson",
                                "Sam Taylor",
                                "Terry Kim",
                            ]
                        ),
                    },
                    "duedate": None,
                    "project": {
                        "id": "10000",
                        "key": "CCS",
                        "name": "System Guardian",
                        "self": "https://example.atlassian.net/rest/api/2/project/10000",
                        "avatarUrls": {
                            "16x16": "https://example.atlassian.net/rest/api/2/universal_avatar/view/type/project/avatar/10423?size=xsmall",
                            "24x24": "https://example.atlassian.net/rest/api/2/universal_avatar/view/type/project/avatar/10423?size=small",
                            "32x32": "https://example.atlassian.net/rest/api/2/universal_avatar/view/type/project/avatar/10423?size=medium",
                            "48x48": "https://example.atlassian.net/rest/api/2/universal_avatar/view/type/project/avatar/10423",
                        },
                        "simplified": True,
                        "projectTypeKey": "software",
                    },
                    "summary": title,
                    "updated": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[
                        :-3
                    ]
                    + "+0800",
                    "watches": {
                        "self": f"https://example.atlassian.net/rest/api/2/issue/{issue_key}/watchers",
                        "isWatching": True,
                        "watchCount": 0,
                    },
                    "assignee": None,
                    "priority": {
                        "id": get_jira_priority_id(severity),
                        "name": severity_to_jira_priority(severity),
                        "self": f"https://example.atlassian.net/rest/api/2/priority/{get_jira_priority_id(severity)}",
                        "iconUrl": f"https://example.atlassian.net/images/icons/priorities/{severity.lower()}.svg",
                    },
                    "progress": {"total": 0, "progress": 0},
                    "reporter": {
                        "self": f"https://example.atlassian.net/rest/api/2/user?accountId=user123456",
                        "active": True,
                        "timeZone": "Asia/Taipei",
                        "accountId": f"user{random.randint(10000, 99999)}",
                        "avatarUrls": {
                            "16x16": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                            "24x24": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                            "32x32": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                            "48x48": "https://secure.gravatar.com/avatar/example?d=https%3A%2F%2Favatar.example.com",
                        },
                        "accountType": "atlassian",
                        "displayName": random.choice(
                            [
                                "John Doe",
                                "Jane Smith",
                                "Alex Johnson",
                                "Sam Taylor",
                                "Terry Kim",
                            ]
                        ),
                    },
                    "security": None,
                    "subtasks": [],
                    "versions": [],
                    "issuetype": {
                        "id": "10001",
                        "name": "Task",
                        "self": "https://example.atlassian.net/rest/api/2/issuetype/10001",
                        "iconUrl": "https://example.atlassian.net/rest/api/2/universal_avatar/view/type/issuetype/avatar/10318?size=medium",
                        "subtask": False,
                        "avatarId": 10318,
                        "entityId": str(uuid.uuid4()),
                        "description": "Tasks track small, distinct pieces of work.",
                        "hierarchyLevel": 0,
                    },
                    "timespent": None,
                    "workratio": -1,
                    "attachment": [],
                    "components": [],
                    "issuelinks": [],
                    "lastViewed": None,
                    "resolution": None,
                    "description": jira_description,
                    "environment": None,
                    "fixVersions": [],
                    "timeestimate": None,
                    "timetracking": {},
                    "resolutiondate": None,
                    "aggregateprogress": {"total": 0, "progress": 0},
                },
            },
            "changelog": {
                "id": f"{random.randint(10000, 99999)}",
                "items": [
                    {
                        "to": get_jira_priority_id(severity),
                        "from": None,
                        "field": "priority",
                        "fieldId": "priority",
                        "toString": severity_to_jira_priority(severity),
                        "fieldtype": "jira",
                        "fromString": None,
                    },
                    {
                        "to": f"user{random.randint(10000, 99999)}",
                        "from": None,
                        "field": "reporter",
                        "fieldId": "reporter",
                        "toString": random.choice(
                            [
                                "John Doe",
                                "Jane Smith",
                                "Alex Johnson",
                                "Sam Taylor",
                                "Terry Kim",
                            ]
                        ),
                        "fieldtype": "jira",
                        "fromString": None,
                        "tmpToAccountId": f"user{random.randint(10000, 99999)}",
                        "tmpFromAccountId": None,
                    },
                    {
                        "to": "10000",
                        "from": None,
                        "field": "Status",
                        "fieldId": "status",
                        "toString": status.capitalize(),
                        "fieldtype": "jira",
                        "fromString": None,
                    },
                    {
                        "to": None,
                        "from": None,
                        "field": "summary",
                        "fieldId": "summary",
                        "toString": title,
                        "fieldtype": "jira",
                        "fromString": None,
                    },
                ],
            },
            "timestamp": int(datetime.datetime.now().timestamp() * 1000),
            "webhookEvent": f"jira:{jira_event_suffix}",
            "issue_event_type_name": f"issue_{jira_event_suffix}",
        }

        # Add keywords as labels
        for keyword in source_keywords:
            if keyword in title.lower() or keyword in jira_description.lower():
                payload["issue"]["fields"]["labels"].append(keyword)

        return payload

    elif source == "datadog":
        # Strengthen keyword presence
        datadog_text = description
        if source_keywords and random.random() < 0.8:
            # Prioritize severity-related keywords
            severity_keywords = ["critical", "emergency", "outage", "down"]
            selected_keywords = [k for k in severity_keywords if k in source_keywords]

            if not selected_keywords:
                selected_keywords = random.sample(
                    source_keywords, min(2, len(source_keywords))
                )

            for keyword in selected_keywords:
                if keyword.upper() not in datadog_text:
                    datadog_text += f" Alert marked as {keyword.upper()}."

        # Create some related keyword-based labels
        tags = [f"key:{value}" for value in keywords]
        for keyword in source_keywords:
            if keyword in title.lower() or keyword in datadog_text.lower():
                tags.append(f"priority:{keyword}")

        payload.update(
            {
                "alert_type": event_type,
                "title": title,
                "text": datadog_text,
                "priority": severity,
                "tags": tags,
                "hostname": random.choice(
                    [
                        "app-server-01",
                        "db-server-02",
                        "api-server-03",
                        "worker-04",
                        "cache-05",
                    ]
                ),
                "status": "resolved" if resolved else "triggered",
            }
        )

    return payload


def severity_to_jira_priority(severity: str) -> str:
    """
    Convert a severity level to a Jira priority.

    :param severity: The severity level to convert
    :type severity: str

    :return: Corresponding Jira priority name
    :rtype: str
    """
    mapping = {
        "critical": "Highest",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }
    return mapping.get(severity, "Medium")


def get_jira_priority_id(severity: str) -> str:
    """
    Convert a severity level to a Jira priority ID.

    :param severity: Severity level (critical, high, medium, low)
    :type severity: str

    :return: Corresponding Jira priority ID string
    :rtype: str
    """
    mapping = {"critical": "1", "high": "2", "medium": "3", "low": "4"}
    return mapping.get(severity, "3")  # Default to Medium (3) if not found


async def send_event_to_api(
    session: aiohttp.ClientSession, event: Dict[str, Any]
) -> bool:
    """
    Send a generated event to the API.

    :param session: The aiohttp ClientSession
    :type session: aiohttp.ClientSession
    :param event: The event to send
    :type event: Dict[str, Any]

    :return: True if the event was successfully sent, False otherwise
    :rtype: bool
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
    else:
        # Handle other event sources as before
        api_payload = {
            "source": source,
            "event_type": event_type,
            "timestamp": event["timestamp"],
            "raw_payload": event["raw_payload"],
        }

    logger.debug(api_payload)

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

    try:
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


async def create_incident_directly(
    session: aiohttp.ClientSession, event: Dict[str, Any]
) -> Optional[int]:
    """
    Create an incident directly in the database through the API.

    :param session: The aiohttp ClientSession
    :type session: aiohttp.ClientSession
    :param event: The event data to use for creating the incident
    :type event: Dict[str, Any]

    :return: The ID of the created incident or None if creation failed
    :rtype: Optional[int]
    """
    # Prepare the payload for incident creation
    incident_payload = {
        "title": event["title"],
        "description": event["description"],
        "severity": event["severity"],
        "status": "resolved" if event["timestamp"] else "open",
        "source": event["source"],
        "created_at": event["timestamp"],
        "resolved_at": event["timestamp"] if event["timestamp"] else None,
    }

    # Endpoint for creating incidents
    endpoint = f"{BASE_URL}/api/incidents/"

    try:
        async with session.post(endpoint, json=incident_payload) as response:
            if response.status == 201:
                result = await response.json()
                return result.get("id")
            else:
                logger.error(f"Failed to create incident: {await response.text()}")
                return None
    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        return None


async def add_resolution_to_incident(
    session: aiohttp.ClientSession, incident_id: int, event: Dict[str, Any]
) -> bool:
    """
    Add resolution information to an incident.

    :param session: The aiohttp ClientSession
    :type session: aiohttp.ClientSession
    :param incident_id: The ID of the incident
    :type incident_id: int
    :param event: The event with resolution data
    :type event: Dict[str, Any]

    :return: True if the resolution was added successfully, False otherwise
    :rtype: bool
    """
    if not event.get("resolution"):
        return True  # No resolution to add

    # Prepare resolution payload
    resolution_payload = {
        "incident_id": incident_id,
        "resolution_text": event["resolution"],
        "resolved_by": random.choice(
            ["John Doe", "Jane Smith", "Alex Johnson", "Sam Taylor", "Terry Kim"]
        ),
        "resolved_at": event["timestamp"],
        "steps_taken": [
            "Investigation of logs and metrics",
            "Identified root cause",
            f"Applied fix: {event['resolution']}",
            "Verified solution resolved the issue",
        ],
    }

    # Endpoint for adding resolutions
    endpoint = f"{BASE_URL}/api/incidents/{incident_id}/resolutions/"

    try:
        async with session.post(endpoint, json=resolution_payload) as response:
            if response.status == 201:
                return True
            else:
                logger.error(f"Failed to add resolution: {await response.text()}")
                return False
    except Exception as e:
        logger.error(f"Error adding resolution: {e}")
        return False


async def generate_and_send_events(num_events: int, historical_days: int = 60) -> None:
    """
    Generate and send a batch of events.

    :param num_events: Number of events to generate
    :param historical_days: Max days in the past to generate events for
    """
    logger.info(f"Generating and sending {num_events} events...")

    # Initialize OpenAI client
    await init_openai_client()

    # Create a session for sending multiple requests
    async with aiohttp.ClientSession() as session:
        # Generate and send events
        for _ in range(num_events):
            # Randomly select a source and event type
            source = random.choice(list(EVENT_SOURCES.keys()))
            event_type = random.choice(EVENT_SOURCES[source])

            # More heavily weight older events to build historical data
            days_ago_weights = [1] * 10 + [0.5] * 20 + [0.3] * (historical_days - 30)
            created_days_ago = random.choices(
                range(historical_days),
                weights=days_ago_weights[:historical_days],
            )[0]

            # Generate the event
            event = await generate_fake_event(
                source=source,
                event_type=event_type,
                created_days_ago=created_days_ago,
            )

            # Send to API
            success = await send_event_to_api(session, event)

            if not success:
                logger.warning(f"Failed to send {source} {event_type} event")

            # Sleep briefly to avoid overwhelming the API
            await asyncio.sleep(0.1)


def main():
    """
    Command line interface for generating fake events.

    :return: None
    :rtype: None
    """
    global BASE_URL

    parser = argparse.ArgumentParser(
        description="Generate fake events for System Guardian"
    )
    parser.add_argument(
        "--events", type=int, default=50, help="Number of events to generate"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Maximum days in the past to generate events",
    )
    parser.add_argument(
        "--url", type=str, help="Base URL of the API (default: http://localhost:8000)"
    )

    args = parser.parse_args()

    if args.url:
        BASE_URL = args.url

    logger.info(f"Generating {args.events} fake events spanning {args.days} days...")
    logger.info(f"API URL: {BASE_URL}")

    # Run the async function
    asyncio.run(generate_and_send_events(args.events, args.days))


if __name__ == "__main__":
    main()

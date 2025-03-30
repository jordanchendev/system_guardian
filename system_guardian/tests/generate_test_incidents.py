#!/usr/bin/env python
"""
Test incident generator for System Guardian.

This script generates test incidents that specifically test the following functionalities:
- Incident detection
- Incident similarity
- Resolution generation
- Severity classification
- Related incidents & insights
- Recommended fix steps
"""
import json
import random
import asyncio
import uuid
import datetime
import argparse
import aiohttp
import time
from typing import List, Dict, Any, Optional
from loguru import logger
from openai import AsyncOpenAI

from system_guardian.settings import settings

# Base URL of your API
BASE_URL = "http://localhost:5566"  # Change this to your API's base URL

# OpenAI client
openai_client = None

# Event sources and their event types for incident testing
EVENT_SOURCES = {
    "github": ["issue", "pull_request"],
    "jira": ["issue_created", "issue_updated", "issue_resolved"],
    "monitoring": ["alert", "metric_alert", "service_check"],
}

# Severity levels
SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

# Categories of incidents for testing
INCIDENT_CATEGORIES = [
    "database_issues",
    "network_problems",
    "application_errors",
    "security_incidents",
    "infrastructure_failures",
    "performance_issues",
    "kubernetes_problems",
    "cloud_service_disruptions",
]

# Test scenarios for each category
TEST_SCENARIOS = {
    "database_issues": [
        {
            "title": "Database connection pool exhaustion causing timeouts",
            "description": "Our application is experiencing timeout errors when connecting to the database. Logs show 'connection limit exceeded' and users are reporting slow response times. This is CRITICAL and requires immediate action.",
            "severity": "critical",
            "keywords": ["database", "connection", "timeout", "exceeded", "pool"],
            "resolution": "Increased max_connections parameter in postgresql.conf and implemented connection pooling with PgBouncer."
        },
        {
            "title": "Database replication lag affecting data consistency",
            "description": "Read replicas are showing outdated data compared to the primary. Monitoring shows replication lag exceeding 30 seconds. This is causing data inconsistency issues in the application.",
            "severity": "high",
            "keywords": ["database", "replication", "lag", "consistency", "replica"],
            "resolution": "Optimized write operations on primary database and increased network bandwidth between primary and replicas."
        },
        {
            "title": "Slow database queries degrading application performance",
            "description": "Several API endpoints are responding slowly due to inefficient database queries. Query times have increased from milliseconds to seconds, affecting user experience.",
            "severity": "medium",
            "keywords": ["database", "slow", "query", "performance", "api"],
            "resolution": "Added missing indexes and optimized JOIN operations in the problematic queries."
        }
    ],
    "network_problems": [
        {
            "title": "Network connectivity issues between microservices",
            "description": "Intermittent connectivity issues between frontend and backend services. Users are experiencing timeouts and error messages. This OUTAGE is affecting all operations.",
            "severity": "critical",
            "keywords": ["network", "connectivity", "timeout", "microservice", "outage"],
            "resolution": "Fixed misconfigured firewall rules that were blocking traffic between service subnets."
        },
        {
            "title": "DNS resolution failures causing service unavailability",
            "description": "Services are unable to resolve hostnames correctly. Error logs show 'unknown host' messages. This is causing intermittent service disruptions.",
            "severity": "high",
            "keywords": ["dns", "resolution", "hostname", "service", "unavailable"],
            "resolution": "Repaired corrupted DNS cache and added redundant DNS servers for failover."
        }
    ],
    "application_errors": [
        {
            "title": "Memory leak in authentication service",
            "description": "The authentication service is experiencing memory leaks, causing it to crash every few hours. Users are being logged out unexpectedly. This is CRITICAL as it affects all users.",
            "severity": "critical",
            "keywords": ["memory", "leak", "crash", "authentication", "service"],
            "resolution": "Fixed memory leak by properly closing database connections in exception handlers."
        },
        {
            "title": "API rate limiting errors in third-party integration",
            "description": "Our integration with the payment processor is hitting rate limits during peak hours. This is causing payment failures and affecting customer purchases.",
            "severity": "high",
            "keywords": ["api", "rate", "limit", "integration", "payment"],
            "resolution": "Implemented request batching and exponential backoff retry strategy for the payment API calls."
        }
    ],
    "security_incidents": [
        {
            "title": "Authentication service breach attempt detected",
            "description": "Multiple failed login attempts from unusual IP ranges detected on the authentication service. Pattern suggests a brute force attack. This is a CRITICAL security incident.",
            "severity": "critical",
            "keywords": ["security", "authentication", "breach", "login", "attack"],
            "resolution": "Blocked suspicious IP ranges and implemented CAPTCHA for repeated failed login attempts."
        },
        {
            "title": "Suspicious activity in admin accounts",
            "description": "Unusual login patterns detected for admin accounts outside of normal business hours. Possible unauthorized access to admin portal.",
            "severity": "high",
            "keywords": ["security", "admin", "suspicious", "login", "unauthorized"],
            "resolution": "Reset admin credentials and enabled multi-factor authentication for all administrative accounts."
        }
    ],
    "infrastructure_failures": [
        {
            "title": "Critical host server failure in production environment",
            "description": "Main application server is unresponsive and cannot be reached via SSH. This is causing a complete OUTAGE of the service. Urgent intervention required.",
            "severity": "critical",
            "keywords": ["server", "failure", "outage", "production", "unresponsive"],
            "resolution": "Hardware failure in the host. Migrated services to backup server and replaced faulty hardware."
        },
        {
            "title": "Disk space reaching critical levels on database server",
            "description": "Database server is running low on disk space (95% used). This will cause write operations to fail if not addressed promptly.",
            "severity": "high",
            "keywords": ["disk", "space", "database", "server", "critical"],
            "resolution": "Cleaned up old logs and database dumps, then added additional storage capacity to the server."
        }
    ],
    "performance_issues": [
        {
            "title": "High CPU usage causing application slowdown",
            "description": "All application servers are showing sustained high CPU usage (>90%). This is causing significant slowdown in response times and affecting user experience.",
            "severity": "high",
            "keywords": ["cpu", "usage", "application", "slowdown", "performance"],
            "resolution": "Identified and optimized CPU-intensive background tasks that were competing with user requests."
        },
        {
            "title": "Memory consumption spike during peak hours",
            "description": "Application servers are experiencing memory consumption spikes during peak usage hours, leading to slower garbage collection and degraded performance.",
            "severity": "medium",
            "keywords": ["memory", "consumption", "peak", "garbage", "collection"],
            "resolution": "Optimized memory usage in data processing routines and adjusted JVM memory settings."
        }
    ],
    "kubernetes_problems": [
        {
            "title": "Multiple Kubernetes nodes showing NotReady status",
            "description": "Several nodes in the Kubernetes cluster are showing NotReady status. Pods are being evicted and new deployments are stuck in Pending state. This is causing service disruption.",
            "severity": "critical",
            "keywords": ["kubernetes", "node", "notready", "pod", "eviction"],
            "resolution": "Fixed corrupted kubelet certificates and restarted kubelet services on affected nodes."
        },
        {
            "title": "Pods stuck in CrashLoopBackOff due to configuration issues",
            "description": "Several critical service pods are stuck in CrashLoopBackOff state. Logs show configuration errors when starting. Services are partially unavailable.",
            "severity": "high",
            "keywords": ["kubernetes", "pod", "crash", "configuration", "service"],
            "resolution": "Fixed incorrect environment variables in the deployment configuration and redeployed affected services."
        }
    ],
    "cloud_service_disruptions": [
        {
            "title": "AWS S3 access issues affecting file uploads",
            "description": "Users are unable to upload files due to S3 access issues. Operations that require file storage are failing. This is causing a major disruption to service.",
            "severity": "high",
            "keywords": ["aws", "s3", "access", "file", "upload"],
            "resolution": "Updated IAM policies that had expired and implemented client-side retry logic for S3 operations."
        },
        {
            "title": "AWS RDS instance performance degradation",
            "description": "The primary RDS instance is showing high CPU utilization and increased latency. Database operations are taking longer than normal, affecting overall application performance.",
            "severity": "medium",
            "keywords": ["aws", "rds", "performance", "latency", "database"],
            "resolution": "Optimized expensive queries and scaled up the RDS instance to handle the increased load."
        }
    ]
}

# Similar incidents for testing incident similarity
SIMILAR_INCIDENTS = [
    # Database connection issues group
    [
        {
            "title": "Database connection pool exhausted in production",
            "description": "Connection pool limit reached in the production database, causing timeouts and errors for users. Application logs show 'connection limit exceeded'.",
            "severity": "critical",
            "category": "database_issues"
        },
        {
            "title": "Database connections maxed out during peak hours",
            "description": "During high traffic periods, we're seeing database connection errors. The pool is exhausted and new connections are being rejected.",
            "severity": "high",
            "category": "database_issues"
        },
        {
            "title": "Connection timeout errors when accessing database",
            "description": "Users report timeout errors when performing data-intensive operations. Logs show database connection pool is being depleted.",
            "severity": "high",
            "category": "database_issues"
        }
    ],
    
    # Memory leak group
    [
        {
            "title": "Memory leak in authentication service",
            "description": "Auth service is consuming more memory over time until it crashes. Requires restart every few hours to maintain service.",
            "severity": "critical",
            "category": "application_errors"
        },
        {
            "title": "Authentication module showing memory growth pattern",
            "description": "Monitoring shows steady memory increase in the auth module until OOM errors occur. Service becomes unresponsive after running for several hours.",
            "severity": "high",
            "category": "application_errors"
        },
        {
            "title": "Out of memory errors in identity service",
            "description": "Identity management service experiences out of memory errors after prolonged use. Memory usage graph shows continuous upward trend without plateaus.",
            "severity": "high",
            "category": "application_errors"
        }
    ],
    
    # Kubernetes node issues
    [
        {
            "title": "Kubernetes nodes showing NotReady status",
            "description": "Multiple nodes in production Kubernetes cluster are in NotReady state. Pods are being evicted causing service disruption.",
            "severity": "critical",
            "category": "kubernetes_problems"
        },
        {
            "title": "K8s cluster node failures affecting deployments",
            "description": "Several nodes in the Kubernetes cluster are failing health checks. New pods are stuck in pending state and cannot be scheduled.",
            "severity": "critical",
            "category": "kubernetes_problems"
        },
        {
            "title": "Worker nodes disconnected from Kubernetes master",
            "description": "Worker nodes are disconnected from the control plane. Kubelet service is failing to maintain connection to the API server.",
            "severity": "high",
            "category": "kubernetes_problems"
        }
    ]
]

async def init_openai_client():
    """Initialize the OpenAI client"""
    global openai_client
    
    if not openai_client:
        try:
            if not settings.openai_api_key:
                logger.warning("No OpenAI API key found in settings. Using template-based generation only.")
                return None
                
            openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI client initialized successfully")
            return openai_client
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            return None
    
    return openai_client

async def generate_test_incident(
    source: str, 
    event_type: str, 
    category: str = None,
    scenario_index: int = None,
    days_ago: int = 0,
    similar_group_index: int = None
) -> Dict[str, Any]:
    """
    Generate a test incident for a specific category and scenario.
    
    Args:
        source: The source of the event (github, jira, etc.)
        event_type: The type of event
        category: The category of the incident
        scenario_index: The index of the scenario to use from the category
        days_ago: Days ago the incident was created
        similar_group_index: Index of similar incident group to use (to test similarity)
        
    Returns:
        Dict containing the event data
    """
    # Choose category if not specified
    if not category:
        category = random.choice(list(TEST_SCENARIOS.keys()))
    
    # Generate a test ID for tracking
    test_id = f"test_{uuid.uuid4()}"
    
    # If we're testing similarity, use a similar incident template
    if similar_group_index is not None and similar_group_index < len(SIMILAR_INCIDENTS):
        similar_group = SIMILAR_INCIDENTS[similar_group_index]
        template = random.choice(similar_group)
        scenario = {
            "title": template["title"],
            "description": template["description"],
            "severity": template["severity"],
            "keywords": [w for w in template["description"].lower().split() if len(w) > 4][:5],
            "resolution": "Pending investigation",
            "category": template["category"]
        }
    else:
        # Get scenarios for the category
        scenarios = TEST_SCENARIOS.get(category, [])
        if not scenarios:
            logger.warning(f"No scenarios found for category: {category}")
            return None
            
        # Choose scenario
        if scenario_index is not None and scenario_index < len(scenarios):
            scenario = scenarios[scenario_index]
        else:
            scenario = random.choice(scenarios)
    
    # Generate timestamp
    if days_ago > 0:
        timestamp = (datetime.datetime.utcnow() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Create raw payload based on source
    if source == "github":
        if event_type == "issue":
            raw_payload = {
                "action": "opened",
                "issue": {
                    "title": scenario["title"],
                    "body": scenario["description"],
                    "state": "open",
                    "labels": [],
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "user": {
                        "login": "test-user",
                        "id": random.randint(10000, 99999),
                    },
                    "number": random.randint(100, 999)
                },
                "repository": {
                    "name": "test-repo",
                    "full_name": "org/test-repo",
                    "owner": {
                        "login": "org"
                    }
                },
                "sender": {
                    "login": "test-user"
                },
                "test_id": test_id,
                "test_category": category,
                "test_severity": scenario["severity"]
            }
        elif event_type == "pull_request":
            raw_payload = {
                "action": "opened",
                "pull_request": {
                    "title": scenario["title"],
                    "body": scenario["description"],
                    "state": "open",
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "user": {
                        "login": "test-user",
                        "id": random.randint(10000, 99999)
                    },
                    "number": random.randint(100, 999)
                },
                "repository": {
                    "name": "test-repo",
                    "full_name": "org/test-repo",
                    "owner": {
                        "login": "org"
                    }
                },
                "sender": {
                    "login": "test-user"
                },
                "test_id": test_id,
                "test_category": category,
                "test_severity": scenario["severity"]
            }
    elif source == "jira":
        issue_id = f"TEST-{random.randint(1000, 9999)}"
        raw_payload = {
            "issue": {
                "id": issue_id,
                "key": issue_id,
                "fields": {
                    "summary": scenario["title"],
                    "description": scenario["description"],
                    "issuetype": {
                        "name": "Bug"
                    },
                    "priority": {
                        "name": severity_to_jira_priority(scenario["severity"])
                    },
                    "status": {
                        "name": "Open" if event_type == "issue_created" else "In Progress"
                    },
                    "created": timestamp,
                    "updated": timestamp,
                    "reporter": {
                        "displayName": "Test User",
                        "emailAddress": "test@example.com"
                    }
                }
            },
            "test_id": test_id,
            "test_category": category,
            "test_severity": scenario["severity"]
        }
    elif source == "monitoring":
        raw_payload = {
            "alert_type": event_type,
            "alert_id": f"alert-{random.randint(10000, 99999)}",
            "title": scenario["title"],
            "message": scenario["description"],
            "severity": scenario["severity"],
            "created": timestamp,
            "tags": scenario["keywords"],
            "source_ip": f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "status": "triggered",
            "test_id": test_id,
            "test_category": category,
            "test_severity": scenario["severity"]
        }
    else:
        raw_payload = {
            "title": scenario["title"],
            "description": scenario["description"],
            "severity": scenario["severity"],
            "timestamp": timestamp,
            "test_id": test_id,
            "test_category": category,
            "test_severity": scenario["severity"]
        }
    
    # Create the standardized event format
    return {
        "source": source,
        "event_type": event_type,
        "timestamp": timestamp,
        "raw_payload": raw_payload
    }

def severity_to_jira_priority(severity: str) -> str:
    """Convert severity to JIRA priority"""
    mapping = {
        "critical": "Highest",
        "high": "High",
        "medium": "Medium",
        "low": "Low"
    }
    return mapping.get(severity, "Medium")

async def send_event_to_api(session: aiohttp.ClientSession, event: Dict[str, Any]) -> bool:
    """
    Send the event to the API for processing.
    
    Args:
        session: The aiohttp client session
        event: The event data to send
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Sending {event['source']} {event['event_type']} event to API")
        
        # Send to the events endpoint
        response = await session.post(
            f"{BASE_URL}/api/events",
            json=event,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status in (200, 201, 202):
            logger.info(f"Successfully sent event to API: {response.status}")
            return True
        else:
            error_text = await response.text()
            logger.error(f"Error sending event to API: {response.status} - {error_text}")
            return False
            
    except Exception as e:
        logger.error(f"Exception sending event to API: {e}")
        return False

async def create_similar_incident_group(session: aiohttp.ClientSession, group_index: int, source: str, event_type: str) -> List[str]:
    """
    Create a group of similar incidents to test similarity detection.
    
    Args:
        session: The aiohttp client session
        group_index: Index of the similar incidents group to use
        source: The source for the events
        event_type: The event type to use
        
    Returns:
        List of test_ids for the created incidents
    """
    if group_index >= len(SIMILAR_INCIDENTS):
        logger.error(f"Invalid group index: {group_index}. Only {len(SIMILAR_INCIDENTS)} groups available.")
        return []
        
    logger.info(f"Creating similar incident group {group_index}")
    
    test_ids = []
    incidents_created = 0
    
    # Create the incidents in the group with different timestamps
    for i in range(len(SIMILAR_INCIDENTS[group_index])):
        # Create with different timestamps (older to newer)
        days_ago = max(0, 10 - i)  # 10, 9, 8, ... days ago
        
        # Generate the incident
        incident = await generate_test_incident(
            source=source,
            event_type=event_type,
            days_ago=days_ago,
            similar_group_index=group_index
        )
        
        if incident:
            # Send to API
            success = await send_event_to_api(session, incident)
            
            if success:
                incidents_created += 1
                test_ids.append(incident["raw_payload"]["test_id"])
                logger.info(f"Created similar incident {i+1}/{len(SIMILAR_INCIDENTS[group_index])}")
            
            # Sleep to avoid overwhelming the API
            await asyncio.sleep(0.5)
    
    logger.info(f"Created {incidents_created}/{len(SIMILAR_INCIDENTS[group_index])} similar incidents")
    return test_ids

async def test_all_functionalities():
    """
    Run tests for all the required functionalities.
    """
    logger.info("Starting comprehensive testing of all functionalities")
    
    # Create a client session for all requests
    async with aiohttp.ClientSession() as session:
        # 1. Test incident detection by creating incidents from different sources
        for source, event_types in EVENT_SOURCES.items():
            event_type = random.choice(event_types)
            category = random.choice(list(TEST_SCENARIOS.keys()))
            
            logger.info(f"Testing incident detection with {source} {event_type} for {category}")
            
            incident = await generate_test_incident(
                source=source,
                event_type=event_type,
                category=category
            )
            
            if incident:
                await send_event_to_api(session, incident)
                await asyncio.sleep(1)  # Give the system time to process
                
        # 2. Test severity classification by creating incidents of different severities
        logger.info("Testing severity classification with incidents of varying severity")
        for severity in SEVERITY_LEVELS:
            # Find a scenario with the target severity
            category = random.choice(list(TEST_SCENARIOS.keys()))
            scenarios = [s for s in TEST_SCENARIOS[category] if s["severity"] == severity]
            
            if scenarios:
                source = random.choice(list(EVENT_SOURCES.keys()))
                event_type = random.choice(EVENT_SOURCES[source])
                
                logger.info(f"Creating {severity} severity incident from {source}")
                
                incident = await generate_test_incident(
                    source=source,
                    event_type=event_type,
                    category=category,
                    scenario_index=TEST_SCENARIOS[category].index(scenarios[0])
                )
                
                if incident:
                    await send_event_to_api(session, incident)
                    await asyncio.sleep(1)  # Give the system time to process
                    
        # 3. Test incident similarity by creating groups of similar incidents
        logger.info("Testing incident similarity with groups of related incidents")
        for group_index in range(len(SIMILAR_INCIDENTS)):
            # Choose random source and event type
            source = random.choice(list(EVENT_SOURCES.keys()))
            event_type = random.choice(EVENT_SOURCES[source])
            
            test_ids = await create_similar_incident_group(
                session=session,
                group_index=group_index,
                source=source,
                event_type=event_type
            )
            
            logger.info(f"Created similar incident group {group_index} with {len(test_ids)} incidents")
            await asyncio.sleep(2)  # Give more time for similarity processing
            
        # 4. Test resolution generation and recommended fix steps
        # (This relies on the knowledge base being populated with resolution data)
        logger.info("Testing resolution generation with incidents that match knowledge base entries")
        for category in INCIDENT_CATEGORIES:
            if category in TEST_SCENARIOS:
                # Pick a source and event type
                source = random.choice(list(EVENT_SOURCES.keys()))
                event_type = random.choice(EVENT_SOURCES[source])
                
                # Create an incident that should match knowledge base entries
                incident = await generate_test_incident(
                    source=source,
                    event_type=event_type,
                    category=category
                )
                
                if incident:
                    await send_event_to_api(session, incident)
                    logger.info(f"Created incident for {category} to test resolution generation")
                    await asyncio.sleep(1)
        
        logger.info("All test incidents have been created successfully.")
        logger.info("Please check the system to verify that the incidents were properly detected, classified, and processed.")

async def main():
    """Main function to run the test incident generator"""
    parser = argparse.ArgumentParser(description="Generate test incidents for System Guardian")
    parser.add_argument("--url", type=str, help=f"Base URL of the API (default: {BASE_URL})")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive tests for all functionalities")
    parser.add_argument("--similarity", type=int, help="Test similarity by creating a group of similar incidents (group_index)")
    parser.add_argument("--category", type=str, choices=INCIDENT_CATEGORIES, help="Category of incident to generate")
    parser.add_argument("--source", type=str, choices=list(EVENT_SOURCES.keys()), help="Source of the event")
    parser.add_argument("--wait", type=int, default=0, help="Wait time in seconds between creating incidents")
    
    args = parser.parse_args()
    
    # Set base URL if provided
    if args.url:
        global BASE_URL
        BASE_URL = args.url
    
    logger.info(f"Using API URL: {BASE_URL}")
    
    # Initialize OpenAI client
    await init_openai_client()
    
    # Create a session for all requests
    async with aiohttp.ClientSession() as session:
        if args.comprehensive:
            # Run comprehensive tests for all functionalities
            await test_all_functionalities()
        elif args.similarity is not None:
            # Test incident similarity with a specific group
            source = args.source or random.choice(list(EVENT_SOURCES.keys()))
            event_type = random.choice(EVENT_SOURCES[source])
            
            await create_similar_incident_group(
                session=session,
                group_index=args.similarity,
                source=source,
                event_type=event_type
            )
        elif args.category:
            # Create a specific category of incident
            source = args.source or random.choice(list(EVENT_SOURCES.keys()))
            event_type = random.choice(EVENT_SOURCES[source])
            
            logger.info(f"Generating incident for category: {args.category} from {source} {event_type}")
            
            incident = await generate_test_incident(
                source=source,
                event_type=event_type,
                category=args.category
            )
            
            if incident:
                await send_event_to_api(session, incident)
                logger.info(f"Created incident for {args.category}")
        else:
            # Default: create a random incident
            source = args.source or random.choice(list(EVENT_SOURCES.keys()))
            event_type = random.choice(EVENT_SOURCES[source])
            category = random.choice(INCIDENT_CATEGORIES)
            
            logger.info(f"Generating random incident from {source} {event_type} for {category}")
            
            incident = await generate_test_incident(
                source=source,
                event_type=event_type,
                category=category
            )
            
            if incident:
                await send_event_to_api(session, incident)
                logger.info(f"Created random incident")

if __name__ == "__main__":
    asyncio.run(main()) 
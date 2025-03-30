#!/usr/bin/env python
"""
Script to generate resolution knowledge for system incidents.

This script creates a set of documents describing how to resolve common system issues.
These documents will be used as a knowledge base in Qdrant for incident resolution.
The documents are stored as individual Markdown files for easy reading and maintenance.
"""
import json
import argparse
import os
import uuid
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Define the knowledge categories
KNOWLEDGE_CATEGORIES = [
    "database_issues",
    "network_problems",
    "application_errors",
    "security_incidents",
    "infrastructure_failures",
    "performance_issues",
    "kubernetes_problems",
    "cloud_service_disruptions",
    "monitoring_alerts",
]

# Resolution documents by category
RESOLUTION_DOCUMENTS = {
    "database_issues": [
        {
            "title": "Database Connection Pool Exhaustion",
            "symptoms": [
                "Application timeout errors when connecting to database",
                "Error logs showing 'connection limit exceeded'",
                "Increasing latency in database operations",
                "Application crashes with database connection errors",
            ],
            "severity": "high",
            "resolution_steps": [
                "1. Check current connection count: `SELECT count(*) FROM pg_stat_activity;`",
                "2. Identify connection sources: `SELECT client_addr, count(*) FROM pg_stat_activity GROUP BY client_addr ORDER BY count(*) DESC;`",
                "3. Check for idle connections: `SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';`",
                "4. Terminate long-running idle connections: `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND current_timestamp - state_change > interval '10 minutes';`",
                "5. Increase max_connections parameter in postgresql.conf",
                "6. Implement connection pooling using PgBouncer or similar tool",
                "7. Review application code for connection leaks",
                "8. Add proper connection release in application exception handlers",
            ],
            "prevention": [
                "Implement connection pooling at application level",
                "Set appropriate connection timeouts and idle connection limits",
                "Monitor connection utilization and set up alerts",
                "Use connection leak detection tools in development and testing",
                "Implement circuit breaker pattern for database access",
            ],
        },
        {
            "title": "Database Replication Lag",
            "symptoms": [
                "Read replicas showing outdated data",
                "Monitoring alerts for replication lag exceeding thresholds",
                "Increasing lag metrics over time",
                "Data inconsistency reports from users",
            ],
            "severity": "medium",
            "resolution_steps": [
                "1. Check current replication lag: `SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;`",
                "2. Identify heavy write operations: `SELECT query, calls, total_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;`",
                "3. Check disk I/O on primary and replicas",
                "4. Ensure replicas have adequate resources (CPU, memory, disk I/O)",
                "5. Reduce write load on primary if possible",
                "6. Add more replicas to distribute read load",
                "7. Consider enhancing network bandwidth between primary and replicas",
                "8. For immediate relief, restart slow replicas if replication is severely lagging",
            ],
            "prevention": [
                "Monitor replication lag continuously",
                "Set up alerts for increasing lag trends",
                "Schedule heavy write operations during low-traffic periods",
                "Ensure replicas have equivalent or better hardware than primary",
                "Implement read/write splitting in application to reduce primary load",
                "Use synchronous replication for critical data if acceptable performance impact",
            ],
        },
        {
            "title": "Database Index Corruption",
            "symptoms": [
                "Slow queries that were previously fast",
                "Error messages about corrupt indexes",
                "Unexpected query results",
                "Database logs showing index-related errors",
            ],
            "severity": "high",
            "resolution_steps": [
                "1. Identify the corrupt index: Check database logs for error messages",
                "2. Verify index corruption: `SELECT pg_stat_user_indexes.schemaname, pg_stat_user_indexes.relname, pg_stat_user_indexes.indexrelname FROM pg_stat_user_indexes, pg_index WHERE pg_index.indisvalid = false AND pg_stat_user_indexes.indexrelid = pg_index.indexrelid;`",
                "3. Drop and recreate the corrupt index: `DROP INDEX <index_name>; CREATE INDEX <index_name> ON <table> (<columns>);`",
                "4. If multiple indexes are corrupt, consider `REINDEX DATABASE <database_name>;`",
                "5. Check system for underlying hardware issues (disk errors, memory problems)",
                "6. Verify database consistency: `ANALYZE VERBOSE;`",
            ],
            "prevention": [
                "Regular database maintenance (VACUUM, ANALYZE)",
                "Monitor index usage and performance",
                "Schedule regular index rebuilds for critical tables",
                "Ensure proper database shutdown procedures are followed",
                "Implement database health checks in monitoring system",
            ],
        },
    ],
    "network_problems": [
        {
            "title": "Network Connectivity Issues Between Services",
            "symptoms": [
                "Timeout errors in service-to-service communication",
                "Increased latency in API responses",
                "Connection refused errors in logs",
                "Intermittent service availability",
            ],
            "severity": "high",
            "resolution_steps": [
                "1. Verify basic connectivity: `ping <service_hostname>`",
                "2. Check if port is open: `telnet <service_hostname> <port>` or `nc -zv <service_hostname> <port>`",
                "3. Verify DNS resolution: `nslookup <service_hostname>`",
                "4. Check network routes: `traceroute <service_hostname>`",
                "5. Inspect firewall rules and security groups",
                "6. Check network ACLs in cloud environment",
                "7. Verify service health status in load balancer",
                "8. Check for network saturation or packet loss: `iperf` between services",
            ],
            "prevention": [
                "Implement service mesh for better network visibility",
                "Use health checks and circuit breakers",
                "Monitor network metrics (latency, packet loss, throughput)",
                "Document network topology and firewall rules",
                "Implement retry with exponential backoff for network operations",
                "Use connection pooling where appropriate",
            ],
        },
        {
            "title": "DNS Resolution Failures",
            "symptoms": [
                "Services unable to resolve hostnames",
                "Error logs showing 'unknown host'",
                "Intermittent connectivity issues",
                "Some services working while others fail",
            ],
            "severity": "critical",
            "resolution_steps": [
                "1. Verify DNS configuration: `cat /etc/resolv.conf`",
                "2. Test DNS resolution: `dig <hostname>` or `nslookup <hostname>`",
                "3. Check DNS server health: `dig @<dns_server> <hostname>`",
                "4. Verify DNS service is running: `systemctl status named` or equivalent",
                "5. Check for DNS cache poisoning: clear DNS cache",
                "6. If using cloud DNS, check cloud provider status",
                "7. Temporarily add hostname entries to /etc/hosts as emergency fix",
                "8. Restart DNS services if necessary",
            ],
            "prevention": [
                "Use multiple DNS servers for redundancy",
                "Implement local DNS caching",
                "Monitor DNS query performance and success rates",
                "Maintain backup of DNS records",
                "Consider using service discovery solutions",
            ],
        },
    ],
    "application_errors": [
        {
            "title": "Memory Leaks in Application",
            "symptoms": [
                "Gradually increasing memory usage over time",
                "OutOfMemoryError exceptions",
                "Application slowdown before crash",
                "Increasing garbage collection frequency",
            ],
            "severity": "high",
            "resolution_steps": [
                "1. Verify memory leak with monitoring tools",
                "2. Take heap dump: `jmap -dump:format=b,file=heap.bin <pid>` (Java) or appropriate tool for your language",
                "3. Analyze heap dump with analyzer tool (e.g., Eclipse MAT for Java)",
                "4. Identify objects with high retention",
                "5. Look for references preventing garbage collection",
                "6. Check for unclosed resources (DB connections, file handles, etc.)",
                "7. Deploy temporary fix: increase heap size or implement scheduled restarts",
                "8. Fix the code by properly releasing resources or removing circular references",
            ],
            "prevention": [
                "Regular memory usage monitoring",
                "Memory leak detection in CI/CD pipeline",
                "Load testing with memory analysis",
                "Code reviews focused on resource management",
                "Use tools like leak canaries or sanitizers in testing",
            ],
        },
        {
            "title": "API Rate Limiting Errors",
            "symptoms": [
                "HTTP 429 (Too Many Requests) responses",
                "Sudden failure of API-dependent functionalities",
                "Error logs showing rate limit exceeded",
                "Increasing latency in API responses before failures",
            ],
            "severity": "medium",
            "resolution_steps": [
                "1. Identify which API is being rate limited",
                "2. Check current rate limit usage and quotas",
                "3. Implement exponential backoff and retry strategy",
                "4. Check for inefficient API usage patterns (e.g., unnecessary polling)",
                "5. Distribute requests across multiple API keys if available",
                "6. Implement request batching where possible",
                "7. Cache API responses where appropriate",
                "8. Contact API provider for quota increase if necessary",
            ],
            "prevention": [
                "Monitor API usage rates and implement alerts before limits are reached",
                "Design applications with rate limits in mind",
                "Implement client-side throttling",
                "Use bulk API endpoints where available",
                "Implement proper caching strategy",
            ],
        },
    ],
    "security_incidents": [
        {
            "title": "Authentication Service Breach Attempt",
            "symptoms": [
                "Multiple failed login attempts from unusual IP ranges",
                "Unusual login patterns (time of day, geolocation)",
                "Attempts to exploit known vulnerabilities in auth system",
                "Unusual account privilege escalation attempts",
            ],
            "severity": "critical",
            "resolution_steps": [
                "1. Temporarily block suspicious IP ranges at firewall or WAF",
                "2. Enable additional authentication factors for admin accounts",
                "3. Analyze logs to identify the attack pattern and scope",
                "4. Check for successful breaches by reviewing user activity logs",
                "5. Reset credentials for any potentially compromised accounts",
                "6. Apply security patches if vulnerability exploitation was attempted",
                "7. Increase logging detail for authentication systems",
                "8. Report incident to security team and follow incident response plan",
            ],
            "prevention": [
                "Implement rate limiting on authentication endpoints",
                "Use captcha or challenge questions for repeated failed attempts",
                "Deploy intrusion detection systems",
                "Implement anomaly detection for login patterns",
                "Regular security penetration testing",
                "Keep authentication systems patched and updated",
            ],
        }
    ],
    "infrastructure_failures": [
        {
            "title": "Host Server Failure",
            "symptoms": [
                "Server unresponsive to ping or SSH",
                "Services hosted on server unavailable",
                "Error logs showing connection timeouts",
                "Monitoring system alerts for host down",
            ],
            "severity": "critical",
            "resolution_steps": [
                "1. Verify if server is truly unreachable: ping, SSH, console access",
                "2. Check physical server status if accessible (power, network)",
                "3. For cloud instances, check cloud provider status dashboard",
                "4. Attempt server restart through management console",
                "5. If restart fails, initiate disaster recovery protocol",
                "6. Provision replacement server if necessary",
                "7. Restore from backup or rehydrate from infrastructure as code",
                "8. Redirect traffic to new server (update DNS, load balancer)",
            ],
            "prevention": [
                "Implement high availability architecture",
                "Use infrastructure as code for quick recovery",
                "Regular backup testing and validation",
                "Implement proper monitoring and alerting",
                "Document recovery procedures and test them regularly",
            ],
        }
    ],
    "kubernetes_problems": [
        {
            "title": "Kubernetes Node Not Ready",
            "symptoms": [
                "Pods stuck in Pending state",
                "Kubectl shows node(s) in NotReady state",
                "Application deployments incomplete",
                "Kubelet service failures",
            ],
            "severity": "high",
            "resolution_steps": [
                "1. Identify affected nodes: `kubectl get nodes`",
                "2. Check node conditions: `kubectl describe node <node-name>`",
                "3. Check kubelet status on node: `systemctl status kubelet`",
                "4. Examine kubelet logs: `journalctl -u kubelet`",
                "5. Check for resource exhaustion: `top`, `df -h`, `free -m`",
                "6. Check container runtime: `systemctl status docker` (or containerd, etc.)",
                "7. Restart kubelet if necessary: `systemctl restart kubelet`",
                "8. Cordon and drain node if persistent issues: `kubectl cordon <node-name>` followed by `kubectl drain <node-name>`",
                "9. For cloud environments, check if node has been terminated by provider",
            ],
            "prevention": [
                "Regular node maintenance and updates",
                "Implement resource requests and limits for all pods",
                "Set up monitoring for node resource usage",
                "Use Pod Disruption Budgets for critical applications",
                "Implement automated node health checks",
            ],
        },
        {
            "title": "Pod Stuck in CrashLoopBackOff",
            "symptoms": [
                "Pod continuously restarting",
                "Pod status showing CrashLoopBackOff",
                "Service unavailability despite pod presence",
                "Application error logs before crash",
            ],
            "severity": "medium",
            "resolution_steps": [
                "1. Identify affected pods: `kubectl get pods -A | grep CrashLoop`",
                "2. Check pod details: `kubectl describe pod <pod-name> -n <namespace>`",
                "3. Examine container logs: `kubectl logs <pod-name> -n <namespace>`",
                "4. Check previous container logs if current container won't start: `kubectl logs <pod-name> -n <namespace> --previous`",
                "5. Verify resource limits are appropriate",
                "6. Check if liveness probe is properly configured",
                "7. Verify container image and tag are correct",
                "8. Check for configuration issues in ConfigMaps or Secrets",
                "9. Temporarily increase initialDelaySeconds for liveness probe as emergency fix",
            ],
            "prevention": [
                "Thorough testing of containers before deployment",
                "Implement health checks in application code",
                "Set appropriate resource requests and limits",
                "Use startup probes for slow-starting applications",
                "Implement proper logging for faster debugging",
            ],
        },
    ],
    "cloud_service_disruptions": [
        {
            "title": "AWS S3 Access Issues",
            "symptoms": [
                "Applications unable to read or write to S3 buckets",
                "Timeout errors when accessing S3",
                "Increased latency in S3 operations",
                "Error logs showing S3 access denied or throttling",
            ],
            "severity": "high",
            "resolution_steps": [
                "1. Check AWS Service Health Dashboard for S3 outages",
                "2. Verify IAM permissions for the accessing role/user",
                "3. Check bucket policy and ACLs",
                "4. Test S3 access using AWS CLI: `aws s3 ls s3://bucket-name/`",
                "5. Check for S3 endpoint connectivity issues",
                "6. Verify that bucket region matches application configuration",
                "7. Implement retry with exponential backoff for S3 operations",
                "8. For critical operations, temporarily switch to backup storage if available",
            ],
            "prevention": [
                "Implement caching layer in front of S3",
                "Use multi-region bucket replication for critical data",
                "Monitor S3 operation latency and error rates",
                "Implement circuit breakers for S3 operations",
                "Have backup storage solution for critical operations",
            ],
        }
    ],
    "performance_issues": [
        {
            "title": "High CPU Usage on Application Servers",
            "symptoms": [
                "Server responding slowly to requests",
                "Monitoring alerts for high CPU usage",
                "Increased response times across multiple services",
                "Process using abnormally high CPU resources",
            ],
            "severity": "medium",
            "resolution_steps": [
                "1. Identify high CPU processes: `top` or `htop`",
                "2. Check system load: `uptime`",
                "3. Get process details: `ps aux | grep <process-id>`",
                "4. For Java applications, take thread dumps: `jstack <pid>`",
                "5. For other applications, use language-specific profiling tools",
                "6. Check for CPU throttling in cloud environments",
                "7. Identify specific endpoints causing high load with APM tools",
                "8. For immediate relief, scale horizontally by adding more servers",
                "9. If single-threaded process, consider increasing instance size (vertical scaling)",
            ],
            "prevention": [
                "Regular performance testing under load",
                "Implement auto-scaling based on CPU metrics",
                "Code profiling during development and testing",
                "Set up monitoring and alerting for CPU usage trends",
                "Optimize critical code paths and database queries",
            ],
        }
    ],
    "monitoring_alerts": [
        {
            "title": "False Positive Monitoring Alerts",
            "symptoms": [
                "Frequent alerts that don't correspond to real issues",
                "Alert fatigue among team members",
                "Alerts triggering during known maintenance windows",
                "Inconsistent alert behavior across similar systems",
            ],
            "severity": "low",
            "resolution_steps": [
                "1. Document pattern of false positives",
                "2. Analyze alert threshold settings",
                "3. Check for environmental factors causing baseline shifts",
                "4. Implement more sophisticated alerting logic (e.g., duration-based thresholds)",
                "5. Add maintenance window functionality to monitoring system",
                "6. Adjust sensitivity of alerts based on historical data",
                "7. Consider implementing anomaly detection instead of static thresholds",
                "8. Create alert suppression rules for known maintenance activities",
            ],
            "prevention": [
                "Regular review and tuning of alert thresholds",
                "Use trending data to set dynamic thresholds",
                "Implement alert correlation to reduce noise",
                "Classify alerts by importance and actionability",
                "Document expected system behavior for different conditions",
            ],
        }
    ],
}


def slugify(text: str) -> str:
    """
    Convert a text string to a slug that can be used in a filename.

    :param text: The text to convert to a slug
    :return: A slugified version of the text
    """
    # Convert to lowercase
    text = text.lower()
    # Replace spaces with hyphens
    text = re.sub(r"\s+", "-", text)
    # Remove any characters that aren't alphanumeric, underscore, or hyphen
    text = re.sub(r"[^a-z0-9_-]", "", text)
    # Replace multiple hyphens with a single hyphen
    text = re.sub(r"-+", "-", text)
    # Remove leading and trailing hyphens
    text = text.strip("-")
    return text


def generate_markdown_document(resolution: Dict[str, Any], category: str) -> str:
    """
    Generate a Markdown document for a resolution.

    :param resolution: The resolution document data
    :param category: The category of the resolution
    :return: A Markdown formatted string
    """
    document_id = str(uuid.uuid4())
    title = resolution["title"]
    severity = resolution["severity"]

    markdown = f"""# {title}

## Metadata
- **ID**: {document_id}
- **Category**: {category}
- **Severity**: {severity}

## Symptoms
"""

    # Add symptoms
    for symptom in resolution["symptoms"]:
        markdown += f"- {symptom}\n"

    # Add resolution steps
    markdown += "\n## Resolution Steps\n\n"
    for step in resolution["resolution_steps"]:
        markdown += f"{step}\n\n"

    # Add prevention measures
    markdown += "## Prevention\n"
    for measure in resolution["prevention"]:
        markdown += f"- {measure}\n"

    return markdown


def save_markdown_document(
    content: str, category: str, title: str, output_dir: str
) -> str:
    """
    Save a Markdown document to a file.

    :param content: The Markdown content to save
    :param category: The category of the resolution
    :param title: The title of the resolution
    :param output_dir: The directory to save the files to
    :return: The path to the saved file
    """
    # Create the output directory if it doesn't exist
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    # Generate a filename based on the title
    filename = f"{slugify(title)}.md"
    file_path = os.path.join(category_dir, filename)

    # Write the content to the file
    with open(file_path, "w") as f:
        f.write(content)

    return file_path


def generate_knowledge_documents(output_dir: str) -> List[str]:
    """
    Generate Markdown knowledge documents from the defined templates.

    :param output_dir: The directory to save the files to
    :return: List of paths to the generated files
    """
    file_paths = []

    for category, resolutions in RESOLUTION_DOCUMENTS.items():
        for resolution in resolutions:
            # Create a knowledge document with appropriate fields
            markdown_content = generate_markdown_document(resolution, category)

            # Save the document to a file
            file_path = save_markdown_document(
                content=markdown_content,
                category=category,
                title=resolution["title"],
                output_dir=output_dir,
            )

            file_paths.append(file_path)

    return file_paths


def generate_index_file(file_paths: List[str], output_dir: str) -> str:
    """
    Generate an index Markdown file listing all documents.

    :param file_paths: List of paths to the generated files
    :param output_dir: The directory where files were saved
    :return: Path to the index file
    """
    index_content = "# Resolution Knowledge Base Index\n\n"

    # Group files by category
    categories = {}
    for path in file_paths:
        rel_path = os.path.relpath(path, output_dir)
        category = os.path.dirname(rel_path)
        if category not in categories:
            categories[category] = []
        categories[category].append(path)

    # Add links to each file, organized by category
    for category, paths in categories.items():
        index_content += f"## {category.replace('_', ' ').title()}\n\n"

        for path in paths:
            # Get the title from the first line of the file
            with open(path, "r") as f:
                title = f.readline().strip("# \n")

            rel_path = os.path.relpath(path, output_dir)
            index_content += f"- [{title}]({rel_path})\n"

        index_content += "\n"

    # Save the index file
    index_path = os.path.join(output_dir, "index.md")
    with open(index_path, "w") as f:
        f.write(index_content)

    return index_path


def generate_metadata_json(file_paths: List[str], output_dir: str) -> str:
    """
    Generate a JSON file with metadata for the knowledge base.

    :param file_paths: List of paths to the generated files
    :param output_dir: The directory where files were saved
    :return: Path to the metadata file
    """
    metadata = []

    for path in file_paths:
        # Extract content
        with open(path, "r") as f:
            content = f.read()

        # Get title from the first line
        title = content.split("\n", 1)[0].strip("# ")

        # Extract other metadata
        category = os.path.dirname(os.path.relpath(path, output_dir))

        # Extract ID
        match = re.search(r"\*\*ID\*\*: ([0-9a-f-]+)", content)
        doc_id = match.group(1) if match else str(uuid.uuid4())

        # Extract severity
        match = re.search(r"\*\*Severity\*\*: (\w+)", content)
        severity = match.group(1) if match else "medium"

        # Extract symptoms
        symptoms = []
        in_symptoms_section = False
        for line in content.split("\n"):
            if line.strip() == "## Symptoms":
                in_symptoms_section = True
                continue
            elif line.strip().startswith("##") and in_symptoms_section:
                in_symptoms_section = False
                continue

            if in_symptoms_section and line.strip().startswith("- "):
                symptoms.append(line.strip("- "))

        metadata.append(
            {
                "id": doc_id,
                "title": title,
                "category": category,
                "severity": severity,
                "file_path": os.path.relpath(path, output_dir),
                "symptoms": symptoms,
            }
        )

    # Save metadata file
    metadata_path = os.path.join(output_dir, "knowledge_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate resolution knowledge documents as Markdown files"
    )
    parser.add_argument(
        "--output-dir",
        default="docs/test_resolution_knowledge",
        help="Output directory for the Markdown files",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate knowledge documents
    file_paths = generate_knowledge_documents(args.output_dir)

    # Generate index file
    index_path = generate_index_file(file_paths, args.output_dir)

    # Generate metadata JSON
    metadata_path = generate_metadata_json(file_paths, args.output_dir)

    print(f"Generated {len(file_paths)} knowledge documents in {args.output_dir}")
    print(f"Index file: {index_path}")
    print(f"Metadata file: {metadata_path}")


if __name__ == "__main__":
    main()

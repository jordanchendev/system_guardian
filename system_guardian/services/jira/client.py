"""JIRA client for creating and managing tickets."""

import logging
from typing import Dict, Any, Optional, List

import aiohttp
import json
from datetime import datetime

from system_guardian.settings import settings

logger = logging.getLogger(__name__)


class JiraClient:
    """JIRA client for creating and managing tickets."""

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        api_token: Optional[str] = None,
        project_key: Optional[str] = None,
        issue_type: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Initialize JIRA client.

        Args:
            url: JIRA instance URL
            username: JIRA username (email)
            api_token: JIRA API token
            project_key: Default JIRA project key
            issue_type: Default JIRA issue type
            timeout: Timeout for requests in seconds
        """
        self.url = url or settings.jira_url
        self.username = username or settings.jira_username
        self.api_token = api_token or settings.jira_api_token
        self.project_key = project_key or settings.jira_project_key
        self.issue_type = issue_type or settings.jira_issue_type
        self.timeout = timeout or settings.jira_timeout

        # Default issue type ID map (will be updated with actual values from API)
        self._issue_type_id_map = {
            "bug": "10004",
            "task": "10002",
            "story": "10001",
            "epic": "10000",
        }

        if not self.url:
            logger.warning("JIRA URL not configured, tickets will not be created")

        if not self.username or not self.api_token:
            logger.warning(
                "JIRA credentials not configured, tickets will not be created"
            )

        self._enabled = settings.jira_enabled

    @property
    def is_configured(self) -> bool:
        """Check if JIRA client is properly configured."""
        return bool(self._enabled and self.url and self.username and self.api_token)

    async def create_ticket(
        self,
        summary: str,
        description: str,
        issue_type: Optional[str] = None,
        project_key: Optional[str] = None,
        labels: Optional[List[str]] = None,
        fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a JIRA ticket.

        Args:
            summary: Ticket summary/title
            description: Ticket description
            issue_type: Issue type (Bug, Task, etc.)
            project_key: Project key
            labels: List of labels
            fields: Additional fields to include

        Returns:
            Response from JIRA API
        """
        if not self.is_configured:
            logger.warning(
                "JIRA is not properly configured, ticket will not be created"
            )
            return {"error": "jira_not_configured"}

        # Use defaults if not specified
        project_key = project_key or self.project_key
        issue_type = issue_type or self.issue_type

        # Log the configuration values being used
        logger.debug(
            f"Creating JIRA ticket with project_key={project_key}, issue_type={issue_type}"
        )

        if not project_key:
            logger.error("No project key specified for JIRA ticket")
            return {"error": "missing_project_key"}

        if not issue_type:
            logger.error("No issue type specified for JIRA ticket")
            return {"error": "missing_issue_type"}

        # Try to find issue type ID first by exact name match
        issue_type_id = None

        # First check if we already have a direct mapping
        issue_type_lower = issue_type.lower()
        issue_type_id = self._issue_type_id_map.get(issue_type_lower)

        # If we still don't have an ID and we have English names like "Task", use known mappings
        if not issue_type_id and issue_type in ["Task", "Bug", "Story", "Epic"]:
            # Standard mappings for common English issue types
            standard_mappings = {
                "task": "10001",
                "bug": "10004",
                "story": "10001",
                "epic": "10000",
            }
            issue_type_id = standard_mappings.get(issue_type_lower)

        # Prepare request payload
        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": description,
            }
        }

        # Always try to use ID first, then fall back to name if ID isn't available
        if issue_type_id:
            payload["fields"]["issuetype"] = {"id": issue_type_id}
            logger.debug(f"Using issue type ID: {issue_type_id} for {issue_type}")
        else:
            # As a fallback, try with the name
            payload["fields"]["issuetype"] = {"name": issue_type}
            logger.debug(
                f"Using issue type name: {issue_type} (no ID mapping available)"
            )

            # For JIRA Cloud with Task (ID: 10001), use it as a safe fallback if we're still guessing
            if settings.jira_url and "atlassian.net" in settings.jira_url:
                logger.warning(
                    f"Using fallback issue type ID 10001 (Task) as a safety measure"
                )
                payload["fields"]["issuetype"] = {
                    "id": "10001"
                }  # Task is usually available

        # Log the payload for debugging
        logger.debug(f"JIRA create ticket payload: {json.dumps(payload)}")

        # Add labels if specified
        if labels:
            payload["fields"]["labels"] = labels

        # Add additional fields if specified
        if fields:
            payload["fields"].update(fields)

        # Send request to JIRA API
        auth = aiohttp.BasicAuth(login=self.username, password=self.api_token)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.url}/rest/api/2/issue",
                    json=payload,
                    auth=auth,
                    headers=headers,
                    timeout=self.timeout,
                ) as response:
                    if response.status == 201:  # Created
                        result = await response.json()
                        logger.info(
                            f"Successfully created JIRA ticket: {result.get('key')}"
                        )
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to create JIRA ticket: {response.status} - {error_text}"
                        )
                        return {
                            "error": f"http_error_{response.status}",
                            "details": error_text,
                        }
        except aiohttp.ClientError as e:
            logger.exception(f"Error creating JIRA ticket: {str(e)}")
            return {"error": "client_error", "details": str(e)}

    async def create_incident_ticket(
        self,
        incident_id: str,
        title: str,
        description: str,
        severity: str,
        source: Optional[str] = None,
        event_type: Optional[str] = None,
        issue_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a JIRA ticket for an incident.

        Args:
            incident_id: Incident ID
            title: Incident title
            description: Incident description
            severity: Incident severity
            source: Event source
            event_type: Event type
            issue_type: Optional issue type override

        Returns:
            Response from JIRA API
        """
        # First, try to initialize issue type map if needed
        if (
            not self._issue_type_id_map or len(self._issue_type_id_map) <= 4
        ):  # Only has default values
            try:
                await self.initialize_issue_type_map()
            except Exception as e:
                logger.warning(f"Failed to initialize issue type map: {str(e)}")

        # Add source and event type to description if available
        full_description = description
        if source and event_type:
            full_description += f"\n\nSource: {source}\nEvent Type: {event_type}"

        # Add incident ID and severity to description
        full_description += f"\n\nIncident ID: {incident_id}"
        full_description += f"\nSeverity: {severity}"
        full_description += f"\nCreated at: {datetime.utcnow().isoformat()}"

        # Format description for JIRA (Markdown-like)
        jira_description = full_description

        # Create labels
        labels = ["system-guardian", f"severity-{severity.lower()}", "auto-created"]
        if source:
            labels.append(f"source-{source}")

        # Create ticket
        summary = f"[Incident #{incident_id}] {title}"

        # Make sure we have a valid issue_type, with fallbacks
        effective_issue_type = issue_type or self.issue_type or "Task"

        logger.info(
            f"Creating incident ticket with issue_type={effective_issue_type}, project_key={self.project_key}"
        )

        return await self.create_ticket(
            summary=summary,
            description=jira_description,
            labels=labels,
            issue_type=effective_issue_type,
            project_key=self.project_key,
        )

    async def get_issue_types(
        self, project_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get available issue types for a project.

        Args:
            project_key: JIRA project key

        Returns:
            Dictionary with issue types information
        """
        if not self.is_configured:
            logger.warning("JIRA is not properly configured, cannot get issue types")
            return {"error": "jira_not_configured"}

        # Use default project key if not specified
        project_key = project_key or self.project_key

        if not project_key:
            logger.error("No project key specified for getting issue types")
            return {"error": "missing_project_key"}

        # Send request to JIRA API
        auth = aiohttp.BasicAuth(login=self.username, password=self.api_token)
        headers = {"Accept": "application/json"}

        try:
            async with aiohttp.ClientSession() as session:
                # Get project metadata including issue types
                async with session.get(
                    f"{self.url}/rest/api/2/project/{project_key}",
                    auth=auth,
                    headers=headers,
                    timeout=self.timeout,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        issue_types = result.get("issueTypes", [])
                        logger.info(
                            f"Retrieved {len(issue_types)} issue types for project {project_key}"
                        )

                        # Log each issue type for debugging
                        for issue_type in issue_types:
                            logger.debug(
                                f"Issue type: {issue_type.get('name')} (ID: {issue_type.get('id')})"
                            )

                        return {"issue_types": issue_types}
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to get issue types: {response.status} - {error_text}"
                        )
                        return {
                            "error": f"http_error_{response.status}",
                            "details": error_text,
                        }
        except aiohttp.ClientError as e:
            logger.exception(f"Error getting issue types: {str(e)}")
            return {"error": "client_error", "details": str(e)}

    async def initialize_issue_type_map(
        self, project_key: Optional[str] = None
    ) -> None:
        """
        Initialize issue type map by fetching available issue types from JIRA API.

        Args:
            project_key: JIRA project key
        """
        logger.info(
            f"Initializing issue type map for project {project_key or self.project_key}"
        )
        result = await self.get_issue_types(project_key)

        if "issue_types" in result:
            # Update issue type map with actual IDs from JIRA
            issue_types = result["issue_types"]
            updated_map = {}

            for issue_type in issue_types:
                name = issue_type.get("name", "").lower()
                id = issue_type.get("id")
                if name and id:
                    updated_map[name] = id
                    logger.debug(f"Added issue type mapping: {name} -> {id}")

            # Update the static map with the actual values
            # Note: This is a simplification, in production code you might want to
            # store this per-instance or in a more structured way
            self._issue_type_id_map = updated_map
            logger.info(f"Issue type map initialized with {len(updated_map)} entries")
        else:
            logger.warning(
                f"Failed to initialize issue type map: {result.get('error', 'unknown error')}"
            )

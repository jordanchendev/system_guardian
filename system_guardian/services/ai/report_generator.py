"""AI-powered report generator service."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, desc

from system_guardian.db.models.incidents import Incident, Event, Resolution
from system_guardian.services.ai.engine import AIEngine
from system_guardian.settings import settings


class ReportFormat:
    """Report format options."""

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


class ReportGenerator:
    """Service for generating AI-powered reports for incidents and events."""

    def __init__(
        self,
        db_session_factory,
        ai_engine: Optional[AIEngine] = None,
    ):
        """
        Initialize the report generator.

        :param db_session_factory: Factory for creating database sessions
        :param ai_engine: Optional AIEngine instance for report generation
        """
        self.db_session_factory = db_session_factory
        self.ai_engine = ai_engine

    async def generate_incident_report(
        self,
        incident_id: int,
        include_events: bool = True,
        include_resolution: bool = True,
        format: str = ReportFormat.MARKDOWN,
    ) -> Dict[str, Any]:
        """
        Generate a detailed report for a specific incident.

        :param incident_id: ID of the incident to generate a report for
        :param include_events: Whether to include related events in the report
        :param include_resolution: Whether to include resolution information
        :param format: Output format (json, markdown, html)
        :return: Report data
        """
        if not self.ai_engine:
            logger.warning("AI Engine not available for report generation")
            return {"error": "AI Engine not available"}

        async with self.db_session_factory() as session:
            # Get the incident
            incident = await session.get(Incident, incident_id)
            if not incident:
                return {"error": f"Incident with ID {incident_id} not found"}

            # Get related events if requested
            events = []
            if include_events:
                events_query = (
                    select(Event)
                    .where(Event.related_incident_id == incident_id)
                    .order_by(Event.created_at)
                )

                events_result = await session.execute(events_query)
                events = events_result.scalars().all()

            # Get resolution if requested
            resolution = None
            if include_resolution:
                resolution_query = select(Resolution).where(
                    Resolution.incident_id == incident_id
                )

                resolution_result = await session.execute(resolution_query)
                resolution = resolution_result.scalars().first()

            # Prepare data for the report
            incident_data = {
                "id": incident.id,
                "title": incident.title,
                "description": incident.description or "",
                "severity": incident.severity,
                "source": incident.source,
                "status": incident.status,
                "created_at": incident.created_at.isoformat(),
                "resolved_at": (
                    incident.resolved_at.isoformat() if incident.resolved_at else None
                ),
            }

            # Prepare events data
            events_data = []
            for event in events:
                event_data = {
                    "id": event.id,
                    "source": event.source,
                    "event_type": event.event_type,
                    "created_at": event.created_at.isoformat(),
                }

                # Extract key content fields
                if isinstance(event.content, dict):
                    content = {}
                    for key in [
                        "title",
                        "description",
                        "message",
                        "text",
                        "error",
                        "status",
                    ]:
                        if key in event.content:
                            content[key] = event.content[key]
                    event_data["content"] = content
                else:
                    event_data["content"] = str(event.content)

                events_data.append(event_data)

            # Prepare resolution data
            resolution_data = None
            if resolution:
                resolution_data = {
                    "suggestion": resolution.suggestion,
                    "is_applied": resolution.is_applied,
                    "confidence": resolution.confidence,
                    "generated_at": (
                        resolution.generated_at.isoformat()
                        if resolution.generated_at
                        else None
                    ),
                }

            # Generate the report using AI
            try:
                # Use specialized model for report generation
                model = (
                    settings.ai_report_generation_model
                    if settings.ai_allow_advanced_models
                    else settings.openai_completion_model
                )

                logger.info(f"Using AI model {model} for incident report generation")

                # Build the system prompt based on the requested format
                system_prompt = "You are an incident report specialist."
                if format == ReportFormat.MARKDOWN:
                    system_prompt += " Generate reports in well-formatted Markdown with proper headings, lists, and formatting."
                elif format == ReportFormat.HTML:
                    system_prompt += " Generate reports in clean HTML with proper headings, lists, and minimal but effective styling."

                # Prepare the AI prompt
                prompt = f"""
                Generate a comprehensive incident report for the following incident:
                
                Incident details:
                {json.dumps(incident_data, indent=2)}
                
                {"Related events:" if events_data else ""}
                {json.dumps(events_data, indent=2) if events_data else ""}
                
                {"Resolution:" if resolution_data else ""}
                {json.dumps(resolution_data, indent=2) if resolution_data else ""}
                
                The report should include:
                1. Executive summary
                2. Incident timeline
                3. Impact assessment
                4. Technical details
                5. Resolution status and steps taken
                6. Lessons learned
                7. Recommendations for preventing similar incidents
                
                """

                # Add format-specific instructions
                if format == ReportFormat.JSON:
                    prompt += """
                    Return your report as a JSON object with the following structure:
                    {
                        "executive_summary": "string",
                        "timeline": "string",
                        "impact": "string",
                        "technical_details": "string",
                        "resolution": "string",
                        "lessons_learned": "string",
                        "recommendations": ["string"]
                    }
                    """
                elif format == ReportFormat.MARKDOWN:
                    prompt += """
                    Format your report as clean, well-structured Markdown with proper headings, lists, code blocks, and emphasis.
                    """
                elif format == ReportFormat.HTML:
                    prompt += """
                    Format your report as clean HTML with proper headings, paragraphs, lists, and minimal inline CSS for readability.
                    """

                # Set the response format based on the requested output format
                response_format = (
                    {"type": "json_object"} if format == ReportFormat.JSON else None
                )

                # Get the appropriate temperature - use a slightly higher temperature for narrative formats
                temperature = settings.ai_default_temperature
                if format in [ReportFormat.MARKDOWN, ReportFormat.HTML]:
                    temperature = min(settings.ai_default_temperature + 0.1, 0.7)

                # Generate the report
                response = await self.ai_engine.llm.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    response_format=response_format,
                )

                # Get the report content
                report_content = response.choices[0].message.content

                # Process the report content based on format
                if format == ReportFormat.JSON:
                    report_data = json.loads(report_content)
                    # Add metadata
                    report_data["metadata"] = {
                        "incident_id": incident_id,
                        "generated_at": datetime.utcnow().isoformat(),
                        "model_used": model,
                        "format": format,
                    }
                    return report_data
                else:
                    # For Markdown and HTML, return the raw content with metadata
                    return {
                        "content": report_content,
                        "metadata": {
                            "incident_id": incident_id,
                            "generated_at": datetime.utcnow().isoformat(),
                            "model_used": model,
                            "format": format,
                        },
                    }

            except Exception as e:
                logger.error(f"Error generating incident report: {e}")
                return {
                    "incident_id": incident_id,
                    "error": f"Failed to generate report: {str(e)}",
                    "generated_at": datetime.utcnow().isoformat(),
                }

    async def generate_summary_report(
        self,
        time_range_days: int = 7,
        format: str = ReportFormat.MARKDOWN,
        include_severity_distribution: bool = True,
        include_resolution_times: bool = True,
        include_recommendations: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a summary report covering incidents within a specified time range.

        :param time_range_days: Number of days to include in the report
        :param format: Output format (json, markdown, html)
        :param include_severity_distribution: Whether to include severity distribution stats
        :param include_resolution_times: Whether to include resolution time stats
        :param include_recommendations: Whether to include recommendations
        :return: Report data
        """
        if not self.ai_engine:
            logger.warning("AI Engine not available for summary report generation")
            return {"error": "AI Engine not available"}

        async with self.db_session_factory() as session:
            # Calculate time range
            start_date = datetime.utcnow() - timedelta(days=time_range_days)

            # Get incidents in the date range
            query = (
                select(Incident)
                .where(Incident.created_at >= start_date)
                .order_by(Incident.created_at.desc())
            )

            result = await session.execute(query)
            incidents = result.scalars().all()

            if not incidents:
                return {
                    "error": "No incidents found in the specified time range",
                    "time_range_days": time_range_days,
                    "generated_at": datetime.utcnow().isoformat(),
                }

            # Calculate basic statistics
            stats = {}

            # Total count
            stats["total_incidents"] = len(incidents)

            # Count by status
            status_counts = {}
            for incident in incidents:
                status = incident.status
                status_counts[status] = status_counts.get(status, 0) + 1
            stats["by_status"] = status_counts

            # Count by severity if requested
            if include_severity_distribution:
                severity_counts = {}
                for incident in incidents:
                    severity = incident.severity
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                stats["by_severity"] = severity_counts

            # Calculate resolution times if requested
            if include_resolution_times:
                resolved_incidents = [i for i in incidents if i.resolved_at]
                resolution_times = []

                for incident in resolved_incidents:
                    if incident.resolved_at and incident.created_at:
                        delta = incident.resolved_at - incident.created_at
                        hours = delta.total_seconds() / 3600
                        resolution_times.append(hours)

                if resolution_times:
                    stats["resolution_times"] = {
                        "average_hours": sum(resolution_times) / len(resolution_times),
                        "min_hours": min(resolution_times),
                        "max_hours": max(resolution_times),
                        "count": len(resolution_times),
                    }
                else:
                    stats["resolution_times"] = {
                        "average_hours": 0,
                        "min_hours": 0,
                        "max_hours": 0,
                        "count": 0,
                    }

            # Prepare incident data for the AI
            incidents_data = []
            for incident in incidents:
                incident_data = {
                    "id": incident.id,
                    "title": incident.title,
                    "description": (
                        incident.description[:200] + "..."
                        if incident.description and len(incident.description) > 200
                        else incident.description
                    ),
                    "severity": incident.severity,
                    "source": incident.source,
                    "status": incident.status,
                    "created_at": incident.created_at.isoformat(),
                    "resolved_at": (
                        incident.resolved_at.isoformat()
                        if incident.resolved_at
                        else None
                    ),
                }
                incidents_data.append(incident_data)

            # Generate the report using AI
            try:
                # Use specialized model for report generation
                model = (
                    settings.ai_report_generation_model
                    if settings.ai_allow_advanced_models
                    else settings.openai_completion_model
                )

                logger.info(f"Using AI model {model} for summary report generation")

                # Build the system prompt based on the requested format
                system_prompt = "You are a system incident report specialist."
                if format == ReportFormat.MARKDOWN:
                    system_prompt += " Generate reports in well-formatted Markdown with proper headings, lists, and formatting."
                elif format == ReportFormat.HTML:
                    system_prompt += " Generate reports in clean HTML with proper headings, lists, and minimal but effective styling."

                # Prepare the AI prompt
                prompt = f"""
                Generate a summary report for {len(incidents)} incidents that occurred in the last {time_range_days} days.
                
                Statistics:
                {json.dumps(stats, indent=2)}
                
                Incident data:
                {json.dumps(incidents_data[:30], indent=2)}  # Limit to 30 incidents for prompt size
                
                The report should include:
                1. Executive summary covering the {time_range_days}-day period
                2. Key metrics and trends
                """

                if include_severity_distribution:
                    prompt += """
                3. Severity distribution analysis
                """

                if include_resolution_times:
                    prompt += """
                4. Resolution time analysis
                """

                if include_recommendations:
                    prompt += """
                5. Recommendations for improving incident management
                """

                # Add format-specific instructions
                if format == ReportFormat.JSON:
                    prompt += """
                    Return your report as a JSON object with the following structure:
                    {
                        "executive_summary": "string",
                        "key_metrics": "string",
                    """

                    if include_severity_distribution:
                        prompt += """
                        "severity_analysis": "string",
                        """

                    if include_resolution_times:
                        prompt += """
                        "resolution_time_analysis": "string",
                        """

                    if include_recommendations:
                        prompt += """
                        "recommendations": ["string"]
                        """

                    prompt += """
                    }
                    """
                elif format == ReportFormat.MARKDOWN:
                    prompt += """
                    Format your report as clean, well-structured Markdown with proper headings, lists, and emphasis.
                    Include tables where appropriate for numerical data.
                    """
                elif format == ReportFormat.HTML:
                    prompt += """
                    Format your report as clean HTML with proper headings, paragraphs, lists, and tables.
                    Add minimal inline CSS for readability.
                    """

                # Set the response format based on the requested output format
                response_format = (
                    {"type": "json_object"} if format == ReportFormat.JSON else None
                )

                # Get the appropriate temperature - use a slightly higher temperature for narrative formats
                temperature = settings.ai_default_temperature
                if format in [ReportFormat.MARKDOWN, ReportFormat.HTML]:
                    temperature = min(settings.ai_default_temperature + 0.1, 0.7)

                # Generate the report
                response = await self.ai_engine.llm.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    response_format=response_format,
                )

                # Get the report content
                report_content = response.choices[0].message.content

                # Process the report content based on format
                if format == ReportFormat.JSON:
                    report_data = json.loads(report_content)
                    # Add metadata
                    report_data["metadata"] = {
                        "time_range_days": time_range_days,
                        "generated_at": datetime.utcnow().isoformat(),
                        "model_used": model,
                        "format": format,
                        "total_incidents": len(incidents),
                    }
                    return report_data
                else:
                    # For Markdown and HTML, return the raw content with metadata
                    return {
                        "content": report_content,
                        "metadata": {
                            "time_range_days": time_range_days,
                            "generated_at": datetime.utcnow().isoformat(),
                            "model_used": model,
                            "format": format,
                            "total_incidents": len(incidents),
                        },
                    }

            except Exception as e:
                logger.error(f"Error generating summary report: {e}")
                return {
                    "error": f"Failed to generate report: {str(e)}",
                    "generated_at": datetime.utcnow().isoformat(),
                    "time_range_days": time_range_days,
                }

    async def generate_operational_recommendations(
        self, time_range_days: int = 30, format: str = ReportFormat.MARKDOWN
    ) -> Dict[str, Any]:
        """
        Generate operational recommendations based on incident history.

        :param time_range_days: Number of days of incident data to analyze
        :param format: Output format (json, markdown, html)
        :return: Recommendations report data
        """
        if not self.ai_engine:
            logger.warning("AI Engine not available for operational recommendations")
            return {"error": "AI Engine not available"}

        async with self.db_session_factory() as session:
            # Calculate time range
            start_date = datetime.utcnow() - timedelta(days=time_range_days)

            # Get incidents in the date range
            query = (
                select(Incident)
                .where(Incident.created_at >= start_date)
                .order_by(Incident.created_at.desc())
            )

            result = await session.execute(query)
            incidents = result.scalars().all()

            if not incidents:
                return {
                    "error": "No incidents found in the specified time range",
                    "time_range_days": time_range_days,
                    "generated_at": datetime.utcnow().isoformat(),
                }

            # Count incidents by source
            source_counts = {}
            for incident in incidents:
                source = incident.source
                source_counts[source] = source_counts.get(source, 0) + 1

            # Get resolutions for analysis
            resolutions_query = (
                select(Resolution, Incident)
                .join(Incident, Resolution.incident_id == Incident.id)
                .where(Incident.created_at >= start_date)
            )

            resolutions_result = await session.execute(resolutions_query)
            resolutions_with_incidents = resolutions_result.all()

            # Prepare data for AI analysis
            incidents_data = []
            for incident in incidents:
                incident_data = {
                    "id": incident.id,
                    "title": incident.title,
                    "description": (
                        incident.description[:200] + "..."
                        if incident.description and len(incident.description) > 200
                        else incident.description
                    ),
                    "severity": incident.severity,
                    "source": incident.source,
                    "status": incident.status,
                    "created_at": incident.created_at.isoformat(),
                    "resolved_at": (
                        incident.resolved_at.isoformat()
                        if incident.resolved_at
                        else None
                    ),
                }
                incidents_data.append(incident_data)

            # Prepare resolutions data
            resolutions_data = []
            for resolution, incident in resolutions_with_incidents:
                if resolution.suggestion:
                    resolution_data = {
                        "incident_id": incident.id,
                        "incident_title": incident.title,
                        "incident_source": incident.source,
                        "incident_severity": incident.severity,
                        "suggestion": (
                            resolution.suggestion[:300] + "..."
                            if len(resolution.suggestion) > 300
                            else resolution.suggestion
                        ),
                        "is_applied": resolution.is_applied,
                    }
                    resolutions_data.append(resolution_data)

            # Generate recommendations using AI
            try:
                # Use specialized model for recommendations
                model = (
                    settings.ai_resolution_generation_model
                    if settings.ai_allow_advanced_models
                    else settings.openai_completion_model
                )

                logger.info(f"Using AI model {model} for operational recommendations")

                # Build the system prompt based on the requested format
                system_prompt = "You are an expert in IT operations and incident management, specializing in providing actionable recommendations for improving system reliability."
                if format == ReportFormat.MARKDOWN:
                    system_prompt += " Generate reports in well-formatted Markdown with proper headings, lists, and formatting."
                elif format == ReportFormat.HTML:
                    system_prompt += " Generate reports in clean HTML with proper headings, lists, and minimal but effective styling."

                # Prepare the AI prompt
                incident_sources = list(source_counts.keys())

                prompt = f"""
                Based on the incident data from the last {time_range_days} days, generate operational recommendations 
                for improving system reliability and reducing incidents.
                
                Key metrics:
                - Total incidents: {len(incidents)}
                - Incident sources: {', '.join(incident_sources)}
                - Source distribution: {json.dumps(source_counts)}
                
                Sample incidents:
                {json.dumps(incidents_data[:20], indent=2)}  # Limit to 20 incidents
                
                Resolution samples:
                {json.dumps(resolutions_data[:10], indent=2)}  # Limit to 10 resolutions
                
                Please provide:
                1. High-priority operational recommendations
                2. System-specific recommendations for each source system
                3. Process improvement recommendations
                4. Monitoring and alerting recommendations
                5. Training recommendations
                6. Long-term strategic recommendations
                
                For each recommendation, explain the rationale and expected benefits.
                """

                # Add format-specific instructions
                if format == ReportFormat.JSON:
                    prompt += """
                    Return your recommendations as a JSON object with the following structure:
                    {
                        "high_priority": [{"recommendation": "string", "rationale": "string", "benefits": "string"}],
                        "system_specific": {"system_name": [{"recommendation": "string", "rationale": "string"}]},
                        "process_improvements": [{"recommendation": "string", "rationale": "string"}],
                        "monitoring": [{"recommendation": "string", "rationale": "string"}],
                        "training": [{"recommendation": "string", "rationale": "string"}],
                        "strategic": [{"recommendation": "string", "rationale": "string"}]
                    }
                    """
                elif format == ReportFormat.MARKDOWN:
                    prompt += """
                    Format your recommendations as clean, well-structured Markdown with proper headings, lists, and emphasis.
                    """
                elif format == ReportFormat.HTML:
                    prompt += """
                    Format your recommendations as clean HTML with proper headings, paragraphs, and lists.
                    """

                # Set the response format based on the requested output format
                response_format = (
                    {"type": "json_object"} if format == ReportFormat.JSON else None
                )

                # Generate the recommendations
                response = await self.ai_engine.llm.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=settings.ai_default_temperature,
                    response_format=response_format,
                )

                # Get the recommendations content
                recommendations_content = response.choices[0].message.content

                # Process based on format
                if format == ReportFormat.JSON:
                    recommendations_data = json.loads(recommendations_content)
                    # Add metadata
                    recommendations_data["metadata"] = {
                        "time_range_days": time_range_days,
                        "generated_at": datetime.utcnow().isoformat(),
                        "model_used": model,
                        "format": format,
                        "analyzed_incidents": len(incidents),
                    }
                    return recommendations_data
                else:
                    # For Markdown and HTML, return the raw content with metadata
                    return {
                        "content": recommendations_content,
                        "metadata": {
                            "time_range_days": time_range_days,
                            "generated_at": datetime.utcnow().isoformat(),
                            "model_used": model,
                            "format": format,
                            "analyzed_incidents": len(incidents),
                        },
                    }

            except Exception as e:
                logger.error(f"Error generating operational recommendations: {e}")
                return {
                    "error": f"Failed to generate recommendations: {str(e)}",
                    "generated_at": datetime.utcnow().isoformat(),
                    "time_range_days": time_range_days,
                }

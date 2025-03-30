"""Incident analyzer for AI-powered incident analysis."""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, text, desc

from system_guardian.db.models.incidents import Incident, Event, Resolution
from system_guardian.services.ai.engine import AIEngine
from system_guardian.settings import settings


class IncidentAnalyzer:
    """Analyzes incidents for patterns and insights."""

    def __init__(self, db_session_factory, ai_engine: Optional[AIEngine] = None):
        """
        Initialize the incident analyzer.

        :param db_session_factory: Factory for creating database sessions
        :param ai_engine: Optional AIEngine instance for advanced analysis
        """
        self.db_session_factory = db_session_factory
        self.ai_engine = ai_engine

    async def analyze_resolution_time(self, time_range: str = "7d") -> Dict:
        """
        Analyze incident resolution times.

        :param time_range: Time range for analysis (e.g., "7d" for 7 days)
        :return: Resolution time statistics
        """
        async with self.db_session_factory() as session:
            # Calculate time range
            days = int(time_range.replace("d", ""))
            start_date = datetime.utcnow() - timedelta(days=days)

            # Query resolved incidents
            query = select(Incident).where(
                and_(
                    Incident.created_at >= start_date, Incident.resolved_at.is_not(None)
                )
            )

            result = await session.execute(query)
            incidents = result.scalars().all()

            # Calculate statistics
            if not incidents:
                return {"average_hours": 0, "min_hours": 0, "max_hours": 0, "count": 0}

            resolution_times = []
            for incident in incidents:
                if incident.resolved_at and incident.created_at:
                    delta = incident.resolved_at - incident.created_at
                    hours = delta.total_seconds() / 3600
                    resolution_times.append(hours)

            if not resolution_times:
                return {"average_hours": 0, "min_hours": 0, "max_hours": 0, "count": 0}

            return {
                "average_hours": sum(resolution_times) / len(resolution_times),
                "min_hours": min(resolution_times),
                "max_hours": max(resolution_times),
                "count": len(resolution_times),
                "by_severity": await self._resolution_time_by_severity(
                    session, start_date
                ),
            }

    async def _resolution_time_by_severity(self, session, start_date) -> Dict:
        """Calculate resolution time by severity level."""
        severities = ["low", "medium", "high", "critical"]
        result = {}

        for severity in severities:
            query = select(Incident).where(
                and_(
                    Incident.created_at >= start_date,
                    Incident.resolved_at.is_not(None),
                    Incident.severity == severity,
                )
            )

            incidents = await session.execute(query)
            incidents = incidents.scalars().all()

            if not incidents:
                result[severity] = {"average_hours": 0, "count": 0}
                continue

            resolution_times = []
            for incident in incidents:
                if incident.resolved_at and incident.created_at:
                    delta = incident.resolved_at - incident.created_at
                    hours = delta.total_seconds() / 3600
                    resolution_times.append(hours)

            if resolution_times:
                result[severity] = {
                    "average_hours": sum(resolution_times) / len(resolution_times),
                    "count": len(resolution_times),
                }
            else:
                result[severity] = {"average_hours": 0, "count": 0}

        return result

    async def identify_common_failures(self, limit: int = 10) -> List[Dict]:
        """
        Identify most common failure patterns.

        :param limit: Maximum number of patterns to return
        :return: List of common failure patterns
        """
        async with self.db_session_factory() as session:
            # Count incidents by source
            sources_query = (
                select(Incident.source, func.count().label("count"))
                .group_by(Incident.source)
                .order_by(desc("count"))
                .limit(limit)
            )

            sources_result = await session.execute(sources_query)
            sources = [{"source": src, "count": count} for src, count in sources_result]

            # If we have AI Engine, enhance the analysis
            if self.ai_engine:
                return await self._ai_enhanced_failure_analysis(session, sources, limit)

            return sources

    async def _ai_enhanced_failure_analysis(
        self, session, basic_sources, limit
    ) -> List[Dict]:
        """
        Use AI to enhance failure pattern analysis.

        :param session: Database session
        :param basic_sources: Basic source count data
        :param limit: Maximum number of patterns to return
        :return: Enhanced failure pattern analysis
        """
        try:
            # Get recent incidents for analysis
            query = (
                select(Incident)
                .order_by(Incident.created_at.desc())
                .limit(50)  # Get a reasonable sample for analysis
            )

            result = await session.execute(query)
            incidents = result.scalars().all()

            incidents_data = []
            for incident in incidents:
                incident_data = {
                    "id": incident.id,
                    "title": incident.title,
                    "description": incident.description,
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

            # Prepare AI prompt
            prompt = f"""
            Analyze the following {len(incidents_data)} incidents and identify common failure patterns.
            Focus on root causes, affected systems, and failure modes.
            
            Incident data:
            {json.dumps(incidents_data, indent=2)}
            
            Please identify the top {limit} failure patterns based on this data.
            For each pattern, provide:
            1. A name/category for the pattern
            2. The likely root cause or trigger
            3. The frequency of occurrence (how many incidents fit this pattern)
            4. The average severity
            5. Recommended preventive measures
            
            Return the results as a JSON array in the following format:
            [
                {{
                    "pattern": "string",
                    "root_cause": "string",
                    "frequency": number,
                    "avg_severity": "string",
                    "preventive_measures": "string"
                }}
            ]
            """

            # Use AI model for trend analysis
            model = (
                settings.ai_trend_analysis_model
                if settings.ai_allow_advanced_models
                else settings.openai_completion_model
            )

            logger.info(f"Using AI model {model} for failure pattern analysis")
            response = await self.ai_engine.llm.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert incident analyzer specializing in identifying patterns and root causes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.ai_default_temperature,
                response_format={"type": "json_object"},
            )

            # Parse the response
            response_text = response.choices[0].message.content
            patterns = json.loads(response_text)

            # Return the enhanced analysis
            return patterns.get("patterns", patterns)

        except Exception as e:
            logger.error(f"Error in AI-enhanced failure analysis: {e}")
            # Fall back to basic analysis
            return basic_sources

    async def calculate_ai_effectiveness(self) -> Dict:
        """
        Calculate how effective AI suggestions have been.

        :return: AI effectiveness metrics
        """
        async with self.db_session_factory() as session:
            # Get all resolutions with AI suggestions
            query = select(Resolution).where(Resolution.suggestion.is_not(None))

            result = await session.execute(query)
            resolutions = result.scalars().all()

            if not resolutions:
                return {
                    "total_suggestions": 0,
                    "applied_suggestions": 0,
                    "effectiveness_rate": 0,
                }

            # Count applied suggestions
            applied = sum(1 for r in resolutions if r.is_applied)

            return {
                "total_suggestions": len(resolutions),
                "applied_suggestions": applied,
                "effectiveness_rate": applied / len(resolutions) if resolutions else 0,
                "average_confidence": (
                    sum(r.confidence for r in resolutions) / len(resolutions)
                    if resolutions
                    else 0
                ),
            }

    async def generate_trend_report(self, days: int = 30) -> Dict:
        """
        Generate a comprehensive trend report using AI analysis.

        :param days: Number of days to analyze
        :return: Trend report data
        """
        if not self.ai_engine:
            logger.warning("AI Engine not available for trend report generation")
            return {"error": "AI Engine not available"}

        async with self.db_session_factory() as session:
            # Calculate time range
            start_date = datetime.utcnow() - timedelta(days=days)

            # Get incidents in the date range
            query = (
                select(Incident)
                .where(Incident.created_at >= start_date)
                .order_by(Incident.created_at.desc())
            )

            result = await session.execute(query)
            incidents = result.scalars().all()

            if not incidents:
                return {"message": "No incidents found in the specified time range"}

            # Prepare incident data
            incidents_data = []
            for incident in incidents:
                incident_data = {
                    "id": incident.id,
                    "title": incident.title,
                    "description": incident.description or "",
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

            # Get basic statistics
            stats = await self._calculate_incident_statistics(session, start_date)

            # Generate trend report using AI
            try:
                # Use AI model for trend analysis
                model = (
                    settings.ai_trend_analysis_model
                    if settings.ai_allow_advanced_models
                    else settings.openai_completion_model
                )

                logger.info(f"Using AI model {model} for trend report generation")

                # Prepare AI prompt
                prompt = f"""
                Generate a comprehensive trend report for the following {len(incidents_data)} incidents that occurred in the last {days} days.
                
                Basic Statistics:
                {json.dumps(stats, indent=2)}
                
                Incident data:
                {json.dumps(incidents_data[:30], indent=2)}  # Limit to 30 incidents for prompt size
                
                Please include in your analysis:
                1. Overall incident frequency trends
                2. Severity distribution trends
                3. Most common sources of incidents
                4. Average resolution times
                5. Key emerging patterns or concerns
                6. Recommendations for reducing incidents
                
                Return the results as a JSON object with the following structure:
                {{
                    "summary": "string",
                    "frequency_trends": "string",
                    "severity_trends": "string",
                    "common_sources": "string",
                    "resolution_time_trends": "string",
                    "emerging_patterns": "string",
                    "recommendations": "string"
                }}
                """

                response = await self.ai_engine.llm.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert incident trend analyst providing actionable insights.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=settings.ai_default_temperature,
                    response_format={"type": "json_object"},
                )

                # Parse the response
                response_text = response.choices[0].message.content
                trend_report = json.loads(response_text)

                # Combine with statistics
                return {
                    "statistics": stats,
                    "analysis": trend_report,
                    "generated_at": datetime.utcnow().isoformat(),
                    "period_days": days,
                    "total_incidents": len(incidents),
                }

            except Exception as e:
                logger.error(f"Error generating trend report: {e}")
                return {
                    "statistics": stats,
                    "error": f"Failed to generate AI analysis: {str(e)}",
                    "generated_at": datetime.utcnow().isoformat(),
                    "period_days": days,
                    "total_incidents": len(incidents),
                }

    async def _calculate_incident_statistics(self, session, start_date) -> Dict:
        """Calculate basic incident statistics."""
        # Count by severity
        severity_query = (
            select(Incident.severity, func.count().label("count"))
            .where(Incident.created_at >= start_date)
            .group_by(Incident.severity)
        )

        severity_result = await session.execute(severity_query)
        severity_counts = {severity: count for severity, count in severity_result}

        # Count by source
        source_query = (
            select(Incident.source, func.count().label("count"))
            .where(Incident.created_at >= start_date)
            .group_by(Incident.source)
            .order_by(desc("count"))
        )

        source_result = await session.execute(source_query)
        source_counts = {source: count for source, count in source_result}

        # Count by status
        status_query = (
            select(Incident.status, func.count().label("count"))
            .where(Incident.created_at >= start_date)
            .group_by(Incident.status)
        )

        status_result = await session.execute(status_query)
        status_counts = {status: count for status, count in status_result}

        # Calculate total
        total_query = (
            select(func.count())
            .select_from(Incident)
            .where(Incident.created_at >= start_date)
        )

        total_result = await session.execute(total_query)
        total_count = total_result.scalar_one()

        return {
            "total_incidents": total_count,
            "by_severity": severity_counts,
            "by_source": source_counts,
            "by_status": status_counts,
        }

    async def perform_root_cause_analysis(self, incident_id: int) -> Dict:
        """
        Perform an AI-powered root cause analysis for a specific incident.

        :param incident_id: ID of the incident to analyze
        :return: Root cause analysis results
        """
        if not self.ai_engine:
            logger.warning("AI Engine not available for root cause analysis")
            return {"error": "AI Engine not available"}

        async with self.db_session_factory() as session:
            # Get the incident
            incident = await session.get(Incident, incident_id)
            if not incident:
                return {"error": f"Incident with ID {incident_id} not found"}

            # Get related events
            events_query = (
                select(Event)
                .where(Event.related_incident_id == incident_id)
                .order_by(Event.created_at)
            )

            events_result = await session.execute(events_query)
            events = events_result.scalars().all()

            # Get resolution if available
            resolution_query = select(Resolution).where(
                Resolution.incident_id == incident_id
            )

            resolution_result = await session.execute(resolution_query)
            resolution = resolution_result.scalars().first()

            # Prepare data for analysis
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
                "resolution": resolution.suggestion if resolution else None,
            }

            events_data = []
            for event in events[
                :10
            ]:  # Limit to first 10 events to keep prompt size reasonable
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

            # Find similar past incidents
            similar_incidents = []
            if self.ai_engine:
                try:
                    incident_text = f"{incident.title}\n{incident.description or ''}"
                    similar = await self.ai_engine.find_similar_incidents(
                        incident_text=incident_text, limit=3, min_similarity_score=0.5
                    )

                    # Filter out the current incident
                    similar_incidents = [
                        s
                        for s in similar
                        if str(s.get("incident_id")) != str(incident_id)
                    ]
                except Exception as e:
                    logger.error(f"Error finding similar incidents: {e}")

            # Use AI for root cause analysis
            try:
                # Use specialized model for root cause analysis
                model = (
                    settings.ai_root_cause_analysis_model
                    if settings.ai_allow_advanced_models
                    else settings.openai_completion_model
                )

                logger.info(f"Using AI model {model} for root cause analysis")

                # Prepare AI prompt
                prompt = f"""
                Perform a detailed root cause analysis for the following incident:
                
                Incident details:
                {json.dumps(incident_data, indent=2)}
                
                Related events ({len(events_data)} events):
                {json.dumps(events_data, indent=2)}
                
                {"Similar past incidents:" if similar_incidents else ""}
                {json.dumps(similar_incidents, indent=2) if similar_incidents else ""}
                
                Please provide a comprehensive root cause analysis that includes:
                1. Identification of primary and contributing factors
                2. The probable sequence of events that led to the incident
                3. System or process weaknesses that were exposed
                4. Whether this appears to be an isolated incident or part of a pattern
                5. Specific recommendations to prevent similar incidents in the future
                
                Return your analysis as a JSON object with the following structure:
                {{
                    "primary_cause": "string",
                    "contributing_factors": ["string"],
                    "sequence_of_events": "string",
                    "system_weaknesses": ["string"],
                    "is_pattern": boolean,
                    "confidence": number,
                    "recommendations": ["string"]
                }}
                """

                response = await self.ai_engine.llm.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in root cause analysis for IT incidents. Provide detailed technical analysis and actionable recommendations.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=settings.ai_default_temperature,
                    response_format={"type": "json_object"},
                )

                # Parse the response
                response_text = response.choices[0].message.content
                analysis = json.loads(response_text)

                # Return the analysis with metadata
                return {
                    "incident_id": incident_id,
                    "analysis": analysis,
                    "generated_at": datetime.utcnow().isoformat(),
                    "model_used": model,
                }

            except Exception as e:
                logger.error(f"Error performing root cause analysis: {e}")
                return {
                    "incident_id": incident_id,
                    "error": f"Failed to generate analysis: {str(e)}",
                    "generated_at": datetime.utcnow().isoformat(),
                }

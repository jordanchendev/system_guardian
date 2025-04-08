"""AI Engine API views."""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Body, Path
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from system_guardian.services.vector_db.qdrant_client import QdrantClient
from system_guardian.services.vector_db.dependencies import get_qdrant_dependency
from system_guardian.services.ai.engine import AIEngine
from system_guardian.services.ai.resolution_generator import ResolutionGenerator
from system_guardian.services.ai.report_generator import ReportGenerator, ReportFormat
from system_guardian.services.ai.incident_analyzer import IncidentAnalyzer
from system_guardian.db.models.incidents import Incident, Resolution
from system_guardian.db.dependencies import get_db_session
from sqlalchemy.future import select
from system_guardian.settings import settings
from system_guardian.services.ai.incident_similarity import IncidentSimilarityService

from .schema import (
    GenerateResolutionRequest,
    GenerateResolutionResponse,
    ApplyResolutionRequest,
    ApplyResolutionResponse,
    ResolutionFeedbackRequest,
    ResolutionFeedbackResponse,
    RelatedIncidentsRequest,
    RelatedIncidentsResponse,
    RelatedIncidentItem,
    InsightItem,
    AIMetricsResponse,
    GenerateIncidentReportRequest,
    GenerateIncidentReportResponse,
    GenerateSummaryReportRequest,
    GenerateSummaryReportResponse,
    GenerateRecommendationsRequest,
    GenerateRecommendationsResponse,
    # Analytics schemas
    ResolutionTimeResponse,
    CommonFailuresResponse,
    AIEffectivenessResponse,
    TrendReportResponse,
    TrendReportRequest,
    RootCauseAnalysisRequest,
    RootCauseAnalysisResponse,
)


router = APIRouter()


@router.post("/generate-resolution", response_model=GenerateResolutionResponse)
async def generate_resolution(
    request: GenerateResolutionRequest = Body(...),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
    db_session: AsyncSession = Depends(get_db_session),
) -> GenerateResolutionResponse:
    """
    Generate a resolution suggestion for an incident using AI.

    :param request: Request containing incident ID and configuration
    :param qdrant_client: Qdrant client dependency
    :param db_session: Database session
    :returns: Generated resolution suggestion
    """
    # Initialize services
    logger.info(f"Received resolution request for incident ID: {request.incident_id}")

    # Initialize AI engine with configuration
    ai_engine = AIEngine(
        vector_db_client=qdrant_client,
        llm_model=request.model or settings.openai_completion_model,
        enable_metrics=True,
    )

    # Initialize resolution generator
    resolution_generator = ResolutionGenerator(ai_engine=ai_engine)

    try:
        # Check if incident exists
        incident_query = select(Incident).where(Incident.id == request.incident_id)
        result = await db_session.execute(incident_query)
        incident = result.scalars().first()

        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident with ID {request.incident_id} not found",
            )

        # Generate resolution using ResolutionGenerator with the session
        logger.debug(
            f"Calling ResolutionGenerator for incident ID: {request.incident_id}, model: {request.model}"
        )
        resolution_data = await resolution_generator.generate_resolution(
            incident_id=request.incident_id,
            session=db_session,
            force_regenerate=request.force_regenerate,
            model=request.model,
            temperature=request.temperature,
        )

        if not resolution_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate resolution for incident #{request.incident_id}",
            )

        # Ensure datetime is converted to string
        if isinstance(resolution_data["generated_at"], datetime):
            resolution_data["generated_at"] = resolution_data[
                "generated_at"
            ].isoformat()

        # Transform the result into the expected response format
        logger.info(
            f"Successfully generated resolution for incident ID: {request.incident_id} with confidence: {resolution_data['confidence']:.2f}"
        )
        return GenerateResolutionResponse(
            id=resolution_data["resolution_id"],
            incident_id=resolution_data["incident_id"],
            suggestion=resolution_data["resolution_text"],
            confidence=resolution_data["confidence"],
            is_applied=(
                resolution_data["is_applied"]
                if "is_applied" in resolution_data
                else False
            ),
            generated_at=resolution_data["generated_at"],
            feedback_score=(
                resolution_data["feedback_score"]
                if "feedback_score" in resolution_data
                else None
            ),
        )
    except ValueError as e:
        # Handle specific validation errors
        logger.error(
            f"Validation error for incident ID {request.incident_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        logger.error(f"HTTP exception occurred for incident ID {request.incident_id}")
        raise
    except Exception as e:
        logger.exception(
            f"Failed to generate resolution for incident ID {request.incident_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate resolution: {str(e)}",
        )


@router.post(
    "/resolutions/{resolution_id}/apply", response_model=ApplyResolutionResponse
)
async def apply_resolution(
    resolution_id: int = Path(..., description="ID of the resolution to apply"),
    request: Optional[ApplyResolutionRequest] = Body(None),
    db_session: AsyncSession = Depends(get_db_session),
) -> ApplyResolutionResponse:
    """
    Mark a resolution as applied.

    :param resolution_id: ID of the resolution
    :param request: Request containing additional notes
    :param db_session: Database session
    :returns: Updated resolution status
    """
    logger.info(f"Received apply resolution request for resolution ID: {resolution_id}")

    try:
        # Get the resolution from the database
        stmt = select(Resolution).where(Resolution.id == resolution_id)
        result = await db_session.execute(stmt)
        resolution = result.scalar_one_or_none()

        if not resolution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resolution with ID {resolution_id} not found",
            )

        # Update the resolution
        resolution.is_applied = True
        applied_at = datetime.utcnow()

        # Commit the changes
        await db_session.commit()

        # Return the response
        return ApplyResolutionResponse(
            resolution_id=resolution.id,
            incident_id=resolution.incident_id,
            is_applied=resolution.is_applied,
            applied_at=applied_at.isoformat(),
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Failed to apply resolution {resolution_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply resolution: {str(e)}",
        )


@router.post(
    "/resolutions/{resolution_id}/feedback", response_model=ResolutionFeedbackResponse
)
async def provide_resolution_feedback(
    resolution_id: int = Path(..., description="ID of the resolution"),
    request: ResolutionFeedbackRequest = Body(...),
    db_session: AsyncSession = Depends(get_db_session),
) -> ResolutionFeedbackResponse:
    """
    Provide feedback on a resolution.

    :param resolution_id: ID of the resolution
    :param request: Feedback request
    :param db_session: Database session
    :returns: Feedback submission status
    """
    logger.info(f"Received feedback for resolution ID: {resolution_id}")

    try:
        # Get the resolution from the database
        stmt = select(Resolution).where(Resolution.id == resolution_id)
        result = await db_session.execute(stmt)
        resolution = result.scalar_one_or_none()

        if not resolution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resolution with ID {resolution_id} not found",
            )

        # Update the feedback score
        resolution.feedback_score = request.feedback_score

        # Commit the changes
        await db_session.commit()

        # Return the response
        return ResolutionFeedbackResponse(
            resolution_id=resolution.id,
            incident_id=resolution.incident_id,
            feedback_score=resolution.feedback_score,
            success=True,
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(
            f"Failed to submit feedback for resolution {resolution_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}",
        )


@router.get(
    "/resolutions/incident/{incident_id}",
    response_model=List[GenerateResolutionResponse],
)
async def get_incident_resolutions(
    incident_id: int = Path(..., description="ID of the incident"),
    db_session: AsyncSession = Depends(get_db_session),
) -> List[GenerateResolutionResponse]:
    """
    Get all resolutions for an incident.

    :param incident_id: ID of the incident
    :param db_session: Database session
    :returns: List of resolutions for the incident
    """
    logger.info(f"Received request for resolutions of incident ID: {incident_id}")

    try:
        # Check if incident exists
        incident_query = select(Incident).where(Incident.id == incident_id)
        result = await db_session.execute(incident_query)
        incident = result.scalars().first()

        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident with ID {incident_id} not found",
            )

        # Get all resolutions for the incident
        stmt = select(Resolution).where(Resolution.incident_id == incident_id)
        result = await db_session.execute(stmt)
        resolutions = result.scalars().all()

        # Convert to response format
        response_items = []
        for res in resolutions:
            generated_at = (
                res.generated_at.isoformat()
                if isinstance(res.generated_at, datetime)
                else res.generated_at
            )
            response_items.append(
                GenerateResolutionResponse(
                    id=res.id,
                    incident_id=res.incident_id,
                    suggestion=res.suggestion,
                    confidence=res.confidence,
                    is_applied=res.is_applied,
                    generated_at=generated_at,
                    feedback_score=res.feedback_score,
                )
            )

        return response_items
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(
            f"Failed to get resolutions for incident {incident_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resolutions: {str(e)}",
        )


@router.post("/related-incidents", response_model=RelatedIncidentsResponse)
async def find_related_incidents(
    request: RelatedIncidentsRequest = Body(...),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
    db_session: AsyncSession = Depends(get_db_session),
) -> RelatedIncidentsResponse:
    """
    Find similar past incidents and provide insights based on them.

    :param request: Parameters for the related incidents search
    :param qdrant_client: Qdrant client dependency
    :param db_session: Database session
    :returns: Related incidents and insights derived from them
    """
    # Initialize services
    query_identifier = request.incident_id or "custom query"
    logger.info(f"Received related incidents request for: {query_identifier}")

    # Initialize AI engine and similarity service
    ai_engine = AIEngine(vector_db_client=qdrant_client, enable_metrics=True)
    similarity_service = IncidentSimilarityService(
        qdrant_client=qdrant_client, ai_engine=ai_engine
    )

    try:
        # Use the IncidentSimilarityService with the provided session
        logger.debug(
            f"Calling IncidentSimilarityService for related incidents: limit={request.limit}, include_resolved={request.include_resolved}"
        )
        result = await similarity_service.find_related_incidents(
            db_session=db_session,
            incident_id=request.incident_id,
            query_text=request.query_text,
            limit=request.limit,
            include_resolved=request.include_resolved,
            min_similarity_score=request.min_similarity_score,
        )

        # Log the results
        logger.info(
            f"Found {len(result['incidents'])} related incidents and {len(result['insights'])} insights"
        )

        # Convert the result to the API response format
        return RelatedIncidentsResponse(
            incidents=[
                RelatedIncidentItem(**incident) for incident in result["incidents"]
            ],
            insights=[InsightItem(**insight) for insight in result["insights"]],
            current_incident=(
                RelatedIncidentItem(**result["current_incident"])
                if result["current_incident"]
                else None
            ),
        )
    except ValueError as e:
        # Handle the specific exceptions from IncidentSimilarityService
        logger.error(f"Validation error for related incidents request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        # Handle any other unexpected errors
        logger.exception(f"Error finding related incidents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finding related incidents: {str(e)}",
        )


@router.get("/metrics", response_model=AIMetricsResponse)
async def get_ai_metrics(
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
) -> AIMetricsResponse:
    """
    Get performance metrics for the AI engine.

    :param qdrant_client: Qdrant client dependency
    :returns: Performance metrics dictionary
    """
    logger.info("Received metrics request")

    # Initialize services

    # Initialize AI engine
    ai_engine = AIEngine(vector_db_client=qdrant_client, enable_metrics=True)

    # Return the metrics
    metrics = ai_engine.get_metrics()
    logger.debug(f"Returning metrics: {metrics}")
    return AIMetricsResponse(metrics=metrics)


@router.post("/generate-incident-report", response_model=GenerateIncidentReportResponse)
async def generate_incident_report(
    request: GenerateIncidentReportRequest = Body(...),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
    db_session: AsyncSession = Depends(get_db_session),
) -> GenerateIncidentReportResponse:
    """
    Generate a detailed incident report using AI.

    :param request: Request parameters for incident report generation
    :param qdrant_client: Qdrant client dependency
    :param db_session: Database session
    :returns: Generated incident report
    """
    # Initialize services
    logger.info(
        f"Received incident report request for incident ID: {request.incident_id}"
    )

    # Initialize AI engine
    ai_engine = AIEngine(
        vector_db_client=qdrant_client,
        llm_model=(
            settings.ai_report_generation_model
            if settings.ai_allow_advanced_models
            else settings.openai_completion_model
        ),
        enable_metrics=True,
    )

    # Initialize report generator with AI engine
    report_generator = ReportGenerator(
        db_session_factory=lambda: db_session, ai_engine=ai_engine
    )

    try:
        # Check if incident exists
        incident_query = select(Incident).where(Incident.id == request.incident_id)
        result = await db_session.execute(incident_query)
        incident = result.scalars().first()

        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident with ID {request.incident_id} not found",
            )

        # Generate incident report
        logger.debug(
            f"Generating incident report for incident ID: {request.incident_id}, format: {request.format}"
        )
        report_result = await report_generator.generate_incident_report(
            incident_id=request.incident_id,
            include_events=request.include_events,
            include_resolution=request.include_resolution,
            format=request.format,
        )

        # Check for errors
        if "error" in report_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=report_result["error"],
            )

        # Format response based on report format
        if request.format == ReportFormat.JSON:
            return GenerateIncidentReportResponse(
                incident_id=request.incident_id,
                content=report_result,
                format=request.format,
                generated_at=report_result.get("metadata", {}).get(
                    "generated_at", datetime.utcnow().isoformat()
                ),
            )
        else:
            return GenerateIncidentReportResponse(
                incident_id=request.incident_id,
                content=report_result.get("content", ""),
                format=request.format,
                generated_at=report_result.get("metadata", {}).get(
                    "generated_at", datetime.utcnow().isoformat()
                ),
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(
            f"Failed to generate incident report for ID {request.incident_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate incident report: {str(e)}",
        )


@router.post("/generate-summary-report", response_model=GenerateSummaryReportResponse)
async def generate_summary_report(
    request: GenerateSummaryReportRequest = Body(...),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
    db_session: AsyncSession = Depends(get_db_session),
) -> GenerateSummaryReportResponse:
    """
    Generate a summary report for incidents over a time period.

    :param request: Request parameters for summary report generation
    :param qdrant_client: Qdrant client dependency
    :param db_session: Database session
    :returns: Generated summary report
    """
    # Initialize services
    logger.info(f"Received summary report request for {request.time_range_days} days")

    # Initialize AI engine
    ai_engine = AIEngine(
        vector_db_client=qdrant_client,
        llm_model=(
            settings.ai_report_generation_model
            if settings.ai_allow_advanced_models
            else settings.openai_completion_model
        ),
        enable_metrics=True,
    )

    # Initialize report generator with AI engine
    report_generator = ReportGenerator(
        db_session_factory=lambda: db_session, ai_engine=ai_engine
    )

    try:
        # Generate summary report
        logger.debug(
            f"Generating summary report for {request.time_range_days} days, format: {request.format}"
        )
        report_result = await report_generator.generate_summary_report(
            time_range_days=request.time_range_days,
            format=request.format,
            include_severity_distribution=request.include_severity_distribution,
            include_resolution_times=request.include_resolution_times,
            include_recommendations=request.include_recommendations,
        )

        # Check for errors
        if "error" in report_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=report_result["error"],
            )

        # Format response based on report format
        if request.format == ReportFormat.JSON:
            return GenerateSummaryReportResponse(
                content=report_result,
                format=request.format,
                time_range_days=request.time_range_days,
                generated_at=report_result.get("metadata", {}).get(
                    "generated_at", datetime.utcnow().isoformat()
                ),
            )
        else:
            return GenerateSummaryReportResponse(
                content=report_result.get("content", ""),
                format=request.format,
                time_range_days=request.time_range_days,
                generated_at=report_result.get("metadata", {}).get(
                    "generated_at", datetime.utcnow().isoformat()
                ),
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Failed to generate summary report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary report: {str(e)}",
        )


@router.post(
    "/generate-recommendations", response_model=GenerateRecommendationsResponse
)
async def generate_recommendations(
    request: GenerateRecommendationsRequest = Body(...),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
    db_session: AsyncSession = Depends(get_db_session),
) -> GenerateRecommendationsResponse:
    """
    Generate operational recommendations based on incident history.

    :param request: Request parameters for recommendations generation
    :param qdrant_client: Qdrant client dependency
    :param db_session: Database session
    :returns: Generated recommendations
    """
    # Initialize services
    logger.info(
        f"Received operational recommendations request for {request.time_range_days} days"
    )

    # Initialize AI engine with appropriate model
    ai_engine = AIEngine(
        vector_db_client=qdrant_client,
        llm_model=(
            settings.ai_resolution_generation_model
            if settings.ai_allow_advanced_models
            else settings.openai_completion_model
        ),
        enable_metrics=True,
    )

    # Initialize report generator with AI engine
    report_generator = ReportGenerator(
        db_session_factory=lambda: db_session, ai_engine=ai_engine
    )

    try:
        # Generate recommendations
        logger.debug(
            f"Generating operational recommendations for {request.time_range_days} days, format: {request.format}"
        )
        recommendations_result = (
            await report_generator.generate_operational_recommendations(
                time_range_days=request.time_range_days, format=request.format
            )
        )

        # Check for errors
        if "error" in recommendations_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=recommendations_result["error"],
            )

        # Format response based on report format
        if request.format == ReportFormat.JSON:
            return GenerateRecommendationsResponse(
                content=recommendations_result,
                format=request.format,
                time_range_days=request.time_range_days,
                generated_at=recommendations_result.get("metadata", {}).get(
                    "generated_at", datetime.utcnow().isoformat()
                ),
            )
        else:
            return GenerateRecommendationsResponse(
                content=recommendations_result.get("content", ""),
                format=request.format,
                time_range_days=request.time_range_days,
                generated_at=recommendations_result.get("metadata", {}).get(
                    "generated_at", datetime.utcnow().isoformat()
                ),
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Failed to generate operational recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate operational recommendations: {str(e)}",
        )


# Analytics endpoints integrated into AI engine
@router.get(
    "/analytics/resolution-time/{time_range}", response_model=ResolutionTimeResponse
)
async def get_resolution_time(
    time_range: str,
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
    db_session: AsyncSession = Depends(get_db_session),
) -> ResolutionTimeResponse:
    """
    Get incident resolution time statistics.

    :param time_range: Time range for analysis (e.g., "7d" for 7 days)
    :param qdrant_client: Qdrant client dependency
    :param db_session: Database session
    :returns: Resolution time statistics
    """
    # Initialize services
    logger.info(f"Analyzing resolution time for range: {time_range}")

    # Create analyzer without AI engine for simple stats
    analyzer = IncidentAnalyzer(db_session_factory=lambda: db_session)

    try:
        result = await analyzer.analyze_resolution_time(time_range)
        return ResolutionTimeResponse(
            average_hours=result.get("average_hours", 0),
            min_hours=result.get("min_hours", 0),
            max_hours=result.get("max_hours", 0),
            count=result.get("count", 0),
            by_severity=result.get("by_severity", {}),
        )
    except Exception as e:
        logger.error(f"Error analyzing resolution time: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze resolution time: {str(e)}",
        )


@router.get("/analytics/common-failures", response_model=CommonFailuresResponse)
async def get_common_failures(
    limit: int = 10,
    use_ai: bool = True,
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
    db_session: AsyncSession = Depends(get_db_session),
) -> CommonFailuresResponse:
    """
    Get most common failure patterns.

    :param limit: Maximum number of patterns to return
    :param use_ai: Whether to use AI for enhanced analysis
    :param qdrant_client: Qdrant client dependency
    :param db_session: Database session
    :returns: Common failure patterns
    """
    # Initialize services
    logger.info(f"Analyzing common failures with limit: {limit}, use_ai: {use_ai}")

    ai_engine = None
    if use_ai:
        # Initialize AI engine
        ai_engine = AIEngine(
            vector_db_client=qdrant_client,
            llm_model=(
                settings.ai_trend_analysis_model
                if settings.ai_allow_advanced_models
                else settings.openai_completion_model
            ),
            enable_metrics=True,
        )

    # Create analyzer
    analyzer = IncidentAnalyzer(
        db_session_factory=lambda: db_session, ai_engine=ai_engine
    )

    try:
        result = await analyzer.identify_common_failures(limit)
        return CommonFailuresResponse(
            patterns=result,
            generated_at=datetime.utcnow().isoformat(),
            ai_enhanced=use_ai,
        )
    except Exception as e:
        logger.error(f"Error identifying common failures: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to identify common failures: {str(e)}",
        )


@router.get("/analytics/ai-effectiveness", response_model=AIEffectivenessResponse)
async def get_ai_effectiveness(
    db_session: AsyncSession = Depends(get_db_session),
) -> AIEffectivenessResponse:
    """
    Get AI effectiveness metrics.

    :param db_session: Database session
    :returns: AI effectiveness metrics
    """
    # Initialize services
    logger.info("Analyzing AI effectiveness")

    # Create analyzer without AI engine for simple stats
    analyzer = IncidentAnalyzer(db_session_factory=lambda: db_session)

    try:
        result = await analyzer.calculate_ai_effectiveness()
        return AIEffectivenessResponse(
            total_suggestions=result.get("total_suggestions", 0),
            applied_suggestions=result.get("applied_suggestions", 0),
            effectiveness_rate=result.get("effectiveness_rate", 0),
            average_confidence=result.get("average_confidence", 0),
        )
    except Exception as e:
        logger.error(f"Error calculating AI effectiveness: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate AI effectiveness: {str(e)}",
        )


@router.post("/analytics/trend-report", response_model=TrendReportResponse)
async def generate_trend_report(
    request: TrendReportRequest = Body(...),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
    db_session: AsyncSession = Depends(get_db_session),
) -> TrendReportResponse:
    """
    Generate a trend report using AI analysis.

    :param request: Request parameters for trend report
    :param qdrant_client: Qdrant client dependency
    :param db_session: Database session
    :returns: Trend report
    """
    # Initialize services
    logger.info(f"Generating trend report for days: {request.days}")

    # Initialize AI engine
    ai_engine = AIEngine(
        vector_db_client=qdrant_client,
        llm_model=(
            settings.ai_trend_analysis_model
            if settings.ai_allow_advanced_models
            else settings.openai_completion_model
        ),
        enable_metrics=True,
    )

    # Create analyzer with AI engine
    analyzer = IncidentAnalyzer(
        db_session_factory=lambda: db_session, ai_engine=ai_engine
    )

    try:
        result = await analyzer.generate_trend_report(request.days)

        # Check for errors
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )

        return TrendReportResponse(
            statistics=result.get("statistics", {}),
            analysis=result.get("analysis", {}),
            generated_at=result.get("generated_at", datetime.utcnow().isoformat()),
            period_days=result.get("period_days", request.days),
            total_incidents=result.get("total_incidents", 0),
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating trend report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate trend report: {str(e)}",
        )


@router.post("/analytics/root-cause-analysis", response_model=RootCauseAnalysisResponse)
async def perform_root_cause_analysis(
    request: RootCauseAnalysisRequest = Body(...),
    qdrant_client: QdrantClient = Depends(get_qdrant_dependency),
    db_session: AsyncSession = Depends(get_db_session),
) -> RootCauseAnalysisResponse:
    """
    Perform root cause analysis for a specific incident.

    :param request: Request parameters for root cause analysis
    :param qdrant_client: Qdrant client dependency
    :param db_session: Database session
    :returns: Root cause analysis results
    """
    # Initialize services
    logger.info(
        f"Performing root cause analysis for incident ID: {request.incident_id}"
    )

    # Initialize AI engine with appropriate model for root cause analysis
    ai_engine = AIEngine(
        vector_db_client=qdrant_client,
        llm_model=(
            settings.ai_root_cause_analysis_model
            if settings.ai_allow_advanced_models
            else settings.openai_completion_model
        ),
        enable_metrics=True,
    )

    # Create analyzer with AI engine
    analyzer = IncidentAnalyzer(
        db_session_factory=lambda: db_session, ai_engine=ai_engine
    )

    try:
        result = await analyzer.perform_root_cause_analysis(request.incident_id)

        # Check for errors
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )

        return RootCauseAnalysisResponse(
            incident_id=result.get("incident_id", request.incident_id),
            analysis=result.get("analysis", {}),
            generated_at=result.get("generated_at", datetime.utcnow().isoformat()),
            model_used=result.get("model_used", "unknown"),
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error performing root cause analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform root cause analysis: {str(e)}",
        )

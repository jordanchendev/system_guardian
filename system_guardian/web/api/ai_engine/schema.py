"""AI Engine API schemas."""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class GenerateResolutionRequest(BaseModel):
    """Request parameters for resolution generation."""
    
    incident_id: int = Field(..., description="ID of the incident to generate a resolution for")
    force_regenerate: bool = Field(False, description="Whether to force regeneration even if a resolution exists")
    model: Optional[str] = Field(None, description="Optional LLM model override")
    temperature: float = Field(0.3, description="Temperature for the generation (0.0-1.0)", ge=0.0, le=1.0)


class GenerateResolutionResponse(BaseModel):
    """Response model for resolution generation."""
    
    id: int = Field(..., description="ID of the generated resolution")
    incident_id: int = Field(..., description="ID of the incident")
    suggestion: str = Field(..., description="Generated resolution suggestion")
    confidence: float = Field(..., description="Confidence score for the resolution (0.0-1.0)")
    is_applied: bool = Field(False, description="Whether the resolution has been applied")
    generated_at: str = Field(..., description="ISO timestamp of when the resolution was generated")
    feedback_score: Optional[int] = Field(None, description="User feedback score if provided")


class ApplyResolutionRequest(BaseModel):
    """Request to mark a resolution as applied."""
    
    resolution_id: int = Field(..., description="ID of the resolution to mark as applied")
    notes: Optional[str] = Field(None, description="Optional notes about the application")


class ApplyResolutionResponse(BaseModel):
    """Response for marking a resolution as applied."""
    
    resolution_id: int = Field(..., description="ID of the resolution")
    incident_id: int = Field(..., description="ID of the incident")
    is_applied: bool = Field(..., description="Whether the resolution is now applied")
    applied_at: str = Field(..., description="ISO timestamp of when the resolution was marked as applied")


class ResolutionFeedbackRequest(BaseModel):
    """Request to provide feedback on a resolution."""
    
    resolution_id: int = Field(..., description="ID of the resolution")
    feedback_score: int = Field(..., description="Feedback score (1-5)", ge=1, le=5)
    comments: Optional[str] = Field(None, description="Optional feedback comments")


class ResolutionFeedbackResponse(BaseModel):
    """Response after providing feedback on a resolution."""
    
    resolution_id: int = Field(..., description="ID of the resolution")
    incident_id: int = Field(..., description="ID of the incident")
    feedback_score: int = Field(..., description="Feedback score that was provided")
    success: bool = Field(..., description="Whether the feedback was successfully recorded")


class RelatedIncidentsRequest(BaseModel):
    """Request parameters for related incidents search."""
    
    incident_id: Optional[int] = Field(None, description="ID of the incident to find related incidents for")
    query_text: Optional[str] = Field(None, description="Optional text to search for related incidents")
    limit: int = Field(5, description="Maximum number of results to return", ge=1, le=25)
    include_resolved: bool = Field(True, description="Whether to include resolved incidents")
    min_similarity_score: float = Field(0.5, description="Minimum similarity score for results", ge=0.0, le=1.0)


class InsightItem(BaseModel):
    """Single insight item."""
    
    type: str = Field(..., description="Type of insight")
    description: str = Field(..., description="Insight description")
    confidence: float = Field(..., description="Confidence score for the insight")


class RelatedIncidentItem(BaseModel):
    """Single related incident item."""
    
    id: int = Field(..., description="Incident ID")
    title: str = Field(..., description="Incident title")
    description: str = Field(..., description="Incident description")
    severity: str = Field(..., description="Incident severity")
    status: str = Field(..., description="Incident status")
    source: str = Field(..., description="Incident source")
    created_at: str = Field(..., description="Incident creation time (ISO format)")
    resolved_at: Optional[str] = Field(None, description="Incident resolution time (ISO format)")
    resolution: Optional[str] = Field(None, description="Incident resolution")
    similarity_score: float = Field(..., description="Similarity score to the query")


class RelatedIncidentsResponse(BaseModel):
    """Response model for related incidents search."""
    
    incidents: List[RelatedIncidentItem] = Field(..., description="List of related incidents")
    insights: List[InsightItem] = Field(..., description="Insights derived from the incidents")
    current_incident: Optional[RelatedIncidentItem] = Field(None, description="Current incident if provided")


class AIMetricsResponse(BaseModel):
    """Response model for AI metrics."""
    
    metrics: Dict[str, Any] = Field(..., description="AI engine performance metrics")


class GenerateIncidentReportRequest(BaseModel):
    """Request parameters for incident report generation."""
    
    incident_id: int = Field(..., description="ID of the incident to generate a report for")
    include_events: bool = Field(True, description="Whether to include events in the report")
    include_resolution: bool = Field(True, description="Whether to include resolution in the report")
    format: str = Field("markdown", description="Output format (json, markdown, html)")


class GenerateIncidentReportResponse(BaseModel):
    """Response model for incident report generation."""
    
    incident_id: int = Field(..., description="ID of the incident")
    content: Union[str, Dict[str, Any]] = Field(..., description="Report content (string for markdown/html, object for json)")
    format: str = Field(..., description="Output format used (json, markdown, html)")
    generated_at: str = Field(..., description="ISO timestamp of when the report was generated")


class GenerateSummaryReportRequest(BaseModel):
    """Request parameters for summary report generation."""
    
    time_range_days: int = Field(7, description="Number of days to include in the report", ge=1, le=90)
    format: str = Field("markdown", description="Output format (json, markdown, html)")
    include_severity_distribution: bool = Field(True, description="Whether to include severity distribution")
    include_resolution_times: bool = Field(True, description="Whether to include resolution times")
    include_recommendations: bool = Field(True, description="Whether to include recommendations")


class GenerateSummaryReportResponse(BaseModel):
    """Response model for summary report generation."""
    
    content: Union[str, Dict[str, Any]] = Field(..., description="Report content (string for markdown/html, object for json)")
    format: str = Field(..., description="Output format used (json, markdown, html)")
    time_range_days: int = Field(..., description="Number of days included in the report")
    generated_at: str = Field(..., description="ISO timestamp of when the report was generated")


class GenerateRecommendationsRequest(BaseModel):
    """Request parameters for recommendations generation."""
    
    time_range_days: int = Field(30, description="Number of days of incident data to analyze", ge=1, le=90)
    format: str = Field("markdown", description="Output format (json, markdown, html)")


class GenerateRecommendationsResponse(BaseModel):
    """Response model for recommendations generation."""
    
    content: Union[str, Dict[str, Any]] = Field(..., description="Recommendations content (string for markdown/html, object for json)")
    format: str = Field(..., description="Output format used (json, markdown, html)")
    time_range_days: int = Field(..., description="Number of days of data analyzed")
    generated_at: str = Field(..., description="ISO timestamp of when the recommendations were generated")


# Analytics schemas integrated into AI Engine module
class SeverityResolutionTime(BaseModel):
    """Resolution time for a specific severity level."""
    
    average_hours: float = Field(..., description="Average resolution time in hours")
    count: int = Field(..., description="Number of incidents")


class ResolutionTimeResponse(BaseModel):
    """Response model for resolution time analysis."""
    
    average_hours: float = Field(..., description="Average resolution time in hours")
    min_hours: float = Field(..., description="Minimum resolution time in hours")
    max_hours: float = Field(..., description="Maximum resolution time in hours")
    count: int = Field(..., description="Number of resolved incidents")
    by_severity: Dict[str, SeverityResolutionTime] = Field({}, description="Resolution time by severity level")


class PatternItem(BaseModel):
    """Single failure pattern item."""
    
    pattern: Optional[str] = Field(None, description="Pattern name/category")
    root_cause: Optional[str] = Field(None, description="Root cause of the pattern")
    frequency: Optional[int] = Field(None, description="Frequency of occurrence")
    avg_severity: Optional[str] = Field(None, description="Average severity of incidents")
    preventive_measures: Optional[str] = Field(None, description="Recommended preventive measures")
    source: Optional[str] = Field(None, description="Source of incidents (for basic analysis)")
    count: Optional[int] = Field(None, description="Count of incidents (for basic analysis)")


class CommonFailuresResponse(BaseModel):
    """Response model for common failures analysis."""
    
    patterns: List[PatternItem] = Field(..., description="List of identified failure patterns")
    generated_at: str = Field(..., description="ISO timestamp of when the analysis was generated")
    ai_enhanced: bool = Field(..., description="Whether AI was used for enhanced analysis")


class AIEffectivenessResponse(BaseModel):
    """Response model for AI effectiveness metrics."""
    
    total_suggestions: int = Field(..., description="Total number of AI-generated suggestions")
    applied_suggestions: int = Field(..., description="Number of suggestions that were applied")
    effectiveness_rate: float = Field(..., description="Rate of applied suggestions (0.0-1.0)")
    average_confidence: float = Field(..., description="Average confidence score of suggestions")


class TrendReportRequest(BaseModel):
    """Request parameters for trend report generation."""
    
    days: int = Field(30, description="Number of days to analyze", ge=1, le=90)


class TrendReportResponse(BaseModel):
    """Response model for trend report."""
    
    statistics: Dict[str, Any] = Field(..., description="Basic statistics about incidents")
    analysis: Dict[str, Any] = Field(..., description="AI-generated analysis of trends")
    generated_at: str = Field(..., description="ISO timestamp of when the report was generated")
    period_days: int = Field(..., description="Number of days analyzed")
    total_incidents: int = Field(..., description="Total number of incidents analyzed")


class RootCauseAnalysisRequest(BaseModel):
    """Request parameters for root cause analysis."""
    
    incident_id: int = Field(..., description="ID of the incident to analyze")


class RootCauseAnalysisResponse(BaseModel):
    """Response model for root cause analysis."""
    
    incident_id: int = Field(..., description="ID of the analyzed incident")
    analysis: Dict[str, Any] = Field(..., description="Root cause analysis results")
    generated_at: str = Field(..., description="ISO timestamp of when the analysis was generated")
    model_used: str = Field(..., description="AI model used for the analysis") 
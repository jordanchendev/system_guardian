"""API endpoints for incident detection configuration."""
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.ext.asyncio import AsyncSession

from system_guardian.db.dependencies import get_db_session
from system_guardian.services.config.incident_rules import (
    ConfigManager, 
    IncidentDetectionConfig
)
from system_guardian.web.api.config.schema import ConfigResponse, StatusResponse


router = APIRouter()


@router.get("/incident-rules", response_model=IncidentDetectionConfig)
async def get_incident_rules(
    config_manager: ConfigManager = Depends(lambda: ConfigManager()),
) -> IncidentDetectionConfig:
    """
    Get current incident detection rule configuration.
    
    :param config_manager: Configuration manager instance
    :returns: Incident detection configuration
    """
    config = await config_manager.load_config()
    return config


@router.post("/incident-rules", response_model=ConfigResponse)
async def update_incident_rules(
    config: IncidentDetectionConfig = Body(...),
    config_manager: ConfigManager = Depends(lambda: ConfigManager()),
) -> ConfigResponse:
    """
    Update incident detection rule configuration.
    
    :param config: New incident detection configuration
    :param config_manager: Configuration manager instance
    :returns: Status of update operation
    """
    # Set config and save
    config_manager._config = config
    success = await config_manager.save_config()
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save configuration"
        )
    
    return ConfigResponse(
        success=True,
        message="Configuration updated successfully",
        config_path=config_manager.config_path
    )


@router.get("/incident-rules/status", response_model=StatusResponse)
async def get_incident_rules_status(
    config_manager: ConfigManager = Depends(lambda: ConfigManager()),
) -> StatusResponse:
    """
    Get a summary of incident detection rule status.
    
    :param config_manager: Configuration manager instance
    :returns: Status of incident detection
    """
    config = await config_manager.load_config()
    
    # Build status summary
    source_configs = {}
    for source, source_rule in config.sources.items():
        source_configs[source] = {
            "enabled": source_rule.enabled,
            "event_types": list(source_rule.conditions.keys()),
            "auto_create_incident": source_rule.auto_create_incident
        }
    
    return StatusResponse(
        enabled=config.enabled,
        source_configs=source_configs
    )


@router.post("/incident-rules/toggle", response_model=ConfigResponse)
async def toggle_incident_detection(
    enabled: bool = Body(..., embed=True),
    config_manager: ConfigManager = Depends(lambda: ConfigManager()),
) -> ConfigResponse:
    """
    Toggle incident detection on or off.
    
    :param enabled: Whether incident detection should be enabled
    :param config_manager: Configuration manager instance
    :returns: Status of update operation
    """
    config = await config_manager.load_config()
    config.enabled = enabled
    
    # Save updated config
    config_manager._config = config
    success = await config_manager.save_config()
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save configuration"
        )
    
    status_str = "enabled" if enabled else "disabled"
    return ConfigResponse(
        success=True,
        message=f"Incident detection {status_str} successfully",
        config_path=config_manager.config_path
    ) 
"""
AI Services Module for System Guardian

This module provides AI-powered services for incident management,
analysis, and resolution.
"""

from typing import Dict, Any, Optional, Type
from loguru import logger

# Import all service classes
from .engine import AIEngine
from .resolution_generator import ResolutionGenerator
from .incident_detector import IncidentDetector
from .incident_analyzer import IncidentAnalyzer
from .report_generator import ReportGenerator
from .incident_similarity import IncidentSimilarityService
from .severity_classifier import SeverityClassifier

# AI Services registry
_service_registry: Dict[str, Type] = {
    'engine': AIEngine,
    'resolution_generator': ResolutionGenerator,
    'incident_detector': IncidentDetector,
    'incident_analyzer': IncidentAnalyzer,
    'report_generator': ReportGenerator,
    'incident_similarity': IncidentSimilarityService,
    'severity_classifier': SeverityClassifier,
}

# Service instances cache
_service_instances: Dict[str, Any] = {}


def get_service(service_name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """
    Get or create an AI service instance by name.
    
    :param service_name: Name of the service to get
    :param config: Optional configuration for the service
    :param kwargs: Additional initialization parameters
    :return: Service instance
    """
    if service_name not in _service_registry:
        raise ValueError(f"Unknown AI service: {service_name}")
    
    # Return cached instance if exists
    if service_name in _service_instances:
        logger.debug(f"Returning cached {service_name} instance")
        return _service_instances[service_name]
    
    # Create new instance
    service_class = _service_registry[service_name]
    
    # For services that depend on AIEngine, ensure we have an engine instance
    if service_name != 'engine' and service_class.__init__.__annotations__.get('ai_engine'):
        engine = get_service('engine', config, **kwargs)
        instance = service_class(ai_engine=engine, **kwargs)
    else:
        instance = service_class(**kwargs)
    
    # Cache the instance
    _service_instances[service_name] = instance
    logger.info(f"Created new {service_name} instance")
    
    return instance


def register_service(service_name: str, service_class: Type) -> None:
    """
    Register a new AI service class.
    
    :param service_name: Name for the service
    :param service_class: Service class to register
    """
    _service_registry[service_name] = service_class
    logger.info(f"Registered new AI service: {service_name}")


def clear_service_cache() -> None:
    """
    Clear the service instances cache.
    """
    _service_instances.clear()
    logger.debug("Cleared AI service cache")
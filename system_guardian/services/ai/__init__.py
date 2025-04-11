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
from .service_factory import AIServiceFactory

# Create a singleton instance of AIServiceFactory
_service_factory = AIServiceFactory()


def get_service(
    service_name: str, config: Optional[Dict[str, Any]] = None, **kwargs
) -> Any:
    """
    Get or create an AI service instance by name.

    :param service_name: Name of the service to get
    :param config: Optional configuration for the service
    :param kwargs: Additional initialization parameters
    :return: Service instance
    """
    return _service_factory.get_service(service_name, **kwargs)


def register_service(service_name: str, service_class: Type) -> None:
    """
    Register a new AI service class.

    :param service_name: Name for the service
    :param service_class: Service class to register
    """
    _service_factory.register_service(service_name, service_class)


def clear_service_cache() -> None:
    """
    Clear the service instances cache.
    """
    _service_factory.clear_instances()


def get_service_names() -> list:
    """
    Get names of all registered services.

    :return: List of service names
    """
    return _service_factory.get_service_names()


def has_service(service_name: str) -> bool:
    """
    Check if a service is registered.

    :param service_name: Name of the service to check
    :return: True if service is registered, False otherwise
    """
    return _service_factory.has_service(service_name)


async def initialize_all() -> Dict[str, bool]:
    """
    Initialize all registered services.

    :return: Dictionary mapping service names to initialization success status
    """
    return await _service_factory.initialize_all()

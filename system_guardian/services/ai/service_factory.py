"""
Factory for creating and managing AI service instances.
"""

from typing import Dict, Any, Optional, Type, List
import importlib
import inspect
from loguru import logger

from system_guardian.services.ai.service_base import AIServiceBase
from system_guardian.services.ai.engine import AIEngine


class AIServiceFactory:
    """
    Factory for creating and managing AI service instances.

    This class provides methods for creating service instances,
    registering new service types, and managing dependencies between services.
    """

    def __init__(self):
        """Initialize the service factory."""
        self._service_registry: Dict[str, Type[AIServiceBase]] = {}
        self._service_instances: Dict[str, AIServiceBase] = {}
        self._engine_instance: Optional[AIEngine] = None
        self._service_dependencies: Dict[str, List[str]] = {}

        # Automatically register built-in services
        self._register_builtin_services()

    def _register_builtin_services(self):
        """Register built-in service classes."""
        try:
            # Import built-in service modules
            from system_guardian.services.ai.resolution_generator import (
                ResolutionGenerator,
            )
            from system_guardian.services.ai.incident_detector import IncidentDetector
            from system_guardian.services.ai.incident_analyzer import IncidentAnalyzer
            from system_guardian.services.ai.report_generator import ReportGenerator
            from system_guardian.services.ai.incident_similarity import (
                IncidentSimilarityService,
            )
            from system_guardian.services.ai.severity_classifier import (
                SeverityClassifier,
            )

            # Register the services
            builtin_services = [
                ("resolution_generator", ResolutionGenerator),
                ("incident_detector", IncidentDetector),
                ("incident_analyzer", IncidentAnalyzer),
                ("report_generator", ReportGenerator),
                ("incident_similarity", IncidentSimilarityService),
                ("severity_classifier", SeverityClassifier),
            ]

            for name, service_class in builtin_services:
                self.register_service(name, service_class)

            logger.debug(f"Registered {len(builtin_services)} built-in services")
        except Exception as e:
            logger.error(f"Error registering built-in services: {str(e)}")

    def register_service(self, name: str, service_class: Type[AIServiceBase]) -> None:
        """
        Register a service class with the factory.

        :param name: Name to register the service under
        :param service_class: Service class to register
        """
        if name in self._service_registry:
            logger.warning(f"Overwriting existing service registration for {name}")

        self._service_registry[name] = service_class

        # Analyze service dependencies
        dependencies = []
        if hasattr(service_class, "__init__"):
            params = inspect.signature(service_class.__init__).parameters
            if "ai_engine" in params:
                dependencies.append("engine")

        self._service_dependencies[name] = dependencies
        logger.info(
            f"Registered service class: {name} with dependencies: {dependencies}"
        )

    def register_engine(self, engine: AIEngine) -> None:
        """
        Register an AIEngine instance to be used by services.

        :param engine: AIEngine instance
        """
        self._engine_instance = engine
        logger.info("Registered AIEngine instance")

    def create_engine(self, **kwargs) -> AIEngine:
        """
        Create and register an AIEngine instance.

        :param kwargs: Parameters for AIEngine initialization
        :return: AIEngine instance
        """
        self._engine_instance = AIEngine(**kwargs)
        logger.info("Created new AIEngine instance")
        return self._engine_instance

    def get_service(self, name: str, **kwargs) -> AIServiceBase:
        """
        Get or create a service instance.

        :param name: Name of the service to get
        :param kwargs: Parameters for service initialization
        :return: Service instance
        """
        # Return cached instance if exists
        if name in self._service_instances:
            logger.debug(f"Returning cached {name} service instance")
            return self._service_instances[name]

        # Check if service is registered
        if name not in self._service_registry:
            raise ValueError(f"Service not registered: {name}")

        # Get service class
        service_class = self._service_registry[name]

        # Check dependencies
        dependencies = self._service_dependencies.get(name, [])
        for dep in dependencies:
            if dep not in self._service_instances:
                logger.debug(f"Initializing dependency {dep} for service {name}")
                self.get_service(dep)

        # Create service instance
        if "ai_engine" in inspect.signature(service_class.__init__).parameters:
            if not self._engine_instance:
                raise ValueError(
                    f"Service {name} requires AIEngine but no engine is registered"
                )

            instance = service_class(ai_engine=self._engine_instance, **kwargs)
        else:
            instance = service_class(**kwargs)

        # Cache the instance
        self._service_instances[name] = instance
        logger.info(f"Created new {name} service instance")

        return instance

    def get_service_names(self) -> List[str]:
        """
        Get names of all registered services.

        :return: List of service names
        """
        return list(self._service_registry.keys())

    def has_service(self, name: str) -> bool:
        """
        Check if a service is registered.

        :param name: Name of the service to check
        :return: True if service is registered, False otherwise
        """
        return name in self._service_registry

    def clear_instances(self) -> None:
        """Clear all cached service instances."""
        self._service_instances.clear()
        logger.debug("Cleared service instances cache")

    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered services.

        :return: Dictionary mapping service names to initialization success status
        """
        results = {}

        # Initialize services in dependency order
        initialized = set()
        while len(initialized) < len(self._service_registry):
            for name in self._service_registry:
                if name in initialized:
                    continue

                # Check if all dependencies are initialized
                dependencies = self._service_dependencies.get(name, [])
                if all(dep in initialized for dep in dependencies):
                    try:
                        service = self.get_service(name)
                        success = await service.initialize()
                        results[name] = success
                        initialized.add(name)
                        logger.info(
                            f"Initialized service {name}: {'success' if success else 'failed'}"
                        )
                    except Exception as e:
                        results[name] = False
                        logger.error(f"Error initializing service {name}: {str(e)}")

        return results

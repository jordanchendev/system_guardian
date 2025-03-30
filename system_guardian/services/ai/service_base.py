"""Base class for all AI services."""

import abc
from typing import Optional, Dict, Any, List
from datetime import datetime
import time
from loguru import logger

class AIServiceBase(abc.ABC):
    """
    Abstract base class for all AI services.
    
    This class provides common functionality used by multiple AI services,
    including metrics tracking, error handling, and integration with AIEngine.
    """
    
    def __init__(
        self, 
        ai_engine = None,
        service_name: str = None,
        enable_metrics: bool = True
    ):
        """
        Initialize the AI service base.
        
        :param ai_engine: Reference to the AIEngine instance
        :param service_name: Name of the service (defaults to class name)
        :param enable_metrics: Whether to track performance metrics
        """
        self.ai_engine = ai_engine
        self.service_name = service_name or self.__class__.__name__
        self.enable_metrics = enable_metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0,
            "last_request_time": None
        }
        logger.debug(f"Initialized {self.service_name}")
    
    def _track_metric(self, metric_name: str, increment: int = 1):
        """
        Track a performance metric if metrics are enabled.
        
        :param metric_name: Name of the metric to track
        :param increment: Value to increment the metric by
        """
        if self.enable_metrics:
            self.metrics[metric_name] = self.metrics.get(metric_name, 0) + increment
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current performance metrics.
        
        :return: Dictionary of metrics
        """
        logger.debug(f"{self.service_name} metrics: {self.metrics}")
        return self.metrics
    
    async def run_with_metrics(self, func, *args, **kwargs):
        """
        Run a function while tracking metrics.
        
        :param func: Async function to run
        :param args: Positional arguments for the function
        :param kwargs: Keyword arguments for the function
        :return: Result of the function
        """
        start_time = time.time()
        self._track_metric("total_requests")
        
        try:
            result = await func(*args, **kwargs)
            self._track_metric("successful_requests")
            return result
        except Exception as e:
            self._track_metric("failed_requests")
            logger.exception(f"Error in {self.service_name}: {str(e)}")
            raise
        finally:
            processing_time = time.time() - start_time
            self._track_metric("total_processing_time", processing_time)
            self.metrics["last_request_time"] = datetime.utcnow().isoformat()
    
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the service. Override this in subclasses.
        
        :return: True if initialization was successful, False otherwise
        """
        pass
    
    def get_version(self) -> str:
        """
        Get the version of the service.
        
        :return: Service version string
        """
        return "1.0.0"  # Default version
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get health information about the service.
        
        :return: Dictionary with health information
        """
        return {
            "name": self.service_name,
            "healthy": True,
            "version": self.get_version(),
            "metrics": self.get_metrics()
        } 
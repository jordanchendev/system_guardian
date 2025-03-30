"""Configuration module for incident detection rules."""
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field, validator
import json
import os
from loguru import logger


class EventCondition(BaseModel):
    """Event condition configuration."""
    
    event_types: List[str] = Field(default_factory=list)
    create_incident_on: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    min_events_threshold: int = 3
    time_window_minutes: int = 60
    field_conditions: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('min_events_threshold')
    def validate_threshold(cls, v):
        """Validate threshold is a positive number."""
        if v < 1:
            raise ValueError('Threshold must be at least 1')
        return v


class SourceRules(BaseModel):
    """Source-specific rules configuration."""
    
    conditions: Dict[str, EventCondition] = Field(default_factory=dict)
    auto_create_incident: bool = True
    enabled: bool = True
    global_keywords: List[str] = Field(default_factory=list)


class IncidentDetectionConfig(BaseModel):
    """Global incident detection configuration."""
    
    sources: Dict[str, SourceRules] = Field(default_factory=dict)
    global_keywords: List[str] = Field(default_factory=list)
    enabled: bool = True
    
    def get_rules_for_event(self, source: str, event_type: str) -> Optional[EventCondition]:
        """
        Get rules for a specific event source and type.
        
        :param source: Event source (e.g., 'github', 'jira')
        :param event_type: Event type (e.g., 'issue', 'pull_request')
        :returns: Event condition rules or None if not found
        """
        if not self.enabled:
            return None
            
        if source not in self.sources:
            return None
            
        source_rules = self.sources[source]
        if not source_rules.enabled:
            return None
        
        # First try direct lookup by event_type
        if event_type in source_rules.conditions:
            return source_rules.conditions[event_type]
            
        # If not found, look for a condition that includes this event_type in its event_types list
        for condition_key, condition in source_rules.conditions.items():
            if event_type in condition.event_types:
                return condition
                
        return None
    
    def get_keywords_for_event(self, source: str, event_type: str) -> Set[str]:
        """
        Get all applicable keywords for an event, including global keywords.
        
        :param source: Event source
        :param event_type: Event type
        :returns: Set of keywords
        """
        keywords = set(self.global_keywords)
        
        if source in self.sources:
            source_rules = self.sources[source]
            keywords.update(source_rules.global_keywords)
            
            # Direct lookup by event_type
            if event_type in source_rules.conditions:
                keywords.update(source_rules.conditions[event_type].keywords)
                return keywords
                
            # Search in event_types lists
            for condition in source_rules.conditions.values():
                if event_type in condition.event_types:
                    keywords.update(condition.keywords)
                    break
                
        return keywords


class ConfigManager:
    """Manager for loading and saving incident detection configuration."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the config manager.
        
        :param config_path: Path to the configuration file
        """
        self.config_path = config_path or os.environ.get(
            "INCIDENT_CONFIG_PATH", 
            "config/incident_rules.json"
        )
        self._config = None
    
    async def load_config(self) -> IncidentDetectionConfig:
        """
        Load configuration from file or create default.
        
        :returns: Incident detection configuration
        """
        if self._config is not None:
            return self._config
            
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    self._config = IncidentDetectionConfig.parse_obj(config_data)
                    logger.info(f"Loaded incident rules configuration from {self.config_path}")
            else:
                self._config = self._create_default_config()
                await self.save_config()
                logger.info(f"Created default incident rules configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading incident configuration: {str(e)}")
            self._config = self._create_default_config()
        return self._config
    
    async def save_config(self) -> bool:
        """
        Save configuration to file.
        
        :returns: True if successful, False otherwise
        """
        if self._config is None:
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self._config.dict(), f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error saving incident configuration: {str(e)}")
            return False
    
    def _create_default_config(self) -> IncidentDetectionConfig:
        """
        Create default configuration.
        
        :returns: Default incident detection configuration
        """
        return IncidentDetectionConfig(
            enabled=True,
            global_keywords=["critical", "urgent", "emergency", "outage", "down"],
            sources={
                "github": SourceRules(
                    enabled=True,
                    global_keywords=["crash", "bug", "error", "broken"],
                    conditions={
                        "issue": EventCondition(
                            event_types=["issues"],
                            create_incident_on=["opened", "edited"],
                            keywords=["critical", "urgent", "broken", "crash", "failure"],
                            min_events_threshold=3,
                            time_window_minutes=60
                        ),
                        "pull_request": EventCondition(
                            event_types=["pull_request"],
                            create_incident_on=["closed"],
                            keywords=["fix", "hotfix", "urgent", "emergency"],
                            field_conditions={"state": "closed", "merged": False},
                            min_events_threshold=2,
                            time_window_minutes=30
                        )
                    }
                ),
                "jira": SourceRules(
                    enabled=True,
                    global_keywords=["bug", "incident", "problem"],
                    conditions={
                        "issue": EventCondition(
                            event_types=["jira:issue_created", "jira:issue_updated"],
                            create_incident_on=["created", "updated"],
                            keywords=["blocker", "critical", "outage", "down", "broken"],
                            min_events_threshold=1,
                            time_window_minutes=30
                        )
                    }
                )
            }
        ) 
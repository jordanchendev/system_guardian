import enum
import os
from pathlib import Path
from tempfile import gettempdir
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from yarl import URL

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):  # noqa: WPS600
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "localhost"
    port: int = 5566
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO
    enable_file_logging: bool = False
    logs_dir: Optional[str] = None
    structured_logging: bool = False

    # Variables for the database
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "system_guardian"
    db_pass: str = "system_guardian"
    db_base: str = "system_guardian"
    db_echo: bool = False

    # Variables for RabbitMQ
    # rabbit_host: str = "system_guardian-rmq"
    rabbit_host: str = "localhost"
    rabbit_port: int = 5672
    rabbit_user: str = "guest"
    rabbit_pass: str = "guest"
    rabbit_vhost: str = "/"

    rabbit_pool_size: int = 2
    rabbit_channel_pool_size: int = 10

    # Qdrant Vector DB settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: Optional[str] = None
    qdrant_timeout: int = 10  # Seconds

    # OpenAI settings
    openai_api_key: str = os.getenv("SYSTEM_GUARDIAN_OPENAI_API_KEY")
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_completion_model: str = "gpt-4o-mini"

    # AI task-specific model settings
    ai_incident_detection_model: str = (
        "gpt-4o-mini"  # Model for incident detection and analysis
    )
    ai_severity_classification_model: str = (
        "gpt-4o-mini"  # Model for severity classification
    )
    ai_similarity_search_model: str = "gpt-4o-mini"  # Model for similarity search
    ai_resolution_generation_model: str = "gpt-4o"  # Model for solution generation
    ai_root_cause_analysis_model: str = "gpt-4o"  # Model for root cause analysis
    ai_report_generation_model: str = "gpt-4o"  # Model for report generation
    ai_trend_analysis_model: str = "gpt-4o-mini"  # Model for trend analysis

    # Whether to allow using advanced models (e.g., GPT-4)
    ai_allow_advanced_models: bool = True

    # Model temperature settings
    ai_default_temperature: float = 0.3  # Low temperature for more consistent results
    ai_creative_temperature: float = (
        0.7  # Higher temperature for tasks requiring creativity
    )

    # Slack settings
    slack_enabled: bool = os.getenv("SYSTEM_GUARDIAN_SLACK_ENABLED", False)
    slack_bot_token: Optional[str] = os.getenv("SYSTEM_GUARDIAN_SLACK_BOT_TOKEN")
    slack_channel_id: Optional[str] = os.getenv(
        "SYSTEM_GUARDIAN_SLACK_CHANNEL_ID", "general"
    )
    slack_username: str = "System Guardian"
    slack_icon_emoji: str = ":robot_face:"
    slack_timeout: int = 30  # Seconds

    # JIRA settings
    jira_enabled: bool = os.getenv("SYSTEM_GUARDIAN_JIRA_ENABLED", False)
    jira_url: Optional[str] = os.getenv("SYSTEM_GUARDIAN_JIRA_URL")
    jira_username: Optional[str] = os.getenv("SYSTEM_GUARDIAN_JIRA_USERNAME")
    jira_api_token: Optional[str] = os.getenv("SYSTEM_GUARDIAN_JIRA_API_TOKEN")
    jira_project_key: str = os.getenv("SYSTEM_GUARDIAN_JIRA_PROJECT_KEY", "INCIDENT")
    jira_issue_type: str = os.getenv("SYSTEM_GUARDIAN_JIRA_ISSUE_TYPE", "Bug")
    jira_timeout: int = 30  # Seconds

    # Add default settings for event detection
    # Controls whether to automatically create incidents for GitHub events
    github_auto_detect_incident: bool = (
        os.getenv("GITHUB_AUTO_DETECT_INCIDENT", "true").lower() == "true"
    )
    # Controls whether to automatically create incidents for JIRA events
    jira_auto_detect_incident: bool = (
        os.getenv("JIRA_AUTO_DETECT_INCIDENT", "false").lower() == "true"
    )
    # Controls whether to automatically create incidents for Datadog events
    datadog_auto_detect_incident: bool = (
        os.getenv("DATADOG_AUTO_DETECT_INCIDENT", "true").lower() == "true"
    )

    @property
    def db_url(self) -> URL:
        """
        Assemble database URL from settings.

        :return: database URL.
        """
        return URL.build(
            scheme="postgresql+asyncpg",
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_pass,
            path=f"/{self.db_base}",
        )

    @property
    def rabbit_url(self) -> URL:
        """
        Assemble RabbitMQ URL from settings.

        :return: rabbit URL.
        """
        return URL.build(
            scheme="amqp",
            host=self.rabbit_host,
            port=self.rabbit_port,
            user=self.rabbit_user,
            password=self.rabbit_pass,
            path=self.rabbit_vhost,
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SYSTEM_GUARDIAN_",
    )


settings = Settings()

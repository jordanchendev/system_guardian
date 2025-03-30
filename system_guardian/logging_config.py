"""Configure logging for the system guardian application."""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from loguru import logger

from system_guardian.settings import settings, LogLevel


def configure_loguru(
    sink=sys.stdout,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    rotation: str = "20 MB",
    retention: str = "1 week",
    format_string: Optional[str] = None,
    serialize: bool = False,
) -> None:
    """
    Configure Loguru logger with given parameters.

    :param sink: Output sink (default: stdout)
    :param level: Log level (default: from settings)
    :param log_file: Optional file path to write logs to
    :param rotation: When to rotate logs (size or time)
    :param retention: How long to keep logs
    :param format_string: Log format string
    :param serialize: Whether to serialize logs as JSON
    """
    # Remove default handlers
    logger.remove()

    # Use the level from settings if not provided
    if level is None:
        level = settings.log_level.value

    # Default format string for general use
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Define a filter for common loggers we want to suppress
    def filter_noisy_loggers(record):
        # Completely disable logs from these namespaces
        blocked_loggers = ["sqlalchemy", "aio_pika", "asyncio", "httpx", "urllib3"]

        # Check if the record comes from these namespaces
        for logger_name in blocked_loggers:
            if logger_name in record["name"]:
                return False

        # Filter out specific content in warning messages
        if (
            "Qdrant client version" in record["message"]
            and "incompatible with server version" in record["message"]
        ):
            return False

        # Allow all other records
        return True

    # Add stdout handler with filter
    logger.add(
        sink=sink,
        level=level,
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True,
        filter=filter_noisy_loggers,
    )

    # Add file handler if requested
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        logger.add(
            sink=log_file,
            level=level,
            format=format_string,
            rotation=rotation,
            retention=retention,
            compression="zip",
            serialize=serialize,
            backtrace=True,
            diagnose=True,
            filter=filter_noisy_loggers,
        )

    logger.debug(f"Configured Loguru with level: {level}")


def configure_sqlalchemy_logging():
    """
    Specialized configuration for SQLAlchemy logging.

    Since SQLAlchemy logs are verbose, this provides separate control.
    """
    # Set up explicitly forbidden loggers
    for unwanted_logger in ["sqlalchemy", "aio_pika", "asyncio"]:
        sql_logger = logging.getLogger(unwanted_logger)
        sql_logger.propagate = False
        sql_logger.handlers = []
        # Add an empty handler to prevent warnings
        sql_logger.addHandler(NoOpHandler())
        sql_logger.setLevel(
            logging.CRITICAL
        )  # Set to highest level to filter most messages

    # Ensure our application also doesn't receive these verbose logs
    orig_log = logger.log

    def filter_log(level, message, *args, **kwargs):
        # Get caller logger name, typically in 'record'
        record = kwargs.get("record", {})
        logger_name = getattr(record, "name", "") if record else ""

        # Filter out messages from unwanted loggers
        if logger_name and any(
            name in logger_name for name in ["sqlalchemy", "aio_pika", "asyncio"]
        ):
            return  # Don't log

        # For all other messages, use the original log function
        return orig_log(level, message, *args, **kwargs)

    # Use filtered version instead of original log method
    logger.log = filter_log


def configure_logging():
    """Configures the application logging."""
    # First, ensure Python's logging system doesn't output anything
    logging.basicConfig(handlers=[logging.NullHandler()])

    # Immediately disable SQLAlchemy logs
    configure_sqlalchemy_logging()

    # First, completely turn off several modules we don't want to see logs from
    for logger_name in [
        "sqlalchemy",
        "aio_pika",
        "asyncio",
        "httpx",
        "urllib3",
        "urllib3.connectionpool",
    ]:
        module_logger = logging.getLogger(logger_name)
        module_logger.setLevel(logging.CRITICAL + 10)  # Higher than CRITICAL
        module_logger.disabled = True
        # Remove any possible handlers
        if module_logger.handlers:
            for handler in module_logger.handlers:
                module_logger.removeHandler(handler)
        # Prevent log propagation upwards
        module_logger.propagate = False

    # Special case for sqlalchemy - completely disable its logs
    logging.getLogger("sqlalchemy").disabled = True
    logging.getLogger("sqlalchemy.engine").disabled = True
    logging.getLogger("sqlalchemy.engine.base.Engine").disabled = True

    # Get log level from environment
    level = settings.log_level.value

    # Configure file log path if enabled
    log_file = None
    if settings.enable_file_logging:
        logs_dir = settings.logs_dir or "logs"
        os.makedirs(logs_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(logs_dir, f"system_guardian_{date_str}.log")

    # Configure Loguru
    configure_loguru(
        level=level,
        log_file=log_file,
        serialize=settings.structured_logging,
    )

    # Set up a NULL handler as fallback
    null_handler = logging.NullHandler()
    logging.getLogger().addHandler(null_handler)

    # Set up interceptor for each SQLAlchemy logger
    sql_interceptor = SQLAlchemyInterceptor()
    for name in [
        "sqlalchemy",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "sqlalchemy.orm",
        "sqlalchemy.dialects",
    ]:
        logger = logging.getLogger(name)
        logger.handlers = [sql_interceptor]  # Replace all handlers

    # Install interceptor - ensure all standard library logs go through loguru redirection
    intercept_handler = InterceptHandler()
    intercept_handler.intercept_all_loggers()

    # Log startup message
    logger.info(f"Logging configured with level {level}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")


# Special interceptor for SQLAlchemy
class SQLAlchemyInterceptor(logging.Handler):
    """Special handler for SQLAlchemy logs"""

    def __init__(self):
        super().__init__()
        self.level = logging.CRITICAL + 10  # Beyond highest level

    def emit(self, record):
        """Never emit logs"""
        pass


# Intercept standard library logging
class InterceptHandler(logging.Handler):
    """
    Intercepts standard library logging and redirects to loguru.

    This handler is needed to capture logs from libraries that use
    the standard logging module.
    """

    def __init__(self):
        super().__init__()
        self.handled_loggers = set()

    def intercept_all_loggers(self):
        """Intercept all existing loggers."""
        # Get standard library root logger
        root_logger = logging.getLogger()

        # Remove any existing handlers
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                if not isinstance(handler, logging.NullHandler):
                    root_logger.removeHandler(handler)

        # Add our handler to the root logger
        root_logger.addHandler(self)
        root_logger.setLevel(logging.WARNING)  # Only focus on important messages

        # Disable specific loggers - use stronger disable method
        for unwanted_logger in ["sqlalchemy", "aio_pika", "asyncio"]:
            for suffix in ["", ".engine", ".pool", ".orm", ".dialects"]:
                log_name = unwanted_logger + suffix
                logging.getLogger(log_name).setLevel(logging.CRITICAL + 10)
                logging.getLogger(log_name).disabled = True
                logging.getLogger(log_name).propagate = False
                # Ensure no handlers
                for handler in logging.getLogger(log_name).handlers[:]:
                    logging.getLogger(log_name).removeHandler(handler)
                # Add empty handler
                logging.getLogger(log_name).addHandler(logging.NullHandler())

        # Iterate through all loggers and configure them
        for logger_name in logging.root.manager.loggerDict:
            # Skip unwanted loggers
            if any(
                name in logger_name for name in ["sqlalchemy", "aio_pika", "asyncio"]
            ):
                continue
            self._intercept_logger(logger_name)

        # Special case for uvicorn loggers
        for logger_name in ("uvicorn", "uvicorn.access"):
            ulogger = logging.getLogger(logger_name)
            # Remove existing handlers
            if ulogger.handlers:
                for handler in ulogger.handlers[:]:
                    ulogger.removeHandler(handler)
            ulogger.addHandler(self)
            if logger_name == "uvicorn.access":
                ulogger.setLevel(logging.CRITICAL)
            else:
                ulogger.setLevel(logging.WARNING)

        logger.debug("Intercepted all standard library loggers")

    def _intercept_logger(self, logger_name: str):
        """Intercept a specific logger."""
        if logger_name in self.handled_loggers:
            return

        log = logging.getLogger(logger_name)

        # Completely disable unwanted logs
        if any(
            name in logger_name
            for name in ["sqlalchemy", "aio_pika", "asyncio", "httpx", "urllib3"]
        ):
            log.setLevel(logging.CRITICAL + 10)
            log.disabled = True
            log.propagate = False
            # Ensure no handlers
            if log.handlers:
                for handler in log.handlers[:]:
                    log.removeHandler(handler)
            # Add empty handler
            log.addHandler(logging.NullHandler())
            return

        # Remove existing handlers
        if log.handlers:
            for handler in log.handlers[:]:
                log.removeHandler(handler)
        log.addHandler(self)
        log.propagate = False
        self.handled_loggers.add(logger_name)

    def __call__(self, record):
        """Handler implementation that redirects to loguru."""
        self.emit(record)

    # Additional methods required for a proper Handler subclass
    def createLock(self):
        """No locking needed for this handler."""
        self.lock = None

    def acquire(self):
        """No lock to acquire."""
        pass

    def release(self):
        """No lock to release."""
        pass

    def setLevel(self, level):
        """Set the logging level."""
        self.level = level

    def setFormatter(self, formatter):
        """Set the formatter for this handler."""
        self.formatter = formatter

    def flush(self):
        """Ensure all logging output has been flushed."""
        pass

    def close(self):
        """Tidy up any resources used by the handler."""
        pass

    def emit(self, record):
        """
        Emit a record - standard logging Handler interface.

        This implements the logging.Handler interface method.
        """
        # Check if the record should be completely ignored
        if any(
            name in record.name.lower()
            for name in ["sqlalchemy", "aio_pika", "httpx", "asyncio", "urllib3"]
        ):
            return

        # Check if the record level has reached the corresponding logger's threshold
        logger_obj = logging.getLogger(record.name)
        if logger_obj.disabled or record.levelno < logger_obj.level:
            return  # Don't handle logs below threshold

        # Avoid specific message content
        if record.getMessage() and (
            "raw sql" in record.getMessage()
            or "generated in" in record.getMessage()
            or "cached since" in record.getMessage()
            or "HTTP Request" in record.getMessage()
            or "Qdrant client version" in record.getMessage()
        ):
            return

        # Get corresponding Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Avoid logging details for uvicorn and fastapi
        if (
            any(prefix in record.name for prefix in ["uvicorn", "fastapi"])
            and record.levelno < logging.WARNING
        ):
            return

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Define empty operation handler, used to suppress unwanted logs
class NoOpHandler(logging.Handler):
    """Empty operation handler, used to suppress unwanted logs"""

    def emit(self, record):
        """Do nothing"""
        pass

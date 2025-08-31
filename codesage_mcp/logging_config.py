"""
Centralized logging configuration for CodeSage MCP server.

This module provides structured logging using structlog with JSON output,
configurable log levels, and multiple sinks (console for development, file for production).
Includes log rotation and rich context for exceptions.
"""

import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from pythonjsonlogger import jsonlogger

from .exceptions import BaseMCPError


def get_environment_info() -> Dict[str, Any]:
    """Get environment information for logging context."""
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "server_version": "1.0.0",  # Update as needed
        "platform": sys.platform,
    }


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = True,
) -> None:
    """
    Configure structured logging with structlog.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file; if None, only console logging
        max_bytes: Max size of log file before rotation
        backup_count: Number of backup files to keep
        json_format: Whether to use JSON format
    """
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",  # structlog will handle formatting
        stream=sys.stdout,
    )

    # Shared processors for structlog
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # JSON formatter
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Human-readable formatter
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )

    # Configure structlog
    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Add file handler if specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(getattr(logging, level.upper()))

        # JSON formatter for file
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        file_handler.setFormatter(json_formatter)

        # Add to root logger
        logging.getLogger().addHandler(file_handler)


def log_exception(
    exc: Exception,
    logger: structlog.stdlib.BoundLogger,
    request_id: Optional[str] = None,
    session_details: Optional[Dict[str, Any]] = None,
    extra_context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an exception with rich context.

    Args:
        exc: The exception to log
        logger: Structlog logger instance
        request_id: Optional request ID
        session_details: Optional session details
        extra_context: Additional context
    """
    context = {
        "timestamp": datetime.utcnow().isoformat(),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "environment": get_environment_info(),
    }

    if isinstance(exc, BaseMCPError):
        context.update({
            "error_code": exc.error_code,
            "error_context": exc.context,
        })

    if request_id:
        context["request_id"] = request_id

    if session_details:
        context["session_details"] = session_details

    if extra_context:
        context.update(extra_context)

    logger.error("Exception occurred", **context)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Decorator for automatic error logging
def log_errors(
    logger_name: str = "codesage_mcp",
    request_id_param: Optional[str] = None,
    session_param: Optional[str] = None,
):
    """
    Decorator to automatically log exceptions in functions.

    Args:
        logger_name: Name of the logger to use
        request_id_param: Parameter name for request ID
        session_param: Parameter name for session details
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                # Extract request_id and session if specified
                request_id = None
                session_details = None

                if request_id_param and request_id_param in kwargs:
                    request_id = kwargs[request_id_param]
                elif request_id_param and len(args) > 0 and hasattr(args[0], request_id_param):
                    request_id = getattr(args[0], request_id_param)

                if session_param and session_param in kwargs:
                    session_details = kwargs[session_param]
                elif session_param and len(args) > 0 and hasattr(args[0], session_param):
                    session_details = getattr(args[0], session_param)

                log_exception(exc, logger, request_id, session_details)
                raise

        def sync_wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                # Extract request_id and session if specified
                request_id = None
                session_details = None

                if request_id_param and request_id_param in kwargs:
                    request_id = kwargs[request_id_param]
                elif request_id_param and len(args) > 0 and hasattr(args[0], request_id_param):
                    request_id = getattr(args[0], request_id_param)

                if session_param and session_param in kwargs:
                    session_details = kwargs[session_param]
                elif session_param and len(args) > 0 and hasattr(args[0], session_param):
                    session_details = getattr(args[0], session_param)

                log_exception(exc, logger, request_id, session_details)
                raise

        if hasattr(func, '__call__') and hasattr(func, '__name__'):
            # Check if it's a coroutine function
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        else:
            return sync_wrapper
    return decorator


# Initialize logging on import if environment variable is set
if os.getenv("CODESAGE_LOG_LEVEL"):
    log_file = os.getenv("CODESAGE_LOG_FILE", "logs/codesage.log")
    setup_logging(
        level=os.getenv("CODESAGE_LOG_LEVEL", "INFO"),
        log_file=log_file,
        json_format=os.getenv("CODESAGE_JSON_LOGS", "true").lower() == "true",
    )
"""
Error Handling Module for CodeSage MCP Server.

This module provides classes for comprehensive error handling, logging, and
error reporting. It includes custom exceptions, structured logging, and
error reporting mechanisms for the MCP server.

Classes:
    CustomException: Base custom exception with error codes
    JSONLogger: Structured JSON logging with performance metrics
    ErrorReporter: Error reporting and aggregation system
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, List
from pathlib import Path

from .exceptions import BaseMCPError

# Set up logger
logger = logging.getLogger(__name__)


class CustomException(BaseMCPError):
    """
    Custom exception class for CodeSage MCP server.

    This exception extends BaseMCPError with additional context and
    error categorization for the MCP server domain.

    Attributes:
        error_code: Unique error code for categorization
        message: Human-readable error message
        context: Additional contextual information
        timestamp: When the error occurred
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Initialize the custom exception.

        Args:
            message: Error message
            error_code: Optional error code (defaults to None)
            context: Optional additional context
            timestamp: Optional timestamp (defaults to current time)
        """
        super().__init__(error_code, message, context, timestamp)

    def __str__(self) -> str:
        """String representation of the exception."""
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "exception_type": "CustomException",
            "stack_trace": traceback.format_exc()
        })
        return base_dict


class JSONLogger:
    """
    Structured JSON logger with performance metrics and error tracking.

    This class provides comprehensive logging capabilities with JSON formatting,
    performance monitoring, and structured error logging for the MCP server.

    Attributes:
        logger: Underlying Python logger instance
        log_file: Optional file path for logging
        json_format: Whether to use JSON formatting
    """

    def __init__(
        self,
        name: str = "codesage_mcp",
        log_file: Optional[str] = None,
        json_format: bool = True,
        level: str = "INFO"
    ):
        """
        Initialize the JSON logger.

        Args:
            name: Logger name
            log_file: Optional file path for logging
            json_format: Whether to use JSON formatting
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.log_file = log_file
        self.json_format = json_format
        self.level = getattr(logging, level.upper(), logging.INFO)

        self._setup_logger()

    def _setup_logger(self):
        """Set up the logger with appropriate handlers and formatters."""
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Set level
        self.logger.setLevel(self.level)

        # Create formatter
        if self.json_format:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if self.log_file:
            try:
                # Create directory if it doesn't exist
                Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(self.level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Failed to set up file logging: {e}")

    def log(self, level: int, message: str, **kwargs):
        """
        Log a message with optional structured data.

        Args:
            level: Logging level (e.g., logging.INFO)
            message: Log message
            **kwargs: Additional structured data
        """
        if self.logger.isEnabledFor(level):
            if kwargs:
                # Add structured data to message if JSON format
                if self.json_format:
                    extra_data = json.dumps(kwargs, default=str)
                    message = f"{message} | {extra_data}"
                else:
                    message = f"{message} | {kwargs}"

            self.logger.log(level, message)

    def log_error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """
        Log an error with exception details.

        Args:
            message: Error message
            exception: Optional exception object
            **kwargs: Additional context
        """
        context = {
            "error_message": message,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }

        if exception:
            context.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "stack_trace": traceback.format_exc()
            })

            if isinstance(exception, BaseMCPError):
                context.update({
                    "error_code": exception.error_code,
                    "error_context": exception.context
                })

        error_msg = f"Error: {message}"
        if self.json_format:
            error_msg += f" | {json.dumps(context, default=str)}"

        self.logger.error(error_msg)

    def log_structured(self, level: int, message: str, **kwargs):
        """
        Log a structured message with key-value pairs.

        Args:
            level: Logging level
            message: Log message
            **kwargs: Structured data as key-value pairs
        """
        self.log(level, message, **kwargs)

    def log_performance(self, operation: str, duration: float, **kwargs):
        """
        Log performance metrics.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            **kwargs: Additional performance context
        """
        context = {
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }

        message = f"Performance: {operation} took {duration:.4f}s"
        if self.json_format:
            message += f" | {json.dumps(context, default=str)}"

        self.logger.info(message)

    def set_file_output(self, filepath: str):
        """
        Set up file output for logging.

        Args:
            filepath: Path to log file
        """
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Remove existing file handlers
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)

            # Add new file handler
            file_handler = logging.FileHandler(filepath)
            file_handler.setLevel(self.level)

            if self.json_format:
                formatter = logging.Formatter(
                    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
                )
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )

            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.log_file = filepath

        except Exception as e:
            self.logger.error(f"Failed to set file output: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics.

        Returns:
            Dictionary with logging statistics
        """
        return {
            "logger_name": self.logger.name,
            "level": logging.getLevelName(self.level),
            "json_format": self.json_format,
            "log_file": self.log_file,
            "handler_count": len(self.logger.handlers)
        }


class ErrorReporter:
    """
    Error reporting and aggregation system.

    This class manages error collection, aggregation, and reporting for the
    MCP server. It supports batch reporting, filtering, and retry mechanisms.

    Attributes:
        errors: List of collected errors
        max_errors: Maximum number of errors to store
        report_endpoint: Optional endpoint for error reporting
    """

    def __init__(self, max_errors: int = 1000, report_endpoint: Optional[str] = None):
        """
        Initialize the error reporter.

        Args:
            max_errors: Maximum number of errors to store
            report_endpoint: Optional endpoint for external error reporting
        """
        self.errors = []
        self.max_errors = max_errors
        self.report_endpoint = report_endpoint
        self.logger = logging.getLogger(__name__)

    def report(self, error_details: Dict[str, Any]) -> None:
        """
        Report an error.

        Args:
            error_details: Dictionary containing error information
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in error_details:
                error_details["timestamp"] = datetime.utcnow().isoformat()

            # Add to error list
            self.errors.append(error_details)

            # Maintain max size
            if len(self.errors) > self.max_errors:
                self.errors.pop(0)

            # Log the error
            self.logger.error(f"Error reported: {error_details.get('message', 'Unknown error')}")

            # Send to external endpoint if configured
            if self.report_endpoint:
                self._send_to_endpoint(error_details)

        except Exception as e:
            self.logger.error(f"Failed to report error: {e}")

    def report_batch(self, errors: List[Dict[str, Any]]) -> None:
        """
        Report multiple errors in batch.

        Args:
            errors: List of error dictionaries
        """
        for error in errors:
            self.report(error)

    def filter_and_report(self, errors: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter errors based on criteria and report matching ones.

        Args:
            errors: List of error dictionaries
            criteria: Filtering criteria

        Returns:
            List of filtered errors that were reported
        """
        filtered_errors = []

        for error in errors:
            if self._matches_criteria(error, criteria):
                filtered_errors.append(error)
                self.report(error)

        return filtered_errors

    def _matches_criteria(self, error: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """
        Check if an error matches the given criteria.

        Args:
            error: Error dictionary
            criteria: Criteria dictionary

        Returns:
            True if error matches criteria
        """
        for key, value in criteria.items():
            if key not in error:
                return False

            error_value = error[key]
            if isinstance(value, dict):
                # Nested criteria
                if not isinstance(error_value, dict):
                    return False
                if not self._matches_criteria(error_value, value):
                    return False
            elif error_value != value:
                return False

        return True

    def get_errors(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get stored errors.

        Args:
            limit: Optional limit on number of errors to return

        Returns:
            List of error dictionaries
        """
        if limit is None:
            return self.errors.copy()
        else:
            return self.errors[-limit:].copy()

    def clear_errors(self) -> None:
        """Clear all stored errors."""
        self.errors.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get error reporting statistics.

        Returns:
            Dictionary with error statistics
        """
        if not self.errors:
            return {
                "total_errors": 0,
                "error_types": {},
                "time_range": None
            }

        # Count error types
        error_types = {}
        for error in self.errors:
            error_type = error.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Get time range
        timestamps = [error.get("timestamp") for error in self.errors if error.get("timestamp")]
        if timestamps:
            time_range = {
                "earliest": min(timestamps),
                "latest": max(timestamps)
            }
        else:
            time_range = None

        return {
            "total_errors": len(self.errors),
            "error_types": error_types,
            "time_range": time_range,
            "max_errors": self.max_errors,
            "report_endpoint": self.report_endpoint
        }

    def _send_to_endpoint(self, error_details: Dict[str, Any]) -> None:
        """
        Send error to external reporting endpoint.

        Args:
            error_details: Error details to send
        """
        # This is a placeholder for external error reporting
        # In a real implementation, you'd make HTTP requests to the endpoint
        self.logger.debug(f"Would send error to {self.report_endpoint}: {error_details}")

    def report_with_retry(self, error_details: Dict[str, Any], max_retries: int = 3) -> None:
        """
        Report an error with retry mechanism.

        Args:
            error_details: Error details to report
            max_retries: Maximum number of retry attempts
        """
        for attempt in range(max_retries + 1):
            try:
                self.report(error_details)
                return
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(f"Error reporting failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    continue
                else:
                    self.logger.error(f"Error reporting failed after {max_retries + 1} attempts: {e}")
                    raise
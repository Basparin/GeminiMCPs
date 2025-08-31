"""
Custom exception classes for CodeSage MCP server.

This module defines a hierarchy of custom exceptions tailored to the MCP server's domains,
providing structured error handling with error codes, timestamps, and contextual information.
"""

from datetime import datetime
from typing import Any, Dict, Optional


class BaseMCPError(Exception):
    """
    Base exception class for all CodeSage MCP errors.

    Provides common attributes for structured error handling and logging.

    Attributes:
        error_code (str): Unique error code for categorization.
        message (str): Human-readable error message.
        context (Dict[str, Any]): Additional contextual information.
        timestamp (datetime): When the error occurred.
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Initialize the base MCP error.

        Args:
            error_code: Unique identifier for the error type.
            message: Descriptive error message.
            context: Optional dictionary with additional error context.
            timestamp: Optional timestamp; defaults to current time.
        """
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.context = context or {}
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary for logging or response formatting.

        Returns:
            Dictionary representation of the error.
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


class ToolExecutionError(BaseMCPError):
    """
    Exception raised when a tool execution fails.

    Used for errors occurring during MCP tool operations.
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the tool execution error.

        Args:
            message: Error message describing the failure.
            tool_name: Name of the tool that failed.
            context: Additional context about the execution failure.
        """
        error_code = "TOOL_EXECUTION_FAILED"
        full_context = {"tool_name": tool_name} if tool_name else {}
        if context:
            full_context.update(context)
        super().__init__(error_code, message, full_context)


class InvalidRequestError(BaseMCPError):
    """
    Exception raised for invalid or malformed requests.

    Used for JSON-RPC request validation failures.
    """

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the invalid request error.

        Args:
            message: Error message describing the validation failure.
            request_id: ID of the invalid request.
            context: Additional context about the request.
        """
        error_code = "INVALID_REQUEST"
        full_context = {"request_id": request_id} if request_id else {}
        if context:
            full_context.update(context)
        super().__init__(error_code, message, full_context)


class IndexingError(BaseMCPError):
    """
    Exception raised during indexing operations.

    Used for errors in FAISS indexing, file processing, or index management.
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the indexing error.

        Args:
            message: Error message describing the indexing failure.
            file_path: Path to the file being indexed.
            operation: Type of indexing operation (e.g., 'load', 'save').
            context: Additional context about the indexing operation.
        """
        error_code = "INDEXING_FAILED"
        full_context = {
            "file_path": file_path,
            "operation": operation,
        }
        if context:
            full_context.update(context)
        super().__init__(error_code, message, full_context)


class ConfigurationError(BaseMCPError):
    """
    Exception raised for configuration-related errors.

    Used for missing or invalid configuration settings.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the configuration error.

        Args:
            message: Error message describing the configuration issue.
            config_key: Name of the problematic configuration key.
            context: Additional context about the configuration.
        """
        error_code = "CONFIGURATION_ERROR"
        full_context = {"config_key": config_key} if config_key else {}
        if context:
            full_context.update(context)
        super().__init__(error_code, message, full_context)
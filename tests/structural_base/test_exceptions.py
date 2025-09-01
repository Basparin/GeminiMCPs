"""
Comprehensive unit tests for custom exception hierarchy.

This module tests the custom exception classes defined in codesage_mcp.exceptions,
focusing on proper error code assignment, context handling, timestamp generation,
and inheritance from BaseMCPError.
"""

import pytest
from datetime import datetime

from codesage_mcp.core.exceptions import (
    BaseMCPError,
    ToolExecutionError,
    InvalidRequestError,
    IndexingError,
    ConfigurationError,
)


class TestBaseMCPError:
    """Test cases for BaseMCPError class."""

    def test_initialization_with_minimal_params(self):
        """Test BaseMCPError initialization with minimal parameters."""
        error = BaseMCPError("TEST_ERROR", "Test message")

        assert error.error_code == "TEST_ERROR"
        assert error.message == "Test message"
        assert error.context == {}
        assert isinstance(error.timestamp, datetime)

    def test_initialization_with_context(self):
        """Test BaseMCPError initialization with context dictionary."""
        context = {"key": "value", "number": 42}
        error = BaseMCPError("TEST_ERROR", "Test message", context=context)

        assert error.context == context

    def test_initialization_with_custom_timestamp(self):
        """Test BaseMCPError initialization with custom timestamp."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        error = BaseMCPError("TEST_ERROR", "Test message", timestamp=custom_time)

        assert error.timestamp == custom_time

    def test_to_dict_conversion(self):
        """Test conversion of error to dictionary format."""
        context = {"operation": "test_op", "user_id": 123}
        timestamp = datetime(2023, 1, 1, 12, 0, 0)

        error = BaseMCPError("TEST_ERROR", "Test message", context=context, timestamp=timestamp)
        result = error.to_dict()

        expected = {
            "error_code": "TEST_ERROR",
            "message": "Test message",
            "context": context,
            "timestamp": timestamp.isoformat(),
        }
        assert result == expected

    def test_exception_inheritance(self):
        """Test that BaseMCPError properly inherits from Exception."""
        error = BaseMCPError("TEST_ERROR", "Test message")

        assert isinstance(error, Exception)
        assert str(error) == "Test message"

    def test_context_immutability(self):
        """Test that context dictionary is not modified externally."""
        context = {"key": "value"}
        error = BaseMCPError("TEST_ERROR", "Test message", context=context)

        # Modify original context
        context["new_key"] = "new_value"

        # Error's context should remain unchanged
        assert "new_key" not in error.context
        assert error.context == {"key": "value"}


class TestToolExecutionError:
    """Test cases for ToolExecutionError class."""

    def test_initialization_without_tool_name(self):
        """Test ToolExecutionError initialization without tool name."""
        error = ToolExecutionError("Tool failed to execute")

        assert error.error_code == "TOOL_EXECUTION_FAILED"
        assert error.message == "Tool failed to execute"
        assert error.context == {}

    def test_initialization_with_tool_name(self):
        """Test ToolExecutionError initialization with tool name."""
        error = ToolExecutionError("Tool failed to execute", tool_name="test_tool")

        assert error.context == {"tool_name": "test_tool"}

    def test_initialization_with_additional_context(self):
        """Test ToolExecutionError initialization with additional context."""
        additional_context = {"param": "value", "timeout": 30}
        error = ToolExecutionError(
            "Tool failed to execute",
            tool_name="test_tool",
            context=additional_context
        )

        expected_context = {"tool_name": "test_tool", "param": "value", "timeout": 30}
        assert error.context == expected_context

    def test_context_merging_priority(self):
        """Test that tool_name takes precedence in context merging."""
        error = ToolExecutionError(
            "Tool failed",
            tool_name="test_tool",
            context={"tool_name": "different_tool", "other": "value"}
        )

        # tool_name from parameter should override context
        assert error.context["tool_name"] == "test_tool"
        assert error.context["other"] == "value"


class TestInvalidRequestError:
    """Test cases for InvalidRequestError class."""

    def test_initialization_without_request_id(self):
        """Test InvalidRequestError initialization without request ID."""
        error = InvalidRequestError("Invalid request format")

        assert error.error_code == "INVALID_REQUEST"
        assert error.message == "Invalid request format"
        assert error.context == {}

    def test_initialization_with_request_id(self):
        """Test InvalidRequestError initialization with request ID."""
        error = InvalidRequestError("Invalid request format", request_id="req_123")

        assert error.context == {"request_id": "req_123"}

    def test_initialization_with_additional_context(self):
        """Test InvalidRequestError initialization with additional context."""
        additional_context = {"field": "code", "expected": "string"}
        error = InvalidRequestError(
            "Invalid request format",
            request_id="req_123",
            context=additional_context
        )

        expected_context = {"request_id": "req_123", "field": "code", "expected": "string"}
        assert error.context == expected_context

    def test_context_merging_with_request_id_override(self):
        """Test context merging when request_id is provided in both places."""
        error = InvalidRequestError(
            "Invalid request",
            request_id="req_123",
            context={"request_id": "different_req", "other": "value"}
        )

        # request_id from parameter should override context
        assert error.context["request_id"] == "req_123"
        assert error.context["other"] == "value"


class TestIndexingError:
    """Test cases for IndexingError class."""

    def test_initialization_minimal(self):
        """Test IndexingError initialization with minimal parameters."""
        error = IndexingError("Indexing operation failed")

        assert error.error_code == "INDEXING_FAILED"
        assert error.message == "Indexing operation failed"
        assert error.context == {}

    def test_initialization_with_file_path(self):
        """Test IndexingError initialization with file path."""
        error = IndexingError("Indexing operation failed", file_path="/path/to/file.py")

        assert error.context == {"file_path": "/path/to/file.py", "operation": None}

    def test_initialization_with_operation(self):
        """Test IndexingError initialization with operation type."""
        error = IndexingError("Indexing operation failed", operation="load")

        assert error.context == {"file_path": None, "operation": "load"}

    def test_initialization_with_all_params(self):
        """Test IndexingError initialization with all parameters."""
        additional_context = {"chunk_size": 1000, "total_chunks": 50}
        error = IndexingError(
            "Indexing operation failed",
            file_path="/path/to/file.py",
            operation="save",
            context=additional_context
        )

        expected_context = {
            "file_path": "/path/to/file.py",
            "operation": "save",
            "chunk_size": 1000,
            "total_chunks": 50
        }
        assert error.context == expected_context

    def test_context_merging_with_file_path_override(self):
        """Test context merging when file_path is provided in both places."""
        error = IndexingError(
            "Indexing failed",
            file_path="/path/to/file.py",
            context={"file_path": "/different/path.py", "other": "value"}
        )

        # file_path from parameter should override context
        assert error.context["file_path"] == "/path/to/file.py"
        assert error.context["other"] == "value"


class TestConfigurationError:
    """Test cases for ConfigurationError class."""

    def test_initialization_without_config_key(self):
        """Test ConfigurationError initialization without config key."""
        error = ConfigurationError("Configuration is invalid")

        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.message == "Configuration is invalid"
        assert error.context == {}

    def test_initialization_with_config_key(self):
        """Test ConfigurationError initialization with config key."""
        error = ConfigurationError("Configuration is invalid", config_key="API_KEY")

        assert error.context == {"config_key": "API_KEY"}

    def test_initialization_with_additional_context(self):
        """Test ConfigurationError initialization with additional context."""
        additional_context = {"expected_type": "string", "provided_type": "int"}
        error = ConfigurationError(
            "Configuration is invalid",
            config_key="API_KEY",
            context=additional_context
        )

        expected_context = {"config_key": "API_KEY", "expected_type": "string", "provided_type": "int"}
        assert error.context == expected_context

    def test_context_merging_with_config_key_override(self):
        """Test context merging when config_key is provided in both places."""
        error = ConfigurationError(
            "Configuration error",
            config_key="API_KEY",
            context={"config_key": "DIFFERENT_KEY", "other": "value"}
        )

        # config_key from parameter should override context
        assert error.context["config_key"] == "API_KEY"
        assert error.context["other"] == "value"


class TestExceptionHierarchyIntegration:
    """Integration tests for exception hierarchy and usage patterns."""

    def test_exception_raising_and_catching_with_context(self):
        """Test raising and catching custom exceptions with proper context."""
        with pytest.raises(ToolExecutionError) as exc_info:
            raise ToolExecutionError(
                "Tool execution failed",
                tool_name="code_analysis",
                context={"input_length": 1000, "timeout": 30}
            )

        error = exc_info.value
        assert error.error_code == "TOOL_EXECUTION_FAILED"
        assert error.context["tool_name"] == "code_analysis"
        assert error.context["input_length"] == 1000
        assert error.context["timeout"] == 30

    def test_multiple_exception_types_in_sequence(self):
        """Test handling multiple exception types in sequence."""
        exceptions_caught = []

        # Test ToolExecutionError
        try:
            raise ToolExecutionError("Tool failed", tool_name="test_tool")
        except ToolExecutionError as e:
            exceptions_caught.append(("ToolExecutionError", e.error_code, e.context))

        # Test InvalidRequestError
        try:
            raise InvalidRequestError("Invalid request", request_id="req_123")
        except InvalidRequestError as e:
            exceptions_caught.append(("InvalidRequestError", e.error_code, e.context))

        # Test IndexingError
        try:
            raise IndexingError("Indexing failed", file_path="/test/file.py", operation="load")
        except IndexingError as e:
            exceptions_caught.append(("IndexingError", e.error_code, e.context))

        # Test ConfigurationError
        try:
            raise ConfigurationError("Config error", config_key="API_KEY")
        except ConfigurationError as e:
            exceptions_caught.append(("ConfigurationError", e.error_code, e.context))

        assert len(exceptions_caught) == 4

        # Verify each exception type
        assert exceptions_caught[0] == ("ToolExecutionError", "TOOL_EXECUTION_FAILED", {"tool_name": "test_tool"})
        assert exceptions_caught[1] == ("InvalidRequestError", "INVALID_REQUEST", {"request_id": "req_123"})
        assert exceptions_caught[2] == ("IndexingError", "INDEXING_FAILED", {"file_path": "/test/file.py", "operation": "load"})
        assert exceptions_caught[3] == ("ConfigurationError", "CONFIGURATION_ERROR", {"config_key": "API_KEY"})

    def test_exception_inheritance_chain(self):
        """Test that all custom exceptions inherit from BaseMCPError and Exception."""
        exceptions = [
            ToolExecutionError("test"),
            InvalidRequestError("test"),
            IndexingError("test"),
            ConfigurationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, BaseMCPError)
            assert isinstance(exc, Exception)
            assert hasattr(exc, 'error_code')
            assert hasattr(exc, 'message')
            assert hasattr(exc, 'context')
            assert hasattr(exc, 'timestamp')
            assert hasattr(exc, 'to_dict')

    def test_timestamp_consistency(self):
        """Test that timestamps are consistent and reasonable."""

        start_time = datetime.utcnow()
        error = BaseMCPError("TEST", "test message")
        end_time = datetime.utcnow()

        # Timestamp should be between start and end
        assert start_time <= error.timestamp <= end_time

        # Timestamp should be recent (within last second)
        time_diff = (datetime.utcnow() - error.timestamp).total_seconds()
        assert time_diff < 1.0

    def test_to_dict_serialization(self):
        """Test that to_dict produces valid JSON-serializable output."""
        import json

        error = ToolExecutionError(
            "Tool failed",
            tool_name="test_tool",
            context={"param": "value", "number": 42}
        )

        result = error.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)

        assert parsed["error_code"] == "TOOL_EXECUTION_FAILED"
        assert parsed["message"] == "Tool failed"
        assert parsed["context"]["tool_name"] == "test_tool"
        assert parsed["context"]["param"] == "value"
        assert parsed["context"]["number"] == 42
        assert "timestamp" in parsed
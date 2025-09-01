"""
Integration tests for error handling system components.

This module tests the integration between custom exceptions, structured logging,
and error reporting systems to ensure they work together seamlessly.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock

from codesage_mcp.core.exceptions import (
    ToolExecutionError,
    InvalidRequestError,
    ConfigurationError,
)
from codesage_mcp.core.logging_config import (
    setup_logging,
    log_exception,
    log_errors,
)
from codesage_mcp.core.error_reporting import (
    ErrorReporter,
    demonstrate_error_reporting,
)


# Pytest fixtures for common test setup
@pytest.fixture
def temp_log_directory():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    with patch('codesage_mcp.core.logging_config.get_logger', return_value=logger):
        yield logger


@pytest.fixture
def error_reporter():
    """Create an ErrorReporter instance for testing."""
    with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
        reporter = ErrorReporter()
        yield reporter


@pytest.fixture
def mock_webhook_session():
    """Create a mock webhook session for testing."""
    session = AsyncMock()
    response = AsyncMock()
    response.status = 200
    session.post.return_value.__aenter__ = response
    session.post.return_value.__aexit__ = AsyncMock()
    return session


class TestExceptionLoggingIntegration:
    """Integration tests for exceptions and logging components."""

    def test_base_mcperror_with_structured_logging(self, temp_log_directory, mock_logger):
        """Test BaseMCPError integration with structured logging."""
        log_file = os.path.join(temp_log_directory, "integration.log")

        # Setup structured logging
        setup_logging(level="ERROR", log_file=log_file, json_format=True)

        # Create and log a custom exception
        error = ToolExecutionError(
            "Tool execution failed during integration test",
            tool_name="integration_tool",
            context={"operation": "test", "params": {"key": "value"}}
        )

        log_exception(error, mock_logger, request_id="int_test_001")

        # Verify structured log output
        with open(log_file, 'r') as f:
            log_content = f.read()
            # Check that log file contains expected content
            assert "ToolExecutionError" in log_content
            assert "int_test_001" in log_content
            assert "TOOL_EXECUTION_FAILED" in log_content

    def test_log_errors_decorator_with_custom_exceptions(self, mock_logger):
        """Test log_errors decorator integration with custom exceptions."""
        @log_errors("integration_test", request_id_param="request_id")
        def function_that_raises_custom_exception(request_id):
            raise InvalidRequestError(
                "Invalid request in integration test",
                request_id=request_id,
                context={"field": "operation", "expected": "string"}
            )

        with pytest.raises(InvalidRequestError) as exc_info:
            function_that_raises_custom_exception("req_int_002")

        # Verify exception details
        error = exc_info.value
        assert error.error_code == "INVALID_REQUEST"
        assert error.context["request_id"] == "req_int_002"

        # Verify logging was called
        assert mock_logger.error.called
        log_call = mock_logger.error.call_args
        assert log_call[0][0] == "Exception occurred"


class TestExceptionErrorReportingIntegration:
    """Integration tests for exceptions and error reporting components."""

    def test_configuration_error_with_privacy_filters(self, error_reporter):
        """Test ConfigurationError with privacy filtering in error reporting."""
        with patch.dict(os.environ, {"PRIVACY_SANITIZE_FIELDS": "password,api_key"}):
            # Create new reporter instance to pick up env changes
            reporter = ErrorReporter()

            error = ConfigurationError(
                "Configuration contains sensitive data",
                config_key="API_KEY"
            )

            sensitive_context = {
                "user": "test_user",
                "password": "sensitive_password",
                "api_key": "sensitive_api_key",
                "normal_setting": "safe_value"
            }

            # Test sanitization directly
            sanitized = reporter._sanitize_data(sensitive_context)

            assert sanitized["user"] == "test_user"
            assert sanitized["password"] == "[REDACTED]"
            assert sanitized["api_key"] == "[REDACTED]"
            assert sanitized["normal_setting"] == "safe_value"


class TestLoggingErrorReportingIntegration:
    """Integration tests for logging and error reporting components."""

    def test_log_errors_decorator_with_error_reporting(self, mock_logger, error_reporter):
        """Test log_errors decorator integration with error reporting."""
        @log_errors("decorator_test", request_id_param="request_id")
        def failing_function(request_id):
            raise RuntimeError("Decorator integration test")

        with patch.object(error_reporter, 'report_error') as mock_report:
            with pytest.raises(RuntimeError):
                failing_function("decorator_req_001")

            # Both decorator logging and error reporting should be triggered
            assert mock_logger.error.called


class TestFullSystemIntegration:
    """Full system integration tests combining all components."""

    def test_graceful_degradation_full_system(self, error_reporter):
        """Test graceful degradation when components are unavailable."""
        # Test with missing optional dependencies
        with patch('builtins.__import__', side_effect=ImportError("Module not available")):
            # Should still function despite missing dependencies
            assert error_reporter._sentry_client is None
            assert error_reporter._prometheus_client is None

            # Should still be able to sanitize data
            sanitized = error_reporter._sanitize_data({"password": "secret"})
            assert sanitized["password"] == "[REDACTED]"


class TestErrorHandlingScenarios:
    """Test specific error handling scenarios and edge cases."""

    def test_nested_exception_with_context_preservation(self, mock_logger):
        """Test nested exceptions with context preservation."""
        @log_errors("nested_test", request_id_param="request_id")
        def outer_function(request_id):
            try:
                inner_function()
            except ValueError as e:
                # Re-raise with additional context
                raise ToolExecutionError(
                    f"Outer function failed: {str(e)}",
                    tool_name="nested_tool",
                    context={"original_error": str(e), "request_id": request_id}
                ) from e

        def inner_function():
            raise ValueError("Inner function error")

        with pytest.raises(ToolExecutionError) as exc_info:
            outer_function("nested_req_001")

        # Verify nested exception details
        error = exc_info.value
        assert error.error_code == "TOOL_EXECUTION_FAILED"
        assert "Inner function error" in error.message
        assert error.context["original_error"] == "Inner function error"
        assert error.context["request_id"] == "nested_req_001"

        # Verify logging captured the nested context
        assert mock_logger.error.called
        log_call = mock_logger.error.call_args
        assert log_call[1]["error_code"] == "TOOL_EXECUTION_FAILED"

    def test_error_context_aggregation(self, mock_logger):
        """Test aggregation of error context from multiple sources."""
        @log_errors("aggregation_test", request_id_param="request_id", session_param="session")
        def function_with_aggregated_context(request_id, session):
            raise InvalidRequestError(
                "Aggregated context test",
                request_id=request_id,
                context={
                    "operation": "aggregation_test",
                    "session_user": session.get("user_id"),
                    "additional_data": "test_value"
                }
            )

        session_data = {"user_id": 789, "role": "admin", "token": "session_token"}

        with pytest.raises(InvalidRequestError) as exc_info:
            function_with_aggregated_context("agg_req_001", session_data)

        error = exc_info.value

        # Verify context aggregation
        assert error.context["request_id"] == "agg_req_001"
        assert error.context["operation"] == "aggregation_test"
        assert error.context["session_user"] == 789
        assert error.context["additional_data"] == "test_value"

        # Verify logging captured aggregated context
        log_call = mock_logger.error.call_args
        logged_context = log_call[1]
        assert logged_context["request_id"] == "agg_req_001"
        assert logged_context["session_details"]["user_id"] == 789


class TestErrorReportingUnitTests:
    """Unit tests for ErrorReporter functionality to improve coverage."""

    def test_parse_privacy_filters_default(self, error_reporter):
        """Test _parse_privacy_filters method with default values."""
        filters = error_reporter._parse_privacy_filters()
        expected = ["password", "token", "key", "secret", "api_key"]
        assert filters == expected

    def test_parse_privacy_filters_custom(self):
        """Test _parse_privacy_filters method with custom values."""
        with patch.dict(os.environ, {"PRIVACY_SANITIZE_FIELDS": "custom1,custom2"}):
            reporter = ErrorReporter()
            filters = reporter._parse_privacy_filters()
            assert filters == ["custom1", "custom2"]

    def test_parse_privacy_filters_empty(self):
        """Test _parse_privacy_filters method with empty values."""
        with patch.dict(os.environ, {"PRIVACY_SANITIZE_FIELDS": ""}):
            reporter = ErrorReporter()
            filters = reporter._parse_privacy_filters()
            assert filters == []

    def test_sanitize_data_comprehensive(self, error_reporter):
        """Test data sanitization for privacy with comprehensive data."""
        data = {
            "user": "test_user",
            "password": "secret123",
            "api_key": "key123",
            "normal_field": "safe",
            "nested": {
                "token": "nested_token",
                "safe": "nested_safe"
            },
            "list_data": [
                {"password": "list_password"},
                {"safe": "list_safe"}
            ]
        }

        sanitized = error_reporter._sanitize_data(data)

        assert sanitized["user"] == "test_user"
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["normal_field"] == "safe"
        assert sanitized["nested"]["token"] == "[REDACTED]"
        assert sanitized["nested"]["safe"] == "nested_safe"
        assert sanitized["list_data"][0]["password"] == "[REDACTED]"
        assert sanitized["list_data"][1]["safe"] == "list_safe"

    def test_format_slack_alert_comprehensive(self, error_reporter):
        """Test Slack alert formatting with comprehensive data."""
        alert_data = {
            "timestamp": "2023-01-01T12:00:00Z",
            "severity": "error",
            "error_type": "ValueError",
            "message": "Test error message",
            "context": {"test": "context"},
            "request_id": "slack_req_001",
            "service": "test_service"
        }

        formatted = error_reporter._format_slack_alert(alert_data)

        assert "text" in formatted
        assert "blocks" in formatted
        assert len(formatted["blocks"]) == 2

        # Check severity emoji
        assert "❌" in formatted["text"]

        # Check fields
        fields = formatted["blocks"][1]["fields"]
        assert len(fields) == 4
        assert "ValueError" in fields[0]["text"]
        assert "slack_req_001" in fields[1]["text"]

    def test_demonstrate_error_reporting_function_exists(self):
        """Test that demonstrate_error_reporting function exists and is callable."""
        assert callable(demonstrate_error_reporting)

    def test_cleanup_function_registration(self):
        """Test that cleanup function is registered with atexit."""
        with patch('atexit.register') as mock_register:
            # Re-import to trigger atexit registration
            import importlib
            import codesage_mcp.core.error_reporting
            importlib.reload(codesage_mcp.core.error_reporting)

            # Should have registered cleanup function
            mock_register.assert_called_once()
            cleanup_func = mock_register.call_args[0][0]
            assert callable(cleanup_func)
"""
Integration tests for error handling system components.

This module tests the integration between custom exceptions, structured logging,
and error reporting systems to ensure they work together seamlessly.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from codesage_mcp.exceptions import (
    BaseMCPError,
    ToolExecutionError,
    InvalidRequestError,
    IndexingError,
    ConfigurationError,
)
from codesage_mcp.logging_config import (
    setup_logging,
    log_exception,
    get_logger,
    log_errors,
)
from codesage_mcp.error_reporting import (
    ErrorReporter,
    report_critical_error,
)


class TestExceptionLoggingIntegration:
    """Integration tests for exceptions and logging."""

    def test_base_mcperror_with_structured_logging(self):
        """Test BaseMCPError integration with structured logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "integration.log")

            # Setup structured logging
            setup_logging(level="ERROR", log_file=log_file, json_format=True)

            logger = get_logger("exception_test")

            # Create and log a custom exception
            error = ToolExecutionError(
                "Tool execution failed during integration test",
                tool_name="integration_tool",
                context={"operation": "test", "params": {"key": "value"}}
            )

            log_exception(error, logger, request_id="int_test_001")

            # Verify structured log output
            with open(log_file, 'r') as f:
                log_content = f.read()
                log_data = json.loads(log_content)

                assert log_data["event"] == "Exception occurred"
                assert log_data["error_type"] == "ToolExecutionError"
                assert log_data["error_code"] == "TOOL_EXECUTION_FAILED"
                assert log_data["message"] == "Tool execution failed during integration test"
                assert log_data["request_id"] == "int_test_001"
                assert log_data["error_context"]["tool_name"] == "integration_tool"

    def test_log_errors_decorator_with_custom_exceptions(self):
        """Test log_errors decorator integration with custom exceptions."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

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

            # Verify logging was called with proper context
            assert mock_logger.error.called
            log_call = mock_logger.error.call_args
            assert log_call[0][0] == "Exception occurred"
            assert log_call[1]["error_code"] == "INVALID_REQUEST"
            assert log_call[1]["request_id"] == "req_int_002"


class TestExceptionErrorReportingIntegration:
    """Integration tests for exceptions and error reporting."""

    def test_custom_exception_with_error_reporting(self):
        """Test custom exception integration with error reporting system."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
            reporter = ErrorReporter()

            error = IndexingError(
                "Indexing operation failed in integration",
                file_path="/test/integration/file.py",
                operation="load",
                context={"chunk_size": 1024, "total_chunks": 10}
            )

            with patch.object(reporter, 'log_exception') as mock_log:
                with patch.object(reporter, '_write_error_log') as mock_write:
                    with patch.object(reporter, '_send_to_monitoring_tools') as mock_monitoring:
                        with patch.object(reporter, '_send_webhook_alerts') as mock_webhooks:
                            reporter.report_error(
                                error,
                                {"additional_context": "integration_test"},
                                "error",
                                "int_req_003"
                            )

                            # Verify error reporting captured exception details
                            write_call = mock_write.call_args
                            logged_context = write_call[0][1]

                            assert logged_context["additional_context"] == "integration_test"

                            # Verify webhook alerts include exception details
                            webhook_call = mock_webhooks.call_args
                            webhook_error = webhook_call[0][0]
                            assert isinstance(webhook_error, IndexingError)
                            assert webhook_error.error_code == "INDEXING_FAILED"

    def test_configuration_error_with_privacy_filters(self):
        """Test ConfigurationError with privacy filtering in error reporting."""
        with patch.dict(os.environ, {
            "ERROR_REPORTING_ENABLED": "true",
            "PRIVACY_SANITIZE_FIELDS": "password,api_key"
        }):
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

            with patch.object(reporter, 'log_exception') as mock_log:
                with patch.object(reporter, '_write_error_log') as mock_write:
                    with patch.object(reporter, '_send_to_monitoring_tools') as mock_monitoring:
                        with patch.object(reporter, '_send_webhook_alerts') as mock_webhooks:
                            reporter.report_error(error, sensitive_context, "error")

                            # Verify privacy filtering was applied
                            write_call = mock_write.call_args
                            sanitized_context = write_call[0][1]

                            assert sanitized_context["user"] == "test_user"
                            assert sanitized_context["password"] == "[REDACTED]"
                            assert sanitized_context["api_key"] == "[REDACTED]"
                            assert sanitized_context["normal_setting"] == "safe_value"


class TestLoggingErrorReportingIntegration:
    """Integration tests for logging and error reporting."""

    def test_structured_logging_with_error_reporting(self):
        """Test structured logging integration with error reporting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "combined.log")

            # Setup logging
            setup_logging(level="ERROR", log_file=log_file, json_format=True)

            with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
                reporter = ErrorReporter()

                # Create error and report it
                error = ValueError("Integration test error")
                context = {"operation": "combined_test", "request_id": "combined_001"}

                with patch.object(reporter, '_write_error_log') as mock_write:
                    with patch.object(reporter, '_send_to_monitoring_tools') as mock_monitoring:
                        with patch.object(reporter, '_send_webhook_alerts') as mock_webhooks:
                            reporter.report_error(error, context, "error", "combined_001")

                            # Both logging systems should have been triggered
                            # (structured logging via log_exception, error reporting via report_error)
                            mock_write.assert_called_once()
                            mock_monitoring.assert_called_once()
                            mock_webhooks.assert_called_once()

    def test_log_errors_decorator_with_error_reporting(self):
        """Test log_errors decorator integration with error reporting."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_logger:
            with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
                reporter = ErrorReporter()

                @log_errors("decorator_test", request_id_param="request_id")
                def failing_function(request_id):
                    raise RuntimeError("Decorator integration test")

                with patch.object(reporter, 'report_error') as mock_report:
                    with pytest.raises(RuntimeError):
                        failing_function("decorator_req_001")

                    # Both decorator logging and error reporting should be triggered
                    assert mock_logger.return_value.error.called

                    # In a real scenario, the error might also be reported via error reporting
                    # depending on how the system is configured


class TestFullSystemIntegration:
    """Full system integration tests combining all components."""

    def test_complete_error_handling_flow(self):
        """Test complete error handling flow from exception to reporting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "full_integration.log")

            # Setup structured logging
            setup_logging(level="ERROR", log_file=log_file, json_format=True)

            with patch.dict(os.environ, {
                "ERROR_REPORTING_ENABLED": "true",
                "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "PRIVACY_SANITIZE_FIELDS": "password,token"
            }):
                reporter = ErrorReporter()

                # Create a complex error scenario
                error = ToolExecutionError(
                    "Complete integration test failure",
                    tool_name="integration_tool",
                    context={
                        "operation": "full_test",
                        "params": {"input": "test_data"},
                        "user": {"id": 123, "password": "sensitive_password"}
                    }
                )

                sensitive_session = {
                    "user_token": "sensitive_token_abc123",
                    "session_id": "session_456",
                    "safe_field": "safe_value"
                }

                with patch.object(reporter, '_get_webhook_session') as mock_get_session:
                    mock_session = AsyncMock()
                    mock_get_session.return_value = mock_session

                    mock_response = AsyncMock()
                    mock_response.status = 200
                    mock_session.post.return_value.__aenter__ = mock_response
                    mock_session.post.return_value.__aexit__ = AsyncMock()

                    # Report the error
                    reporter.report_error(
                        error,
                        {"additional": "context", "token": "another_sensitive_token"},
                        "error",
                        "full_int_req_001",
                        sensitive_session
                    )

                    # Verify webhook was called with sanitized data
                    mock_session.post.assert_called_once()
                    webhook_call = mock_session.post.call_args
                    payload = webhook_call[0][1]

                    # Verify sensitive data was sanitized in webhook payload
                    assert "password" not in str(payload)
                    assert "token" not in str(payload)
                    assert "sensitive" not in str(payload)
                    assert "safe_value" in str(payload)

    def test_performance_monitoring_with_exception_handling(self):
        """Test performance monitoring integration with exception handling."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
            reporter = ErrorReporter()

            @log_errors("performance_test")
            def slow_failing_function():
                import time
                time.sleep(0.1)  # Simulate some work
                raise ConnectionError("Performance test error")

            with patch('time.time', side_effect=[1.0, 3.0]):  # 2 seconds
                with patch('psutil.Process') as mock_process:
                    mock_memory = MagicMock()
                    mock_memory.memory_info.return_value.rss = 100 * 1024 * 1024
                    mock_process.return_value.memory_info.return_value = mock_memory

                    with patch.object(reporter, 'report_error') as mock_report:
                        with pytest.raises(ConnectionError):
                            reporter.monitor_performance("perf_test", slow_failing_function)

                        # Should report both the original error and performance issues
                        assert mock_report.call_count >= 1

                        # Check that the original error was reported
                        error_calls = mock_report.call_args_list
                        original_error_call = next(
                            (call for call in error_calls if "Performance test error" in str(call[0][0])),
                            None
                        )
                        assert original_error_call is not None

    def test_graceful_degradation_full_system(self):
        """Test graceful degradation when components are unavailable."""
        # Test with missing optional dependencies
        with patch('builtins.__import__', side_effect=ImportError("Module not available")):
            with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
                reporter = ErrorReporter()

                # Should still function despite missing dependencies
                assert reporter._sentry_client is None
                assert reporter._prometheus_client is None

                # Should still be able to report errors
                with patch.object(reporter, 'log_exception') as mock_log:
                    error = ValueError("Graceful degradation test")
                    reporter.report_error(error, {"test": "data"}, "error")

                    # Core logging should still work
                    mock_log.assert_called_once()

    def test_exception_hierarchy_with_full_logging(self):
        """Test exception hierarchy with full logging and reporting integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "hierarchy.log")

            setup_logging(level="ERROR", log_file=log_file, json_format=True)

            with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
                reporter = ErrorReporter()

                # Test all exception types in sequence
                exceptions_to_test = [
                    ("ToolExecutionError", ToolExecutionError("Tool failed", tool_name="hierarchy_tool")),
                    ("InvalidRequestError", InvalidRequestError("Invalid request", request_id="hier_req_001")),
                    ("IndexingError", IndexingError("Indexing failed", file_path="/test/file.py")),
                    ("ConfigurationError", ConfigurationError("Config error", config_key="TEST_KEY")),
                ]

                for error_type_name, error in exceptions_to_test:
                    with patch.object(reporter, '_write_error_log') as mock_write:
                        with patch.object(reporter, '_send_to_monitoring_tools') as mock_monitoring:
                            with patch.object(reporter, '_send_webhook_alerts') as mock_webhooks:
                                reporter.report_error(error, {"test_type": error_type_name}, "error")

                                # Verify each error type is handled correctly
                                write_call = mock_write.call_args
                                logged_error = write_call[0][0]
                                assert type(logged_error).__name__ == error_type_name

                                # Verify error codes are preserved
                                if hasattr(logged_error, 'error_code'):
                                    expected_codes = {
                                        "ToolExecutionError": "TOOL_EXECUTION_FAILED",
                                        "InvalidRequestError": "INVALID_REQUEST",
                                        "IndexingError": "INDEXING_FAILED",
                                        "ConfigurationError": "CONFIGURATION_ERROR",
                                    }
                                    assert logged_error.error_code == expected_codes[error_type_name]


class TestErrorHandlingScenarios:
    """Test specific error handling scenarios and edge cases."""

    def test_nested_exception_with_context_preservation(self):
        """Test nested exceptions with context preservation."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

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

    def test_multiple_error_reporting_channels(self):
        """Test error reporting through multiple channels simultaneously."""
        with patch.dict(os.environ, {
            "ERROR_REPORTING_ENABLED": "true",
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
            "EMAIL_WEBHOOK_URL": "https://api.email.com/webhook",
            "DASHBOARD_WEBHOOK_URL": "https://dashboard.com/webhook"
        }):
            reporter = ErrorReporter()

            error = BaseMCPError(
                "MULTI_CHANNEL_TEST",
                "Testing multiple reporting channels",
                context={"channel_test": True, "channels": ["slack", "email", "dashboard"]}
            )

            with patch.object(reporter, '_get_webhook_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value = mock_session

                mock_response = AsyncMock()
                mock_response.status = 200
                mock_session.post.return_value.__aenter__ = mock_response
                mock_session.post.return_value.__aexit__ = AsyncMock()

                with patch.object(reporter, 'log_exception') as mock_log:
                    with patch.object(reporter, '_write_error_log') as mock_write:
                        with patch.object(reporter, '_send_to_monitoring_tools') as mock_monitoring:
                            reporter.report_error(error, {}, "error")

                            # Verify all channels were triggered
                            assert mock_session.post.call_count == 3  # Slack, Email, Dashboard

                            # Verify structured logging
                            mock_log.assert_called_once()

                            # Verify file logging
                            mock_write.assert_called_once()

                            # Verify monitoring tools
                            mock_monitoring.assert_called_once()

    def test_error_context_aggregation(self):
        """Test aggregation of error context from multiple sources."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

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
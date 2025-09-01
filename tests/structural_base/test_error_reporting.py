"""
Comprehensive unit tests for error reporting system.

This module tests the error reporting functionality in codesage_mcp.core.error_reporting,
focusing on error reporting with integrations, webhook alerts, performance monitoring,
privacy filtering, and graceful degradation when services are unavailable.
"""

import pytest
import json
import asyncio
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

import aiohttp

from codesage_mcp.core.error_reporting import (
    ErrorReporter,
    report_critical_error,
    error_reporter,
)
from codesage_mcp.core.exceptions import ToolExecutionError


class TestErrorReporterInitialization:
    """Test cases for ErrorReporter initialization and setup."""

    def test_error_reporter_initialization(self):
        """Test ErrorReporter initialization with default settings."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "false"}):
            reporter = ErrorReporter()

            assert reporter.logger is not None
            assert reporter._sentry_client is None
            assert reporter._prometheus_client is None
            assert reporter._webhook_session is None
            assert isinstance(reporter._privacy_filters, list)

    def test_error_reporter_with_sentry_integration(self):
        """Test ErrorReporter initialization with Sentry integration."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true", "SENTRY_DSN": "test_dsn"}):
            with patch('sentry_sdk.init') as mock_sentry_init:
                reporter = ErrorReporter()

                mock_sentry_init.assert_called_once_with(dsn="test_dsn")
                assert reporter._sentry_client is not None

    def test_error_reporter_with_prometheus_integration(self):
        """Test ErrorReporter initialization with Prometheus integration."""
        with patch.dict(os.environ, {
            "ERROR_REPORTING_ENABLED": "true",
            "PROMETHEUS_PUSHGATEWAY_URL": "http://test:9091"
        }):
            with patch('prometheus_client.CollectorRegistry') as mock_registry:
                with patch('prometheus_client.push_to_gateway'):
                    reporter = ErrorReporter()

                    assert reporter._prometheus_client is not None
                    assert reporter._prometheus_registry is not None

    def test_error_reporter_missing_integrations(self):
        """Test ErrorReporter when integrations are not available."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
            with patch('builtins.__import__', side_effect=ImportError("Module not found")):
                reporter = ErrorReporter()

                # Should not raise errors, just log warnings
                assert reporter._sentry_client is None
                assert reporter._prometheus_client is None

    def test_privacy_filters_parsing(self):
        """Test parsing of privacy filter fields."""
        with patch.dict(os.environ, {
            "ERROR_REPORTING_ENABLED": "false",
            "PRIVACY_SANITIZE_FIELDS": "password,token,api_key,secret"
        }):
            reporter = ErrorReporter()

            expected_filters = ["password", "token", "api_key", "secret"]
            assert reporter._privacy_filters == expected_filters

    def test_privacy_filters_empty(self):
        """Test privacy filters with empty configuration."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "false"}):
            reporter = ErrorReporter()

            assert reporter._privacy_filters == []


class TestSanitizeData:
    """Test cases for _sanitize_data method."""

    def test_sanitize_data_no_filters(self):
        """Test _sanitize_data with no privacy filters."""
        reporter = ErrorReporter()
        reporter._privacy_filters = []

        data = {"user": "test", "password": "secret123"}
        result = reporter._sanitize_data(data)

        assert result == data  # Should not modify if no filters

    def test_sanitize_data_with_filters(self):
        """Test _sanitize_data with privacy filters configured."""
        reporter = ErrorReporter()
        reporter._privacy_filters = ["password", "token", "key"]

        data = {
            "user": "test_user",
            "password": "secret123",
            "api_token": "token_abc",
            "api_key": "key_xyz",
            "normal_field": "safe_value"
        }

        result = reporter._sanitize_data(data)

        assert result["user"] == "test_user"
        assert result["password"] == "[REDACTED]"
        assert result["api_token"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["normal_field"] == "safe_value"

    def test_sanitize_data_nested_dict(self):
        """Test _sanitize_data with nested dictionary structures."""
        reporter = ErrorReporter()
        reporter._privacy_filters = ["password", "secret"]

        data = {
            "user": {
                "name": "test",
                "credentials": {
                    "password": "secret123",
                    "username": "user"
                }
            },
            "api": {
                "secret": "api_secret",
                "endpoint": "https://api.example.com"
            }
        }

        result = reporter._sanitize_data(data)

        assert result["user"]["name"] == "test"
        assert result["user"]["credentials"]["password"] == "[REDACTED]"
        assert result["user"]["credentials"]["username"] == "user"
        assert result["api"]["secret"] == "[REDACTED]"
        assert result["api"]["endpoint"] == "https://api.example.com"

    def test_sanitize_data_with_list(self):
        """Test _sanitize_data with list containing dictionaries."""
        reporter = ErrorReporter()
        reporter._privacy_filters = ["token"]

        data = {
            "users": [
                {"name": "user1", "token": "token1"},
                {"name": "user2", "token": "token2"}
            ],
            "safe_list": ["item1", "item2"]
        }

        result = reporter._sanitize_data(data)

        assert result["users"][0]["name"] == "user1"
        assert result["users"][0]["token"] == "[REDACTED]"
        assert result["users"][1]["name"] == "user2"
        assert result["users"][1]["token"] == "[REDACTED]"
        assert result["safe_list"] == ["item1", "item2"]

    def test_sanitize_data_case_insensitive(self):
        """Test _sanitize_data with case-insensitive matching."""
        reporter = ErrorReporter()
        reporter._privacy_filters = ["password"]

        data = {"PASSWORD": "secret", "Password": "secret2", "passWORD": "secret3"}

        result = reporter._sanitize_data(data)

        assert result["PASSWORD"] == "[REDACTED]"
        assert result["Password"] == "[REDACTED]"
        assert result["passWORD"] == "[REDACTED]"


class TestReportError:
    """Test cases for report_error method."""

    def test_report_error_disabled_reporting(self):
        """Test report_error when error reporting is disabled."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "false"}):
            reporter = ErrorReporter()

            with patch.object(reporter, 'log_exception') as mock_log:
                with patch.object(reporter, '_write_error_log') as mock_write:
                    with patch.object(reporter, '_send_to_monitoring_tools') as mock_monitoring:
                        with patch.object(reporter, '_send_webhook_alerts') as mock_webhooks:
                            error = ValueError("Test error")
                            reporter.report_error(error, {"test": True}, "error", "req_123")

                            # Should not call any reporting methods
                            assert not mock_write.called
                            assert not mock_monitoring.called
                            assert not mock_webhooks.called

    def test_report_error_full_flow(self):
        """Test report_error with full reporting flow enabled."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
            reporter = ErrorReporter()

            with patch.object(reporter, 'log_exception') as mock_log:
                with patch.object(reporter, '_write_error_log') as mock_write:
                    with patch.object(reporter, '_send_to_monitoring_tools') as mock_monitoring:
                        with patch.object(reporter, '_send_webhook_alerts') as mock_webhooks:
                            error = ToolExecutionError("Tool failed", tool_name="test_tool")
                            context = {"operation": "test", "password": "secret"}
                            session = {"user_id": 123, "token": "session_token"}

                            reporter.report_error(error, context, "error", "req_123", session)

                            # Verify all methods were called
                            mock_write.assert_called_once()
                            mock_monitoring.assert_called_once()
                            mock_webhooks.assert_called_once()

                            # Verify sanitization was applied
                            write_call = mock_write.call_args
                            sanitized_context = write_call[0][1]  # Second argument is context
                            assert sanitized_context["operation"] == "test"
                            assert sanitized_context["password"] == "[REDACTED]"

    def test_report_error_with_base_mcperror(self):
        """Test report_error with BaseMCPError containing error code."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
            reporter = ErrorReporter()

            with patch.object(reporter, 'log_exception') as mock_log:
                with patch.object(reporter, '_write_error_log') as mock_write:
                    with patch.object(reporter, '_send_to_monitoring_tools') as mock_monitoring:
                        with patch.object(reporter, '_send_webhook_alerts') as mock_webhooks:
                            error = ToolExecutionError("Tool failed", tool_name="test_tool")

                            reporter.report_error(error, {}, "error")

                            # Verify log_exception was called with proper context
                            mock_log.assert_called_once()
                            log_call = mock_log.call_args
                            assert log_call[0][0] == error  # First arg is error
                            assert log_call[0][2] is None  # request_id
                            assert log_call[0][3] == {}  # session_details (sanitized)


class TestWriteErrorLog:
    """Test cases for _write_error_log method."""

    def test_write_error_log_success(self):
        """Test successful writing of error log."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test_errors.log")

            with patch.dict(os.environ, {"ERROR_LOG_FILE": log_file}):
                reporter = ErrorReporter()

                error = ValueError("Test error")
                context = {"test": True, "value": 123}
                timestamp = datetime(2023, 1, 1, 12, 0, 0)

                with patch('codesage_mcp.core.error_reporting.datetime') as mock_datetime:
                    mock_datetime.utcnow.return_value = timestamp

                    reporter._write_error_log(error, context, "error", "req_123")

                # Verify log file was created and contains correct data
                assert os.path.exists(log_file)

                with open(log_file, 'r') as f:
                    content = f.read()
                    log_entry = json.loads(content)

                    assert log_entry["severity"] == "error"
                    assert log_entry["error_type"] == "ValueError"
                    assert log_entry["message"] == "Test error"
                    assert log_entry["context"] == context
                    assert log_entry["request_id"] == "req_123"
                    assert log_entry["timestamp"] == timestamp.isoformat()

    def test_write_error_log_creates_directory(self):
        """Test that _write_error_log creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "logs")
            log_file = os.path.join(log_dir, "errors.log")

            with patch.dict(os.environ, {"ERROR_LOG_FILE": log_file}):
                reporter = ErrorReporter()

                error = RuntimeError("Directory creation test")
                reporter._write_error_log(error, {}, "warning")

                assert os.path.exists(log_dir)
                assert os.path.exists(log_file)

    def test_write_error_log_handles_write_error(self):
        """Test _write_error_log handles file write errors gracefully."""
        with patch.dict(os.environ, {"ERROR_LOG_FILE": "/invalid/path/errors.log"}):
            reporter = ErrorReporter()

            with patch.object(reporter.logger, 'error') as mock_log_error:
                error = Exception("Test error")
                reporter._write_error_log(error, {}, "error")

                # Should log the error but not raise
                mock_log_error.assert_called_once()


class TestSendToMonitoringTools:
    """Test cases for _send_to_monitoring_tools method."""

    def test_send_to_monitoring_tools_sentry_enabled(self):
        """Test sending error to Sentry when configured."""
        reporter = ErrorReporter()
        reporter._sentry_client = MagicMock()

        error = ValueError("Sentry test error")
        context = {"operation": "test"}

        with patch.object(reporter._sentry_client, 'configure_scope') as mock_scope:
            mock_scope.return_value.__enter__ = MagicMock()
            mock_scope.return_value.__exit__ = MagicMock()

            reporter._send_to_monitoring_tools(error, context, "error")

            # Verify Sentry was called
            mock_scope.assert_called_once()

    def test_send_to_monitoring_tools_sentry_error(self):
        """Test handling of Sentry reporting errors."""
        reporter = ErrorReporter()
        reporter._sentry_client = MagicMock()

        error = ValueError("Sentry error test")

        with patch.object(reporter._sentry_client, 'configure_scope') as mock_scope:
            mock_scope.side_effect = Exception("Sentry failed")

            with patch.object(reporter.logger, 'error') as mock_log_error:
                reporter._send_to_monitoring_tools(error, {}, "error")

                # Should log the Sentry error
                mock_log_error.assert_called_once()

    def test_send_to_monitoring_tools_prometheus_enabled(self):
        """Test sending error to Prometheus when configured."""
        reporter = ErrorReporter()
        reporter._prometheus_client = MagicMock()
        reporter._prometheus_push = MagicMock()

        error = RuntimeError("Prometheus test error")

        with patch('prometheus_client.Counter') as mock_counter:
            mock_counter_instance = MagicMock()
            mock_counter.return_value = mock_counter_instance

            reporter._send_to_monitoring_tools(error, {}, "error")

            # Verify Prometheus counter was created and incremented
            mock_counter.assert_called_once()
            mock_counter_instance.labels.assert_called_once_with(
                error_type="RuntimeError",
                severity="error"
            )
            mock_counter_instance.labels().inc.assert_called_once()
            reporter._prometheus_push.assert_called_once()

    def test_send_to_monitoring_tools_prometheus_error(self):
        """Test handling of Prometheus reporting errors."""
        reporter = ErrorReporter()
        reporter._prometheus_client = MagicMock()

        with patch('prometheus_client.Counter', side_effect=Exception("Prometheus failed")):
            with patch.object(reporter.logger, 'error') as mock_log_error:
                reporter._send_to_monitoring_tools(ValueError("Test"), {}, "error")

                # Should log the Prometheus error
                mock_log_error.assert_called_once()


class TestSendWebhookAlerts:
    """Test cases for _send_webhook_alerts method."""

    def test_send_webhook_alerts_slack_enabled(self):
        """Test sending Slack webhook alerts."""
        reporter = ErrorReporter()

        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch.object(reporter, '_get_webhook_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value = mock_session

                mock_response = AsyncMock()
                mock_response.status = 200
                mock_session.post.return_value.__aenter__ = mock_response
                mock_session.post.return_value.__aexit__ = AsyncMock()

                error = ValueError("Slack test error")
                reporter._send_webhook_alerts(error, {}, "error", "req_123")

                # Verify webhook was called
                mock_session.post.assert_called_once()
                call_args = mock_session.post.call_args
                assert call_args[0][0] == "https://hooks.slack.com/test"

                # Verify Slack message format
                payload = call_args[0][1]
                assert "text" in payload
                assert "❌" in payload["text"]  # Error emoji
                assert "CodeSage MCP" in payload["text"]

    def test_send_webhook_alerts_email_enabled(self):
        """Test sending email webhook alerts."""
        reporter = ErrorReporter()

        with patch.dict(os.environ, {"EMAIL_WEBHOOK_URL": "https://api.email.com/webhook"}):
            with patch.object(reporter, '_get_webhook_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value = mock_session

                mock_response = AsyncMock()
                mock_response.status = 200
                mock_session.post.return_value.__aenter__ = mock_response
                mock_session.post.return_value.__aexit__ = AsyncMock()

                error = RuntimeError("Email test error")
                reporter._send_webhook_alerts(error, {"user": "test@example.com"}, "warning", "req_456")

                # Verify webhook was called
                mock_session.post.assert_called_once()
                call_args = mock_session.post.call_args
                assert call_args[0][0] == "https://api.email.com/webhook"

    def test_send_webhook_alerts_multiple_webhooks(self):
        """Test sending to multiple webhook types simultaneously."""
        reporter = ErrorReporter()

        with patch.dict(os.environ, {
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
            "EMAIL_WEBHOOK_URL": "https://api.email.com/webhook",
            "DASHBOARD_WEBHOOK_URL": "https://dashboard.com/webhook"
        }):
            with patch.object(reporter, '_get_webhook_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value = mock_session

                mock_response = AsyncMock()
                mock_response.status = 200
                mock_session.post.return_value.__aenter__ = mock_response
                mock_session.post.return_value.__aexit__ = AsyncMock()

                error = Exception("Multi-webhook test")
                reporter._send_webhook_alerts(error, {}, "critical")

                # Verify all three webhooks were called
                assert mock_session.post.call_count == 3

    def test_send_webhook_alerts_failure_handling(self):
        """Test handling of webhook failures."""
        reporter = ErrorReporter()

        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch.object(reporter, '_get_webhook_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value = mock_session

                # Simulate webhook failure
                mock_session.post.side_effect = aiohttp.ClientError("Connection failed")

                with patch.object(reporter.logger, 'error') as mock_log_error:
                    error = ValueError("Webhook failure test")
                    reporter._send_webhook_alerts(error, {}, "error")

                    # Should log the webhook error
                    mock_log_error.assert_called_once()

    def test_send_webhook_alerts_no_webhooks_configured(self):
        """Test _send_webhook_alerts when no webhooks are configured."""
        reporter = ErrorReporter()

        # No webhook URLs configured
        with patch.object(reporter, '_get_webhook_session') as mock_get_session:
            error = ValueError("No webhooks test")
            reporter._send_webhook_alerts(error, {}, "error")

            # Should not attempt to get session or send webhooks
            assert not mock_get_session.called


class TestMonitorPerformance:
    """Test cases for monitor_performance method."""

    def test_monitor_performance_sync_function_success(self):
        """Test monitor_performance with successful sync function."""
        reporter = ErrorReporter()

        def test_function(x, y):
            return x + y

        with patch('time.time', side_effect=[1.0, 2.5]):  # 1.5 seconds
            with patch('psutil.Process') as mock_process:
                mock_memory = MagicMock()
                mock_memory.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
                mock_process.return_value.memory_info.return_value = mock_memory

                result = reporter.monitor_performance("test_operation", test_function, 2, 3)

                assert result == 5

    def test_monitor_performance_async_function_success(self):
        """Test monitor_performance with successful async function."""
        reporter = ErrorReporter()

        async def async_test_function(x, y):
            await asyncio.sleep(0.1)
            return x * y

        with patch('time.time', side_effect=[1.0, 2.0]):  # 1.0 seconds
            with patch('psutil.Process') as mock_process:
                mock_memory = MagicMock()
                mock_memory.memory_info.return_value.rss = 50 * 1024 * 1024  # 50MB
                mock_process.return_value.memory_info.return_value = mock_memory

                result = reporter.monitor_performance("async_test", async_test_function, 4, 5)

                assert result == 20

    def test_monitor_performance_slow_operation(self):
        """Test monitor_performance detects and reports slow operations."""
        reporter = ErrorReporter()

        def slow_function():
            return "completed"

        with patch('time.time', side_effect=[1.0, 8.0]):  # 7 seconds - exceeds threshold
            with patch('psutil.Process') as mock_process:
                mock_memory = MagicMock()
                mock_memory.memory_info.return_value.rss = 10 * 1024 * 1024
                mock_process.return_value.memory_info.return_value = mock_memory

                with patch.object(reporter, 'report_error') as mock_report:
                    result = reporter.monitor_performance("slow_operation", slow_function)

                    assert result == "completed"

                    # Should report slow operation
                    mock_report.assert_called()
                    slow_call = mock_report.call_args_list[0]
                    assert "Slow operation" in slow_call[0][0].args[0]
                    assert slow_call[0][2] == "warning"

    def test_monitor_performance_memory_spike(self):
        """Test monitor_performance detects and reports memory spikes."""
        reporter = ErrorReporter()

        def memory_function():
            return "done"

        with patch('time.time', side_effect=[1.0, 2.0]):
            with patch('psutil.Process') as mock_process:
                mock_memory = MagicMock()
                # Simulate 600MB memory spike
                mock_memory.memory_info.side_effect = [
                    MagicMock(rss=100 * 1024 * 1024),  # Start: 100MB
                    MagicMock(rss=700 * 1024 * 1024)   # End: 700MB
                ]
                mock_process.return_value.memory_info = mock_memory

                with patch.object(reporter, 'report_error') as mock_report:
                    result = reporter.monitor_performance("memory_operation", memory_function)

                    assert result == "done"

                    # Should report memory spike
                    mock_report.assert_called()
                    memory_call = mock_report.call_args_list[0]
                    assert "Memory spike" in memory_call[0][0].args[0]
                    assert memory_call[0][2] == "warning"

    def test_monitor_performance_function_error(self):
        """Test monitor_performance handles function errors."""
        reporter = ErrorReporter()

        def failing_function():
            raise ValueError("Function failed")

        with patch('time.time', side_effect=[1.0, 1.5]):
            with patch('psutil.Process') as mock_process:
                mock_memory = MagicMock()
                mock_memory.memory_info.return_value.rss = 50 * 1024 * 1024
                mock_process.return_value.memory_info.return_value = mock_memory

                with patch.object(reporter, 'report_error') as mock_report:
                    with pytest.raises(ValueError):
                        reporter.monitor_performance("failing_operation", failing_function)

                    # Should report the original error
                    mock_report.assert_called_once()
                    error_call = mock_report.call_args
                    assert error_call[0][0].args[0] == "Function failed"
                    assert error_call[0][2] == "error"


class TestErrorReporterIntegration:
    """Integration tests for ErrorReporter with other components."""

    def test_error_reporter_with_base_mcperror_integration(self):
        """Test ErrorReporter integration with BaseMCPError."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
            reporter = ErrorReporter()

            error = ToolExecutionError(
                "Integration test error",
                tool_name="integration_tool",
                context={"operation": "test", "sensitive": "secret_data"}
            )

            with patch.object(reporter, 'log_exception') as mock_log:
                with patch.object(reporter, '_write_error_log') as mock_write:
                    with patch.object(reporter, '_send_to_monitoring_tools') as mock_monitoring:
                        with patch.object(reporter, '_send_webhook_alerts') as mock_webhooks:
                            reporter.report_error(error, {"additional": "context"}, "error", "int_req_001")

                            # Verify BaseMCPError context is included
                            log_call = mock_log.call_args
                            logged_error = log_call[0][0]
                            assert isinstance(logged_error, ToolExecutionError)
                            assert logged_error.error_code == "TOOL_EXECUTION_FAILED"

    def test_graceful_degradation_missing_dependencies(self):
        """Test graceful degradation when optional dependencies are missing."""
        with patch.dict(os.environ, {"ERROR_REPORTING_ENABLED": "true"}):
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                reporter = ErrorReporter()

                # Should initialize without errors
                assert reporter._sentry_client is None
                assert reporter._prometheus_client is None

                # Should still be able to report errors
                with patch.object(reporter, 'log_exception') as mock_log:
                    error = ValueError("Graceful degradation test")
                    reporter.report_error(error)

                    mock_log.assert_called_once()

    def test_webhook_session_reuse(self):
        """Test that webhook session is reused across multiple calls."""
        reporter = ErrorReporter()

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False

            # First call
            session1 = reporter._get_webhook_session()
            # Second call
            session2 = reporter._get_webhook_session()

            # Should return the same session
            assert session1 is session2
            # Should only create session once
            mock_session_class.assert_called_once()


class TestGlobalFunctions:
    """Test cases for global error reporting functions."""

    def test_report_critical_error(self):
        """Test report_critical_error convenience function."""
        with patch.object(error_reporter, 'report_error') as mock_report:
            error = RuntimeError("Critical error")
            context = {"severity": "high"}

            report_critical_error(error, context, "critical_req_001")

            mock_report.assert_called_once_with(
                error,
                context,
                "critical",
                "critical_req_001"
            )

    def test_error_reporter_cleanup(self):
        """Test error reporter cleanup on exit."""
        with patch.object(error_reporter, '_webhook_session') as mock_session:
            mock_session.closed = False

            # Simulate cleanup
            with patch('asyncio.create_task') as mock_create_task:
                # Import the cleanup function
                from codesage_mcp.core.error_reporting import cleanup
                cleanup()

                # Should attempt to close session
                mock_create_task.assert_called_once()


class TestPrivacySanitizationIntegration:
    """Integration tests for privacy sanitization across the system."""

    def test_full_error_reporting_with_sensitive_data(self):
        """Test complete error reporting flow with sensitive data sanitization."""
        with patch.dict(os.environ, {
            "ERROR_REPORTING_ENABLED": "true",
            "PRIVACY_SANITIZE_FIELDS": "password,token,api_key"
        }):
            reporter = ErrorReporter()

            sensitive_context = {
                "user_id": 123,
                "password": "super_secret_password",
                "api_token": "token_abcdef123456",
                "api_key": "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz",
                "normal_field": "safe_value",
                "nested": {
                    "another_password": "nested_secret",
                    "safe_nested": "safe"
                }
            }

            error = ValueError("Sensitive data test")

            with patch.object(reporter, 'log_exception') as mock_log:
                with patch.object(reporter, '_write_error_log') as mock_write:
                    with patch.object(reporter, '_send_to_monitoring_tools') as mock_monitoring:
                        with patch.object(reporter, '_send_webhook_alerts') as mock_webhooks:
                            reporter.report_error(error, sensitive_context, "error")

                            # Verify all sensitive data was sanitized
                            write_call = mock_write.call_args
                            sanitized = write_call[0][1]

                            assert sanitized["user_id"] == 123
                            assert sanitized["password"] == "[REDACTED]"
                            assert sanitized["api_token"] == "[REDACTED]"
                            assert sanitized["api_key"] == "[REDACTED]"
                            assert sanitized["normal_field"] == "safe_value"
                            assert sanitized["nested"]["another_password"] == "[REDACTED]"
                            assert sanitized["nested"]["safe_nested"] == "safe"
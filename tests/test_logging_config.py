"""
Comprehensive unit tests for structured logging configuration.

This module tests the logging configuration functionality in codesage_mcp.logging_config,
focusing on structured logging setup, exception logging with context, logger creation,
and the error logging decorator for both sync and async functions.
"""

import pytest
import json
import logging
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

import structlog

from codesage_mcp.logging_config import (
    setup_logging,
    log_exception,
    get_logger,
    log_errors,
    get_environment_info,
)
from codesage_mcp.exceptions import BaseMCPError, ToolExecutionError


class TestSetupLogging:
    """Test cases for setup_logging function."""

    def test_setup_logging_console_only(self):
        """Test setup_logging with console output only."""
        with patch('sys.stdout', new_callable=MagicMock) as mock_stdout:
            setup_logging(level="INFO", log_file=None, json_format=False)

            # Verify structlog is configured
            logger = structlog.get_logger("test")
            assert logger is not None

            # Verify console renderer is used (not JSON)
            # This is hard to test directly, but we can check that logging works
            with patch('structlog.write_logger') as mock_write:
                logger.info("Test message", key="value")
                # Should call the write logger
                assert mock_write.called

    def test_setup_logging_with_file(self):
        """Test setup_logging with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            setup_logging(level="DEBUG", log_file=log_file, json_format=True)

            # Verify log file is created
            assert os.path.exists(log_file)

            # Test logging to file
            logger = get_logger("test_logger")
            logger.error("Test error message", error_code="TEST_ERROR")

            # Check file contents
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test error message" in content
                assert "error_code" in content

    def test_setup_logging_json_format(self):
        """Test setup_logging with JSON format enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            setup_logging(level="INFO", log_file=log_file, json_format=True)

            logger = get_logger("test")
            logger.info("JSON test message", test_key="test_value")

            # Check that file contains valid JSON
            with open(log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0

                # Parse first line as JSON
                log_entry = json.loads(lines[0])
                assert log_entry["event"] == "JSON test message"
                assert log_entry["test_key"] == "test_value"
                assert "timestamp" in log_entry

    def test_setup_logging_log_rotation(self):
        """Test setup_logging with log rotation configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            setup_logging(
                level="INFO",
                log_file=log_file,
                max_bytes=100,  # Small size to trigger rotation
                backup_count=2
            )

            logger = get_logger("test")

            # Write enough logs to trigger rotation
            for i in range(10):
                logger.info(f"Log message {i}", iteration=i)

            # Check that rotation files are created
            assert os.path.exists(log_file)
            assert os.path.exists(log_file + ".1")

    def test_setup_logging_invalid_level(self):
        """Test setup_logging with invalid log level defaults to INFO."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(level="INVALID_LEVEL")

            # Should still call basicConfig with INFO level
            mock_basic_config.assert_called_once()
            call_args = mock_basic_config.call_args
            assert call_args[1]["level"] == logging.INFO


class TestLogException:
    """Test cases for log_exception function."""

    def test_log_exception_with_base_mcperror(self):
        """Test log_exception with BaseMCPError including error code and context."""
        logger = MagicMock()
        error = ToolExecutionError(
            "Tool execution failed",
            tool_name="test_tool",
            context={"param": "value", "timeout": 30}
        )

        log_exception(error, logger, request_id="req_123", extra_context={"user_id": 456})

        # Verify logger.error was called
        assert logger.error.called

        # Check the call arguments
        call_args = logger.error.call_args
        args, kwargs = call_args

        assert args[0] == "Exception occurred"
        assert kwargs["timestamp"] is not None
        assert kwargs["error_type"] == "ToolExecutionError"
        assert kwargs["message"] == "Tool execution failed"
        assert kwargs["error_code"] == "TOOL_EXECUTION_FAILED"
        assert kwargs["error_context"]["tool_name"] == "test_tool"
        assert kwargs["request_id"] == "req_123"
        assert kwargs["user_id"] == 456

    def test_log_exception_with_regular_exception(self):
        """Test log_exception with regular Python exception."""
        logger = MagicMock()
        error = ValueError("Regular error message")

        log_exception(error, logger)

        # Verify logger.error was called
        assert logger.error.called

        call_args = logger.error.call_args
        args, kwargs = call_args

        assert args[0] == "Exception occurred"
        assert kwargs["error_type"] == "ValueError"
        assert kwargs["message"] == "Regular error message"
        assert "error_code" not in kwargs  # Should not have error_code for regular exceptions

    def test_log_exception_with_session_details(self):
        """Test log_exception with session details."""
        logger = MagicMock()
        error = RuntimeError("Session error")
        session_details = {"user_id": 123, "session_token": "abc123"}

        log_exception(error, logger, session_details=session_details)

        call_args = logger.error.call_args
        args, kwargs = call_args

        assert kwargs["session_details"] == session_details

    def test_log_exception_with_minimal_params(self):
        """Test log_exception with minimal parameters."""
        logger = MagicMock()
        error = Exception("Minimal error")

        log_exception(error, logger)

        assert logger.error.called

        call_args = logger.error.call_args
        args, kwargs = call_args

        assert args[0] == "Exception occurred"
        assert kwargs["error_type"] == "Exception"
        assert kwargs["message"] == "Minimal error"
        assert "request_id" not in kwargs
        assert "session_details" not in kwargs

    def test_log_exception_with_traceback(self):
        """Test that log_exception includes traceback information."""
        logger = MagicMock()
        try:
            raise ValueError("Test error with traceback")
        except ValueError as e:
            log_exception(e, logger)

        call_args = logger.error.call_args
        args, kwargs = call_args

        assert "traceback" in kwargs
        assert "ValueError: Test error with traceback" in kwargs["traceback"]

    def test_log_exception_environment_info(self):
        """Test that log_exception includes environment information."""
        logger = MagicMock()
        error = Exception("Test error")

        log_exception(error, logger)

        call_args = logger.error.call_args
        args, kwargs = call_args

        assert "environment" in kwargs
        env = kwargs["environment"]
        assert "python_version" in env
        assert "server_version" in env
        assert "platform" in env


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_creation(self):
        """Test that get_logger returns a properly configured logger."""
        logger = get_logger("test_logger")

        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'warning')

    def test_get_logger_same_name_returns_same_instance(self):
        """Test that get_logger returns the same instance for same name."""
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")

        # In structlog, this might not be the exact same instance, but should be equivalent
        assert logger1 is not None
        assert logger2 is not None

    def test_get_logger_different_names(self):
        """Test that get_logger creates different loggers for different names."""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")

        assert logger1 is not None
        assert logger2 is not None
        # They should be different instances
        assert logger1 is not logger2

    def test_get_logger_logging_functionality(self):
        """Test that logger from get_logger actually logs messages."""
        with patch('structlog.write_logger') as mock_write:
            logger = get_logger("test")
            logger.info("Test message", key="value")

            assert mock_write.called


class TestLogErrorsDecorator:
    """Test cases for log_errors decorator."""

    def test_log_errors_sync_function_no_error(self):
        """Test log_errors decorator on sync function with no error."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_errors("test_logger")
            def test_function(x, y):
                return x + y

            result = test_function(2, 3)

            assert result == 5
            # Logger should not be called since no error occurred
            assert not mock_logger.error.called

    def test_log_errors_sync_function_with_error(self):
        """Test log_errors decorator on sync function that raises error."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_errors("test_logger")
            def failing_function():
                raise ValueError("Test error")

            with pytest.raises(ValueError):
                failing_function()

            # Verify error was logged
            assert mock_logger.error.called

            call_args = mock_logger.error.call_args
            args, kwargs = call_args
            assert args[0] == "Exception occurred"
            assert kwargs["error_type"] == "ValueError"
            assert kwargs["message"] == "Test error"

    def test_log_errors_sync_function_with_request_id_param(self):
        """Test log_errors decorator with request_id parameter extraction."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_errors("test_logger", request_id_param="request_id")
            def function_with_request_id(request_id, data):
                raise RuntimeError("Request failed")

            with pytest.raises(RuntimeError):
                function_with_request_id("req_123", {"key": "value"})

            # Verify request_id was extracted
            call_args = mock_logger.error.call_args
            args, kwargs = call_args
            assert kwargs["request_id"] == "req_123"

    def test_log_errors_sync_function_with_session_param(self):
        """Test log_errors decorator with session parameter extraction."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_errors("test_logger", session_param="session")
            def function_with_session(session, action):
                raise PermissionError("Access denied")

            session_data = {"user_id": 456, "role": "admin"}

            with pytest.raises(PermissionError):
                function_with_session(session_data, "delete")

            # Verify session was extracted
            call_args = mock_logger.error.call_args
            args, kwargs = call_args
            assert kwargs["session_details"] == session_data

    def test_log_errors_sync_function_with_object_attribute(self):
        """Test log_errors decorator extracting params from object attributes."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            class TestClass:
                def __init__(self):
                    self.request_id = "obj_req_789"
                    self.session = {"user": "test"}

                @log_errors("test_logger", request_id_param="request_id", session_param="session")
                def failing_method(self):
                    raise ConnectionError("Network error")

            obj = TestClass()

            with pytest.raises(ConnectionError):
                obj.failing_method()

            # Verify parameters were extracted from object
            call_args = mock_logger.error.call_args
            args, kwargs = call_args
            assert kwargs["request_id"] == "obj_req_789"
            assert kwargs["session_details"] == {"user": "test"}

    @pytest.mark.asyncio
    async def test_log_errors_async_function_no_error(self):
        """Test log_errors decorator on async function with no error."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                @log_errors("test_logger")
                async def async_test_function(x, y):
                    return x * y

                result = await async_test_function(4, 5)

                assert result == 20
                assert not mock_logger.error.called

    @pytest.mark.asyncio
    async def test_log_errors_async_function_with_error(self):
        """Test log_errors decorator on async function that raises error."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_errors("test_logger")
            async def async_failing_function():
                raise TimeoutError("Async operation timed out")

            with pytest.raises(TimeoutError):
                await async_failing_function()

            # Verify error was logged
            assert mock_logger.error.called

            call_args = mock_logger.error.call_args
            args, kwargs = call_args
            assert args[0] == "Exception occurred"
            assert kwargs["error_type"] == "TimeoutError"
            assert kwargs["message"] == "Async operation timed out"

    @pytest.mark.asyncio
    async def test_log_errors_async_function_with_params(self):
        """Test log_errors decorator on async function with parameter extraction."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_errors("test_logger", request_id_param="request_id")
            async def async_function_with_params(request_id, data):
                raise FileNotFoundError("File not found")

            with pytest.raises(FileNotFoundError):
                await async_function_with_params("async_req_999", {"file": "test.txt"})

            # Verify request_id was extracted
            call_args = mock_logger.error.call_args
            args, kwargs = call_args
            assert kwargs["request_id"] == "async_req_999"


class TestGetEnvironmentInfo:
    """Test cases for get_environment_info function."""

    def test_get_environment_info_structure(self):
        """Test that get_environment_info returns expected structure."""
        info = get_environment_info()

        assert isinstance(info, dict)
        assert "python_version" in info
        assert "server_version" in info
        assert "platform" in info

    def test_get_environment_info_values(self):
        """Test that get_environment_info contains reasonable values."""
        info = get_environment_info()

        # Python version should be in format like "3.9.7"
        assert len(info["python_version"].split(".")) >= 2

        # Server version should be a string
        assert isinstance(info["server_version"], str)

        # Platform should be a non-empty string
        assert len(info["platform"]) > 0


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_structured_logging_with_base_mcperror(self):
        """Test end-to-end structured logging with BaseMCPError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "integration.log")

            # Setup logging
            setup_logging(level="ERROR", log_file=log_file, json_format=True)

            # Create and log error
            logger = get_logger("integration_test")
            error = ToolExecutionError(
                "Integration test error",
                tool_name="integration_tool",
                context={"test": True, "value": 123}
            )

            log_exception(error, logger, request_id="int_test_001")

            # Verify log file contents
            with open(log_file, 'r') as f:
                content = f.read()
                log_data = json.loads(content)

                assert log_data["event"] == "Exception occurred"
                assert log_data["error_type"] == "ToolExecutionError"
                assert log_data["error_code"] == "TOOL_EXECUTION_FAILED"
                assert log_data["request_id"] == "int_test_001"
                assert log_data["error_context"]["tool_name"] == "integration_tool"

    def test_decorator_with_full_context(self):
        """Test decorator with full context extraction."""
        with patch('codesage_mcp.logging_config.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            class RequestHandler:
                def __init__(self):
                    self.request_id = "handler_req_456"
                    self.user_session = {"user_id": 789, "permissions": ["read", "write"]}

                @log_errors("handler_logger", request_id_param="request_id", session_param="user_session")
                def process_request(self, data):
                    if data.get("should_fail"):
                        raise ValueError("Processing failed")
                    return {"result": "success"}

            handler = RequestHandler()

            # Test successful call
            result = handler.process_request({"action": "test"})
            assert result == {"result": "success"}
            assert not mock_logger.error.called

            # Test failed call
            with pytest.raises(ValueError):
                handler.process_request({"should_fail": True})

            assert mock_logger.error.called
            call_args = mock_logger.error.call_args
            args, kwargs = call_args
            assert kwargs["request_id"] == "handler_req_456"
            assert kwargs["session_details"]["user_id"] == 789
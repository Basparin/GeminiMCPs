#!/usr/bin/env python3
"""
Test Gemini CLI Compatibility Layer

This module tests the Gemini CLI compatibility features including:
- Tools format adaptation (array vs object)
- Error code format adaptation (string vs numeric)
- Response structure compatibility
- Format detection and adaptation
"""

import json
import pytest
from unittest.mock import Mock, MagicMock

from codesage_mcp.gemini_compatibility import (
    GeminiCompatibilityHandler,
    ResponseFormat,
    create_gemini_compatible_error_response,
    adapt_response_for_gemini
)


class TestGeminiCompatibilityHandler:
    """Test the Gemini compatibility handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    def test_detect_standard_mcp_format(self):
        """Test detection of standard MCP format."""
        headers = {"user-agent": "curl/7.68.0"}
        body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_detect_gemini_array_tools_format(self):
        """Test detection of Gemini array tools format."""
        headers = {"user-agent": "node"}
        body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        # Sample tools object for testing (this is what get_all_tools_definitions_as_object() returns)
        tools_object = {
            "tool1": {"name": "tool1", "description": "Test tool 1"},
            "tool2": {"name": "tool2", "description": "Test tool 2"}
        }

        # The result should be wrapped in a "tools" key (like the actual tools/list response)
        result = {"tools": tools_object}

        # First request - should detect array format immediately (Gemini detection is immediate)
        response = self.handler.create_compatible_response(
            result=result,
            request_headers=headers,
            request_body=body
        )
        # Check that the tools are returned in array format (Gemini)
        assert isinstance(response["tools"], list)
        assert len(response["tools"]) == 2
        assert response["tools"][0]["name"] == "tool1"
        assert response["tools"][1]["name"] == "tool2"

    def test_detect_gemini_numeric_errors_format(self):
        """Test detection of Gemini numeric errors format."""
        headers = {"user-agent": "node"}
        body = {"jsonrpc": "2.0", "method": "tools/call", "id": 1}

        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.GEMINI_NUMERIC_ERRORS

    def test_adapt_tools_response_array_format(self):
        """Test adapting tools response to array format."""
        tools_object = {
            "tool1": {"name": "tool1", "description": "Test tool 1"},
            "tool2": {"name": "tool2", "description": "Test tool 2"}
        }

        adapted = self.handler.adapt_tools_response(tools_object, ResponseFormat.GEMINI_ARRAY_TOOLS)

        assert "tools" in adapted
        assert isinstance(adapted["tools"], list)
        assert len(adapted["tools"]) == 2
        assert adapted["tools"][0]["name"] == "tool1"
        assert adapted["tools"][1]["name"] == "tool2"

    def test_adapt_tools_response_object_format(self):
        """Test adapting tools response to object format."""
        tools_object = {
            "tool1": {"name": "tool1", "description": "Test tool 1"},
            "tool2": {"name": "tool2", "description": "Test tool 2"}
        }

        adapted = self.handler.adapt_tools_response(tools_object, ResponseFormat.STANDARD_MCP)

        assert adapted == tools_object

    def test_adapt_error_response_numeric_format(self):
        """Test adapting error response to numeric format."""
        error_response = {"code": "INVALID_PARAMS", "message": "Invalid parameters"}

        adapted = self.handler.adapt_error_response(error_response, ResponseFormat.GEMINI_NUMERIC_ERRORS)

        assert adapted["code"] == -32602  # Numeric code for INVALID_PARAMS
        assert adapted["message"] == "Invalid parameters"

    def test_adapt_error_response_string_format(self):
        """Test adapting error response to string format."""
        error_response = {"code": "INVALID_PARAMS", "message": "Invalid parameters"}

        adapted = self.handler.adapt_error_response(error_response, ResponseFormat.STANDARD_MCP)

        assert adapted["code"] == "INVALID_PARAMS"
        assert adapted["message"] == "Invalid parameters"

    def test_create_compatible_response_success(self):
        """Test creating a compatible success response."""
        headers = {"user-agent": "node"}
        body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        response = self.handler.create_compatible_response(
            result={"tools": []},
            error=None,
            request_id=1,
            request_headers=headers,
            request_body=body
        )

        assert response == {"tools": []}

    def test_create_compatible_response_error(self):
        """Test creating a compatible error response."""
        headers = {"user-agent": "node"}
        body = {"jsonrpc": "2.0", "method": "tools/call", "id": 1}

        error = {"code": "TOOL_NOT_FOUND", "message": "Tool not found"}

        response = self.handler.create_compatible_response(
            result=None,
            error=error,
            request_id=1,
            request_headers=headers,
            request_body=body
        )

        assert response["error"]["code"] == -32001  # Numeric code for TOOL_NOT_FOUND

    def test_get_tools_definitions_array(self):
        """Test converting tools object to array format."""
        tools_object = {
            "tool1": {"name": "tool1", "description": "Test tool 1"},
            "tool2": {"name": "tool2", "description": "Test tool 2"}
        }

        tools_array = self.handler.get_tools_definitions_array(tools_object)

        assert isinstance(tools_array, list)
        assert len(tools_array) == 2
        assert tools_array[0]["name"] == "tool1"
        assert tools_array[1]["name"] == "tool2"

    def test_get_tools_definitions_object(self):
        """Test returning tools in object format."""
        tools_object = {
            "tool1": {"name": "tool1", "description": "Test tool 1"},
            "tool2": {"name": "tool2", "description": "Test tool 2"}
        }

        result = self.handler.get_tools_definitions_object(tools_object)

        assert result == tools_object


class TestCompatibilityFunctions:
    """Test standalone compatibility functions."""

    def test_create_gemini_compatible_error_response_string(self):
        """Test creating Gemini compatible error response with string code."""
        error = create_gemini_compatible_error_response(
            "INVALID_PARAMS", "Invalid parameters", ResponseFormat.STANDARD_MCP
        )

        assert error["code"] == "INVALID_PARAMS"
        assert error["message"] == "Invalid parameters"

    def test_create_gemini_compatible_error_response_numeric(self):
        """Test creating Gemini compatible error response with numeric code."""
        error = create_gemini_compatible_error_response(
            "INVALID_PARAMS", "Invalid parameters"
        )

        assert error["code"] == -32602  # Numeric code for INVALID_PARAMS
        assert error["message"] == "Invalid parameters"

    def test_adapt_response_for_gemini_success(self):
        """Test adapting response for Gemini success case."""
        headers = {"user-agent": "curl/7.68.0"}
        body = {"jsonrpc": "2.0", "method": "initialize", "id": 1}

        response = adapt_response_for_gemini(
            result={"protocolVersion": "2025-06-18"},
            error=None,
            request_id=1,
            request_headers=headers,
            request_body=body
        )

        assert response["result"]["protocolVersion"] == "2025-06-18"

    def test_adapt_response_for_gemini_error(self):
        """Test adapting response for Gemini error case."""
        headers = {"user-agent": "node"}
        body = {"jsonrpc": "2.0", "method": "tools/call", "id": 1}

        error = {"code": "TOOL_NOT_FOUND", "message": "Tool not found"}

        response = adapt_response_for_gemini(
            result=None,
            error=error,
            request_id=1,
            request_headers=headers,
            request_body=body
        )

        assert response["error"]["code"] == -32001  # Numeric code


class TestJSONRPCCompatibility:
    """Test JSON-RPC response compatibility."""

    def test_json_serialization_compatibility(self):
        """Test that responses can be properly JSON serialized."""
        handler = GeminiCompatibilityHandler()

        # Test success response
        success_response = handler.create_compatible_response(
            result={"status": "ok"},
            error=None,
            request_id=123
        )

        json_str = json.dumps(success_response)
        parsed = json.loads(json_str)

        assert parsed["result"]["status"] == "ok"

        # Test error response with Gemini CLI headers to trigger numeric conversion
        headers = {"user-agent": "node"}
        body = {"jsonrpc": "2.0", "method": "tools/call", "id": 456}

        error_response = handler.create_compatible_response(
            result=None,
            error={"code": "INVALID_REQUEST", "message": "Invalid request"},
            request_id=456,
            request_headers=headers,
            request_body=body
        )

        json_str = json.dumps(error_response)
        parsed = json.loads(json_str)

        assert parsed["error"]["code"] == -32600  # Numeric code for INVALID_REQUEST


if __name__ == "__main__":
    pytest.main([__file__])
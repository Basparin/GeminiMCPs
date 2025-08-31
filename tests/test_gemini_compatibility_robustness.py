#!/usr/bin/env python3
"""
Comprehensive Robustness Tests for Gemini CLI Compatibility Layer

This module provides extensive tests for edge cases, error handling, and robustness
scenarios for the GeminiCompatibilityHandler, including:
- Edge cases with None values, empty inputs, and malformed data
- Error handling for invalid types, exceptions, and invalid error codes
- Robustness tests for large data, request history limits, and concurrent scenarios
"""

import json
import pytest
import threading
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from codesage_mcp.gemini_compatibility import (
    GeminiCompatibilityHandler,
    ResponseFormat,
    GeminiCliVersion,
    create_gemini_compatible_error_response,
    adapt_response_for_gemini,
    get_compatibility_handler
)


class TestEdgeCases:
    """Test edge cases for GeminiCompatibilityHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    def test_detect_response_format_none_headers(self):
        """Test detect_response_format with None headers."""
        format_type = self.handler.detect_response_format(None, {"method": "tools/list"})
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_detect_response_format_none_body(self):
        """Test detect_response_format with None body."""
        format_type = self.handler.detect_response_format({"user-agent": "node"}, None)
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_detect_response_format_both_none(self):
        """Test detect_response_format with both None."""
        format_type = self.handler.detect_response_format(None, None)
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_detect_response_format_empty_dicts(self):
        """Test detect_response_format with empty dicts."""
        format_type = self.handler.detect_response_format({}, {})
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_detect_response_format_missing_user_agent(self):
        """Test detect_response_format with missing user-agent."""
        headers = {"content-type": "application/json"}
        body = {"method": "tools/list"}
        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_detect_response_format_invalid_user_agent_type(self):
        """Test detect_response_format with invalid user-agent type."""
        headers = {"user-agent": 12345}
        body = {"method": "tools/list"}
        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_detect_response_format_unknown_method(self):
        """Test detect_response_format with unknown method."""
        headers = {"user-agent": "node"}
        body = {"method": "unknown/method"}
        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_detect_response_format_missing_method(self):
        """Test detect_response_format with missing method."""
        headers = {"user-agent": "node"}
        body = {"jsonrpc": "2.0"}
        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_adapt_tools_response_none_input(self):
        """Test adapt_tools_response with None input."""
        result = self.handler.adapt_tools_response(None, ResponseFormat.GEMINI_ARRAY_TOOLS)
        assert result == {"tools": []}

    def test_adapt_tools_response_empty_dict(self):
        """Test adapt_tools_response with empty dict."""
        result = self.handler.adapt_tools_response({}, ResponseFormat.GEMINI_ARRAY_TOOLS)
        assert result == {"tools": []}

    def test_adapt_tools_response_list_input(self):
        """Test adapt_tools_response with list input."""
        tools_list = [{"name": "tool1"}]
        result = self.handler.adapt_tools_response(tools_list, ResponseFormat.GEMINI_ARRAY_TOOLS)
        assert result == {"tools": tools_list}

    def test_adapt_tools_response_already_wrapped(self):
        """Test adapt_tools_response with already wrapped tools."""
        wrapped_tools = {"tools": [{"name": "tool1"}]}
        result = self.handler.adapt_tools_response(wrapped_tools, ResponseFormat.GEMINI_ARRAY_TOOLS)
        assert result == wrapped_tools

    def test_adapt_tools_response_nested_structure(self):
        """Test adapt_tools_response with nested structure."""
        nested_tools = {"category": {"tool1": {"name": "tool1"}}}
        result = self.handler.adapt_tools_response(nested_tools, ResponseFormat.GEMINI_ARRAY_TOOLS)
        assert "tools" in result
        assert isinstance(result["tools"], list)
        assert len(result["tools"]) == 1

    def test_adapt_error_response_none_input(self):
        """Test adapt_error_response with None input."""
        result = self.handler.adapt_error_response(None, ResponseFormat.GEMINI_NUMERIC_ERRORS)
        assert result is None

    def test_adapt_error_response_missing_code(self):
        """Test adapt_error_response with missing code."""
        error = {"message": "Test error"}
        result = self.handler.adapt_error_response(error, ResponseFormat.GEMINI_NUMERIC_ERRORS)
        assert result["code"] == -32603  # Default INTERNAL_ERROR

    def test_adapt_error_response_invalid_code_type(self):
        """Test adapt_error_response with invalid code type."""
        error = {"code": 123, "message": "Test error"}
        result = self.handler.adapt_error_response(error, ResponseFormat.GEMINI_NUMERIC_ERRORS)
        assert result["code"] == 123  # Should remain unchanged

    def test_adapt_error_response_unknown_string_code(self):
        """Test adapt_error_response with unknown string code."""
        error = {"code": "UNKNOWN_ERROR", "message": "Test error"}
        result = self.handler.adapt_error_response(error, ResponseFormat.GEMINI_NUMERIC_ERRORS)
        assert result["code"] == -32603  # Default INTERNAL_ERROR

    def test_create_compatible_response_none_result(self):
        """Test create_compatible_response with None result."""
        response = self.handler.create_compatible_response(result=None)
        assert response["result"] is None

    def test_create_compatible_response_none_error(self):
        """Test create_compatible_response with None error."""
        response = self.handler.create_compatible_response(error=None)
        assert "error" not in response

    def test_create_compatible_response_none_id(self):
        """Test create_compatible_response with None id."""
        response = self.handler.create_compatible_response(request_id=None)
        assert response["id"] is None

    def test_create_compatible_response_invalid_result_type(self):
        """Test create_compatible_response with invalid result type."""
        response = self.handler.create_compatible_response(result=set([1, 2, 3]))
        assert response["result"] == set([1, 2, 3])

    def test_get_tools_definitions_array_none_input(self):
        """Test get_tools_definitions_array with None input."""
        with pytest.raises(AttributeError):
            self.handler.get_tools_definitions_array(None)

    def test_get_tools_definitions_array_non_dict_input(self):
        """Test get_tools_definitions_array with non-dict input."""
        with pytest.raises(TypeError):
            self.handler.get_tools_definitions_array("not a dict")

    def test_get_tools_definitions_object_none_input(self):
        """Test get_tools_definitions_object with None input."""
        result = self.handler.get_tools_definitions_object(None)
        assert result is None

    def test_get_tools_definitions_object_non_dict_input(self):
        """Test get_tools_definitions_object with non-dict input."""
        result = self.handler.get_tools_definitions_object("not a dict")
        assert result == "not a dict"


class TestErrorHandling:
    """Test error handling scenarios for GeminiCompatibilityHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    def test_detect_response_format_keyerror(self):
        """Test detect_response_format handles KeyError gracefully."""
        headers = {"user-agent": "node"}
        body = {}  # Missing 'method' key
        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_adapt_tools_response_typeerror(self):
        """Test adapt_tools_response handles TypeError."""
        # Non-iterable input for dict.items()
        with pytest.raises(TypeError):
            self.handler.adapt_tools_response(123, ResponseFormat.GEMINI_ARRAY_TOOLS)

    def test_adapt_error_response_typeerror(self):
        """Test adapt_error_response handles TypeError."""
        error = {"code": []}  # List instead of string/int
        result = self.handler.adapt_error_response(error, ResponseFormat.GEMINI_NUMERIC_ERRORS)
        assert result["code"] == []  # Should remain unchanged

    def test_create_compatible_response_json_serialization_error(self):
        """Test create_compatible_response with non-serializable data."""
        class NonSerializable:
            pass

        response = self.handler.create_compatible_response(result=NonSerializable())
        # Should not raise exception, but json.dumps might fail later
        assert "result" in response

    def test_create_gemini_compatible_error_response_invalid_format(self):
        """Test create_gemini_compatible_error_response with invalid format."""
        error = create_gemini_compatible_error_response(
            "INVALID_PARAMS", "Test", "invalid_format"
        )
        assert error["code"] == "INVALID_PARAMS"  # Should keep string

    def test_adapt_response_for_gemini_invalid_headers_type(self):
        """Test adapt_response_for_gemini with invalid headers type."""
        with pytest.raises(TypeError):
            adapt_response_for_gemini(request_headers="invalid")


class TestRobustness:
    """Test robustness scenarios for GeminiCompatibilityHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    def test_request_history_limit(self):
        """Test that request history is limited to 10 items."""
        for i in range(15):
            self.handler.detect_response_format({}, {"method": f"test{i}"})

        assert len(self.handler.request_history) == 10
        # Should contain the last 10 requests
        assert self.handler.request_history[-1]["method"] == "test14"

    def test_large_tools_object(self):
        """Test handling of large tools object."""
        large_tools = {f"tool{i}": {"name": f"tool{i}", "description": f"Description {i}"}
                      for i in range(1000)}

        result = self.handler.adapt_tools_response(large_tools, ResponseFormat.GEMINI_ARRAY_TOOLS)
        assert len(result["tools"]) == 1000
        assert result["tools"][0]["name"] == "tool0"
        assert result["tools"][-1]["name"] == "tool999"

    def test_large_error_response(self):
        """Test handling of large error response."""
        large_message = "Error: " + "x" * 10000
        error = {"code": "INTERNAL_ERROR", "message": large_message}

        result = self.handler.adapt_error_response(error, ResponseFormat.GEMINI_NUMERIC_ERRORS)
        assert result["code"] == -32603
        assert len(result["message"]) == len(large_message)

    def test_concurrent_access(self):
        """Test concurrent access to the handler."""
        results = []

        def worker(worker_id):
            for i in range(100):
                format_type = self.handler.detect_response_format(
                    {"user-agent": "node"},
                    {"method": "tools/list", "id": i}
                )
                results.append(format_type)

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All results should be GEMINI_ARRAY_TOOLS
        assert all(r == ResponseFormat.GEMINI_ARRAY_TOOLS for r in results)
        assert len(results) == 500  # 5 threads * 100 iterations

    def test_malformed_jsonrpc_request(self):
        """Test handling of malformed JSON-RPC request."""
        # Missing jsonrpc field
        headers = {"user-agent": "node"}
        body = {"method": "tools/list", "id": 1}

        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.GEMINI_ARRAY_TOOLS

    def test_invalid_request_id_types(self):
        """Test handling of various request ID types."""
        # Test with different ID types
        for request_id in [1, "string", None, [], {}]:
            response = self.handler.create_compatible_response(request_id=request_id)
            assert response["id"] == request_id

    def test_empty_tools_list_adaptation(self):
        """Test adaptation of empty tools list."""
        empty_tools = {}
        result = self.handler.adapt_tools_response(empty_tools, ResponseFormat.GEMINI_ARRAY_TOOLS)
        assert result == {"tools": []}

    def test_nested_error_structure(self):
        """Test handling of nested error structure."""
        nested_error = {
            "code": "INVALID_PARAMS",
            "message": "Invalid parameters",
            "data": {"field": "name", "value": None}
        }

        result = self.handler.adapt_error_response(nested_error, ResponseFormat.GEMINI_NUMERIC_ERRORS)
        assert result["code"] == -32602
        assert "data" in result

    def test_unicode_characters_in_tools(self):
        """Test handling of Unicode characters in tools."""
        unicode_tools = {
            "工具1": {"name": "工具1", "description": "测试工具"},
            "tool2": {"name": "tool2", "description": "Test tool with émojis 🚀"}
        }

        result = self.handler.adapt_tools_response(unicode_tools, ResponseFormat.GEMINI_ARRAY_TOOLS)
        assert len(result["tools"]) == 2
        assert result["tools"][0]["name"] == "工具1"
        assert "🚀" in result["tools"][1]["description"]

    def test_extremely_long_method_name(self):
        """Test handling of extremely long method name."""
        long_method = "a" * 10000
        headers = {"user-agent": "node"}
        body = {"method": long_method}

        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.STANDARD_MCP  # Not a known method

    def test_multiple_format_detection_consistency(self):
        """Test consistency of format detection across multiple calls."""
        headers = {"user-agent": "node"}
        body = {"method": "tools/list"}

        # Make multiple calls
        formats = []
        for _ in range(10):
            format_type = self.handler.detect_response_format(headers, body)
            formats.append(format_type)

        # All should be the same
        assert all(f == ResponseFormat.GEMINI_ARRAY_TOOLS for f in formats)

    def test_memory_efficiency_large_history(self):
        """Test memory efficiency with large request history."""
        # Simulate large request bodies
        large_body = {"method": "test", "params": {"data": "x" * 1000}}

        for i in range(20):
            self.handler.detect_response_format({}, large_body)

        # History should be trimmed to 10
        assert len(self.handler.request_history) == 10
        # Each item should be the large body
        assert len(self.handler.request_history[0]["params"]["data"]) == 1000


class TestGlobalHandler:
    """Test the global compatibility handler instance."""

    def test_get_compatibility_handler_singleton(self):
        """Test that get_compatibility_handler returns a singleton."""
        handler1 = get_compatibility_handler()
        handler2 = get_compatibility_handler()

        assert handler1 is handler2
        assert isinstance(handler1, GeminiCompatibilityHandler)

    def test_global_handler_functionality(self):
        """Test that global handler works correctly."""
        handler = get_compatibility_handler()

        # Test basic functionality
        format_type = handler.detect_response_format({"user-agent": "curl"}, {})
        assert format_type == ResponseFormat.STANDARD_MCP


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    def test_full_request_response_cycle(self):
        """Test a full request-response cycle."""
        # Simulate Gemini CLI request
        headers = {"user-agent": "node"}
        body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        # Detect format
        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.GEMINI_ARRAY_TOOLS

        # Create response
        tools = {"tool1": {"name": "tool1"}}
        response = self.handler.create_compatible_response(
            result={"tools": tools},
            request_headers=headers,
            request_body=body
        )

        # Should be adapted to array format
        assert isinstance(response["tools"], list)
        assert response["tools"][0]["name"] == "tool1"

    def test_error_response_cycle(self):
        """Test error response cycle."""
        headers = {"user-agent": "node"}
        body = {"jsonrpc": "2.0", "method": "tools/call", "id": 1}

        # Detect format
        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.GEMINI_NUMERIC_ERRORS

        # Create error response
        error = {"code": "TOOL_NOT_FOUND", "message": "Tool not found"}
        response = self.handler.create_compatible_response(
            error=error,
            request_headers=headers,
            request_body=body
        )

        # Should have numeric error code
        assert response["error"]["code"] == -32001

    def test_mixed_request_types(self):
        """Test handling mixed request types in sequence."""
        requests = [
            ({"user-agent": "node"}, {"method": "tools/list"}),
            ({"user-agent": "curl"}, {"method": "initialize"}),
            ({"user-agent": "node"}, {"method": "tools/call"}),
            ({}, {"method": "unknown"})
        ]

        expected_formats = [
            ResponseFormat.GEMINI_ARRAY_TOOLS,
            ResponseFormat.STANDARD_MCP,
            ResponseFormat.GEMINI_NUMERIC_ERRORS,
            ResponseFormat.STANDARD_MCP
        ]

        for (headers, body), expected in zip(requests, expected_formats):
            format_type = self.handler.detect_response_format(headers, body)
            assert format_type == expected


if __name__ == "__main__":
    pytest.main([__file__])
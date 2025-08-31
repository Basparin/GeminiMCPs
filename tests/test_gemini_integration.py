#!/usr/bin/env python3
"""
Integration Tests for Gemini CLI Compatibility Layer

This module provides comprehensive integration tests that simulate real FastAPI
request/response cycles with different Gemini CLI versions and scenarios.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from codesage_mcp.gemini_compatibility import (
    GeminiCompatibilityHandler,
    ResponseFormat,
    adapt_response_for_gemini,
    GeminiCliVersion
)


class TestFastAPIIntegration:
    """Test integration with FastAPI request/response cycle."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    @pytest.mark.asyncio
    async def test_gemini_cli_tools_list_integration(self, gemini_cli_headers, tools_list_request_body, sample_tools_object):
        """Test full integration cycle for Gemini CLI tools/list request."""
        # Mock FastAPI request
        request = Mock()
        request.headers = gemini_cli_headers
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/mcp"

        # Simulate receiving the request body
        request_body = tools_list_request_body

        # Process through compatibility handler
        response = adapt_response_for_gemini(
            result={"tools": sample_tools_object},
            request_headers=request.headers,
            request_body=request_body
        )

        # Verify response format for Gemini CLI
        assert "tools" in response
        assert isinstance(response["tools"], list)
        assert len(response["tools"]) == 2
        assert response["tools"][0]["name"] == "code_analysis"
        assert response["tools"][1]["name"] == "suggest_improvements"

    @pytest.mark.asyncio
    async def test_gemini_cli_tools_call_integration(self, gemini_cli_headers, tools_call_request_body, sample_error_response):
        """Test full integration cycle for Gemini CLI tools/call request with error."""
        request = Mock()
        request.headers = gemini_cli_headers
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/mcp"

        request_body = tools_call_request_body

        # Process error response
        response = adapt_response_for_gemini(
            error=sample_error_response,
            request_headers=request.headers,
            request_body=request_body
        )

        # Verify error format for Gemini CLI (numeric codes)
        assert "jsonrpc" in response
        assert "error" in response
        assert response["error"]["code"] == -32602  # INVALID_PARAMS
        assert response["error"]["message"] == "Invalid parameters provided"

    @pytest.mark.asyncio
    async def test_standard_mcp_integration(self, standard_mcp_headers, initialize_request_body):
        """Test integration with standard MCP client."""
        request = Mock()
        request.headers = standard_mcp_headers
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/mcp"

        request_body = initialize_request_body

        # Process success response
        response = adapt_response_for_gemini(
            result={"protocolVersion": "2024-11-05", "capabilities": {}},
            request_headers=request.headers,
            request_body=request_body
        )

        # Verify standard MCP format (JSON-RPC wrapper)
        assert "jsonrpc" in response
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"

    @pytest.mark.asyncio
    async def test_mixed_client_types_integration(self, gemini_cli_headers, standard_mcp_headers, sample_tools_object):
        """Test handling different client types in sequence."""
        # First request from Gemini CLI
        gemini_request = Mock()
        gemini_request.headers = gemini_cli_headers
        gemini_request.method = "POST"
        gemini_request.url = Mock()
        gemini_request.url.path = "/mcp"

        gemini_body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        gemini_response = adapt_response_for_gemini(
            result={"tools": sample_tools_object},
            request_headers=gemini_request.headers,
            request_body=gemini_body
        )

        # Should be array format
        assert isinstance(gemini_response["tools"], list)

        # Second request from standard MCP client
        mcp_request = Mock()
        mcp_request.headers = standard_mcp_headers
        mcp_request.method = "POST"
        mcp_request.url = Mock()
        mcp_request.url.path = "/mcp"

        mcp_body = {"jsonrpc": "2.0", "method": "tools/list", "id": 2}

        mcp_response = adapt_response_for_gemini(
            result={"tools": sample_tools_object},
            request_headers=mcp_request.headers,
            request_body=mcp_body
        )

        # Should be object format
        assert "jsonrpc" in mcp_response
        assert "result" in mcp_response
        assert isinstance(mcp_response["result"]["tools"], dict)


class TestGeminiCliVersions:
    """Test compatibility with different Gemini CLI versions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    def test_gemini_v1_0_detection(self):
        """Test detection of Gemini CLI v1.0."""
        headers = {"user-agent": "gemini-cli/1.0.0 node/18.17.0"}
        body = {"method": "tools/list"}

        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.GEMINI_ARRAY_TOOLS

    def test_gemini_v2_0_detection(self):
        """Test detection of Gemini CLI v2.0."""
        headers = {"user-agent": "gemini-cli/2.0.0 node/20.5.0"}
        body = {"method": "tools/call"}

        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.GEMINI_NUMERIC_ERRORS

    def test_gemini_latest_detection(self):
        """Test detection of latest Gemini CLI."""
        headers = {"user-agent": "gemini-cli/latest node/21.0.0"}
        body = {"method": "initialize"}

        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.GEMINI_NUMERIC_ERRORS

    def test_gemini_cli_user_agents(self):
        """Test various Gemini CLI user agent strings."""
        # All Gemini CLI user agents behave the same way - format determined by method
        test_cases = [
            ("gemini-cli/1.0.0", ResponseFormat.GEMINI_ARRAY_TOOLS),
            ("Gemini-CLI/2.0.0", ResponseFormat.GEMINI_ARRAY_TOOLS),  # Same as others for tools/list
            ("node gemini-cli", ResponseFormat.GEMINI_ARRAY_TOOLS),
            ("gemini node/18", ResponseFormat.GEMINI_ARRAY_TOOLS),
        ]

        for user_agent, expected_format in test_cases:
            headers = {"user-agent": user_agent}
            body = {"method": "tools/list"}

            format_type = self.handler.detect_response_format(headers, body)
            assert format_type == expected_format, f"Failed for user-agent: {user_agent}"

    def test_gemini_cli_v1_0_tools_response(self):
        """Test tools response format for Gemini CLI v1.0."""
        headers = {"user-agent": "gemini-cli/1.0.0"}
        body = {"method": "tools/list"}

        tools = {"tool1": {"name": "tool1"}, "tool2": {"name": "tool2"}}
        response = self.handler.create_compatible_response(
            result={"tools": tools},
            request_headers=headers,
            request_body=body
        )

        # v1.0 expects array format
        assert isinstance(response["tools"], list)
        assert len(response["tools"]) == 2

    def test_gemini_cli_v2_0_error_response(self):
        """Test error response format for Gemini CLI v2.0."""
        headers = {"user-agent": "gemini-cli/2.0.0"}
        body = {"method": "tools/call"}

        error = {"code": "INVALID_PARAMS", "message": "Invalid parameters"}
        response = self.handler.create_compatible_response(
            error=error,
            request_headers=headers,
            request_body=body
        )

        # v2.0 expects numeric error codes
        assert response["error"]["code"] == -32602

    def test_gemini_cli_beta_version_detection(self):
        """Test detection of Gemini CLI beta versions."""
        # All versions behave the same for tools/list method
        test_cases = [
            ("gemini-cli/1.0.0-beta.1", ResponseFormat.GEMINI_ARRAY_TOOLS),
            ("gemini-cli/2.0.0-rc.1", ResponseFormat.GEMINI_ARRAY_TOOLS),  # Same for tools/list
            ("gemini-cli/0.9.0-alpha", ResponseFormat.GEMINI_ARRAY_TOOLS),
        ]

        for user_agent, expected in test_cases:
            headers = {"user-agent": user_agent}
            body = {"method": "tools/list"}
            format_type = self.handler.detect_response_format(headers, body)
            assert format_type == expected, f"Failed for beta user-agent: {user_agent}"

    def test_gemini_cli_version_compatibility_matrix(self):
        """Test compatibility matrix for different Gemini CLI versions."""
        version_matrix = {
            "1.0.0": {
                "tools/list": ResponseFormat.GEMINI_ARRAY_TOOLS,
                "tools/call": ResponseFormat.GEMINI_NUMERIC_ERRORS,
                "initialize": ResponseFormat.GEMINI_NUMERIC_ERRORS,
            },
            "2.0.0": {
                "tools/list": ResponseFormat.GEMINI_ARRAY_TOOLS,
                "tools/call": ResponseFormat.GEMINI_NUMERIC_ERRORS,
                "initialize": ResponseFormat.GEMINI_NUMERIC_ERRORS,
            },
            "latest": {
                "tools/list": ResponseFormat.GEMINI_ARRAY_TOOLS,
                "tools/call": ResponseFormat.GEMINI_NUMERIC_ERRORS,
                "initialize": ResponseFormat.GEMINI_NUMERIC_ERRORS,
            }
        }

        for version, method_formats in version_matrix.items():
            for method, expected_format in method_formats.items():
                headers = {"user-agent": f"gemini-cli/{version}"}
                body = {"method": method}

                format_type = self.handler.detect_response_format(headers, body)
                assert format_type == expected_format, f"Failed for version {version}, method {method}"

    def test_gemini_cli_different_node_versions(self):
        """Test compatibility with different Node.js versions in Gemini CLI."""
        node_versions = ["14.17.0", "16.13.0", "18.17.0", "20.5.0", "21.0.0"]

        for node_version in node_versions:
            headers = {"user-agent": f"gemini-cli/1.0.0 node/{node_version}"}
            body = {"method": "tools/list"}

            format_type = self.handler.detect_response_format(headers, body)
            assert format_type == ResponseFormat.GEMINI_ARRAY_TOOLS, f"Failed for Node.js {node_version}"

    def test_gemini_cli_with_additional_headers(self):
        """Test Gemini CLI detection with additional headers."""
        headers = {
            "user-agent": "gemini-cli/1.0.0 node/18.17.0",
            "content-type": "application/json",
            "accept": "application/json",
            "x-custom-header": "value"
        }
        body = {"method": "tools/list"}

        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.GEMINI_ARRAY_TOOLS

    def test_gemini_cli_case_insensitive_detection(self):
        """Test case insensitive detection of Gemini CLI."""
        user_agents = [
            "GEMINI-CLI/1.0.0",
            "Gemini-Cli/1.0.0",
            "gemini-cli/1.0.0",
            "GeMiNi-ClI/1.0.0"
        ]

        for user_agent in user_agents:
            headers = {"user-agent": user_agent}
            body = {"method": "tools/list"}

            format_type = self.handler.detect_response_format(headers, body)
            assert format_type == ResponseFormat.GEMINI_ARRAY_TOOLS, f"Failed for user-agent: {user_agent}"

    def test_gemini_cli_version_fallback(self):
        """Test fallback behavior for unknown Gemini CLI versions."""
        # Test with a very high version number
        headers = {"user-agent": "gemini-cli/99.99.99"}
        body = {"method": "tools/list"}

        format_type = self.handler.detect_response_format(headers, body)
        # Should still be detected as Gemini CLI
        assert format_type == ResponseFormat.GEMINI_ARRAY_TOOLS

    def test_gemini_cli_mixed_with_other_tools(self):
        """Test Gemini CLI detection when mixed with other tools."""
        user_agents = [
            "gemini-cli/1.0.0 curl/7.88.1",
            "node/18.17.0 gemini-cli/1.0.0 python/3.9",
            "gemini-cli/1.0.0 (compatible; some-tool/1.0)",
        ]

        for user_agent in user_agents:
            headers = {"user-agent": user_agent}
            body = {"method": "tools/list"}

            format_type = self.handler.detect_response_format(headers, body)
            assert format_type == ResponseFormat.GEMINI_ARRAY_TOOLS, f"Failed for mixed user-agent: {user_agent}"


class TestNetworkFailureSimulation:
    """Test network failure and error scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    @patch('json.loads')
    def test_corrupted_json_request_handling(self, mock_json_loads, corrupted_json_strings):
        """Test handling of corrupted JSON in requests."""
        for corrupted_json in corrupted_json_strings:
            mock_json_loads.side_effect = json.JSONDecodeError("Invalid JSON", corrupted_json, 0)

            # Should handle gracefully and default to standard format
            format_type = self.handler.detect_response_format({"user-agent": "node"}, {})
            assert format_type == ResponseFormat.STANDARD_MCP

    def test_network_timeout_simulation(self):
        """Test handling of network timeout scenarios."""
        # Simulate timeout by having None headers/body
        format_type = self.handler.detect_response_format(None, None)
        assert format_type == ResponseFormat.STANDARD_MCP

    def test_partial_request_data(self):
        """Test handling of partial or incomplete request data."""
        # Missing method
        headers = {"user-agent": "node"}
        body = {"jsonrpc": "2.0", "id": 1}

        format_type = self.handler.detect_response_format(headers, body)
        assert format_type == ResponseFormat.STANDARD_MCP

        # Empty body
        format_type = self.handler.detect_response_format(headers, {})
        assert format_type == ResponseFormat.STANDARD_MCP


class TestLargeCodebaseScenarios:
    """Test scenarios with large codebases and extensive tool definitions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    def test_extensive_tools_definitions(self, large_tools_object):
        """Test handling of extensive tools definitions."""
        # Adapt to Gemini array format
        result = self.handler.adapt_tools_response(large_tools_object, ResponseFormat.GEMINI_ARRAY_TOOLS)

        assert "tools" in result
        assert isinstance(result["tools"], list)
        assert len(result["tools"]) == 1000

        # Verify first and last tools
        assert result["tools"][0]["name"] == "tool_0"
        assert result["tools"][-1]["name"] == "tool_999"

    def test_large_response_adaptation(self, large_tools_object):
        """Test adaptation performance with large responses."""
        import time

        start_time = time.time()

        # Perform adaptation
        result = self.handler.adapt_tools_response(large_tools_object, ResponseFormat.GEMINI_ARRAY_TOOLS)

        end_time = time.time()
        adaptation_time = end_time - start_time

        # Should complete within reasonable time (less than 1 second for 1000 tools)
        assert adaptation_time < 1.0
        assert len(result["tools"]) == 1000

    def test_memory_usage_large_tools(self, large_tools_object):
        """Test memory usage with large tools objects."""
        import sys

        # Get memory usage before
        initial_memory = sys.getsizeof(large_tools_object)

        # Adapt the tools
        result = self.handler.adapt_tools_response(large_tools_object, ResponseFormat.GEMINI_ARRAY_TOOLS)

        # Get memory usage after
        final_memory = sys.getsizeof(result)

        # Adapted result should be reasonable size
        assert final_memory > 0
        assert final_memory < initial_memory * 2  # Should not double memory usage


class TestConcurrentRequests:
    """Test handling of concurrent requests."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    def test_concurrent_format_detection(self):
        """Test concurrent format detection."""
        import threading
        import concurrent.futures

        results = []

        def detect_format(client_type):
            if client_type == "gemini":
                headers = {"user-agent": "node"}
                body = {"method": "tools/list"}
            else:
                headers = {"user-agent": "curl"}
                body = {"method": "initialize"}

            format_type = self.handler.detect_response_format(headers, body)
            results.append(format_type)

        # Run concurrent detections
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(10):
                client_type = "gemini" if i % 2 == 0 else "standard"
                futures.append(executor.submit(detect_format, client_type))

            # Wait for all to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()

        # Verify results
        assert len(results) == 10
        gemini_count = sum(1 for r in results if r == ResponseFormat.GEMINI_ARRAY_TOOLS)
        standard_count = sum(1 for r in results if r == ResponseFormat.STANDARD_MCP)
        assert gemini_count == 5
        assert standard_count == 5


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

    def test_complete_tools_workflow(self, gemini_cli_headers, sample_tools_object):
        """Test complete tools listing workflow."""
        # Step 1: Client sends tools/list request
        request_body = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1
        }

        # Step 2: Server detects format
        format_type = self.handler.detect_response_format(gemini_cli_headers, request_body)
        assert format_type == ResponseFormat.GEMINI_ARRAY_TOOLS

        # Step 3: Server adapts tools response
        adapted_tools = self.handler.adapt_tools_response(sample_tools_object, format_type)

        # Step 4: Server creates compatible response
        response = self.handler.create_compatible_response(
            result=adapted_tools,
            request_headers=gemini_cli_headers,
            request_body=request_body
        )

        # Step 5: Verify final response
        assert "tools" in response
        assert isinstance(response["tools"], list)
        assert len(response["tools"]) == 2

    def test_complete_error_workflow(self, gemini_cli_headers, sample_error_response):
        """Test complete error handling workflow."""
        # Step 1: Client sends tools/call request
        request_body = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 2
        }

        # Step 2: Server detects format
        format_type = self.handler.detect_response_format(gemini_cli_headers, request_body)
        assert format_type == ResponseFormat.GEMINI_NUMERIC_ERRORS

        # Step 3: Server adapts error response
        adapted_error = self.handler.adapt_error_response(sample_error_response, format_type)

        # Step 4: Server creates compatible response
        response = self.handler.create_compatible_response(
            error=adapted_error,
            request_headers=gemini_cli_headers,
            request_body=request_body
        )

        # Step 5: Verify final response
        assert "jsonrpc" in response
        assert "error" in response
        assert response["error"]["code"] == -32602  # Numeric code


if __name__ == "__main__":
    pytest.main([__file__])
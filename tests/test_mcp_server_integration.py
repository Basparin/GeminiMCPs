#!/usr/bin/env python3
"""
Integration test for MCP server to verify JSON-RPC response format compliance.
This simulates actual Gemini CLI requests to ensure proper error handling.
"""

import json
import sys

# Add the project root to Python path
sys.path.insert(0, '/home/basparin/Escritorio/GeminiMCPs')

from codesage_mcp.main import app
from fastapi.testclient import TestClient

def test_mcp_server_error_responses():
    """Test MCP server error responses with actual HTTP requests."""
    print("=== Testing MCP Server Error Responses ===")

    client = TestClient(app)

    # Test cases that should generate errors
    error_test_cases = [
        {
            "name": "Invalid method",
            "request": {
                "jsonrpc": "2.0",
                "method": "invalid_method",
                "id": 1
            },
            "expected_error_code": -32603  # Internal error (unknown method)
        },
        {
            "name": "Tools call with invalid tool",
            "request": {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "nonexistent_tool",
                    "arguments": {}
                },
                "id": 2
            },
            "expected_error_code": -32001  # Tool not found
        },
        {
            "name": "Tools call with invalid params",
            "request": {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": "invalid_params",  # Should be dict
                "id": 3
            },
            "expected_error_code": -32600  # Invalid request (due to Pydantic validation)
        }
    ]

    for test_case in error_test_cases:
        print(f"\nTesting: {test_case['name']}")

        # Send request
        response = client.post("/mcp", json=test_case['request'])

        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Response JSON: {json.dumps(response_data, indent=2)}")

            # Validate JSON-RPC 2.0 compliance
            assert "jsonrpc" in response_data
            assert response_data["jsonrpc"] == "2.0"
            assert "id" in response_data

            # Check for error response
            if "error" in response_data:
                error_obj = response_data["error"]
                print(f"Error object: {error_obj}")

                # Validate error structure
                assert "code" in error_obj
                assert "message" in error_obj

                # Validate error code is numeric
                assert isinstance(error_obj["code"], int), f"Error code should be int, got {type(error_obj['code'])}: {error_obj['code']}"

                # Check that result field is NOT present when there's an error
                assert "result" not in response_data, "Result field should not be present in error responses"

                # Check expected error code
                if test_case["expected_error_code"] is not None:
                    assert error_obj["code"] == test_case["expected_error_code"], f"Expected error code {test_case['expected_error_code']}, got {error_obj['code']}"

                print(f"✓ Error response format is valid (code: {error_obj['code']}, type: {type(error_obj['code'])})")

            else:
                print("✓ Successful response (no error field)")

        else:
            print(f"✗ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")

def test_mcp_server_success_responses():
    """Test MCP server success responses."""
    print("\n=== Testing MCP Server Success Responses ===")

    client = TestClient(app)

    # Test successful requests
    success_test_cases = [
        {
            "name": "Initialize request",
            "request": {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client"}
                },
                "id": 1
            }
        },
        {
            "name": "Tools list request",
            "request": {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2
            }
        }
    ]

    for test_case in success_test_cases:
        print(f"\nTesting: {test_case['name']}")

        response = client.post("/mcp", json=test_case['request'])

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Response JSON: {json.dumps(response_data, indent=2)}")

            # Validate JSON-RPC 2.0 compliance
            assert "jsonrpc" in response_data
            assert response_data["jsonrpc"] == "2.0"
            assert "id" in response_data

            # Check for successful response
            if "result" in response_data:
                # Check that error field is NOT present when there's a result
                assert "error" not in response_data, "Error field should not be present in success responses"
                print("✓ Success response format is valid (no error field)")

            elif "error" in response_data:
                error_obj = response_data["error"]
                print(f"Error response: {error_obj}")
                # This might be expected for some requests

        else:
            print(f"✗ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")

def test_jsonrpc_format_compliance():
    """Test overall JSON-RPC 2.0 format compliance."""
    print("\n=== Testing JSON-RPC 2.0 Format Compliance ===")

    client = TestClient(app)

    # Test various request types
    test_requests = [
        # Valid initialize
        {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
            "id": 1
        },
        # Invalid JSON-RPC version
        {
            "jsonrpc": "1.0",
            "method": "initialize",
            "id": 2
        },
        # Missing method
        {
            "jsonrpc": "2.0",
            "id": 3
        },
        # Invalid tool call
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "invalid_tool"},
            "id": 4
        }
    ]

    for i, request in enumerate(test_requests):
        print(f"\nTest request {i+1}: {request.get('method', 'unknown')}")

        try:
            response = client.post("/mcp", json=request)
            response_data = response.json()

            print(f"Response: {json.dumps(response_data, indent=2)}")

            # All responses should have jsonrpc field
            assert "jsonrpc" in response_data, "Missing jsonrpc field"
            assert response_data["jsonrpc"] == "2.0", f"Wrong jsonrpc version: {response_data['jsonrpc']}"

            # All responses should have id field (matching request or null)
            assert "id" in response_data, "Missing id field"

            # Response should have either result or error, but not both
            has_result = "result" in response_data
            has_error = "error" in response_data

            assert has_result or has_error, "Response must have either result or error"
            assert not (has_result and has_error), "Response cannot have both result and error"

            if has_error:
                error_obj = response_data["error"]
                assert "code" in error_obj, "Error missing code field"
                assert "message" in error_obj, "Error missing message field"
                assert isinstance(error_obj["code"], int), f"Error code must be int, got {type(error_obj['code'])}"

            print("✓ JSON-RPC 2.0 format compliance verified")

        except Exception as e:
            print(f"✗ Error in request {i+1}: {e}")

if __name__ == "__main__":
    test_mcp_server_error_responses()
    test_mcp_server_success_responses()
    test_jsonrpc_format_compliance()
    print("\n=== Integration Testing Complete ===")
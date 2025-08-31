#!/usr/bin/env python3
"""
Direct test of MCP server response format to identify Gemini CLI compatibility issues.
This script simulates actual JSON-RPC requests and examines the raw responses.
"""

import json
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/basparin/Escritorio/GeminiMCPs')

from codesage_mcp.main import JSONRPCResponse
from codesage_mcp.gemini_compatibility import create_gemini_compatible_error_response

def test_direct_response_creation():
    """Test direct creation of JSON-RPC responses."""
    print("=== Testing Direct Response Creation ===")

    # Test successful response
    success_response = JSONRPCResponse(
        jsonrpc="2.0",
        result={"tools": []},
        error=None,
        id=123
    )

    success_dict = success_response.dict()
    print("Successful response dict:")
    print(json.dumps(success_dict, indent=2))
    print(f"Has 'error' field: {'error' in success_dict}")
    print(f"Has 'result' field: {'result' in success_dict}")
    print()

    # Test error response
    error_data = create_gemini_compatible_error_response("TOOL_NOT_FOUND", "Tool not found")
    error_response = JSONRPCResponse(
        jsonrpc="2.0",
        result=None,
        error=error_data,
        id=456
    )

    error_dict = error_response.dict()
    print("Error response dict:")
    print(json.dumps(error_dict, indent=2))
    print(f"Has 'error' field: {'error' in error_dict}")
    print(f"Has 'result' field: {'result' in error_dict}")
    print(f"Error field content: {error_dict.get('error', 'NOT FOUND')}")
    print()

def test_raw_jsonrpc_structure():
    """Test raw JSON-RPC structure compliance."""
    print("=== Testing Raw JSON-RPC Structure ===")

    # Test successful response structure
    success_raw = {
        "jsonrpc": "2.0",
        "result": {"status": "ok"},
        "id": 1
    }
    print("Raw successful response:")
    print(json.dumps(success_raw, indent=2))
    print()

    # Test error response structure
    error_raw = {
        "jsonrpc": "2.0",
        "error": {
            "code": -32601,
            "message": "Method not found"
        },
        "id": 2
    }
    print("Raw error response:")
    print(json.dumps(error_raw, indent=2))
    print()

def test_error_field_variations():
    """Test different error field structures that might cause issues."""
    print("=== Testing Error Field Variations ===")

    # Test with different error structures
    error_cases = [
        # Standard MCP error
        {
            "code": -32601,
            "message": "Method not found"
        },
        # Error with data field
        {
            "code": -32601,
            "message": "Method not found",
            "data": {"method": "unknown_method"}
        },
        # String error code (should be converted)
        {
            "code": "TOOL_NOT_FOUND",
            "message": "Tool not found"
        },
        # Error with extra fields (potential issue)
        {
            "code": -32601,
            "message": "Method not found",
            "extra_field": "this might cause issues"
        }
    ]

    for i, error_case in enumerate(error_cases):
        print(f"Error case {i+1}:")
        print(json.dumps(error_case, indent=2))

        # Test with JSONRPCResponse using create_compatible_response (proper way)
        response = JSONRPCResponse.create_compatible_response(
            result=None,
            error=error_case,
            request_id=100 + i
        )

        response_dict = response.dict()
        print(f"Response dict: {json.dumps(response_dict, indent=2)}")
        print(f"Error code type: {type(response_dict['error']['code'])}")
        print()

def test_json_serialization():
    """Test JSON serialization of responses."""
    print("=== Testing JSON Serialization ===")

    # Create error response
    error_data = create_gemini_compatible_error_response("TOOL_NOT_FOUND", "Tool not found")
    error_response = JSONRPCResponse(
        jsonrpc="2.0",
        result=None,
        error=error_data,
        id=789
    )

    # Serialize to JSON
    json_str = json.dumps(error_response.dict())
    print(f"JSON string: {json_str}")

    # Parse back
    parsed = json.loads(json_str)
    print(f"Parsed back: {json.dumps(parsed, indent=2)}")

    # Check for any unexpected fields
    expected_fields = {'jsonrpc', 'error', 'id'}
    actual_fields = set(parsed.keys())
    print(f"Expected fields: {expected_fields}")
    print(f"Actual fields: {actual_fields}")
    print(f"Extra fields: {actual_fields - expected_fields}")
    print()

if __name__ == "__main__":
    test_direct_response_creation()
    test_raw_jsonrpc_structure()
    test_error_field_variations()
    test_json_serialization()
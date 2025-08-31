#!/usr/bin/env python3
"""
Simple test to verify JSON-RPC response format compliance.
This script tests the actual server responses to ensure the error field
is properly excluded from successful responses.
"""

import json
import requests
import sys

def test_jsonrpc_format():
    """Test JSON-RPC response format."""
    server_url = "http://127.0.0.1:8000"

    # Test 1: Successful initialize request
    print("Testing successful initialize request...")
    init_request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {},
        "id": 1
    }

    try:
        response = requests.post(f"{server_url}/mcp", json=init_request)
        response.raise_for_status()
        result = response.json()

        print(f"Response: {json.dumps(result, indent=2)}")

        # Check that error field is NOT present in successful response
        if 'error' in result:
            print("❌ FAIL: 'error' field found in successful response")
            return False
        else:
            print("✅ PASS: 'error' field correctly excluded from successful response")

        # Check required fields are present
        required_fields = ['jsonrpc', 'result', 'id']
        for field in required_fields:
            if field not in result:
                print(f"❌ FAIL: Missing required field '{field}'")
                return False

        print("✅ PASS: All required fields present")

    except Exception as e:
        print(f"❌ FAIL: Error testing initialize: {e}")
        return False

    # Test 2: Error response (invalid method)
    print("\nTesting error response...")
    error_request = {
        "jsonrpc": "2.0",
        "method": "invalid_method",
        "params": {},
        "id": 2
    }

    try:
        response = requests.post(f"{server_url}/mcp", json=error_request)
        response.raise_for_status()
        result = response.json()

        print(f"Response: {json.dumps(result, indent=2)}")

        # Check that error field IS present in error response
        if 'error' not in result:
            print("❌ FAIL: 'error' field missing from error response")
            return False
        else:
            print("✅ PASS: 'error' field correctly included in error response")

        # Check that result is null when there's an error
        if result.get('result') is not None:
            print("❌ FAIL: 'result' should be null when there's an error")
            return False
        else:
            print("✅ PASS: 'result' is correctly null in error response")

    except Exception as e:
        print(f"❌ FAIL: Error testing invalid method: {e}")
        return False

    print("\n🎉 All JSON-RPC format tests PASSED!")
    return True

if __name__ == "__main__":
    success = test_jsonrpc_format()
    sys.exit(0 if success else 1)
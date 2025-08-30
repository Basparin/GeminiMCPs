#!/usr/bin/env python3
"""
Test script to verify Gemini CLI integration with CodeSage MCP Server.
This script simulates Gemini CLI requests to test the MCP integration.
"""

import json
import requests
import sys
import time
from typing import Dict, Any

# MCP Server URL
MCP_URL = "http://127.0.0.1:8002/mcp"

def test_initialize():
    """Test the initialize request."""
    print("Testing initialize request...")

    request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1
    }

    try:
        response = requests.post(MCP_URL, json=request, headers={
            'User-Agent': 'gemini-cli/1.0.0',
            'Content-Type': 'application/json'
        })
        response.raise_for_status()

        result = response.json()
        print(f"✅ Initialize response: {json.dumps(result, indent=2)}")

        # Validate response structure
        assert 'jsonrpc' in result
        assert 'id' in result
        assert 'result' in result
        assert 'error' not in result  # Should not have error in successful response

        return True
    except Exception as e:
        print(f"❌ Initialize failed: {e}")
        return False

def test_tools_list():
    """Test the tools/list request."""
    print("\nTesting tools/list request...")

    request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 2
    }

    try:
        response = requests.post(MCP_URL, json=request, headers={
            'User-Agent': 'gemini-cli/1.0.0',
            'Content-Type': 'application/json'
        })
        response.raise_for_status()

        result = response.json()
        print(f"✅ Tools/list response: {json.dumps(result, indent=2)}")

        # Validate response structure
        assert 'jsonrpc' in result
        assert 'id' in result
        assert 'result' in result
        assert 'error' not in result  # Should not have error in successful response

        # Check if tools are properly formatted
        if 'tools' in result['result']:
            tools = result['result']['tools']
            print(f"Found {len(tools)} tools")
            for tool in tools[:3]:  # Show first 3 tools
                print(f"  - {tool.get('name', 'unknown')}: {tool.get('description', '')[:50]}...")
        elif isinstance(result['result'], dict):
            print(f"Tools returned as object with {len(result['result'])} entries")
        else:
            print(f"Unexpected tools format: {type(result['result'])}")

        return True
    except Exception as e:
        print(f"❌ Tools/list failed: {e}")
        return False

def test_tools_call():
    """Test a tools/call request."""
    print("\nTesting tools/call request...")

    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 3,
        "params": {
            "name": "get_configuration"
        }
    }

    try:
        response = requests.post(MCP_URL, json=request, headers={
            'User-Agent': 'gemini-cli/1.0.0',
            'Content-Type': 'application/json'
        })
        response.raise_for_status()

        result = response.json()
        print(f"✅ Tools/call response: {json.dumps(result, indent=2)}")

        # Validate response structure
        assert 'jsonrpc' in result
        assert 'id' in result
        assert 'result' in result
        assert 'error' not in result  # Should not have error in successful response

        return True
    except Exception as e:
        print(f"❌ Tools/call failed: {e}")
        return False

def test_error_response():
    """Test error response handling."""
    print("\nTesting error response...")

    # Send invalid request to trigger error
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 4,
        "params": {
            "name": "nonexistent_tool"
        }
    }

    try:
        response = requests.post(MCP_URL, json=request, headers={
            'User-Agent': 'gemini-cli/1.0.0',
            'Content-Type': 'application/json'
        })
        response.raise_for_status()

        result = response.json()
        print(f"✅ Error response: {json.dumps(result, indent=2)}")

        # Validate error response structure
        assert 'jsonrpc' in result
        assert 'id' in result
        assert 'error' in result
        assert 'result' not in result  # Should not have result in error response

        return True
    except Exception as e:
        print(f"❌ Error response test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting Gemini CLI Integration Tests")
    print("=" * 50)

    # Check if server is running
    try:
        response = requests.get("http://127.0.0.1:8002/")
        if response.status_code != 200:
            print("❌ MCP Server is not running. Please start it first:")
            print("   uvicorn codesage_mcp.main:app --host 127.0.0.1 --port 8002")
            sys.exit(1)
    except:
        print("❌ Cannot connect to MCP Server. Please start it first:")
        print("   uvicorn codesage_mcp.main:app --host 127.0.0.1 --port 8002")
        sys.exit(1)

    print("✅ MCP Server is running")

    # Run tests
    tests = [
        test_initialize,
        test_tools_list,
        test_tools_call,
        test_error_response
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        time.sleep(0.1)  # Small delay between tests

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Gemini CLI integration should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the server logs for more details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
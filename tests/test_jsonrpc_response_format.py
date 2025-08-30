#!/usr/bin/env python3
"""
Test JSON-RPC Response Format Compliance

This module tests that the JSON-RPC response format complies with the JSON-RPC 2.0 specification,
specifically that the 'error' field is only present when there is an actual error.
"""

import json
import pytest
from codesage_mcp.main import JSONRPCResponse


class TestJSONRPCResponseFormat:
    """Test JSON-RPC response format compliance."""

    def test_successful_response_excludes_error_field(self):
        """Test that successful responses don't include the error field."""
        response = JSONRPCResponse(
            jsonrpc="2.0",
            result={"tools": []},
            error=None,
            id=123
        )

        response_dict = response.dict()

        # Verify error field is not present
        assert 'error' not in response_dict

        # Verify required fields are present
        assert response_dict['jsonrpc'] == '2.0'
        assert response_dict['result'] == {"tools": []}
        assert response_dict['id'] == 123

    def test_error_response_includes_error_field(self):
        """Test that error responses include the error field."""
        error_data = {"code": -32601, "message": "Method not found"}
        response = JSONRPCResponse(
            jsonrpc="2.0",
            result=None,
            error=error_data,
            id=123
        )

        response_dict = response.dict()

        # Verify error field is present
        assert 'error' in response_dict
        assert response_dict['error'] == error_data

        # Verify result field is not present (JSON-RPC 2.0 spec: must not contain result when error is present)
        assert 'result' not in response_dict

        # Verify required fields are present
        assert response_dict['jsonrpc'] == '2.0'
        assert response_dict['id'] == 123

    def test_json_serialization_compliance(self):
        """Test that JSON serialization produces correct format."""
        # Test successful response
        success_response = JSONRPCResponse(
            jsonrpc="2.0",
            result={"status": "ok"},
            error=None,
            id=456
        )

        success_json = json.dumps(success_response.dict())
        success_data = json.loads(success_json)

        # Should not have error field
        assert 'error' not in success_data
        assert success_data['result'] == {"status": "ok"}

        # Test error response
        error_response = JSONRPCResponse(
            jsonrpc="2.0",
            result=None,
            error={"code": -32700, "message": "Parse error"},
            id=789
        )

        error_json = json.dumps(error_response.dict())
        error_data = json.loads(error_json)

        # Should have error field
        assert 'error' in error_data
        assert error_data['error']['code'] == -32700
        # Should not have result field (JSON-RPC 2.0 spec)
        assert 'result' not in error_data

    def test_response_with_null_id(self):
        """Test response with null id (notification responses)."""
        response = JSONRPCResponse(
            jsonrpc="2.0",
            result=None,
            error=None,
            id=None
        )

        response_dict = response.dict()

        # Should not have error field
        assert 'error' not in response_dict
        assert response_dict['id'] is None

    def test_complete_response_structure(self):
        """Test complete response structure matches JSON-RPC 2.0 spec."""
        # Successful response structure
        success = JSONRPCResponse(
            jsonrpc="2.0",
            result={"data": "test"},
            error=None,
            id=1
        )

        success_dict = success.dict()
        expected_success_keys = {'jsonrpc', 'result', 'id'}
        assert set(success_dict.keys()) == expected_success_keys

        # Error response structure
        error = JSONRPCResponse(
            jsonrpc="2.0",
            result=None,
            error={"code": 1, "message": "test error"},
            id=2
        )

        error_dict = error.dict()
        expected_error_keys = {'jsonrpc', 'error', 'id'}  # result should NOT be present when there's an error
        assert set(error_dict.keys()) == expected_error_keys


if __name__ == "__main__":
    pytest.main([__file__])
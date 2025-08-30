"""Gemini CLI Compatibility Layer for CodeSage MCP Server.

This module provides compatibility handling for different Gemini CLI versions
and response format expectations, including:
- Tools format (array vs object)
- Error code format (number vs string)
- Response structure variations
- JSON-RPC format adaptations
"""

import logging
from typing import Dict, List, Any, Union, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class GeminiCliVersion(Enum):
    """Known Gemini CLI versions with different compatibility requirements."""
    V1_0 = "1.0"
    V2_0 = "2.0"
    LATEST = "latest"


class ResponseFormat(Enum):
    """Supported response format types."""
    STANDARD_MCP = "standard_mcp"
    GEMINI_ARRAY_TOOLS = "gemini_array_tools"
    GEMINI_NUMERIC_ERRORS = "gemini_numeric_errors"


class GeminiCompatibilityHandler:
    """Handles Gemini CLI compatibility for different versions and formats."""

    # Standard MCP error codes (string format)
    MCP_ERROR_CODES = {
        "INVALID_REQUEST": -32600,
        "METHOD_NOT_FOUND": -32601,
        "INVALID_PARAMS": -32602,
        "INTERNAL_ERROR": -32603,
        "SERVER_ERROR": -32000,
        "TOOL_NOT_FOUND": -32001,
        "TOOL_EXECUTION_ERROR": -32002,
        "FILE_NOT_FOUND": -32003,
        "INVALID_INPUT": -32004,
        "PERMISSION_DENIED": -32005,
    }

    def __init__(self):
        self.request_history = []

    def detect_response_format(self, request_headers: Dict[str, Any],
                                request_body: Dict[str, Any]) -> ResponseFormat:
        """Detect the expected response format based on request characteristics."""

        # Track this request in history
        self.request_history.append(request_body)
        # Keep only last 10 requests to avoid memory issues
        if len(self.request_history) > 10:
            self.request_history = self.request_history[-10:]

        # Check for Gemini CLI specific headers
        user_agent = request_headers.get('user-agent', '').lower()
        if 'gemini' in user_agent or 'node' in user_agent:
            # Check request patterns that indicate format preferences
            method = request_body.get('method', '')

            # tools/list requests from Gemini CLI expect array format
            if method == 'tools/list':
                return ResponseFormat.GEMINI_ARRAY_TOOLS

            # Check for error response patterns
            if method in ['tools/call', 'initialize']:
                return ResponseFormat.GEMINI_NUMERIC_ERRORS

        return ResponseFormat.STANDARD_MCP

    def adapt_tools_response(self, tools_object: Dict[str, Any],
                            format_type: ResponseFormat) -> Dict[str, Any]:
        """Adapt tools response format based on detected requirements."""

        if format_type == ResponseFormat.GEMINI_ARRAY_TOOLS:
            # Convert object format to array format
            tools_array = []
            for tool_name, tool_def in tools_object.items():
                tools_array.append(tool_def)
            logger.debug(f"Adapted tools to array format with {len(tools_array)} tools")
            return {"tools": tools_array}

        # For standard MCP or other formats, ensure consistent structure
        # Gemini CLI may expect tools to be wrapped even in standard format
        if format_type == ResponseFormat.STANDARD_MCP:
            tools_array = []
            for tool_name, tool_def in tools_object.items():
                tools_array.append(tool_def)
            logger.debug(f"Adapted tools to standard array format with {len(tools_array)} tools")
            return {"tools": tools_array}

        # Fallback to object format
        logger.debug(f"Using fallback object format for tools")
        return tools_object

    def adapt_error_response(self, error_response: Dict[str, Any],
                           format_type: ResponseFormat) -> Dict[str, Any]:
        """Adapt error response format for compatibility."""

        if format_type == ResponseFormat.GEMINI_NUMERIC_ERRORS:
            # Convert string error codes to numeric codes
            error_code = error_response.get('code', 'INTERNAL_ERROR')
            if isinstance(error_code, str):
                # Map string codes to numeric codes
                numeric_code = self.MCP_ERROR_CODES.get(error_code, -32603)
                error_response['code'] = numeric_code

        return error_response

    def create_compatible_response(self,
                                   result: Optional[Any] = None,
                                   error: Optional[Dict[str, Any]] = None,
                                   request_id: Union[str, int, None] = None) -> Dict[str, Any]:
        """Create a response compatible with the detected Gemini CLI format."""

        # Start with base response structure
        response = {
            "jsonrpc": "2.0",
        }

        # Only include 'id' if it's not None (for requests that expect a response)
        if request_id is not None:
            response["id"] = request_id

        if error is not None:
            # Adapt error format based on detected format (assuming standard MCP for now)
            # This part might need further refinement if specific error code adaptations are truly needed
            adapted_error = self.adapt_error_response(error, ResponseFormat.STANDARD_MCP)
            response["error"] = adapted_error
            # Do not include 'result' field when there's an error (JSON-RPC 2.0 spec)
            logger.debug(f"Created error response: {response}")
        else:
            # For successful responses, only include result field
            response["result"] = result
            logger.debug(f"Created success response with result type: {type(result)}")

        return response

    def get_tools_definitions_array(self, tools_object: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert tools object to array format for Gemini CLI compatibility."""
        tools_array = []
        for tool_name, tool_def in tools_object.items():
            tools_array.append(tool_def)
        return tools_array

    def get_tools_definitions_object(self, tools_object: Dict[str, Any]) -> Dict[str, Any]:
        """Return tools in standard object format."""
        return tools_object


# Global compatibility handler instance
compatibility_handler = GeminiCompatibilityHandler()


def get_compatibility_handler() -> GeminiCompatibilityHandler:
    """Get the global Gemini compatibility handler instance."""
    return compatibility_handler


def create_gemini_compatible_error_response(error_code: Union[str, int],
                                           message: str) -> Dict[str, Any]:
    """Create an error response compatible with Gemini CLI expectations."""

    # Convert string codes to numeric if needed
    if isinstance(error_code, str):
        numeric_code = GeminiCompatibilityHandler.MCP_ERROR_CODES.get(error_code, -32603)
        error_code = numeric_code

    return {
        "code": error_code,
        "message": message
    }


def adapt_response_for_gemini(result: Optional[Any] = None,
                            error: Optional[Dict[str, Any]] = None,
                            request_id: Union[str, int, None] = None) -> Dict[str, Any]:
    """Adapt response for Gemini CLI compatibility."""
    return compatibility_handler.create_compatible_response(
        result=result,
        error=error,
        request_id=request_id
    )
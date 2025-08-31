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

    def detect_response_format(self, request_headers: Optional[Dict[str, Any]],
                                request_body: Optional[Dict[str, Any]]) -> ResponseFormat:
        """Detect the expected response format based on request characteristics."""

        # Handle None values and validate types by defaulting to empty dicts
        if request_headers is None or not isinstance(request_headers, dict):
            logger.warning(f"Invalid request_headers type: {type(request_headers)}, defaulting to empty dict")
            request_headers = {}
        if request_body is None or not isinstance(request_body, dict):
            logger.warning(f"Invalid request_body type: {type(request_body)}, defaulting to empty dict")
            request_body = {}

        # Track this request in history
        self.request_history.append(request_body)
        # Keep only last 10 requests to avoid memory issues
        if len(self.request_history) > 10:
            self.request_history = self.request_history[-10:]

        # Check for Gemini CLI specific headers
        user_agent = request_headers.get('user-agent', '')
        if isinstance(user_agent, str):
            user_agent = user_agent.lower()
        else:
            logger.warning(f"Invalid user-agent type: {type(user_agent)}, expected str, defaulting to empty")
            user_agent = ''
        if 'gemini' in user_agent or 'node' in user_agent:
            # Check request patterns that indicate format preferences
            method = request_body.get('method', '')

            # tools/list requests from Gemini CLI expect array format
            if method == 'tools/list':
                return ResponseFormat.GEMINI_ARRAY_TOOLS

            # Check for error response patterns
            if method in ['tools/call', 'initialize']:
                return ResponseFormat.GEMINI_NUMERIC_ERRORS

        format_type = ResponseFormat.STANDARD_MCP
        logger.debug(f"Detected response format: {format_type}")
        return format_type

    def adapt_tools_response(self, tools_object: Dict[str, Any],
                            format_type: ResponseFormat) -> Dict[str, Any]:
        """Adapt tools response format based on detected requirements."""

        logger.debug(f"Adapting tools response for format: {format_type}")

        # Handle None input defensively
        if tools_object is None:
            logger.debug("Tools object is None, returning empty tools array")
            return {"tools": []}

        if format_type == ResponseFormat.GEMINI_ARRAY_TOOLS:
            # Check if tools are already in array format
            if isinstance(tools_object, list):
                logger.debug("Tools already in array format, no adaptation needed")
                return {"tools": tools_object}
            # Check if tools are already wrapped in array format
            if isinstance(tools_object, dict) and isinstance(tools_object.get("tools"), list):
                logger.debug("Tools already in wrapped array format, no adaptation needed")
                return tools_object
            # Convert object format to array format
            if isinstance(tools_object, dict):
                tools_array = []
                for tool_name, tool_def in tools_object.items():
                    tools_array.append(tool_def)
                logger.debug(f"Adapted tools to array format with {len(tools_array)} tools")
                return {"tools": tools_array}
            # For invalid types, raise TypeError to maintain expected behavior
            else:
                logger.error(f"Invalid tools_object type: {type(tools_object)}, expected dict or list")
                raise TypeError(f"Invalid tools_object type: {type(tools_object)}, expected dict or list")

        # For standard MCP or other formats, ensure consistent structure
        # Gemini CLI may expect tools to be wrapped even in standard format
        if format_type == ResponseFormat.STANDARD_MCP:
            logger.debug(f"Using standard object format for tools")
            return tools_object # Return the original object for standard MCP

        # Fallback to object format
        logger.debug(f"Using fallback object format for tools")
        return tools_object

    def adapt_error_response(self, error_response: Dict[str, Any],
                            format_type: ResponseFormat) -> Dict[str, Any]:
        """Adapt error response format for compatibility."""

        logger.debug(f"Adapting error response for format: {format_type}")

        # Handle None input defensively
        if error_response is None:
            logger.debug("Error response is None, returning None")
            return None

        # Ensure error_response is a dict before processing
        if not isinstance(error_response, dict):
            logger.warning(f"Invalid error_response type: {type(error_response)}, returning as is")
            return error_response

        if format_type == ResponseFormat.GEMINI_NUMERIC_ERRORS:
            # Convert string error codes to numeric codes
            error_code = error_response.get('code', 'INTERNAL_ERROR')
            if isinstance(error_code, str):
                # Map string codes to numeric codes
                numeric_code = self.MCP_ERROR_CODES.get(error_code, -32603)
                logger.debug(f"Converting error code '{error_code}' to numeric {numeric_code}")
                error_response['code'] = numeric_code

        return error_response

    def create_compatible_response(self,
                                     result: Optional[Any] = None,
                                     error: Optional[Dict[str, Any]] = None,
                                     request_id: Union[str, int, None] = None,
                                     request_headers: Optional[Dict[str, Any]] = None,
                                     request_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a response compatible with the detected Gemini CLI format."""

        logger.debug("Creating compatible response")

        # Validate inputs
        if error is not None and not isinstance(error, dict):
            logger.warning(f"Invalid error type: {type(error)}, expected dict or None")
            error = None  # Reset to None to avoid issues

        if request_id is not None and not isinstance(request_id, (str, int)):
            logger.warning(f"Invalid request_id type: {type(request_id)}, expected str, int, or None")

        format_type = self.detect_response_format(request_headers, request_body)
        logger.debug(f"Using format type: {format_type}")

        # For Gemini formats, return the payload directly at the top level for success, but wrap errors
        if format_type in [ResponseFormat.GEMINI_ARRAY_TOOLS, ResponseFormat.GEMINI_NUMERIC_ERRORS]:
            if error is not None:
                adapted_error = self.adapt_error_response(error, format_type)
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": adapted_error
                }
                logger.debug(f"Created wrapped Gemini error response: {response}")
                return response
            else:
                logger.debug(f"Created unwrapped Gemini success response with result type: {type(result)}")
                if format_type == ResponseFormat.GEMINI_ARRAY_TOOLS and isinstance(result, dict) and "tools" in result:
                    result = self.adapt_tools_response(result["tools"], format_type)
                return result

        # Standard MCP format with JSON-RPC wrapper
        response = {
            "jsonrpc": "2.0",
        }

        # Always include 'id' field, even if None, for JSON-RPC 2.0 compliance in error responses
        response["id"] = request_id

        if error is not None:
            adapted_error = self.adapt_error_response(error, format_type)
            response["error"] = adapted_error
            # Do not include 'result' field when there's an error (JSON-RPC 2.0 spec)
            logger.debug(f"Created standard error response: {response}")
        else:
            # For successful responses, only include result field
            response["result"] = result
            logger.debug(f"Created standard success response with result type: {type(result)}")

        return response

    def get_tools_definitions_array(self, tools_object: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert tools object to array format for Gemini CLI compatibility."""
        logger.debug("Converting tools object to array format")
        if tools_object is None:
            logger.error("Tools object is None, cannot convert to array")
            raise AttributeError("'NoneType' object has no attribute 'items'")
        if not isinstance(tools_object, dict):
            logger.error(f"Invalid tools_object type: {type(tools_object)}, expected dict")
            raise TypeError(f"Expected dict, got {type(tools_object)}")
        tools_array = []
        for tool_name, tool_def in tools_object.items():
            tools_array.append(tool_def)
        logger.debug(f"Converted {len(tools_array)} tools to array format")
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
                                           message: str,
                                           format_type: Optional[ResponseFormat] = None) -> Dict[str, Any]:
    """Create an error response compatible with Gemini CLI expectations."""

    logger.debug(f"Creating Gemini compatible error response with code: {error_code}")

    # Default to GEMINI_NUMERIC_ERRORS for backward compatibility
    if format_type is None:
        format_type = ResponseFormat.GEMINI_NUMERIC_ERRORS

    # Convert string codes to numeric if needed based on format type
    if format_type == ResponseFormat.GEMINI_NUMERIC_ERRORS and isinstance(error_code, str):
        numeric_code = GeminiCompatibilityHandler.MCP_ERROR_CODES.get(error_code, -32603)
        logger.debug(f"Converting error code '{error_code}' to numeric {numeric_code}")
        error_code = numeric_code
    # For STANDARD_MCP, keep string codes as-is

    return {
        "code": error_code,
        "message": message
    }


def adapt_response_for_gemini(result: Optional[Any] = None,
                             error: Optional[Dict[str, Any]] = None,
                             request_id: Union[str, int, None] = None,
                             request_headers: Optional[Dict[str, Any]] = None,
                             request_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Adapt response for Gemini CLI compatibility."""
    logger.debug("Adapting response for Gemini compatibility")
    if request_headers is not None and not isinstance(request_headers, dict):
        logger.error(f"Invalid request_headers type: {type(request_headers)}, expected dict or None")
        raise TypeError(f"request_headers must be dict or None, got {type(request_headers)}")
    if request_body is not None and not isinstance(request_body, dict):
        logger.error(f"Invalid request_body type: {type(request_body)}, expected dict or None")
        raise TypeError(f"request_body must be dict or None, got {type(request_body)}")
    return compatibility_handler.create_compatible_response(
        result=result,
        error=error,
        request_id=request_id,
        request_headers=request_headers,
        request_body=request_body
    )
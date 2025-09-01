"""
API Handling Module for CodeSage MCP Server.

This module provides classes for handling FastAPI applications, JSON-RPC processing,
and MCP protocol implementation. These classes are designed to work with the
MCP (Model Context Protocol) specification and provide a clean interface for
API interactions.

Classes:
    FastAPIApp: Manages FastAPI application setup and routes
    JSONRPCProcessor: Handles JSON-RPC request/response processing
    MCPProtocol: Implements MCP protocol handshake and message handling
"""

from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .logging_config import setup_logging, get_logger, log_exception
from .gemini_compatibility import (
    get_compatibility_handler,
    ResponseFormat
)
from codesage_mcp.features.performance_monitoring.performance_monitor import get_performance_monitor, get_usage_analyzer
from codesage_mcp.features.user_feedback.user_feedback import get_user_feedback_collector
from codesage_mcp.features.performance_monitoring.trend_analysis import get_trend_analyzer
from codesage_mcp.features.performance_monitoring.auto_performance_tuner import get_auto_performance_tuner
from codesage_mcp.features.caching.adaptive_cache_manager import get_adaptive_cache_manager
from codesage_mcp.features.memory_management.memory_manager import get_memory_manager
from codesage_mcp.features.memory_management.workload_pattern_recognition import get_workload_pattern_recognition


class GeminiCompatibleJSONResponse(JSONResponse):
    """Custom JSON response that excludes None fields for Gemini CLI compatibility."""

    def render(self, content: Any) -> bytes:
        # Recursively remove None fields from the response
        def remove_none_fields(obj):
            if isinstance(obj, dict):
                return {k: remove_none_fields(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [remove_none_fields(item) for item in obj]
            else:
                return obj

        cleaned_content = remove_none_fields(content)
        return super().render(cleaned_content)


class FastAPIApp:
    """
    Manages FastAPI application setup and routes for MCP server.

    This class encapsulates the FastAPI application configuration, middleware,
    and route registration for the MCP server. It provides methods for
    initializing the app and registering MCP-specific routes.

    Attributes:
        app: The FastAPI application instance
    """

    def __init__(self):
        """Initialize the FastAPI application with MCP configuration."""
        self.app = FastAPI()
        self._setup_middleware()
        self._register_routes()

    def _setup_middleware(self):
        """Set up middleware for the FastAPI application."""
        # Only set up middleware if app has the middleware method (not a mock)
        if hasattr(self.app, 'middleware'):
            @self.app.middleware("http")
            async def log_requests(request: Request, call_next):
                logger = get_logger(__name__)
                logger.info("Incoming request", method=request.method, url=str(request.url), headers=dict(request.headers))
                response = await call_next(request)
                return response

    def _register_routes(self):
        """Register MCP-specific routes."""
        # Only register routes if app has the required methods (not a mock)
        if hasattr(self.app, 'get') and hasattr(self.app, 'post'):
            # Root endpoints
            @self.app.get("/")
            async def root_get():
                return {"message": "CodeSage MCP Server is running!"}

            @self.app.post("/")
            async def root_post():
                return {"message": "CodeSage MCP Server is running!"}

            # Metrics endpoint
            @self.app.get("/metrics")
            async def metrics():
                """
                Prometheus metrics endpoint for monitoring and alerting.
                Returns metrics in Prometheus exposition format.
                """
            try:
                # Get current performance metrics
                performance_monitor = get_performance_monitor()
                current_metrics = performance_monitor.get_current_metrics()
                report = performance_monitor.get_performance_report()

                # Format metrics for Prometheus
                prometheus_output = []

                # Add basic info metric
                prometheus_output.append("# HELP codesage_mcp_info CodeSage MCP Server information")
                prometheus_output.append("# TYPE codesage_mcp_info gauge")
                prometheus_output.append('codesage_mcp_info{version="0.1.0"} 1')

                # Add uptime metric
                uptime_seconds = report.get("uptime_seconds", 0)
                prometheus_output.append("# HELP codesage_mcp_uptime_seconds Time since server start")
                prometheus_output.append("# TYPE codesage_mcp_uptime_seconds counter")
                prometheus_output.append(f"codesage_mcp_uptime_seconds {uptime_seconds}")

                # Add request metrics
                prometheus_output.append("# HELP codesage_mcp_requests_total Total number of requests processed")
                prometheus_output.append("# TYPE codesage_mcp_requests_total counter")
                prometheus_output.append(f"codesage_mcp_requests_total {performance_monitor.request_count}")

                prometheus_output.append("# HELP codesage_mcp_errors_total Total number of errors")
                prometheus_output.append("# TYPE codesage_mcp_errors_total counter")
                prometheus_output.append(f"codesage_mcp_errors_total {performance_monitor.error_count}")

                # Add response time metrics
                response_time = current_metrics.get("response_time_ms", {}).get("value")
                if response_time is not None:
                    prometheus_output.append("# HELP codesage_mcp_response_time_ms Current response time in milliseconds")
                    prometheus_output.append("# TYPE codesage_mcp_response_time_ms gauge")
                    prometheus_output.append(f"codesage_mcp_response_time_ms {response_time}")

                # Add throughput metric
                throughput = current_metrics.get("throughput_rps", {}).get("value")
                if throughput is not None:
                    prometheus_output.append("# HELP codesage_mcp_requests_per_second Current requests per second")
                    prometheus_output.append("# TYPE codesage_mcp_requests_per_second gauge")
                    prometheus_output.append(f"codesage_mcp_requests_per_second {throughput}")

                # Add resource utilization metrics
                memory_percent = current_metrics.get("memory_usage_percent", {}).get("value")
                if memory_percent is not None:
                    prometheus_output.append("# HELP codesage_mcp_memory_usage_percent Current memory usage percentage")
                    prometheus_output.append("# TYPE codesage_mcp_memory_usage_percent gauge")
                    prometheus_output.append(f"codesage_mcp_memory_usage_percent {memory_percent}")

                cpu_percent = current_metrics.get("cpu_usage_percent", {}).get("value")
                if cpu_percent is not None:
                    prometheus_output.append("# HELP codesage_mcp_cpu_usage_percent Current CPU usage percentage")
                    prometheus_output.append("# TYPE codesage_mcp_cpu_usage_percent gauge")
                    prometheus_output.append(f"codesage_mcp_cpu_usage_percent {cpu_percent}")

                # Add error rate metric
                error_rate = current_metrics.get("error_rate_percent", {}).get("value")
                if error_rate is not None:
                    prometheus_output.append("# HELP codesage_mcp_error_rate_percent Current error rate percentage")
                    prometheus_output.append("# TYPE codesage_mcp_error_rate_percent gauge")
                    prometheus_output.append(f"codesage_mcp_error_rate_percent {error_rate}")

                # Add performance score
                performance_score = report.get("performance_score", 0)
                prometheus_output.append("# HELP codesage_mcp_performance_score Overall performance score (0-100)")
                prometheus_output.append("# TYPE codesage_mcp_performance_score gauge")
                prometheus_output.append(f"codesage_mcp_performance_score {performance_score}")

                # Add alert count
                alerts = report.get("recent_alerts", [])
                prometheus_output.append("# HELP codesage_mcp_active_alerts Number of active alerts")
                prometheus_output.append("# TYPE codesage_mcp_active_alerts gauge")
                prometheus_output.append(f"codesage_mcp_active_alerts {len(alerts)}")

                # Return response with correct content type
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse("\n".join(prometheus_output) + "\n", media_type="text/plain; version=0.0.4; charset=utf-8")

            except Exception as e:
                log_exception(e, get_logger(__name__))
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse(f"# Error generating metrics: {str(e)}\n", media_type="text/plain; version=0.0.4; charset=utf-8")

    def register_routes(self):
        """Register additional MCP-specific routes."""
        # Only register routes if app has the add_api_route method
        if hasattr(self.app, 'add_api_route'):
            # Register MCP-specific routes
            self.app.add_api_route("/mcp/initialize", self._handle_mcp_initialize, methods=["POST"])
            self.app.add_api_route("/mcp/tools", self._handle_mcp_tools, methods=["GET", "POST"])
        # This method can be extended to register more routes
        pass

    def add_api_route(self, path: str, endpoint, **kwargs):
        """Add an API route (for compatibility with FastAPI interface)."""
        # This is a compatibility method for testing
        pass

    def _handle_mcp_initialize(self):
        """Handle MCP initialize request."""
        return {"status": "initialized"}

    def _handle_mcp_tools(self):
        """Handle MCP tools request."""
        return {"tools": []}


class JSONRPCRequest(BaseModel):
    """JSON-RPC request model."""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Any] = None
    id: Optional[Any] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC response model."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Any] = None
    id: Optional[Any] = None

    def model_dump(self, *args, **kwargs):
        """Override model_dump method to comply with JSON-RPC 2.0 specification."""
        # Use exclude_none=True to prevent None fields from being serialized
        data = super().model_dump(*args, exclude_none=True, **kwargs)

        # Additional validation to prevent schema issues
        if 'error' in data and 'result' in data:
            logger = get_logger(__name__)
            logger.error("CRITICAL: Response contains both 'error' and 'result' fields after model_dump!")
            data.pop('result', None)

        return data

    def dict(self, *args, **kwargs):
        """Deprecated dict method for backward compatibility."""
        return self.model_dump(*args, **kwargs)

    @classmethod
    def create_compatible_response(cls,
                                   result: Optional[Any] = None,
                                   error: Optional[Dict[str, Any]] = None,
                                   request_id: Optional[Any] = None) -> 'JSONRPCResponse':
        """Create a response using Gemini compatibility handler."""
        compatibility_handler = get_compatibility_handler()

        # Ensure error codes are numeric (JSON-RPC 2.0 specification)
        if error is not None and isinstance(error.get('code'), str):
            # Convert string error code to numeric using compatibility handler
            error = compatibility_handler.adapt_error_response(error, ResponseFormat.GEMINI_NUMERIC_ERRORS)

        # Use compatibility handler to create adapted response
        adapted_response = compatibility_handler.create_compatible_response(
            result=result,
            error=error,
            request_id=request_id
        )

        # Create response object from adapted data, but filter out None values
        filtered_response = {k: v for k, v in adapted_response.items() if v is not None}
        return cls(**filtered_response)


class JSONRPCProcessor:
    """
    Handles JSON-RPC request/response processing for MCP server.

    This class provides methods for processing JSON-RPC requests, validating
    their format, and generating appropriate responses. It supports both
    single requests and batch processing.

    Attributes:
        compatibility_handler: Handler for Gemini CLI compatibility
    """

    def __init__(self):
        """Initialize the JSON-RPC processor."""
        self.compatibility_handler = get_compatibility_handler()

    def process(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a JSON-RPC request.

        Args:
            request_data: The JSON-RPC request data

        Returns:
            JSON-RPC response data

        Raises:
            ValidationError: If request format is invalid
        """
        try:
            # Validate request format
            jsonrpc_request = JSONRPCRequest(**request_data)

            # Process based on method
            if jsonrpc_request.method == "initialize":
                return self._handle_initialize(jsonrpc_request)
            elif jsonrpc_request.method == "tools/list":
                return self._handle_tools_list(jsonrpc_request)
            elif jsonrpc_request.method == "tools/call":
                return self._handle_tools_call(jsonrpc_request)
            else:
                return self._create_error_response(-32601, "Method not found", jsonrpc_request.id)

        except ValidationError as e:
            return self._create_error_response(-32600, f"Invalid Request: {e}", request_data.get('id'))
        except Exception as e:
            logger = get_logger(__name__)
            log_exception(e, logger)
            return self._create_error_response(-32000, f"Server error: {e}", request_data.get('id'))

    def process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of JSON-RPC requests.

        Args:
            requests: List of JSON-RPC request data

        Returns:
            List of JSON-RPC response data
        """
        responses = []
        for request in requests:
            response = self.process(request)
            responses.append(response)
        return responses

    def _handle_initialize(self, request: JSONRPCRequest) -> Dict[str, Any]:
        """Handle initialize request."""
        response_result = {
            "protocolVersion": "2025-06-18",
            "serverInfo": {"name": "CodeSage MCP Server", "version": "0.1.0"},
            "capabilities": {
                "tools": {
                    "listChanged": False
                },
                "prompts": {
                    "listChanged": False
                }
            },
        }
        return JSONRPCResponse.create_compatible_response(
            result=response_result,
            request_id=request.id
        ).model_dump()

    def _handle_tools_list(self, request: JSONRPCRequest) -> Dict[str, Any]:
        """Handle tools/list request."""
        # This would normally return the list of available tools
        # For now, return empty list
        response_result = {"tools": []}
        return JSONRPCResponse.create_compatible_response(
            result=response_result,
            request_id=request.id
        ).model_dump()

    def _handle_tools_call(self, request: JSONRPCRequest) -> Dict[str, Any]:
        """Handle tools/call request."""
        if not request.params or not isinstance(request.params, dict):
            return self._create_error_response(-32602, "Invalid params", request.id)

        tool_name = request.params.get("name")
        if not tool_name:
            return self._create_error_response(-32602, "Tool name required", request.id)

        # This would normally execute the tool
        # For now, return method not found
        return self._create_error_response(-32601, f"Tool '{tool_name}' not found", request.id)

    def _create_error_response(self, code: int, message: str, request_id: Optional[Any] = None) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        error = {"code": code, "message": message}
        return JSONRPCResponse.create_compatible_response(
            error=error,
            request_id=request_id
        ).model_dump()


class MCPProtocol:
    """
    Implements MCP protocol handshake and message handling.

    This class handles the MCP protocol initialization, capability negotiation,
    and message processing. It provides methods for handling different types
    of MCP messages and maintaining protocol state.

    Attributes:
        capabilities: Server capabilities
        initialized: Whether the protocol has been initialized
    """

    def __init__(self):
        """Initialize the MCP protocol handler."""
        self.capabilities = {
            "tools": {
                "listChanged": False
            },
            "prompts": {
                "listChanged": False
            }
        }
        self.initialized = False
        self.logger = get_logger(__name__)

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the server capabilities.

        Returns:
            Dictionary of server capabilities
        """
        return self.capabilities.copy()

    def handle_initialize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle initialize request.

        Args:
            request: Initialize request data

        Returns:
            Initialize response data
        """
        try:
            params = request.get("params", {})
            protocol_version = params.get("protocolVersion", "2024-11-05")

            response = {
                "protocolVersion": protocol_version,
                "capabilities": self.capabilities,
                "serverInfo": {
                    "name": "CodeSage",
                    "version": "1.0.0"
                }
            }

            self.initialized = True
            self.logger.info("MCP protocol initialized", protocol_version=protocol_version)

            return response

        except Exception as e:
            self.logger.exception("Failed to handle initialize request", error=str(e))
            raise

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an MCP request.

        Args:
            request: Request data

        Returns:
            Response data
        """
        try:
            method = request.get("method")
            request_id = request.get("id")

            if method == "initialize":
                return self.handle_initialize(request)
            elif method == "notifications/initialized":
                return {"result": None}
            else:
                # For other methods, delegate to JSON-RPC processor
                processor = JSONRPCProcessor()
                return processor.process(request)
        except Exception as e:
            # Handle exceptions and format as JSON-RPC error
            return self.format_error(-32000, f"Server error: {e}", request_id)

    def format_error(self, error_code: int, message: str, request_id: Optional[Any] = None) -> Dict[str, Any]:
        """
        Format an error response.

        Args:
            error_code: Error code
            message: Error message
            request_id: Request ID

        Returns:
            Formatted error response
        """
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": error_code,
                "message": message
            },
            "id": request_id
        }


# Initialize logging
setup_logging(
    level="INFO",
    log_file="logs/codesage.log",
    json_format=True
)
logger = get_logger(__name__)

# Initialize performance monitoring
performance_monitor = get_performance_monitor()
usage_analyzer = get_usage_analyzer()
feedback_collector = get_user_feedback_collector()
trend_analyzer = get_trend_analyzer()

# Initialize self-optimization features
auto_performance_tuner = get_auto_performance_tuner()
adaptive_cache_manager = get_adaptive_cache_manager()
memory_manager = get_memory_manager()
workload_pattern_recognition = get_workload_pattern_recognition()
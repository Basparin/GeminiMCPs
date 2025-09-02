"""
CES CodeSage Integration Module

Provides integration between CES and CodeSage MCP server.
Handles MCP protocol communication, tool orchestration, and
CES-specific extensions to CodeSage functionality.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import aiohttp
from dataclasses import dataclass
import time
from collections import deque


@dataclass
class MCPRequest:
    """MCP protocol request structure"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


@dataclass
class MCPResponse:
    """MCP protocol response structure"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


class CodeSageIntegration:
    """
    Integration layer between CES and CodeSage MCP server.

    Provides:
    - MCP protocol communication
    - Tool discovery and execution
    - CES-specific tool extensions
    - Connection management and error handling
    """

    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 30):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Connection state
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
        self.server_info: Optional[Dict[str, Any]] = None
        self.available_tools: Dict[str, Dict[str, Any]] = {}

        # Enhanced features
        self.retry_attempts = 3
        self.retry_delay = 1.0
        self.request_history = deque(maxlen=1000)
        self.connection_pool_size = 10
        self.active_requests = 0
        self.max_concurrent_requests = 5

        self.logger.info(f"CodeSage Integration initialized for {server_url}")

    async def connect(self) -> bool:
        """
        Establish connection to CodeSage MCP server

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.session:
                await self.session.close()

            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

            # Test connection with initialize request
            init_request = MCPRequest(
                method="initialize",
                params={},
                id="ces_init"
            )

            response = await self._send_request(init_request)

            if response and response.result:
                self.server_info = response.result
                self.connected = True

                # Get available tools
                await self._discover_tools()

                self.logger.info(f"Connected to CodeSage MCP server: {self.server_info}")
                return True
            else:
                self.logger.error("Failed to initialize connection to CodeSage MCP server")
                return False

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Close connection to CodeSage MCP server"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        self.logger.info("Disconnected from CodeSage MCP server")

    async def _send_request(self, request: MCPRequest) -> Optional[MCPResponse]:
        """Send MCP request to server"""
        if not self.session:
            self.logger.error("No active session")
            return None

        try:
            url = f"{self.server_url}/mcp"
            headers = {"Content-Type": "application/json"}

            request_data = {
                "jsonrpc": request.jsonrpc,
                "method": request.method,
                "id": request.id
            }

            if request.params is not None:
                request_data["params"] = request.params

            async with self.session.post(url, json=request_data, headers=headers) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    return MCPResponse(**response_data)
                else:
                    self.logger.error(f"HTTP error {resp.status}: {await resp.text()}")
                    return None

        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return None

    async def _send_request_with_retry(self, request: MCPRequest, max_retries: int = None) -> Optional[MCPResponse]:
        """Send MCP request with retry logic"""
        if max_retries is None:
            max_retries = self.retry_attempts

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._send_request(request)
                if response:
                    return response

                if attempt < max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    self.logger.error(f"All retry attempts failed: {e}")

        return None

    async def _discover_tools(self):
        """Discover available tools from CodeSage server"""
        try:
            tools_request = MCPRequest(
                method="tools/list",
                params={},
                id="ces_tools_discovery"
            )

            response = await self._send_request(tools_request)

            if response and response.result:
                # Handle different tool response formats
                tools_data = response.result

                if isinstance(tools_data, dict):
                    # Object format: {"tool_name": {...}}
                    self.available_tools = tools_data
                elif isinstance(tools_data, list):
                    # Array format: [{"name": "...", ...}]
                    self.available_tools = {tool["name"]: tool for tool in tools_data}
                else:
                    self.logger.warning(f"Unexpected tools format: {type(tools_data)}")

                self.logger.info(f"Discovered {len(self.available_tools)} tools")
            else:
                self.logger.error("Failed to discover tools")

        except Exception as e:
            self.logger.error(f"Tool discovery failed: {e}")

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool on the CodeSage server

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self.connected:
            return {
                "status": "error",
                "error": "Not connected to CodeSage server",
                "timestamp": datetime.now().isoformat()
            }

        if tool_name not in self.available_tools:
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' not available",
                "available_tools": list(self.available_tools.keys()),
                "timestamp": datetime.now().isoformat()
            }

        try:
            tool_request = MCPRequest(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments
                },
                id=f"ces_tool_{tool_name}_{int(datetime.now().timestamp())}"
            )

            response = await self._send_request(tool_request)

            if response and response.result is not None:
                return {
                    "status": "success",
                    "result": response.result,
                    "tool": tool_name,
                    "timestamp": datetime.now().isoformat()
                }
            elif response and response.error:
                return {
                    "status": "error",
                    "error": response.error,
                    "tool": tool_name,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "error": "No response from tool execution",
                    "tool": tool_name,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "tool": tool_name,
                "timestamp": datetime.now().isoformat()
            }

    async def execute_tools_batch(self, tool_requests: List[Dict[str, Any]],
                                progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Execute multiple tools in batch with progress tracking

        Args:
            tool_requests: List of {"tool_name": str, "arguments": dict} items
            progress_callback: Optional callback for progress updates

        Returns:
            List of tool execution results
        """
        if not self.connected:
            return [{
                "status": "error",
                "error": "Not connected to CodeSage server",
                "timestamp": datetime.now().isoformat()
            }] * len(tool_requests)

        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def execute_single_tool(tool_req: Dict[str, Any], index: int):
            async with semaphore:
                tool_name = tool_req["tool_name"]
                arguments = tool_req.get("arguments", {})

                result = await self.execute_tool(tool_name, arguments)
                results.append((index, result))

                if progress_callback:
                    progress_callback(index + 1, len(tool_requests), tool_name, result)

                return result

        # Execute tools concurrently with concurrency control
        tasks = [execute_single_tool(req, i) for i, req in enumerate(tool_requests)]
        await asyncio.gather(*tasks)

        # Sort results by original order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    async def execute_tool_with_progress(self, tool_name: str, arguments: Dict[str, Any],
                                       progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Execute a tool with progress tracking for long-running operations

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            progress_callback: Optional callback for progress updates

        Returns:
            Tool execution result
        """
        if progress_callback:
            progress_callback(0, 100, "Starting tool execution", {"status": "starting"})

        result = await self.execute_tool(tool_name, arguments)

        if progress_callback:
            progress_callback(100, 100, "Tool execution completed", result)

        return result

    async def handle_notification(self, notification: Dict[str, Any]) -> None:
        """
        Handle incoming MCP notifications

        Args:
            notification: MCP notification message
        """
        try:
            method = notification.get("method", "")
            params = notification.get("params", {})

            if method == "notifications/tools/list_changed":
                self.logger.info("Tool list changed, refreshing available tools")
                await self._discover_tools()
            elif method == "notifications/initialized":
                self.logger.info("CodeSage server initialized")
            elif method.startswith("custom/"):
                # Handle custom notifications
                self.logger.info(f"Received custom notification: {method}")
            else:
                self.logger.debug(f"Unhandled notification: {method}")

        except Exception as e:
            self.logger.error(f"Error handling notification: {e}")

    def get_request_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent request history"""
        return list(self.request_history)[-limit:]

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "connected": self.connected,
            "server_url": self.server_url,
            "active_requests": self.active_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
            "retry_attempts": self.retry_attempts,
            "available_tools_count": len(self.available_tools),
            "request_history_size": len(self.request_history)
        }

    async def get_server_status(self) -> Dict[str, Any]:
        """Get CodeSage server status"""
        if not self.connected:
            return {"status": "disconnected"}

        try:
            # Try to get basic server info
            status_request = MCPRequest(
                method="initialize",
                params={},
                id="ces_status_check"
            )

            response = await self._send_request(status_request)

            if response and response.result:
                return {
                    "status": "connected",
                    "server_info": response.result,
                    "tools_count": len(self.available_tools),
                    "last_check": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to get server status",
                    "last_check": datetime.now().isoformat()
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available tools"""
        return self.available_tools.copy()

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        return self.available_tools.get(tool_name)

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics from CodeSage server

        Returns:
            Dictionary containing performance metrics
        """
        if not self.connected:
            return {
                "status": "disconnected",
                "error": "Not connected to CodeSage server",
                "timestamp": datetime.now().isoformat()
            }

        try:
            # Get performance metrics using available tools
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "server_status": "connected",
                "connection_stats": self.get_connection_stats(),
                "tool_count": len(self.available_tools),
                "active_requests": self.active_requests,
                "request_history_size": len(self.request_history)
            }

            # Try to get specific performance metrics if tools are available
            if "get_performance_metrics" in self.available_tools:
                perf_result = await self.execute_tool("get_performance_metrics", {})
                if perf_result.get("status") == "success":
                    metrics["performance_data"] = perf_result.get("result", {})
                else:
                    metrics["performance_data"] = {"error": "Failed to retrieve performance data"}

            # Add system-level metrics
            import psutil
            try:
                metrics["system_metrics"] = {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage_percent": psutil.disk_usage('/').percent
                }
            except ImportError:
                metrics["system_metrics"] = {"error": "psutil not available"}

            return metrics

        except Exception as e:
            self.logger.error(f"Performance metrics retrieval failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def is_healthy(self) -> bool:
        """
        Check if CodeSage integration is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if connected and has tools available
            return self.connected and len(self.available_tools) > 0
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on CodeSage integration"""
        health_status = {
            "component": "CodeSage Integration",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # Connection check
        health_status["checks"]["connection"] = {
            "status": "healthy" if self.connected else "unhealthy",
            "details": f"Connected to {self.server_url}" if self.connected else "Not connected"
        }

        # Tools check
        tools_count = len(self.available_tools)
        health_status["checks"]["tools"] = {
            "status": "healthy" if tools_count > 0 else "warning",
            "details": f"{tools_count} tools available"
        }

        # Server response check
        if self.connected:
            server_status = await self.get_server_status()
            health_status["checks"]["server_response"] = {
                "status": "healthy" if server_status.get("status") == "connected" else "unhealthy",
                "details": server_status.get("status", "unknown")
            }
        else:
            health_status["checks"]["server_response"] = {
                "status": "skipped",
                "details": "Cannot check server response when disconnected"
            }

        # Overall health
        all_healthy = all(
            check["status"] in ["healthy", "skipped"]
            for check in health_status["checks"].values()
        )
        health_status["overall_status"] = "healthy" if all_healthy else "unhealthy"

        return health_status


class CESToolExtensions:
    """
    CES-specific tool extensions that build on CodeSage functionality.

    Provides higher-level tools that combine multiple CodeSage tools
    and add CES-specific intelligence.
    """

    def __init__(self, codesage_integration: CodeSageIntegration):
        self.codesage = codesage_integration
        self.logger = logging.getLogger(__name__)

    async def analyze_codebase_intelligence(self, codebase_path: str) -> Dict[str, Any]:
        """
        Intelligent codebase analysis combining multiple CodeSage tools

        Args:
            codebase_path: Path to the codebase to analyze

        Returns:
            Comprehensive analysis results
        """
        results = {
            "codebase_path": codebase_path,
            "timestamp": datetime.now().isoformat(),
            "analyses": {}
        }

        try:
            # Get file structure
            if "get_file_structure" in self.codesage.available_tools:
                structure_result = await self.codesage.execute_tool(
                    "get_file_structure",
                    {"codebase_path": codebase_path, "file_path": "."}
                )
                results["analyses"]["structure"] = structure_result

            # Count lines of code
            if "count_lines_of_code" in self.codesage.available_tools:
                loc_result = await self.codesage.execute_tool(
                    "count_lines_of_code",
                    {"codebase_path": codebase_path}
                )
                results["analyses"]["lines_of_code"] = loc_result

            # Analyze dependencies
            if "get_dependencies_overview" in self.codesage.available_tools:
                deps_result = await self.codesage.execute_tool(
                    "get_dependencies_overview",
                    {"codebase_path": codebase_path}
                )
                results["analyses"]["dependencies"] = deps_result

            # Get performance metrics
            if "get_performance_metrics" in self.codesage.available_tools:
                perf_result = await self.codesage.execute_tool(
                    "get_performance_metrics",
                    {}
                )
                results["analyses"]["performance"] = perf_result

            results["status"] = "success"

        except Exception as e:
            self.logger.error(f"Codebase intelligence analysis failed: {e}")
            results["status"] = "error"
            results["error"] = str(e)

        return results

    async def intelligent_task_execution(self, task_description: str,
                                       codebase_path: str) -> Dict[str, Any]:
        """
        Execute tasks with CES intelligence using CodeSage tools

        Args:
            task_description: Description of the task to execute
            codebase_path: Path to the codebase

        Returns:
            Task execution results with CES enhancements
        """
        execution_results = {
            "task": task_description,
            "codebase_path": codebase_path,
            "timestamp": datetime.now().isoformat(),
            "ces_enhancements": {}
        }

        try:
            # Analyze codebase first for context
            codebase_analysis = await self.analyze_codebase_intelligence(codebase_path)
            execution_results["ces_enhancements"]["codebase_context"] = codebase_analysis

            # Determine appropriate tools based on task
            task_lower = task_description.lower()

            if "analyze" in task_lower or "review" in task_lower:
                # Use analysis tools
                if "analyze_codebase_improvements" in self.codesage.available_tools:
                    analysis_result = await self.codesage.execute_tool(
                        "analyze_codebase_improvements",
                        {"codebase_path": codebase_path}
                    )
                    execution_results["ces_enhancements"]["code_analysis"] = analysis_result

            elif "test" in task_lower:
                # Use testing tools
                if "generate_unit_tests" in self.codesage.available_tools:
                    test_result = await self.codesage.execute_tool(
                        "generate_unit_tests",
                        {"file_path": codebase_path}  # This would need refinement
                    )
                    execution_results["ces_enhancements"]["test_generation"] = test_result

            elif "document" in task_lower:
                # Use documentation tools
                if "auto_document_tool" in self.codesage.available_tools:
                    doc_result = await self.codesage.execute_tool(
                        "auto_document_tool",
                        {"tool_name": "analyze_codebase"}  # Example
                    )
                    execution_results["ces_enhancements"]["documentation"] = doc_result

            execution_results["status"] = "success"

        except Exception as e:
            self.logger.error(f"Intelligent task execution failed: {e}")
            execution_results["status"] = "error"
            execution_results["error"] = str(e)

        return execution_results

    async def batch_analyze_codebase(self, codebase_path: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        Perform batch analysis of codebase using multiple tools concurrently

        Args:
            codebase_path: Path to the codebase
            analysis_types: List of analysis types to perform

        Returns:
            Batch analysis results
        """
        if analysis_types is None:
            analysis_types = ["structure", "lines_of_code", "dependencies", "performance"]

        # Prepare tool requests for batch execution
        tool_requests = []

        if "structure" in analysis_types and "get_file_structure" in self.codesage.available_tools:
            tool_requests.append({
                "tool_name": "get_file_structure",
                "arguments": {"codebase_path": codebase_path, "file_path": "."}
            })

        if "lines_of_code" in analysis_types and "count_lines_of_code" in self.codesage.available_tools:
            tool_requests.append({
                "tool_name": "count_lines_of_code",
                "arguments": {"codebase_path": codebase_path}
            })

        if "dependencies" in analysis_types and "get_dependencies_overview" in self.codesage.available_tools:
            tool_requests.append({
                "tool_name": "get_dependencies_overview",
                "arguments": {"codebase_path": codebase_path}
            })

        if "performance" in analysis_types and "get_performance_metrics" in self.codesage.available_tools:
            tool_requests.append({
                "tool_name": "get_performance_metrics",
                "arguments": {}
            })

        # Progress tracking
        progress_updates = []

        def progress_callback(completed: int, total: int, tool_name: str, result: Dict[str, Any]):
            progress_updates.append({
                "completed": completed,
                "total": total,
                "tool_name": tool_name,
                "status": result.get("status", "unknown")
            })
            self.logger.info(f"Batch analysis progress: {completed}/{total} - {tool_name}")

        # Execute batch analysis
        results = await self.codesage.execute_tools_batch(tool_requests, progress_callback)

        return {
            "codebase_path": codebase_path,
            "analysis_types": analysis_types,
            "batch_results": results,
            "progress_updates": progress_updates,
            "timestamp": datetime.now().isoformat(),
            "status": "success" if all(r.get("status") == "success" for r in results) else "partial_success"
        }

    async def intelligent_workflow_execution(self, workflow_steps: List[Dict[str, Any]],
                                          progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Execute a complex workflow with intelligent tool orchestration

        Args:
            workflow_steps: List of workflow steps with tool requirements
            progress_callback: Optional progress callback

        Returns:
            Workflow execution results
        """
        workflow_results = {
            "workflow_steps": workflow_steps,
            "step_results": [],
            "timestamp": datetime.now().isoformat(),
            "status": "running"
        }

        try:
            for i, step in enumerate(workflow_steps):
                step_name = step.get("name", f"Step {i+1}")
                tool_requests = step.get("tools", [])

                if progress_callback:
                    progress_callback(i, len(workflow_steps), step_name, {"status": "starting"})

                # Execute step tools in batch
                step_results = await self.codesage.execute_tools_batch(tool_requests)

                step_result = {
                    "step_name": step_name,
                    "tool_results": step_results,
                    "status": "success" if all(r.get("status") == "success" for r in step_results) else "error"
                }

                workflow_results["step_results"].append(step_result)

                if progress_callback:
                    progress_callback(i + 1, len(workflow_steps), step_name, step_result)

                # Check if workflow should continue
                if step_result["status"] != "success" and step.get("required", True):
                    workflow_results["status"] = "failed"
                    workflow_results["failed_at_step"] = i
                    break

            if workflow_results["status"] == "running":
                workflow_results["status"] = "completed"

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            workflow_results["status"] = "error"
            workflow_results["error"] = str(e)

        return workflow_results
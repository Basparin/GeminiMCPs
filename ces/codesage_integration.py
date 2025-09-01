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


@dataclass
class MCPRequest:
    """MCP protocol request structure"""
    jsonrpc: str = "2.0"
    method: str
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
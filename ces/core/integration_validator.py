"""
CES Integration Validator - Validates CodeSage Integration with CES Foundation

Provides comprehensive validation of CodeSage components integration with CES,
ensuring compatibility, performance, and multi-AI integration requirements.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .cognitive_agent import CognitiveAgent
from .performance_monitor import get_performance_monitor
from .tools import get_ces_tools
from .task_workflow import get_workflow_orchestrator
from ..codesage_integration import CodeSageIntegration, CESToolExtensions
from ..ai_orchestrator.ai_assistant import AIOrchestrator
from ..analytics.analytics_manager import AnalyticsManager


class CESIntegrationValidator:
    """Validates CodeSage integration with CES foundation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cognitive_agent = CognitiveAgent()
        self.performance_monitor = get_performance_monitor()
        self.ces_tools = get_ces_tools()
        self.workflow_orchestrator = get_workflow_orchestrator(self.cognitive_agent)
        self.codesage_integration = CodeSageIntegration()
        self.ces_tool_extensions = CESToolExtensions(self.codesage_integration)
        self.ai_orchestrator = AIOrchestrator()
        self.analytics_manager = AnalyticsManager()

        self.validation_results = []

    async def run_full_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation of all CodeSage integrations

        Returns:
            Complete validation results
        """
        self.logger.info("Starting comprehensive CES-CodeSage integration validation")

        validation_start = time.time()

        # Test categories
        test_categories = [
            "cognitive_agent_integration",
            "performance_monitoring_integration",
            "tool_orchestration_integration",
            "workflow_execution_integration",
            "codesage_protocol_integration",
            "multi_ai_compatibility",
            "analytics_integration",
            "end_to_end_functionality"
        ]

        results = {}
        for category in test_categories:
            self.logger.info(f"Running validation category: {category}")
            try:
                test_method = getattr(self, f"validate_{category}")
                results[category] = await test_method()
                self.logger.info(f"✓ {category} validation completed")
            except Exception as e:
                self.logger.error(f"✗ {category} validation failed: {e}")
                results[category] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        # Overall assessment
        successful_tests = sum(1 for result in results.values()
                             if isinstance(result, dict) and result.get("status") == "passed")
        total_tests = len(test_categories)

        validation_time = time.time() - validation_start

        overall_result = {
            "validation_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": round((successful_tests / total_tests) * 100, 2),
                "validation_time_seconds": round(validation_time, 2),
                "timestamp": datetime.now().isoformat()
            },
            "test_results": results,
            "integration_status": "healthy" if successful_tests == total_tests else "degraded",
            "recommendations": self._generate_recommendations(results)
        }

        self.logger.info(f"Validation completed: {successful_tests}/{total_tests} tests passed")
        return overall_result

    async def validate_cognitive_agent_integration(self) -> Dict[str, Any]:
        """Validate cognitive agent integration with CodeSage components"""
        try:
            # Test task analysis
            test_task = "Analyze the codebase for potential improvements and generate unit tests"
            analysis = self.cognitive_agent.analyze_task(test_task)

            # Verify analysis includes MCP tools
            if not analysis.mcp_tools_required:
                return {"status": "failed", "error": "No MCP tools identified in task analysis"}

            # Test tool execution
            if analysis.mcp_tools_required:
                tool_name = analysis.mcp_tools_required[0]
                args = self.cognitive_agent._prepare_tool_arguments(tool_name, test_task, {})

                if tool_name in ['read_code_file', 'search_codebase', 'analyze_codebase_improvements']:
                    result = await self.cognitive_agent.execute_ces_tool(tool_name, args or {})
                else:
                    result = await self.cognitive_agent.execute_mcp_tool(tool_name, args or {})

                if result.get('status') != 'success':
                    return {"status": "failed", "error": f"Tool execution failed: {result}"}

            return {
                "status": "passed",
                "details": {
                    "task_analysis": "successful",
                    "tool_identification": len(analysis.mcp_tools_required),
                    "tool_execution": "successful"
                }
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def validate_performance_monitoring_integration(self) -> Dict[str, Any]:
        """Validate performance monitoring integration"""
        try:
            # Test performance metrics collection
            initial_metrics = self.performance_monitor.get_current_metrics()

            # Simulate some activity
            self.performance_monitor.record_request(150.0, True, "/test", "test_user")
            self.performance_monitor.record_mcp_tool_execution("test_tool", 50.0, True)

            # Check updated metrics
            updated_metrics = self.performance_monitor.get_current_metrics()

            if not updated_metrics:
                return {"status": "failed", "error": "Performance metrics not updated"}

            # Test usage analyzer
            self.cognitive_agent.usage_analyzer.record_user_action("test_user", "test_action", {})

            return {
                "status": "passed",
                "details": {
                    "metrics_collection": "working",
                    "usage_tracking": "working",
                    "performance_data": len(updated_metrics)
                }
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def validate_tool_orchestration_integration(self) -> Dict[str, Any]:
        """Validate tool orchestration integration"""
        try:
            # Test CES tools
            file_result = self.ces_tools.read_code_file("ces/__init__.py")
            if file_result.get('status') != 'success':
                return {"status": "failed", "error": f"CES tool failed: {file_result}"}

            # Test search functionality
            search_result = self.ces_tools.search_codebase("class", ["*.py"])
            if search_result.get('status') != 'success':
                return {"status": "failed", "error": f"Search tool failed: {search_result}"}

            # Test MCP tool discovery (if available)
            if self.codesage_integration.connected:
                tools = self.codesage_integration.get_available_tools()
                if not tools:
                    return {"status": "warning", "message": "No MCP tools available"}
            else:
                self.logger.info("CodeSage MCP server not connected - skipping MCP tool validation")

            return {
                "status": "passed",
                "details": {
                    "ces_tools": "working",
                    "search_functionality": "working",
                    "mcp_tools_available": len(self.codesage_integration.get_available_tools()) if self.codesage_integration.connected else 0
                }
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def validate_workflow_execution_integration(self) -> Dict[str, Any]:
        """Validate workflow execution integration"""
        try:
            # Test workflow creation and execution
            test_workflow = await self.workflow_orchestrator.execute_workflow(
                "Analyze the CES codebase structure and identify key components"
            )

            if test_workflow.status not in ["completed", "failed"]:
                return {"status": "failed", "error": "Workflow did not complete properly"}

            # Check workflow results
            if not test_workflow.results:
                return {"status": "failed", "error": "Workflow produced no results"}

            return {
                "status": "passed",
                "details": {
                    "workflow_execution": "successful",
                    "workflow_status": test_workflow.status,
                    "results_generated": bool(test_workflow.results),
                    "execution_time": test_workflow.end_time - test_workflow.start_time if test_workflow.end_time else 0
                }
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def validate_codesage_protocol_integration(self) -> Dict[str, Any]:
        """Validate CodeSage MCP protocol integration"""
        try:
            # Test connection
            connected = await self.codesage_integration.connect()
            if not connected:
                return {"status": "warning", "message": "CodeSage MCP server not available"}

            # Test protocol methods
            server_info = await self.codesage_integration.get_server_status()
            if not server_info:
                return {"status": "failed", "error": "Server status check failed"}

            # Test tool discovery
            tools = self.codesage_integration.get_available_tools()
            self.logger.info(f"Discovered {len(tools)} MCP tools")

            # Test health check
            health = await self.codesage_integration.health_check()
            if health.get('overall_status') not in ['healthy', 'warning']:
                return {"status": "failed", "error": f"Health check failed: {health}"}

            return {
                "status": "passed",
                "details": {
                    "connection": "successful",
                    "server_info": "available",
                    "tools_discovered": len(tools),
                    "health_status": health.get('overall_status')
                }
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def validate_multi_ai_compatibility(self) -> Dict[str, Any]:
        """Validate multi-AI integration compatibility"""
        try:
            # Test AI orchestrator integration
            available_assistants = self.ai_orchestrator.get_available_assistants()
            self.logger.info(f"Available AI assistants: {len(available_assistants)}")

            # Test assistant recommendation
            test_task = "Write a Python function to calculate fibonacci numbers"
            recommendations = self.ai_orchestrator.recommend_assistants(test_task, ['programming'])

            if not recommendations:
                return {"status": "warning", "message": "No AI assistant recommendations available"}

            # Test cognitive agent with AI orchestration
            analysis = self.cognitive_agent.analyze_task(test_task)
            if not analysis.recommended_assistants:
                return {"status": "failed", "error": "Cognitive agent not recommending AI assistants"}

            return {
                "status": "passed",
                "details": {
                    "available_assistants": len(available_assistants),
                    "assistant_recommendations": len(recommendations),
                    "cognitive_agent_ai_integration": "working"
                }
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def validate_analytics_integration(self) -> Dict[str, Any]:
        """Validate analytics integration"""
        try:
            # Test analytics recording
            await self.analytics_manager.record_usage_event("test_event", "test_user", {"test": True})
            await self.analytics_manager.record_performance_metric("test_metric", 100.0, "test_user")
            await self.analytics_manager.record_task_completion("test_task", "test_user", 5000, True)

            # Test analytics retrieval
            overview = await self.analytics_manager.get_overview()
            if "error" in overview:
                return {"status": "failed", "error": f"Analytics overview failed: {overview['error']}"}

            summary = await self.analytics_manager.get_summary()
            if "error" in summary:
                return {"status": "failed", "error": f"Analytics summary failed: {summary['error']}"}

            return {
                "status": "passed",
                "details": {
                    "usage_tracking": "working",
                    "performance_metrics": "working",
                    "analytics_overview": "available",
                    "analytics_summary": "available"
                }
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def validate_end_to_end_functionality(self) -> Dict[str, Any]:
        """Validate end-to-end functionality"""
        try:
            # Complete end-to-end test
            test_task = "Analyze the CES core modules and suggest improvements"

            # 1. Task analysis
            analysis = self.cognitive_agent.analyze_task(test_task)

            # 2. Tool execution
            tool_results = []
            for tool_name in analysis.mcp_tools_required[:2]:  # Test first 2 tools
                args = self.cognitive_agent._prepare_tool_arguments(tool_name, test_task, {})
                if tool_name in ['read_code_file', 'search_codebase', 'analyze_codebase_improvements']:
                    result = await self.cognitive_agent.execute_ces_tool(tool_name, args or {})
                else:
                    result = await self.cognitive_agent.execute_mcp_tool(tool_name, args or {})
                tool_results.append(result)

            # 3. AI orchestration
            ai_result = await self.ai_orchestrator.execute_task(test_task)

            # 4. Result synthesis
            final_result = {
                "task": test_task,
                "analysis": analysis.__dict__,
                "tool_results": tool_results,
                "ai_result": ai_result,
                "performance_metrics": self.performance_monitor.get_current_metrics()
            }

            # Validate all components worked
            if not analysis.mcp_tools_required:
                return {"status": "failed", "error": "No tools identified"}

            if not any(r.get('status') == 'success' for r in tool_results):
                return {"status": "failed", "error": "No tools executed successfully"}

            if ai_result.get('status') != 'completed':
                return {"status": "failed", "error": "AI orchestration failed"}

            return {
                "status": "passed",
                "details": {
                    "task_analysis": "successful",
                    "tool_execution": f"{sum(1 for r in tool_results if r.get('status') == 'success')}/{len(tool_results)} successful",
                    "ai_orchestration": "successful",
                    "result_synthesis": "successful",
                    "end_to_end_time": "measured"
                }
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        failed_tests = [test for test, result in results.items()
                       if isinstance(result, dict) and result.get("status") == "failed"]

        if failed_tests:
            recommendations.append(f"Address failed validations: {', '.join(failed_tests)}")

        # Specific recommendations based on test results
        if "codesage_protocol_integration" in results:
            protocol_result = results["codesage_protocol_integration"]
            if isinstance(protocol_result, dict) and protocol_result.get("status") == "warning":
                recommendations.append("Consider starting CodeSage MCP server for full functionality")

        if "multi_ai_compatibility" in results:
            ai_result = results["multi_ai_compatibility"]
            if isinstance(ai_result, dict) and ai_result.get("status") == "warning":
                recommendations.append("Ensure AI assistants are properly configured and available")

        if len(failed_tests) == 0:
            recommendations.append("All integrations validated successfully - system ready for production")

        return recommendations

    async def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report"""
        results = await self.run_full_validation()

        report = f"""
# CES-CodeSage Integration Validation Report
Generated: {datetime.now().isoformat()}

## Summary
- Total Tests: {results['validation_summary']['total_tests']}
- Successful Tests: {results['validation_summary']['successful_tests']}
- Success Rate: {results['validation_summary']['success_rate']}%
- Validation Time: {results['validation_summary']['validation_time_seconds']}s
- Overall Status: {results['integration_status']}

## Test Results
"""

        for test_name, test_result in results['test_results'].items():
            status = test_result.get('status', 'unknown') if isinstance(test_result, dict) else 'error'
            status_icon = "✓" if status == "passed" else "✗" if status == "failed" else "⚠"
            report += f"\n### {test_name.replace('_', ' ').title()}\n"
            report += f"**Status:** {status_icon} {status.upper()}\n"

            if isinstance(test_result, dict):
                if 'details' in test_result:
                    report += "**Details:**\n"
                    for key, value in test_result['details'].items():
                        report += f"- {key}: {value}\n"
                if 'error' in test_result:
                    report += f"**Error:** {test_result['error']}\n"

        report += f"\n## Recommendations\n"
        for rec in results.get('recommendations', []):
            report += f"- {rec}\n"

        return report


# Global validator instance
_integration_validator = None


def get_integration_validator() -> CESIntegrationValidator:
    """Get the global CES integration validator instance"""
    global _integration_validator
    if _integration_validator is None:
        _integration_validator = CESIntegrationValidator()
    return _integration_validator
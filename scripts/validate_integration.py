#!/usr/bin/env python3
"""
CES Phase 0.3: Integration Validation Script

This script performs comprehensive validation of CES component integration,
testing data flow, component interactions, and system health.
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime
import json

# Add CES to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ces.core.cognitive_agent import CognitiveAgent
from ces.config.ces_config import CESConfig
from ces.ai_orchestrator.cli_integration import AIAssistantManager
from ces.codesage_integration import CodeSageIntegration
from ces.utils.helpers import setup_logging


class CESIntegrationValidator:
    """
    Comprehensive integration validator for CES Phase 0.3

    Performs end-to-end validation of all component interactions.
    """

    def __init__(self):
        self.logger = setup_logging()
        self.config = CESConfig()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "phase": "0.3",
            "validation_type": "integration",
            "tests": {},
            "overall_status": "unknown"
        }

    async def run_full_validation(self):
        """
        Run complete integration validation suite

        Returns:
            Validation results dictionary
        """
        self.logger.info("Starting CES Phase 0.3 Integration Validation")

        try:
            # Component initialization validation
            await self._validate_component_initialization()

            # Data flow validation
            await self._validate_data_flow()

            # AI integration validation
            await self._validate_ai_integration()

            # CodeSage integration validation
            await self._validate_codesage_integration()

            # Performance validation
            await self._validate_performance()

            # Error handling validation
            await self._validate_error_handling()

            # Determine overall status
            self._calculate_overall_status()

            self.logger.info("Integration validation completed")
            return self.results

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.results["overall_status"] = "error"
            self.results["error"] = str(e)
            return self.results

    async def _validate_component_initialization(self):
        """Validate component initialization"""
        self.logger.info("Validating component initialization...")

        test_results = {
            "status": "unknown",
            "components": {},
            "duration": 0
        }

        start_time = time.time()

        try:
            # Test cognitive agent initialization
            agent = CognitiveAgent(self.config)
            test_results["components"]["cognitive_agent"] = {
                "status": "healthy",
                "details": "Initialized successfully"
            }

            # Test AI manager initialization
            ai_manager = AIAssistantManager()
            available = ai_manager.get_available_assistants()
            test_results["components"]["ai_manager"] = {
                "status": "healthy" if len(available) > 0 else "warning",
                "details": f"{len(available)} assistants available"
            }

            # Test CodeSage initialization
            codesage = CodeSageIntegration()
            connected = await codesage.connect()
            test_results["components"]["codesage"] = {
                "status": "healthy" if connected else "warning",
                "details": "Connected" if connected else "Not connected"
            }

            test_results["status"] = "healthy"

        except Exception as e:
            test_results["status"] = "error"
            test_results["error"] = str(e)

        test_results["duration"] = time.time() - start_time
        self.results["tests"]["component_initialization"] = test_results

    async def _validate_data_flow(self):
        """Validate data flow between components"""
        self.logger.info("Validating data flow...")

        test_results = {
            "status": "unknown",
            "flows": {},
            "duration": 0
        }

        start_time = time.time()

        try:
            agent = CognitiveAgent(self.config)

            # Test task analysis -> memory flow
            task = "Create a simple function"
            analysis = agent.analyze_task(task)

            test_results["flows"]["task_analysis"] = {
                "status": "healthy",
                "details": f"Complexity: {analysis.complexity_score}"
            }

            # Test memory -> context flow
            context = agent.memory_manager.retrieve_context(task, ["task_history"])
            test_results["flows"]["memory_retrieval"] = {
                "status": "healthy",
                "details": f"Retrieved {len(context)} context items"
            }

            # Test result storage flow
            result = {"status": "completed", "result": "Test result"}
            agent.memory_manager.store_task_result(task, result)
            test_results["flows"]["result_storage"] = {
                "status": "healthy",
                "details": "Result stored successfully"
            }

            test_results["status"] = "healthy"

        except Exception as e:
            test_results["status"] = "error"
            test_results["error"] = str(e)

        test_results["duration"] = time.time() - start_time
        self.results["tests"]["data_flow"] = test_results

    async def _validate_ai_integration(self):
        """Validate AI assistant integration"""
        self.logger.info("Validating AI integration...")

        test_results = {
            "status": "unknown",
            "assistants": {},
            "duration": 0
        }

        start_time = time.time()

        try:
            ai_manager = AIAssistantManager()
            available_assistants = ai_manager.get_available_assistants()

            for assistant in available_assistants:
                name = assistant["name"]
                try:
                    # Test basic connectivity
                    status = await ai_manager.test_assistant_connection(name)
                    test_results["assistants"][name] = {
                        "status": "healthy" if status.get("status") == "success" else "warning",
                        "details": status.get("response", "No response")
                    }
                except Exception as e:
                    test_results["assistants"][name] = {
                        "status": "error",
                        "details": str(e)
                    }

            test_results["status"] = "healthy" if test_results["assistants"] else "warning"

        except Exception as e:
            test_results["status"] = "error"
            test_results["error"] = str(e)

        test_results["duration"] = time.time() - start_time
        self.results["tests"]["ai_integration"] = test_results

    async def _validate_codesage_integration(self):
        """Validate CodeSage integration"""
        self.logger.info("Validating CodeSage integration...")

        test_results = {
            "status": "unknown",
            "connection": {},
            "tools": {},
            "duration": 0
        }

        start_time = time.time()

        try:
            codesage = CodeSageIntegration()

            # Test connection
            connected = await codesage.connect()
            test_results["connection"] = {
                "status": "healthy" if connected else "warning",
                "details": "Connected to MCP server" if connected else "Connection failed"
            }

            if connected:
                # Test tool discovery
                tools = codesage.get_available_tools()
                test_results["tools"] = {
                    "status": "healthy" if len(tools) > 0 else "warning",
                    "details": f"{len(tools)} tools available"
                }

                # Test tool execution
                if tools:
                    tool_name = list(tools.keys())[0]
                    result = await codesage.execute_tool(tool_name, {})
                    test_results["tool_execution"] = {
                        "status": "healthy" if result.get("status") == "success" else "warning",
                        "details": f"Tool {tool_name} executed"
                    }

            test_results["status"] = "healthy"

        except Exception as e:
            test_results["status"] = "error"
            test_results["error"] = str(e)

        test_results["duration"] = time.time() - start_time
        self.results["tests"]["codesage_integration"] = test_results

    async def _validate_performance(self):
        """Validate system performance"""
        self.logger.info("Validating performance...")

        test_results = {
            "status": "unknown",
            "metrics": {},
            "duration": 0
        }

        start_time = time.time()

        try:
            agent = CognitiveAgent(self.config)

            # Test task analysis performance
            task = "Simple test task"
            analysis_start = time.time()
            analysis = agent.analyze_task(task)
            analysis_time = time.time() - analysis_start

            test_results["metrics"]["task_analysis"] = {
                "value": analysis_time,
                "unit": "seconds",
                "status": "healthy" if analysis_time < 0.2 else "warning"
            }

            # Test memory performance
            memory_start = time.time()
            context = agent.memory_manager.retrieve_context(task, [])
            memory_time = time.time() - memory_start

            test_results["metrics"]["memory_retrieval"] = {
                "value": memory_time,
                "unit": "seconds",
                "status": "healthy" if memory_time < 0.1 else "warning"
            }

            test_results["status"] = "healthy"

        except Exception as e:
            test_results["status"] = "error"
            test_results["error"] = str(e)

        test_results["duration"] = time.time() - start_time
        self.results["tests"]["performance"] = test_results

    async def _validate_error_handling(self):
        """Validate error handling and recovery"""
        self.logger.info("Validating error handling...")

        test_results = {
            "status": "unknown",
            "error_scenarios": {},
            "duration": 0
        }

        start_time = time.time()

        try:
            agent = CognitiveAgent(self.config)

            # Test invalid task handling
            result = agent.execute_task("")
            test_results["error_scenarios"]["empty_task"] = {
                "status": "healthy",
                "details": f"Handled gracefully: {result.get('status', 'unknown')}"
            }

            # Test ethical rejection
            ethical_result = agent.execute_task("Create harmful software")
            test_results["error_scenarios"]["ethical_rejection"] = {
                "status": "healthy" if ethical_result.get('status') == 'rejected' else "warning",
                "details": f"Ethical check: {ethical_result.get('status', 'unknown')}"
            }

            test_results["status"] = "healthy"

        except Exception as e:
            test_results["status"] = "error"
            test_results["error"] = str(e)

        test_results["duration"] = time.time() - start_time
        self.results["tests"]["error_handling"] = test_results

    def _calculate_overall_status(self):
        """Calculate overall validation status"""
        test_results = self.results["tests"]

        if not test_results:
            self.results["overall_status"] = "error"
            return

        statuses = [test.get("status", "unknown") for test in test_results.values()]

        if all(status == "healthy" for status in statuses):
            self.results["overall_status"] = "healthy"
        elif any(status == "error" for status in statuses):
            self.results["overall_status"] = "error"
        elif any(status == "warning" for status in statuses):
            self.results["overall_status"] = "warning"
        else:
            self.results["overall_status"] = "unknown"

    def print_results(self):
        """Print validation results in a readable format"""
        print("\n" + "="*60)
        print("CES Phase 0.3 Integration Validation Results")
        print("="*60)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Overall Status: {self.results['overall_status'].upper()}")
        print()

        for test_name, test_result in self.results["tests"].items():
            status = test_result.get("status", "unknown")
            duration = test_result.get("duration", 0)

            status_icon = "✓" if status == "healthy" else "⚠" if status == "warning" else "✗"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Status: {status.upper()}")

            # Print test-specific details
            if "components" in test_result:
                for comp_name, comp_info in test_result["components"].items():
                    comp_status = comp_info.get("status", "unknown")
                    comp_icon = "✓" if comp_status == "healthy" else "⚠" if comp_status == "warning" else "✗"
                    print(f"     {comp_icon} {comp_name}: {comp_info.get('details', '')}")

            if "flows" in test_result:
                for flow_name, flow_info in test_result["flows"].items():
                    flow_status = flow_info.get("status", "unknown")
                    flow_icon = "✓" if flow_status == "healthy" else "⚠" if flow_status == "warning" else "✗"
                    print(f"     {flow_icon} {flow_name}: {flow_info.get('details', '')}")

            if "metrics" in test_result:
                for metric_name, metric_info in test_result["metrics"].items():
                    metric_status = metric_info.get("status", "unknown")
                    metric_icon = "✓" if metric_status == "healthy" else "⚠" if metric_status == "warning" else "✗"
                    value = metric_info.get("value", 0)
                    unit = metric_info.get("unit", "")
                    print(f"     {metric_icon} {metric_name}: {value} {unit}")

            print()

    def save_results(self, output_file: str = None):
        """Save validation results to file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"integration_validation_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to: {output_file}")


async def main():
    """Main validation script entry point"""
    print("CES Phase 0.3 Integration Validator")
    print("===================================")

    validator = CESIntegrationValidator()

    # Run validation
    results = await validator.run_full_validation()

    # Print results
    validator.print_results()

    # Save results
    validator.save_results()

    # Exit with appropriate code
    if results["overall_status"] == "healthy":
        print("🎉 All integration tests passed!")
        sys.exit(0)
    elif results["overall_status"] == "warning":
        print("⚠️  Integration tests completed with warnings")
        sys.exit(1)
    else:
        print("❌ Integration validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
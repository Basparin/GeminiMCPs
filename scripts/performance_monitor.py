#!/usr/bin/env python3
"""
CES Phase 0.3: Performance Monitoring Script

This script establishes and monitors performance baselines for CES components,
providing real-time performance metrics and trend analysis.
"""

import asyncio
import sys
import time
import psutil
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, List

# Add CES to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ces.core.cognitive_agent import CognitiveAgent
from ces.config.ces_config import CESConfig
from ces.ai_orchestrator.cli_integration import AIAssistantManager
from ces.utils.helpers import setup_logging


class CESPerformanceMonitor:
    """
    Performance monitoring and baseline establishment for CES Phase 0.3

    Tracks key performance metrics and establishes baseline expectations.
    """

    def __init__(self):
        self.logger = setup_logging()
        self.config = CESConfig()
        self.baselines = self._load_baselines()
        self.current_metrics = {}

    def _load_baselines(self) -> Dict[str, Any]:
        """Load performance baselines"""
        return {
            "task_analysis": {"p50": 0.1, "p95": 0.2, "p99": 0.5, "unit": "seconds"},
            "ai_response": {"p50": 0.3, "p95": 0.5, "p99": 1.0, "unit": "seconds"},
            "memory_retrieval": {"p50": 0.05, "p95": 0.1, "p99": 0.2, "unit": "seconds"},
            "memory_usage": {"max": 256, "unit": "MB"},
            "cpu_usage": {"max": 50, "unit": "percent"},
            "error_rate": {"max": 0.01, "unit": "ratio"}
        }

    async def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive performance tests

        Returns:
            Performance test results
        """
        self.logger.info("Running CES performance tests...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "phase": "0.3",
            "test_type": "performance_baseline",
            "metrics": {},
            "compliance": {},
            "recommendations": []
        }

        # Test individual components
        results["metrics"]["task_analysis"] = await self._test_task_analysis_performance()
        results["metrics"]["memory_system"] = await self._test_memory_performance()
        results["metrics"]["ai_integration"] = await self._test_ai_performance()
        results["metrics"]["system_resources"] = self._test_system_resources()

        # Check compliance with baselines
        results["compliance"] = self._check_baseline_compliance(results["metrics"])

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results["compliance"])

        return results

    async def _test_task_analysis_performance(self) -> Dict[str, Any]:
        """Test task analysis performance"""
        self.logger.info("Testing task analysis performance...")

        agent = CognitiveAgent(self.config)
        test_tasks = [
            "Create a simple function",
            "Implement complex authentication with OAuth, JWT, and database integration",
            "Write unit tests for the user management system",
            "Design the architecture for a web application with microservices",
            "Optimize database queries for better performance"
        ]

        analysis_times = []

        for task in test_tasks:
            start_time = time.time()
            analysis = agent.analyze_task(task)
            analysis_time = time.time() - start_time
            analysis_times.append(analysis_time)

        # Calculate percentiles
        analysis_times.sort()
        n = len(analysis_times)

        return {
            "samples": n,
            "p50": analysis_times[int(n * 0.5)],
            "p95": analysis_times[int(n * 0.95)],
            "p99": analysis_times[int(n * 0.99)] if n > 100 else analysis_times[-1],
            "min": min(analysis_times),
            "max": max(analysis_times),
            "avg": sum(analysis_times) / n,
            "unit": "seconds"
        }

    async def _test_memory_performance(self) -> Dict[str, Any]:
        """Test memory system performance"""
        self.logger.info("Testing memory system performance...")

        agent = CognitiveAgent(self.config)
        memory = agent.memory_manager

        # Test retrieval performance
        retrieval_times = []
        for i in range(10):
            task = f"Test task {i}"
            memory.store_task_result(task, {"status": "completed", "result": f"Result {i}"})

            start_time = time.time()
            context = memory.retrieve_context(task, ["task_history"])
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)

        # Calculate percentiles
        retrieval_times.sort()
        n = len(retrieval_times)

        return {
            "retrieval": {
                "samples": n,
                "p50": retrieval_times[int(n * 0.5)],
                "p95": retrieval_times[int(n * 0.95)],
                "avg": sum(retrieval_times) / n,
                "unit": "seconds"
            },
            "storage_count": 10
        }

    async def _test_ai_performance(self) -> Dict[str, Any]:
        """Test AI assistant performance"""
        self.logger.info("Testing AI assistant performance...")

        ai_manager = AIAssistantManager()
        available_assistants = ai_manager.get_available_assistants()

        results = {}

        for assistant in available_assistants:
            name = assistant["name"]
            try:
                # Test response time
                start_time = time.time()
                test_result = await ai_manager.test_assistant_connection(name)
                response_time = time.time() - start_time

                results[name] = {
                    "available": True,
                    "response_time": response_time,
                    "status": test_result.get("status", "unknown")
                }
            except Exception as e:
                results[name] = {
                    "available": False,
                    "error": str(e)
                }

        return results

    def _test_system_resources(self) -> Dict[str, Any]:
        """Test system resource usage"""
        self.logger.info("Testing system resource usage...")

        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=1.0)

        return {
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "cpu_usage_percent": cpu_percent,
            "timestamp": datetime.now().isoformat()
        }

    def _check_baseline_compliance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with performance baselines"""
        compliance = {}

        # Task analysis compliance
        if "task_analysis" in metrics:
            ta_metrics = metrics["task_analysis"]
            ta_baseline = self.baselines["task_analysis"]

            compliance["task_analysis"] = {
                "p95_compliant": ta_metrics["p95"] <= ta_baseline["p95"],
                "p99_compliant": ta_metrics["p99"] <= ta_baseline["p99"],
                "actual_p95": ta_metrics["p95"],
                "target_p95": ta_baseline["p95"]
            }

        # Memory compliance
        if "memory_system" in metrics:
            mem_metrics = metrics["memory_system"]
            mem_baseline = self.baselines["memory_retrieval"]

            if "retrieval" in mem_metrics:
                retrieval = mem_metrics["retrieval"]
                compliance["memory_retrieval"] = {
                    "p95_compliant": retrieval["p95"] <= mem_baseline["p95"],
                    "actual_p95": retrieval["p95"],
                    "target_p95": mem_baseline["p95"]
                }

        # System resource compliance
        if "system_resources" in metrics:
            sys_metrics = metrics["system_resources"]
            mem_baseline = self.baselines["memory_usage"]
            cpu_baseline = self.baselines["cpu_usage"]

            compliance["memory_usage"] = {
                "compliant": sys_metrics["memory_usage_mb"] <= mem_baseline["max"],
                "actual": sys_metrics["memory_usage_mb"],
                "target": mem_baseline["max"]
            }

            compliance["cpu_usage"] = {
                "compliant": sys_metrics["cpu_usage_percent"] <= cpu_baseline["max"],
                "actual": sys_metrics["cpu_usage_percent"],
                "target": cpu_baseline["max"]
            }

        return compliance

    def _generate_recommendations(self, compliance: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        for metric_name, metric_compliance in compliance.items():
            if not metric_compliance.get("compliant", True):
                if "task_analysis" in metric_name:
                    recommendations.append(
                        f"Task analysis performance ({metric_compliance['actual_p95']:.3f}s) exceeds "
                        f"target ({metric_compliance['target_p95']:.3f}s). Consider optimizing NLP processing."
                    )
                elif "memory_retrieval" in metric_name:
                    recommendations.append(
                        f"Memory retrieval performance ({metric_compliance['actual_p95']:.3f}s) exceeds "
                        f"target ({metric_compliance['target_p95']:.3f}s). Consider database indexing improvements."
                    )
                elif "memory_usage" in metric_name:
                    recommendations.append(
                        f"Memory usage ({metric_compliance['actual']:.1f}MB) exceeds "
                        f"target ({metric_compliance['target']}MB). Consider memory optimization."
                    )
                elif "cpu_usage" in metric_name:
                    recommendations.append(
                        f"CPU usage ({metric_compliance['actual']:.1f}%) exceeds "
                        f"target ({metric_compliance['target']}%). Consider performance profiling."
                    )

        if not recommendations:
            recommendations.append("All performance metrics are within acceptable ranges.")

        return recommendations

    def print_report(self, results: Dict[str, Any]):
        """Print performance report"""
        print("\n" + "="*60)
        print("CES Phase 0.3 Performance Baseline Report")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print()

        # Print metrics
        print("Performance Metrics:")
        print("-" * 20)

        for metric_name, metric_data in results["metrics"].items():
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            if isinstance(metric_data, dict):
                for key, value in metric_data.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {metric_data}")

        # Print compliance
        print("\n\nBaseline Compliance:")
        print("-" * 20)

        for metric_name, compliance in results["compliance"].items():
            status = "✓ PASS" if compliance.get("compliant", False) else "✗ FAIL"
            print(f"{status} {metric_name.replace('_', ' ').title()}")

            if not compliance.get("compliant", False):
                if "actual" in compliance and "target" in compliance:
                    print(f"    Actual: {compliance['actual']}, Target: {compliance['target']}")

        # Print recommendations
        if results["recommendations"]:
            print("\n\nRecommendations:")
            print("-" * 15)
            for rec in results["recommendations"]:
                print(f"• {rec}")

    def save_report(self, results: Dict[str, Any], output_file: str = None):
        """Save performance report to file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"performance_report_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nReport saved to: {output_file}")


async def main():
    """Main performance monitoring script"""
    print("CES Phase 0.3 Performance Monitor")
    print("=================================")

    monitor = CESPerformanceMonitor()

    # Run performance tests
    results = await monitor.run_performance_tests()

    # Print report
    monitor.print_report(results)

    # Save report
    monitor.save_report(results)

    # Check if all metrics are compliant
    compliance_results = results.get("compliance", {})
    all_compliant = all(
        comp.get("compliant", False)
        for comp in compliance_results.values()
        if "compliant" in comp
    )

    if all_compliant:
        print("\n🎉 All performance metrics meet baseline requirements!")
        sys.exit(0)
    else:
        print("\n⚠️  Some performance metrics do not meet baseline requirements")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
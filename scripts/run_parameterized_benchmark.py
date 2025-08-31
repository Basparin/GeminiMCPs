#!/usr/bin/env python3
"""
Parameterized Benchmark Runner for CodeSage MCP Server.

This script runs benchmarks with different configurations and parameters
to test various scenarios and workloads.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import time


class ParameterizedBenchmarkRunner:
    """Runner for parameterized benchmarks with different configurations."""

    def __init__(self, server_url: str = "http://localhost:8000/mcp"):
        self.server_url = server_url
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)

    def run_benchmark_with_config(self, config_name: str, users: int = 10,
                                iterations: int = 5) -> Dict[str, Any]:
        """Run benchmark with specific configuration."""
        print(f"Running benchmark with config: {config_name}")
        print(f"Users: {users}, Iterations: {iterations}")

        # Get configuration parameters
        config = self._get_config_parameters(config_name)

        # Override with command line parameters
        config["concurrent_users"] = users
        config["test_iterations"] = iterations

        # Run the benchmark
        results = self._execute_benchmark(config)

        # Save results
        self._save_results(config_name, config, results)

        return results

    def _get_config_parameters(self, config_name: str) -> Dict[str, Any]:
        """Get configuration parameters for a specific config."""
        configs = {
            "default": {
                "concurrent_users": 10,
                "test_iterations": 5,
                "duration_seconds": 60,
                "load_scenario": "sustained",
                "codebase_size": "medium",
                "enable_edge_cases": True
            },
            "high_load": {
                "concurrent_users": 20,
                "test_iterations": 5,
                "duration_seconds": 120,
                "load_scenario": "sustained",
                "codebase_size": "medium",
                "enable_edge_cases": False
            },
            "stress_test": {
                "concurrent_users": 50,
                "test_iterations": 10,
                "duration_seconds": 180,
                "load_scenario": "stress",
                "codebase_size": "large",
                "enable_edge_cases": True
            },
            "small_codebase": {
                "concurrent_users": 10,
                "test_iterations": 5,
                "duration_seconds": 60,
                "load_scenario": "sustained",
                "codebase_size": "small",
                "enable_edge_cases": True
            },
            "medium_codebase": {
                "concurrent_users": 10,
                "test_iterations": 5,
                "duration_seconds": 90,
                "load_scenario": "sustained",
                "codebase_size": "medium",
                "enable_edge_cases": True
            },
            "large_codebase": {
                "concurrent_users": 5,
                "test_iterations": 3,
                "duration_seconds": 120,
                "load_scenario": "sustained",
                "codebase_size": "large",
                "enable_edge_cases": False
            },
            "burst_load": {
                "concurrent_users": 15,
                "test_iterations": 5,
                "duration_seconds": 60,
                "load_scenario": "bursty",
                "codebase_size": "medium",
                "enable_edge_cases": True
            },
            "pattern_recognition": {
                "concurrent_users": 10,
                "test_iterations": 5,
                "duration_seconds": 120,
                "load_scenario": "pattern_recognition",
                "codebase_size": "medium",
                "enable_edge_cases": True
            }
        }

        return configs.get(config_name, configs["default"])

    def _execute_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the benchmark with given configuration."""
        results = {
            "config": config,
            "timestamp": time.time(),
            "results": {}
        }

        try:
            # Set environment variables for the benchmark
            env = os.environ.copy()
            env.update({
                "BENCHMARK_CONFIG": json.dumps(config),
                "SERVER_URL": self.server_url
            })

            # Run comprehensive benchmark
            print("Running comprehensive performance benchmark...")
            cmd = [
                sys.executable,
                "tests/benchmark_performance.py",
                self.server_url
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=config.get("duration_seconds", 300) + 60  # Add buffer
            )

            results["results"]["comprehensive"] = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }

            # Run modular tool benchmarks
            print("Running modular tool benchmarks...")
            cmd = [
                sys.executable,
                "tests/benchmark_mcp_tools.py",
                self.server_url,
                str(config.get("test_iterations", 5))
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300
            )

            results["results"]["modular_tools"] = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }

        except subprocess.TimeoutExpired:
            results["results"]["timeout"] = True
            print("Benchmark timed out")
        except Exception as e:
            results["results"]["error"] = str(e)
            print(f"Benchmark failed: {e}")

        return results

    def _save_results(self, config_name: str, config: Dict[str, Any],
                     results: Dict[str, Any]):
        """Save benchmark results to file."""
        timestamp = int(time.time())
        filename = f"parameterized_benchmark_{config_name}_{timestamp}.json"

        result_data = {
            "config_name": config_name,
            "config": config,
            "results": results,
            "timestamp": timestamp
        }

        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)

        print(f"Results saved to: {filepath}")

    def run_multiple_configs(self, configs: List[str]) -> List[Dict[str, Any]]:
        """Run benchmarks for multiple configurations."""
        all_results = []

        for config_name in configs:
            try:
                result = self.run_benchmark_with_config(config_name)
                all_results.append(result)
            except Exception as e:
                print(f"Failed to run config {config_name}: {e}")
                all_results.append({
                    "config_name": config_name,
                    "error": str(e),
                    "timestamp": time.time()
                })

        return all_results

    def generate_comparison_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a comparison report for multiple benchmark runs."""
        report = []
        report.append("# Parameterized Benchmark Comparison Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        for result in results:
            if "error" in result:
                report.append(f"## {result['config_name']} - FAILED")
                report.append(f"Error: {result['error']}")
                continue

            config = result.get("config", {})
            report.append(f"## {result['config_name']}")
            report.append(f"- Concurrent Users: {config.get('concurrent_users', 'N/A')}")
            report.append(f"- Test Iterations: {config.get('test_iterations', 'N/A')}")
            report.append(f"- Duration: {config.get('duration_seconds', 'N/A')}s")
            report.append(f"- Load Scenario: {config.get('load_scenario', 'N/A')}")
            report.append(f"- Codebase Size: {config.get('codebase_size', 'N/A')}")

            # Add success status
            comprehensive = result.get("results", {}).get("comprehensive", {})
            modular = result.get("results", {}).get("modular_tools", {})

            report.append(f"- Comprehensive Benchmark: {'PASS' if comprehensive.get('success') else 'FAIL'}")
            report.append(f"- Modular Tools Benchmark: {'PASS' if modular.get('success') else 'FAIL'}")
            report.append("")

        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run parameterized benchmarks")
    parser.add_argument("--config", default="default",
                       help="Benchmark configuration name")
    parser.add_argument("--users", type=int, default=10,
                       help="Number of concurrent users")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of test iterations")
    parser.add_argument("--server-url", default="http://localhost:8000/mcp",
                       help="MCP server URL")
    parser.add_argument("--multiple", nargs="+",
                       help="Run multiple configurations")
    parser.add_argument("--output-report", help="Output comparison report file")

    args = parser.parse_args()

    runner = ParameterizedBenchmarkRunner(args.server_url)

    if args.multiple:
        # Run multiple configurations
        results = runner.run_multiple_configs(args.multiple)

        if args.output_report:
            report = runner.generate_comparison_report(results)
            with open(args.output_report, 'w') as f:
                f.write(report)
            print(f"Comparison report saved to: {args.output_report}")
    else:
        # Run single configuration
        result = runner.run_benchmark_with_config(
            args.config,
            args.users,
            args.iterations
        )
        print("Benchmark completed successfully!")


if __name__ == "__main__":
    main()
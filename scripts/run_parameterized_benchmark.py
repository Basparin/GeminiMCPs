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
import logging

# Hardware-adaptive imports
try:
    from tests.hardware_utils import (
        get_hardware_profile,
        get_adaptive_config,
        check_safety_requirements,
        log_system_info,
        detect_cpu_cores,
        detect_available_ram,
        HardwareProfile
    )
except ImportError:
    print("WARNING: hardware_utils module not found. Running without hardware adaptation.")
    print("Install psutil and ensure tests/hardware_utils.py is available for optimal performance.")
    # Fallback definitions
    HardwareProfile = str
    def get_hardware_profile() -> str:
        return 'medium'
    def get_adaptive_config(profile: str) -> Dict[str, Any]:
        return {'max_workers': 4, 'timeout_seconds': 30}
    def check_safety_requirements(profile: str = None) -> bool:
        return True
    def log_system_info():
        pass
    def detect_cpu_cores() -> int:
        return 4
    def detect_available_ram() -> float:
        return 8.0


class ParameterizedBenchmarkRunner:
    """Runner for parameterized benchmarks with hardware-adaptive configurations."""

    def __init__(self, server_url: str = "http://localhost:8000/mcp"):
        self.server_url = server_url
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)

        # Hardware-adaptive configuration
        self.hardware_profile = get_hardware_profile()
        self.adaptive_config = get_adaptive_config(self.hardware_profile)
        self.cpu_cores = detect_cpu_cores()
        self.available_ram_gb = detect_available_ram()

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Log hardware information
        self.logger.info("Initializing ParameterizedBenchmarkRunner with hardware-adaptive configuration")
        log_system_info()

        # Check safety requirements
        if not check_safety_requirements(self.hardware_profile):
            self.logger.warning(f"System does not meet minimum safety requirements for hardware profile '{self.hardware_profile}'")
            self.logger.warning("Benchmarks may be resource-intensive or fail on this system.")

        self.logger.info(f"Hardware Profile: {self.hardware_profile} (CPU: {self.cpu_cores}, RAM: {self.available_ram_gb:.1f}GB)")
        self.logger.info(f"Adaptive Config: {self.adaptive_config}")

    def _get_adaptive_concurrent_users(self, base_users: int) -> int:
        """Get adaptive concurrent users based on hardware profile."""
        scaling_factors = {
            'light': 0.3,   # Reduce users significantly on low-end hardware
            'medium': 0.7,  # Moderate reduction for mid-range hardware
            'full': 1.0     # Full user count on high-end hardware
        }

        scaling_factor = scaling_factors.get(self.hardware_profile, 1.0)
        adaptive_users = max(1, int(base_users * scaling_factor))

        self.logger.debug(f"Adaptive concurrent users: {base_users} -> {adaptive_users} "
                         f"(scaling factor: {scaling_factor:.1f}x for {self.hardware_profile} hardware)")
        return adaptive_users

    def _get_adaptive_duration(self, base_duration: int) -> int:
        """Get adaptive test duration based on hardware profile."""
        duration_scaling_factors = {
            'light': 0.4,   # Shorter duration on low-end hardware
            'medium': 0.7,  # Moderate duration for mid-range hardware
            'full': 1.0     # Full duration on high-end hardware
        }

        scaling_factor = duration_scaling_factors.get(self.hardware_profile, 1.0)
        adaptive_duration = max(10, int(base_duration * scaling_factor))

        self.logger.debug(f"Adaptive duration: {base_duration}s -> {adaptive_duration}s "
                         f"(scaling factor: {scaling_factor:.1f}x for {self.hardware_profile} hardware)")
        return adaptive_duration

    def _get_adaptive_iterations(self, base_iterations: int) -> int:
        """Get adaptive test iterations based on hardware profile."""
        iteration_scaling_factors = {
            'light': 0.3,   # Reduce iterations significantly on low-end hardware
            'medium': 0.6,  # Moderate reduction for mid-range hardware
            'full': 1.0     # Full iterations on high-end hardware
        }

        scaling_factor = iteration_scaling_factors.get(self.hardware_profile, 1.0)
        adaptive_iterations = max(1, int(base_iterations * scaling_factor))

        self.logger.debug(f"Adaptive iterations: {base_iterations} -> {adaptive_iterations} "
                         f"(scaling factor: {scaling_factor:.1f}x for {self.hardware_profile} hardware)")
        return adaptive_iterations

    def _check_config_safety(self, config: Dict[str, Any]) -> bool:
        """Check if configuration is safe for current hardware."""
        concurrent_users = config.get('concurrent_users', 1)
        duration_seconds = config.get('duration_seconds', 60)

        # Safety thresholds based on hardware profile
        safety_limits = {
            'light': {'max_users': 5, 'max_duration': 120, 'warning_threshold': 0.8},
            'medium': {'max_users': 20, 'max_duration': 300, 'warning_threshold': 0.9},
            'full': {'max_users': 100, 'max_duration': 600, 'warning_threshold': 0.95}
        }

        limits = safety_limits.get(self.hardware_profile, safety_limits['medium'])
        is_safe = True

        if concurrent_users > limits['max_users']:
            self.logger.warning(f"Configuration '{config.get('name', 'unknown')}' exceeds safe concurrent users limit: "
                              f"{concurrent_users} > {limits['max_users']} for {self.hardware_profile} hardware")
            is_safe = False

        if duration_seconds > limits['max_duration']:
            self.logger.warning(f"Configuration '{config.get('name', 'unknown')}' exceeds safe duration limit: "
                              f"{duration_seconds}s > {limits['max_duration']}s for {self.hardware_profile} hardware")
            is_safe = False

        # Resource usage estimation
        estimated_memory_mb = concurrent_users * 50  # Rough estimate: 50MB per user
        if estimated_memory_mb > (self.available_ram_gb * 1024 * limits['warning_threshold']):
            self.logger.warning(f"Configuration may cause high memory usage: ~{estimated_memory_mb}MB estimated "
                              f"vs {self.available_ram_gb * 1024 * limits['warning_threshold']:.0f}MB safe limit")

        return is_safe

    def run_benchmark_with_config(self, config_name: str, users: int = 10,
                                iterations: int = 5) -> Dict[str, Any]:
        """Run benchmark with specific configuration."""
        print(f"Running benchmark with config: {config_name}")
        print(f"Requested: Users: {users}, Iterations: {iterations}")

        # Get configuration parameters (with hardware adaptation)
        config = self._get_config_parameters(config_name)

        # Apply command line overrides with adaptive scaling
        if users != 10:  # Only override if user specified custom value
            config["concurrent_users"] = self._get_adaptive_concurrent_users(users)
            config["_original_concurrent_users"] = users
            print(f"Applied adaptive scaling to custom users: {users} -> {config['concurrent_users']}")

        if iterations != 5:  # Only override if user specified custom value
            config["test_iterations"] = self._get_adaptive_iterations(iterations)
            config["_original_test_iterations"] = iterations
            print(f"Applied adaptive scaling to custom iterations: {iterations} -> {config['test_iterations']}")

        print(f"Final config: Users: {config['concurrent_users']}, Iterations: {config['test_iterations']}, "
              f"Duration: {config['duration_seconds']}s")

        # Run the benchmark
        results = self._execute_benchmark(config)

        # Save results
        self._save_results(config_name, config, results)

        return results

    def _get_config_parameters(self, config_name: str) -> Dict[str, Any]:
        """Get configuration parameters for a specific config with hardware-adaptive scaling."""
        base_configs = {
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

        config = base_configs.get(config_name, base_configs["default"]).copy()
        config["name"] = config_name  # Add config name for safety checks

        # Apply hardware-adaptive scaling
        original_users = config["concurrent_users"]
        original_duration = config["duration_seconds"]
        original_iterations = config["test_iterations"]

        config["concurrent_users"] = self._get_adaptive_concurrent_users(original_users)
        config["duration_seconds"] = self._get_adaptive_duration(original_duration)
        config["test_iterations"] = self._get_adaptive_iterations(original_iterations)

        # Store original values for reporting
        config["_original_concurrent_users"] = original_users
        config["_original_duration_seconds"] = original_duration
        config["_original_test_iterations"] = original_iterations

        # Check configuration safety
        if not self._check_config_safety(config):
            self.logger.warning(f"Configuration '{config_name}' may be resource-intensive for {self.hardware_profile} hardware")
            self.logger.info("Consider using a lighter configuration or upgrading hardware for better performance")

        self.logger.info(f"Applied hardware-adaptive scaling to config '{config_name}':")
        self.logger.info(f"  Concurrent users: {original_users} -> {config['concurrent_users']}")
        self.logger.info(f"  Duration: {original_duration}s -> {config['duration_seconds']}s")
        self.logger.info(f"  Iterations: {original_iterations} -> {config['test_iterations']}")

        return config

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
        """Save benchmark results to file with hardware information."""
        timestamp = int(time.time())
        filename = f"parameterized_benchmark_{config_name}_{timestamp}.json"

        result_data = {
            "config_name": config_name,
            "config": config,
            "results": results,
            "timestamp": timestamp,
            "hardware_info": {
                "profile": self.hardware_profile,
                "cpu_cores": self.cpu_cores,
                "available_ram_gb": self.available_ram_gb,
                "adaptive_config": self.adaptive_config
            }
        }

        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)

        print(f"Results saved to: {filepath}")
        print(f"Hardware profile used: {self.hardware_profile} (CPU: {self.cpu_cores}, RAM: {self.available_ram_gb:.1f}GB)")

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
        """Generate a comparison report for multiple benchmark runs with hardware information."""
        report = []
        report.append("# Parameterized Benchmark Comparison Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Add hardware summary if available
        if results and "hardware_info" in results[0]:
            hw_info = results[0]["hardware_info"]
            report.append("## Hardware Configuration")
            report.append(f"- Profile: {hw_info.get('profile', 'unknown')}")
            report.append(f"- CPU Cores: {hw_info.get('cpu_cores', 'unknown')}")
            report.append(f"- Available RAM: {hw_info.get('available_ram_gb', 'unknown')}GB")
            report.append("")

        for result in results:
            if "error" in result:
                report.append(f"## {result['config_name']} - FAILED")
                report.append(f"Error: {result['error']}")
                continue

            config = result.get("config", {})
            report.append(f"## {result['config_name']}")

            # Show adaptive scaling information
            original_users = config.get('_original_concurrent_users')
            if original_users and original_users != config.get('concurrent_users'):
                report.append(f"- Concurrent Users: {config.get('concurrent_users', 'N/A')} (adapted from {original_users})")
            else:
                report.append(f"- Concurrent Users: {config.get('concurrent_users', 'N/A')}")

            original_iterations = config.get('_original_test_iterations')
            if original_iterations and original_iterations != config.get('test_iterations'):
                report.append(f"- Test Iterations: {config.get('test_iterations', 'N/A')} (adapted from {original_iterations})")
            else:
                report.append(f"- Test Iterations: {config.get('test_iterations', 'N/A')}")

            original_duration = config.get('_original_duration_seconds')
            if original_duration and original_duration != config.get('duration_seconds'):
                report.append(f"- Duration: {config.get('duration_seconds', 'N/A')}s (adapted from {original_duration}s)")
            else:
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
    parser = argparse.ArgumentParser(description="Run parameterized benchmarks with hardware-adaptive scaling")
    parser.add_argument("--config", default="default",
                        help="Benchmark configuration name")
    parser.add_argument("--users", type=int, default=10,
                        help="Number of concurrent users (will be scaled based on hardware)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of test iterations (will be scaled based on hardware)")
    parser.add_argument("--server-url", default="http://localhost:8000/mcp",
                        help="MCP server URL")
    parser.add_argument("--multiple", nargs="+",
                        help="Run multiple configurations")
    parser.add_argument("--output-report", help="Output comparison report file")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable hardware-adaptive scaling (use original values)")
    parser.add_argument("--hardware-info", action="store_true",
                        help="Display detailed hardware information and exit")

    args = parser.parse_args()

    # Create runner with hardware detection
    runner = ParameterizedBenchmarkRunner(args.server_url)

    # Display hardware info if requested
    if args.hardware_info:
        print("\n" + "="*60)
        print("HARDWARE DETECTION SUMMARY")
        print("="*60)
        print(f"Hardware Profile: {runner.hardware_profile}")
        print(f"CPU Cores: {runner.cpu_cores}")
        print(f"Available RAM: {runner.available_ram_gb:.1f}GB")
        print(f"Adaptive Configuration: {runner.adaptive_config}")
        print("="*60 + "\n")
        return

    # Disable adaptive scaling if requested
    if args.no_adaptive:
        print("WARNING: Hardware-adaptive scaling disabled. Using original configuration values.")
        print("This may result in resource-intensive operations on lower-end hardware.")
        # Override the scaling methods to return original values
        runner._get_adaptive_concurrent_users = lambda x: x
        runner._get_adaptive_duration = lambda x: x
        runner._get_adaptive_iterations = lambda x: x

    if args.multiple:
        # Run multiple configurations
        print(f"Running {len(args.multiple)} configurations with hardware-adaptive scaling...")
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
        print(f"Hardware profile used: {runner.hardware_profile}")


if __name__ == "__main__":
    main()
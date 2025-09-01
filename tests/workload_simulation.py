#!/usr/bin/env python3
"""
Workload Simulation Tool for CodeSage MCP Server

This script simulates various real-world usage patterns to validate the
self-optimization system of the CodeSage MCP Server. It supports different
load scenarios and configurable parameters for comprehensive testing.
"""

import argparse
import asyncio
import json
import logging
import random
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import requests
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

# Import hardware detection utilities
from .hardware_utils import get_hardware_profile, get_adaptive_config, log_system_info


@dataclass
class SimulationConfig:
    """Configuration for the workload simulation."""
    server_url: str = "http://localhost:8000/mcp"
    concurrent_users: int = 10
    target_rps: float = 5.0
    duration_seconds: int = 60
    load_scenario: str = "sustained"
    request_mix_ratios: Dict[str, float] = None
    usage_pattern: str = "development"
    log_level: str = "INFO"
    output_file: Optional[str] = None

    def __post_init__(self):
        if self.request_mix_ratios is None:
            self.request_mix_ratios = {
                "read_code_file": 0.3,
                "search_codebase": 0.2,
                "semantic_search_codebase": 0.15,
                "get_file_structure": 0.1,
                "summarize_code_section": 0.1,
                "suggest_code_improvements": 0.05,
                "get_performance_metrics": 0.05,
                "get_cache_statistics": 0.05
            }


def get_adaptive_simulation_config(hardware_profile: str) -> Dict[str, Any]:
    """
    Get adaptive simulation configuration based on detected hardware capabilities.

    Args:
        hardware_profile: Hardware profile ('light', 'medium', 'full')

    Returns:
        Dict with adaptive values for concurrent_users, target_rps, duration_seconds
    """
    # Base configurations for each hardware profile
    configs = {
        'light': {
            'concurrent_users': 2,      # Reduced for low-end hardware
            'target_rps': 1.0,          # Lower RPS to prevent overload
            'duration_seconds': 30      # Shorter duration for resource conservation
        },
        'medium': {
            'concurrent_users': 5,      # Moderate concurrent users
            'target_rps': 2.5,          # Balanced RPS
            'duration_seconds': 45      # Medium duration
        },
        'full': {
            'concurrent_users': 10,     # Full capacity for high-end hardware
            'target_rps': 5.0,          # Higher RPS for powerful systems
            'duration_seconds': 60      # Standard duration
        }
    }

    return configs.get(hardware_profile, configs['medium'])  # Default to medium if unknown


def apply_hardware_adaptive_config(config: SimulationConfig, hardware_profile: str) -> SimulationConfig:
    """
    Apply hardware-adaptive adjustments to the simulation configuration.

    Args:
        config: Original simulation configuration
        hardware_profile: Detected hardware profile

    Returns:
        SimulationConfig: Updated configuration with adaptive settings
    """
    adaptive_config = get_adaptive_simulation_config(hardware_profile)

    # Only adjust if the config uses default values (not overridden by command line)
    # We'll assume defaults are being used unless explicitly set differently
    if config.concurrent_users == 10:  # Default value
        config.concurrent_users = adaptive_config['concurrent_users']
    if config.target_rps == 5.0:  # Default value
        config.target_rps = adaptive_config['target_rps']
    if config.duration_seconds == 60:  # Default value
        config.duration_seconds = adaptive_config['duration_seconds']

    return config


class OperationType(Enum):
    """Types of operations for timeout configuration."""
    FAST = "fast"              # Cache hits, simple reads (100-500ms)
    MEDIUM = "medium"          # Searches, file operations (1-2s)
    SLOW = "slow"              # LLM analysis, complex computations (5-10s)
    VERY_SLOW = "very_slow"    # Large indexing, batch operations (30-60s)


def _classify_operation_type(request_data: Dict[str, Any]) -> OperationType:
    """Classify the operation type based on the request."""
    tool_name = request_data.get("params", {}).get("name", "")

    # Fast operations
    if tool_name in ["read_code_file", "get_file_structure"]:
        return OperationType.FAST

    # Medium operations
    elif tool_name in ["search_codebase", "semantic_search_codebase", "get_performance_metrics"]:
        return OperationType.MEDIUM

    # Slow operations
    elif tool_name in ["suggest_code_improvements", "summarize_code_section", "find_duplicate_code"]:
        return OperationType.SLOW

    # Very slow operations
    elif tool_name in ["index_codebase", "generate_unit_tests", "profile_code_performance"]:
        return OperationType.VERY_SLOW

    # Default to medium
    else:
        return OperationType.MEDIUM


def _get_adaptive_timeout(operation_type: OperationType, target_rps: float) -> float:
    """Get adaptive timeout based on operation type and load."""
    # Base timeouts (similar to config.py)
    base_timeouts = {
        OperationType.FAST: 0.5,      # 500ms
        OperationType.MEDIUM: 2.0,    # 2 seconds
        OperationType.SLOW: 10.0,     # 10 seconds
        OperationType.VERY_SLOW: 60.0 # 60 seconds
    }

    base_timeout = base_timeouts.get(operation_type, 2.0)

    # Adjust based on load (higher RPS = more aggressive timeouts)
    load_factor = min(target_rps / 100.0, 3.0)  # Max 3x adjustment
    if load_factor > 1.0:
        base_timeout *= (1.0 + (load_factor - 1.0) * 0.5)  # Increase timeout under load

    return base_timeout


class MetricsCollector:
    """Collects and analyzes simulation metrics."""

    def __init__(self):
        self.response_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        self.requests_per_second = deque(maxlen=100)
        self.start_time = time.time()
        self.lock = threading.Lock()

    def record_request(self, response_time: float, success: bool):
        """Record a completed request."""
        with self.lock:
            self.response_times.append(response_time)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1

            current_time = time.time()
            self.requests_per_second.append(current_time)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_requests = len(self.response_times)
        if total_requests == 0:
            return {"error": "No requests recorded"}

        with self.lock:
            avg_response_time = statistics.mean(self.response_times)
            if total_requests >= 2:
                median_response_time = statistics.median(self.response_times)
                p95_response_time = statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
            else:
                median_response_time = avg_response_time
                p95_response_time = avg_response_time
            success_rate = (self.success_count / total_requests) * 100

            # Calculate RPS
            if len(self.requests_per_second) > 1:
                time_span = self.requests_per_second[-1] - self.requests_per_second[0]
                actual_rps = len(self.requests_per_second) / time_span if time_span > 0 else 0
            else:
                actual_rps = 0

            return {
                "total_requests": total_requests,
                "successful_requests": self.success_count,
                "failed_requests": self.error_count,
                "success_rate_percent": success_rate,
                "avg_response_time_ms": avg_response_time,
                "median_response_time_ms": median_response_time,
                "p95_response_time_ms": p95_response_time,
                "actual_rps": actual_rps,
                "duration_seconds": time.time() - self.start_time
            }


class RequestGenerator:
    """Generates realistic MCP tool requests based on usage patterns."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.request_id = 0
        self.usage_patterns = self._define_usage_patterns()

    def _define_usage_patterns(self) -> Dict[str, Dict[str, float]]:
        """Define tool usage patterns for different scenarios."""
        return {
            "development": {
                "read_code_file": 0.4,
                "search_codebase": 0.15,
                "get_file_structure": 0.15,
                "suggest_code_improvements": 0.1,
                "get_performance_metrics": 0.05,
                "index_codebase": 0.05,
                "get_cache_statistics": 0.05,
                "summarize_code_section": 0.05
            },
            "code_review": {
                "read_code_file": 0.3,
                "semantic_search_codebase": 0.25,
                "find_duplicate_code": 0.15,
                "summarize_code_section": 0.15,
                "suggest_code_improvements": 0.1,
                "get_file_structure": 0.05
            },
            "refactoring": {
                "search_codebase": 0.2,
                "find_duplicate_code": 0.2,
                "suggest_code_improvements": 0.2,
                "read_code_file": 0.15,
                "get_dependencies_overview": 0.1,
                "generate_unit_tests": 0.1,
                "get_file_structure": 0.05
            },
            "onboarding": {
                "get_file_structure": 0.3,
                "read_code_file": 0.25,
                "get_dependencies_overview": 0.15,
                "summarize_code_section": 0.15,
                "list_undocumented_functions": 0.1,
                "count_lines_of_code": 0.05
            },
            "ci_cd": {
                "index_codebase": 0.3,
                "search_codebase": 0.2,
                "get_performance_metrics": 0.15,
                "get_cache_statistics": 0.15,
                "profile_code_performance": 0.1,
                "get_file_structure": 0.05
            }
        }

    def _get_sample_file_paths(self) -> List[str]:
        """Get sample file paths for testing."""
        return [
            "codesage_mcp/main.py",
            "codesage_mcp/config.py",
            "codesage_mcp/cache.py",
            "codesage_mcp/codebase_manager.py",
            "codesage_mcp/searching.py",
            "tests/test_main.py",
            "README.md",
            "pyproject.toml"
        ]

    def _get_sample_queries(self) -> List[str]:
        """Get sample search queries."""
        return [
            "def main",
            "class Config",
            "import os",
            "function cache",
            "performance metrics",
            "error handling",
            "database connection",
            "API endpoint"
        ]

    def generate_request(self) -> Dict[str, Any]:
        """Generate a random MCP tool request based on the current pattern."""
        self.request_id += 1

        # Select tool based on usage pattern
        pattern = self.usage_patterns.get(self.config.usage_pattern, self.usage_patterns["development"])
        tools = list(pattern.keys())
        weights = list(pattern.values())
        tool_name = random.choices(tools, weights=weights)[0]

        # Generate tool-specific parameters
        params = self._generate_tool_params(tool_name)

        return {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "id": self.request_id
        }

    def _generate_tool_params(self, tool_name: str) -> Dict[str, Any]:
        """Generate parameters for a specific tool."""
        file_paths = self._get_sample_file_paths()
        queries = self._get_sample_queries()

        params = {}

        if tool_name == "read_code_file":
            params["file_path"] = random.choice(file_paths)
        elif tool_name in ["search_codebase", "semantic_search_codebase"]:
            params["codebase_path"] = "."
            params["pattern" if tool_name == "search_codebase" else "query"] = random.choice(queries)
            if tool_name == "search_codebase":
                params["file_types"] = ["*.py", "*.md"]
        elif tool_name == "get_file_structure":
            params["codebase_path"] = "."
            params["file_path"] = random.choice(file_paths)
        elif tool_name == "summarize_code_section":
            params["file_path"] = random.choice(file_paths)
            params["start_line"] = random.randint(1, 50)
            params["end_line"] = params["start_line"] + random.randint(10, 50)
            params["llm_model"] = random.choice(["groq", "openrouter", "google"])
        elif tool_name == "suggest_code_improvements":
            params["file_path"] = random.choice(file_paths)
            params["start_line"] = random.randint(1, 30)
            params["end_line"] = params["start_line"] + random.randint(5, 20)
        elif tool_name == "index_codebase":
            params["path"] = "."
        elif tool_name == "find_duplicate_code":
            params["codebase_path"] = "."
        elif tool_name == "get_dependencies_overview":
            params["codebase_path"] = "."
        elif tool_name == "list_undocumented_functions":
            params["file_path"] = random.choice([f for f in file_paths if f.endswith(".py")])
        elif tool_name == "count_lines_of_code":
            params["codebase_path"] = "."
        elif tool_name == "profile_code_performance":
            params["file_path"] = random.choice([f for f in file_paths if f.endswith(".py")])
        elif tool_name == "generate_unit_tests":
            params["file_path"] = random.choice([f for f in file_paths if f.endswith(".py")])
        # Add more tool params as needed

        return params


class LoadScenario:
    """Implements different load scenarios."""

    @staticmethod
    def ramp_up(current_time: float, total_duration: float) -> float:
        """Gradually increase load from 0 to target over the duration."""
        progress = min(current_time / total_duration, 1.0)
        return progress

    @staticmethod
    def bursty(current_time: float, total_duration: float) -> float:
        """Bursty load with periodic spikes."""
        cycle = (current_time % 10) / 10  # 10-second cycles
        if cycle < 0.7:  # 70% normal load
            return 0.5
        else:  # 30% high load
            return 2.0

    @staticmethod
    def sustained(current_time: float, total_duration: float) -> float:
        """Constant load at target level."""
        return 1.0

    @staticmethod
    def stress(current_time: float, total_duration: float) -> float:
        """High constant load to test limits."""
        return 3.0

    @staticmethod
    def pattern_recognition(current_time: float, total_duration: float) -> float:
        """Adaptive load based on pattern recognition simulation."""
        # Simulate pattern recognition by varying load based on time
        hour_of_day = (current_time / 3600) % 24
        if 9 <= hour_of_day <= 17:  # Business hours
            return 1.5
        elif 18 <= hour_of_day <= 22:  # Evening
            return 0.8
        else:  # Night
            return 0.3


async def worker_simulation(worker_id: int, config: SimulationConfig,
                          metrics: MetricsCollector, generator: RequestGenerator,
                          stop_event: threading.Event):
    """Simulate a single user sending requests."""
    session = requests.Session()

    while not stop_event.is_set():
        try:
            # Generate and send request
            request_data = generator.generate_request()
            start_time = time.time()

            # Use adaptive timeout based on operation type
            operation_type = _classify_operation_type(request_data)
            adaptive_timeout = _get_adaptive_timeout(operation_type, config.target_rps)

            response = session.post(config.server_url, json=request_data, timeout=adaptive_timeout)
            response_time = (time.time() - start_time) * 1000

            success = response.status_code == 200
            if success:
                try:
                    response_data = response.json()
                    success = "error" not in response_data
                except (ValueError, json.JSONDecodeError):
                    success = False

            metrics.record_request(response_time, success)

            # Log the request
            logging.debug(f"Worker {worker_id}: {request_data['params']['name']} - "
                         f"{response_time:.2f}ms - {'SUCCESS' if success else 'FAILED'}")

        except Exception as e:
            logging.error(f"Worker {worker_id} error: {e}")
            # Use adaptive timeout for failure recording
            failure_timeout = adaptive_timeout * 1000  # Convert to milliseconds
            metrics.record_request(failure_timeout, False)

        # Sleep to control RPS (will be adjusted by load scenario)
        time.sleep(1.0 / config.target_rps)


def run_simulation(config: SimulationConfig) -> Dict[str, Any]:
    """Run the workload simulation."""
    logging.basicConfig(level=getattr(logging, config.log_level),
                       format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Starting workload simulation: {config.load_scenario} scenario")
    logging.info(f"Configuration: {config.concurrent_users} users, "
                f"{config.target_rps} RPS target, {config.duration_seconds}s duration")

    metrics = MetricsCollector()
    generator = RequestGenerator(config)
    stop_event = threading.Event()

    # Get load scenario function
    scenario_funcs = {
        "ramp_up": LoadScenario.ramp_up,
        "bursty": LoadScenario.bursty,
        "sustained": LoadScenario.sustained,
        "stress": LoadScenario.stress,
        "pattern_recognition": LoadScenario.pattern_recognition
    }

    load_func = scenario_funcs.get(config.load_scenario, LoadScenario.sustained)

    def worker_wrapper(worker_id: int):
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(worker_simulation(worker_id, config, metrics,
                                                generator, stop_event))

    # Start workers
    with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
        futures = [executor.submit(worker_wrapper, i) for i in range(config.concurrent_users)]

        start_time = time.time()
        try:
            while time.time() - start_time < config.duration_seconds:
                # Adjust load based on scenario
                current_time = time.time() - start_time
                load_multiplier = load_func(current_time, config.duration_seconds)

                # Log progress every 10 seconds
                if int(current_time) % 10 == 0 and int(current_time) > 0:
                    summary = metrics.get_summary()
                    logging.info(f"Progress: {current_time:.1f}s - "
                               f"RPS: {summary.get('actual_rps', 0):.2f} - "
                               f"Success: {summary.get('success_rate_percent', 0):.1f}%")

                time.sleep(1)

        finally:
            stop_event.set()
            # Wait for workers to finish
            for future in futures:
                future.result(timeout=5)

    # Get final results
    results = metrics.get_summary()
    results["config"] = {
        "concurrent_users": config.concurrent_users,
        "target_rps": config.target_rps,
        "duration_seconds": config.duration_seconds,
        "load_scenario": config.load_scenario,
        "usage_pattern": config.usage_pattern
    }

    # Add hardware profile information to results
    hardware_profile = get_hardware_profile()
    results["hardware_profile"] = hardware_profile
    results["adaptive_config_applied"] = True

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CodeSage MCP Server Workload Simulator")
    parser.add_argument("--server-url", default="http://localhost:8000/mcp",
                        help="MCP server URL")
    parser.add_argument("--users", type=int, default=10,
                        help="Number of concurrent users")
    parser.add_argument("--rps", type=float, default=5.0,
                        help="Target requests per second")
    parser.add_argument("--duration", type=int, default=60,
                        help="Simulation duration in seconds")
    parser.add_argument("--scenario", choices=["ramp_up", "bursty", "sustained", "stress", "pattern_recognition"],
                        default="sustained", help="Load scenario")
    parser.add_argument("--pattern", choices=["development", "code_review", "refactoring", "onboarding", "ci_cd"],
                        default="development", help="Usage pattern")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable hardware-adaptive configuration")

    args = parser.parse_args()

    # Detect hardware and log system info
    hardware_profile = get_hardware_profile()
    log_system_info()

    config = SimulationConfig(
        server_url=args.server_url,
        concurrent_users=args.users,
        target_rps=args.rps,
        duration_seconds=args.duration,
        load_scenario=args.scenario,
        usage_pattern=args.pattern,
        log_level=args.log_level,
        output_file=args.output
    )

    # Apply hardware-adaptive configuration unless disabled
    if not args.no_adaptive:
        config = apply_hardware_adaptive_config(config, hardware_profile)
        logging.info(f"Applied hardware-adaptive configuration for profile: {hardware_profile}")
        logging.info(f"Adaptive settings: {config.concurrent_users} users, "
                    f"{config.target_rps} RPS, {config.duration_seconds}s duration")
    else:
        logging.info("Hardware-adaptive configuration disabled")

    try:
        results = run_simulation(config)

        # Print results
        print("\n" + "="*60)
        print("WORKLOAD SIMULATION RESULTS")
        print("="*60)
        print(f"Hardware Profile: {results.get('hardware_profile', 'unknown')}")
        print(f"Scenario: {results['config']['load_scenario']}")
        print(f"Pattern: {results['config']['usage_pattern']}")
        print(f"Duration: {results['config']['duration_seconds']}s")
        print(f"Users: {results['config']['concurrent_users']}")
        print(f"Target RPS: {results['config']['target_rps']}")
        print()
        print(f"Total Requests: {results['total_requests']}")
        print(f"Successful: {results['successful_requests']}")
        print(f"Failed: {results['failed_requests']}")
        print(f"Success Rate: {results['success_rate_percent']:.2f}%")
        print(f"Actual RPS: {results['actual_rps']:.2f}")
        print()
        print("Response Times (ms):")
        print(f"  Average: {results['avg_response_time_ms']:.2f}")
        print(f"  Median: {results['median_response_time_ms']:.2f}")
        print(f"  95th Percentile: {results['p95_response_time_ms']:.2f}")
        print("="*60)

        # Save to file if specified
        if config.output_file:
            with open(config.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {config.output_file}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()
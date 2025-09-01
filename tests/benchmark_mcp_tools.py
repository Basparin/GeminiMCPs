#!/usr/bin/env python3
"""
Modular MCP Tools Benchmark Suite for CodeSage MCP Server.

This module provides specialized benchmarks for individual MCP tools, access patterns,
and edge cases to ensure comprehensive performance validation.
"""

import time
import statistics
import requests
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import concurrent.futures

from benchmark_performance import BenchmarkResult, PerformanceBenchmarker


@dataclass
class ModularBenchmarkConfig:
    """Configuration for modular benchmark tests."""
    server_url: str = "http://localhost:8000/mcp"
    test_iterations: int = 5
    concurrent_users: int = 1
    timeout_seconds: int = 30
    enable_edge_cases: bool = True


class ModularMCPToolBenchmarker:
    """Modular benchmarker for MCP tools with specialized test scenarios."""

    def __init__(self, config: ModularBenchmarkConfig = None):
        self.config = config or ModularBenchmarkConfig()
        self.base_benchmarker = PerformanceBenchmarker()

    def benchmark_code_reading_tools(self) -> List[BenchmarkResult]:
        """Benchmark code reading and file access tools."""
        results = []

        # Create test codebase
        codebase_dir = self.base_benchmarker.create_test_codebase("medium")

        try:
            tools_to_test = [
                {
                    "name": "read_code_file",
                    "args": {"file_path": f"{codebase_dir}/module_0.py"},
                    "description": "Read single file"
                },
                {
                    "name": "get_file_structure",
                    "args": {
                        "codebase_path": str(codebase_dir),
                        "file_path": f"{codebase_dir}/module_0.py"
                    },
                    "description": "Get file structure"
                },
                {
                    "name": "list_undocumented_functions",
                    "args": {"file_path": f"{codebase_dir}/module_0.py"},
                    "description": "List undocumented functions"
                }
            ]

            for tool in tools_to_test:
                latencies = self._run_tool_benchmark(tool["name"], tool["args"])

                if latencies:
                    results.extend(self._create_tool_results(
                        f"code_reading_{tool['name']}",
                        latencies,
                        tool["description"]
                    ))

        finally:
            shutil.rmtree(codebase_dir)

        return results

    def benchmark_search_and_analysis_tools(self) -> List[BenchmarkResult]:
        """Benchmark search and code analysis tools."""
        results = []

        # Create test codebase
        codebase_dir = self.base_benchmarker.create_test_codebase("medium")

        try:
            tools_to_test = [
                {
                    "name": "search_codebase",
                    "args": {
                        "codebase_path": str(codebase_dir),
                        "pattern": "def get_module_id",
                        "file_types": ["*.py"]
                    },
                    "description": "Text search in codebase"
                },
                {
                    "name": "semantic_search_codebase",
                    "args": {
                        "codebase_path": str(codebase_dir),
                        "query": "function that returns module id",
                        "top_k": 5
                    },
                    "description": "Semantic search in codebase"
                },
                {
                    "name": "find_duplicate_code",
                    "args": {
                        "codebase_path": str(codebase_dir),
                        "min_similarity": 0.8
                    },
                    "description": "Find duplicate code"
                },
                {
                    "name": "get_dependencies_overview",
                    "args": {"codebase_path": str(codebase_dir)},
                    "description": "Analyze dependencies"
                },
                {
                    "name": "count_lines_of_code",
                    "args": {"codebase_path": str(codebase_dir)},
                    "description": "Count lines of code"
                }
            ]

            for tool in tools_to_test:
                latencies = self._run_tool_benchmark(tool["name"], tool["args"])

                if latencies:
                    results.extend(self._create_tool_results(
                        f"search_analysis_{tool['name']}",
                        latencies,
                        tool["description"]
                    ))

        finally:
            shutil.rmtree(codebase_dir)

        return results

    def benchmark_performance_monitoring_tools(self) -> List[BenchmarkResult]:
        """Benchmark performance monitoring and metrics tools."""
        results = []

        tools_to_test = [
            {
                "name": "get_performance_metrics",
                "args": {},
                "description": "Get performance metrics"
            },
            {
                "name": "get_cache_statistics",
                "args": {},
                "description": "Get cache statistics"
            },
            {
                "name": "get_performance_report",
                "args": {},
                "description": "Get performance report"
            },
            {
                "name": "get_usage_patterns",
                "args": {},
                "description": "Get usage patterns"
            }
        ]

        for tool in tools_to_test:
            latencies = self._run_tool_benchmark(tool["name"], tool["args"])

            if latencies:
                results.extend(self._create_tool_results(
                    f"performance_monitoring_{tool['name']}",
                    latencies,
                    tool["description"]
                ))

        return results

    def benchmark_access_patterns(self) -> List[BenchmarkResult]:
        """Benchmark different access patterns and usage scenarios."""
        results = []

        # Pattern 1: Sequential file reading
        print("Testing sequential file access pattern...")
        seq_results = self._benchmark_sequential_access()
        results.extend(seq_results)

        # Pattern 2: Random file access
        print("Testing random file access pattern...")
        rand_results = self._benchmark_random_access()
        results.extend(rand_results)

        # Pattern 3: Burst access pattern
        print("Testing burst access pattern...")
        burst_results = self._benchmark_burst_access()
        results.extend(burst_results)

        # Pattern 4: Mixed tool usage
        print("Testing mixed tool usage pattern...")
        mixed_results = self._benchmark_mixed_usage()
        results.extend(mixed_results)

        return results

    def _benchmark_sequential_access(self) -> List[BenchmarkResult]:
        """Benchmark sequential file access pattern."""
        results = []

        codebase_dir = self.base_benchmarker.create_test_codebase("medium")

        try:
            # Read files sequentially
            latencies = []
            for i in range(min(10, len(list(codebase_dir.glob("*.py"))))):
                file_path = f"{codebase_dir}/module_{i}.py"
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": i,
                    "params": {
                        "name": "read_code_file",
                        "arguments": {"file_path": file_path}
                    }
                }

                start_time = time.time()
                response = requests.post(self.config.server_url, json=request_data,
                                       timeout=self.config.timeout_seconds)
                latency = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    latencies.append(latency)

            if latencies:
                results.extend(self._create_tool_results(
                    "access_pattern_sequential",
                    latencies,
                    "Sequential file access"
                ))

        finally:
            shutil.rmtree(codebase_dir)

        return results

    def _benchmark_random_access(self) -> List[BenchmarkResult]:
        """Benchmark random file access pattern."""
        results = []

        codebase_dir = self.base_benchmarker.create_test_codebase("medium")

        try:
            # Read files in random order
            import random
            py_files = list(codebase_dir.glob("*.py"))
            random_files = random.sample(py_files, min(10, len(py_files)))

            latencies = []
            for file_path in random_files:
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": 1,
                    "params": {
                        "name": "read_code_file",
                        "arguments": {"file_path": str(file_path)}
                    }
                }

                start_time = time.time()
                response = requests.post(self.config.server_url, json=request_data,
                                       timeout=self.config.timeout_seconds)
                latency = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    latencies.append(latency)

            if latencies:
                results.extend(self._create_tool_results(
                    "access_pattern_random",
                    latencies,
                    "Random file access"
                ))

        finally:
            shutil.rmtree(codebase_dir)

        return results

    def _benchmark_burst_access(self) -> List[BenchmarkResult]:
        """Benchmark burst access pattern (multiple requests in quick succession)."""
        results = []

        codebase_dir = self.base_benchmarker.create_test_codebase("small")

        try:
            # Send multiple requests in burst
            latencies = []
            for i in range(20):  # 20 requests in burst
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": i,
                    "params": {
                        "name": "read_code_file",
                        "arguments": {"file_path": f"{codebase_dir}/module_0.py"}
                    }
                }

                start_time = time.time()
                response = requests.post(self.config.server_url, json=request_data,
                                       timeout=self.config.timeout_seconds)
                latency = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    latencies.append(latency)

            if latencies:
                results.extend(self._create_tool_results(
                    "access_pattern_burst",
                    latencies,
                    "Burst file access (20 requests)"
                ))

        finally:
            shutil.rmtree(codebase_dir)

        return results

    def _benchmark_mixed_usage(self) -> List[BenchmarkResult]:
        """Benchmark mixed tool usage pattern."""
        results = []

        codebase_dir = self.base_benchmarker.create_test_codebase("medium")

        try:
            # Mix of different tool calls
            tool_sequence = [
                ("read_code_file", {"file_path": f"{codebase_dir}/module_0.py"}),
                ("search_codebase", {
                    "codebase_path": str(codebase_dir),
                    "pattern": "def",
                    "file_types": ["*.py"]
                }),
                ("get_file_structure", {
                    "codebase_path": str(codebase_dir),
                    "file_path": f"{codebase_dir}/module_0.py"
                }),
                ("get_performance_metrics", {}),
            ]

            latencies = []
            for tool_name, args in tool_sequence * 3:  # Repeat sequence 3 times
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": 1,
                    "params": {
                        "name": tool_name,
                        "arguments": args
                    }
                }

                start_time = time.time()
                response = requests.post(self.config.server_url, json=request_data,
                                       timeout=self.config.timeout_seconds)
                latency = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    latencies.append(latency)

            if latencies:
                results.extend(self._create_tool_results(
                    "access_pattern_mixed",
                    latencies,
                    "Mixed tool usage"
                ))

        finally:
            shutil.rmtree(codebase_dir)

        return results

    def benchmark_edge_cases_and_stress(self) -> List[BenchmarkResult]:
        """Benchmark edge cases and stress scenarios."""
        results = []

        if not self.config.enable_edge_cases:
            return results

        # Edge Case 1: Very large files
        print("Testing large file handling...")
        large_results = self._benchmark_large_files()
        results.extend(large_results)

        # Edge Case 2: Concurrent access to same resource
        print("Testing concurrent access...")
        concurrent_results = self._benchmark_concurrent_access()
        results.extend(concurrent_results)

        # Edge Case 3: Invalid inputs
        print("Testing invalid inputs...")
        invalid_results = self._benchmark_invalid_inputs()
        results.extend(invalid_results)

        # Edge Case 4: Memory pressure simulation
        print("Testing memory pressure...")
        memory_results = self._benchmark_memory_pressure()
        results.extend(memory_results)

        return results

    def _benchmark_large_files(self) -> List[BenchmarkResult]:
        """Benchmark handling of very large files."""
        results = []

        # Create a large test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Create a file with ~1MB of content
            for i in range(10000):
                f.write(f"""
def function_{i}():
    \"\"\"Function {i} documentation.\"\"\"
    x = {i}
    y = x * 2
    z = y + x
    result = z ** 2
    return result

class Class_{i}:
    def __init__(self):
        self.value = {i}

    def compute(self):
        return self.value * {i}

    def get_info(self):
        return {{
            "id": self.value,
            "computed": self.compute(),
            "description": "Large class {i} with lots of methods"
        }}
""")
            large_file_path = f.name

        try:
            latencies = self._run_tool_benchmark("read_code_file", {"file_path": large_file_path})

            if latencies:
                results.extend(self._create_tool_results(
                    "edge_case_large_file",
                    latencies,
                    "Large file reading (~1MB)"
                ))

        finally:
            Path(large_file_path).unlink()

        return results

    def _benchmark_concurrent_access(self) -> List[BenchmarkResult]:
        """Benchmark concurrent access to the same resource."""
        results = []

        codebase_dir = self.base_benchmarker.create_test_codebase("small")

        try:
            def concurrent_worker(worker_id: int):
                latencies = []
                for i in range(10):
                    request_data = {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "id": f"{worker_id}_{i}",
                        "params": {
                            "name": "read_code_file",
                            "arguments": {"file_path": f"{codebase_dir}/module_0.py"}
                        }
                    }

                    start_time = time.time()
                    response = requests.post(self.config.server_url, json=request_data,
                                           timeout=self.config.timeout_seconds)
                    latency = (time.time() - start_time) * 1000

                    if response.status_code == 200:
                        latencies.append(latency)

                return latencies

            # Run concurrent workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(concurrent_worker, i) for i in range(5)]
                all_latencies = []
                for future in concurrent.futures.as_completed(futures):
                    all_latencies.extend(future.result())

            if all_latencies:
                results.extend(self._create_tool_results(
                    "edge_case_concurrent_access",
                    all_latencies,
                    "Concurrent access to same file"
                ))

        finally:
            shutil.rmtree(codebase_dir)

        return results

    def _benchmark_invalid_inputs(self) -> List[BenchmarkResult]:
        """Benchmark handling of invalid inputs."""
        results = []

        invalid_requests = [
            {
                "name": "read_code_file",
                "args": {"file_path": "/nonexistent/file.py"},
                "description": "Nonexistent file"
            },
            {
                "name": "search_codebase",
                "args": {"codebase_path": "/nonexistent/path", "pattern": "test"},
                "description": "Invalid codebase path"
            },
            {
                "name": "get_file_structure",
                "args": {"codebase_path": "/tmp", "file_path": "/nonexistent/file.py"},
                "description": "Invalid file path"
            }
        ]

        for invalid_req in invalid_requests:
            latencies = self._run_tool_benchmark(invalid_req["name"], invalid_req["args"])

            if latencies:
                results.extend(self._create_tool_results(
                    f"edge_case_invalid_{invalid_req['name']}",
                    latencies,
                    f"Invalid input: {invalid_req['description']}"
                ))

        return results

    def _benchmark_memory_pressure(self) -> List[BenchmarkResult]:
        """Benchmark performance under memory pressure."""
        results = []

        # Create multiple large codebases and test access
        latencies = []
        temp_dirs = []

        try:
            # Create 5 medium codebases
            for i in range(5):
                temp_dir = self.base_benchmarker.create_test_codebase("medium")
                temp_dirs.append(temp_dir)

                # Test search on each codebase
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": i,
                    "params": {
                        "name": "search_codebase",
                        "arguments": {
                            "codebase_path": str(temp_dir),
                            "pattern": "def",
                            "file_types": ["*.py"]
                        }
                    }
                }

                start_time = time.time()
                response = requests.post(self.config.server_url, json=request_data,
                                       timeout=self.config.timeout_seconds)
                latency = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    latencies.append(latency)

        finally:
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir)

        if latencies:
            results.extend(self._create_tool_results(
                "edge_case_memory_pressure",
                latencies,
                "Memory pressure with multiple codebases"
            ))

        return results

    def _run_tool_benchmark(self, tool_name: str, args: Dict[str, Any]) -> List[float]:
        """Run benchmark for a specific tool and return latencies."""
        latencies = []

        for i in range(self.config.test_iterations):
            request_data = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": i,
                "params": {
                    "name": tool_name,
                    "arguments": args
                }
            }

            try:
                start_time = time.time()
                response = requests.post(self.config.server_url, json=request_data,
                                       timeout=self.config.timeout_seconds)
                latency = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    latencies.append(latency)

            except Exception as e:
                print(f"Tool benchmark failed for {tool_name}: {e}")

        return latencies

    def _create_tool_results(self, test_name: str, latencies: List[float],
                           description: str) -> List[BenchmarkResult]:
        """Create benchmark results from latency data."""
        if not latencies:
            return []

        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        if len(latencies) >= 2:
            p95_latency = statistics.quantiles(latencies, n=20)[18]
        else:
            p95_latency = max_latency

        return [
            BenchmarkResult(
                test_name=test_name,
                metric_name="avg_latency",
                value=avg_latency,
                unit="milliseconds",
                target=2000.0,  # 2 seconds
                achieved=avg_latency <= 2000.0,
                metadata={"description": description, "iterations": len(latencies)}
            ),
            BenchmarkResult(
                test_name=test_name,
                metric_name="max_latency",
                value=max_latency,
                unit="milliseconds",
                target=5000.0,  # 5 seconds
                achieved=max_latency <= 5000.0,
                metadata={"description": description, "iterations": len(latencies)}
            ),
            BenchmarkResult(
                test_name=test_name,
                metric_name="p95_latency",
                value=p95_latency,
                unit="milliseconds",
                target=3000.0,  # 3 seconds
                achieved=p95_latency <= 3000.0,
                metadata={"description": description, "iterations": len(latencies)}
            )
        ]

    def run_modular_benchmark_suite(self) -> List[BenchmarkResult]:
        """Run the complete modular benchmark suite."""
        print("Starting modular MCP tools benchmark suite...")
        print(f"Server URL: {self.config.server_url}")
        print(f"Test iterations: {self.config.test_iterations}")
        print()

        all_results = []

        # Run all benchmark categories
        print("1. Benchmarking code reading tools...")
        all_results.extend(self.benchmark_code_reading_tools())

        print("2. Benchmarking search and analysis tools...")
        all_results.extend(self.benchmark_search_and_analysis_tools())

        print("3. Benchmarking performance monitoring tools...")
        all_results.extend(self.benchmark_performance_monitoring_tools())

        print("4. Benchmarking access patterns...")
        all_results.extend(self.benchmark_access_patterns())

        if self.config.enable_edge_cases:
            print("5. Benchmarking edge cases and stress scenarios...")
            all_results.extend(self.benchmark_edge_cases_and_stress())

        print(f"\nCompleted {len(all_results)} benchmark tests")
        return all_results


def run_modular_benchmarks(server_url: str = "http://localhost:8000/mcp",
                          test_iterations: int = 5) -> List[BenchmarkResult]:
    """Run the modular benchmark suite with specified configuration."""
    config = ModularBenchmarkConfig(
        server_url=server_url,
        test_iterations=test_iterations
    )

    benchmarker = ModularMCPToolBenchmarker(config)
    return benchmarker.run_modular_benchmark_suite()


if __name__ == "__main__":
    import sys

    server_url = "http://localhost:8000/mcp"
    test_iterations = 5

    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    if len(sys.argv) > 2:
        test_iterations = int(sys.argv[2])

    print("CodeSage MCP Modular Tools Benchmark Suite")
    print("=" * 50)

    results = run_modular_benchmarks(server_url, test_iterations)

    # Print summary
    passed = sum(1 for r in results if r.achieved)
    total = len(results)

    print("\nSUMMARY:")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    # Print detailed results
    print("\nDETAILED RESULTS:")
    for result in results:
        status = "✓ PASS" if result.achieved else "✗ FAIL"
        print(f"{status} {result.test_name} - {result.metric_name}: {result.value:.2f} {result.unit}")
        if not result.achieved and result.target:
            print(f"      Target: {result.target} {result.unit}")
#!/usr/bin/env python3
"""
Performance Benchmarks for Gemini CLI Compatibility Layer

This module provides comprehensive performance benchmarks for the GeminiCompatibilityHandler,
measuring response adaptation times, memory usage, and throughput under load.
"""

import pytest
from memory_profiler import memory_usage
import time
import psutil
import os

from codesage_mcp.core.gemini_compatibility import (
    GeminiCompatibilityHandler,
    ResponseFormat
)

# Hardware-adaptive imports
from tests.hardware_utils import (
    get_hardware_profile,
    get_adaptive_config,
    log_system_info
)


class TestPerformanceBenchmarks:
    """Performance benchmarks for GeminiCompatibilityHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

        # Hardware-adaptive configuration
        self.hardware_profile = get_hardware_profile()
        self.adaptive_config = get_adaptive_config(self.hardware_profile)

        # Adaptive scaling based on hardware profile
        if self.hardware_profile == 'light':
            self.adaptive_tools_count = 100
            self.adaptive_thread_count = 2
            self.adaptive_payload_size = 1000  # 1KB
        elif self.hardware_profile == 'medium':
            self.adaptive_tools_count = 500
            self.adaptive_thread_count = 5
            self.adaptive_payload_size = 5000  # 5KB
        else:  # full
            self.adaptive_tools_count = 1000
            self.adaptive_thread_count = 10
            self.adaptive_payload_size = 10000  # 10KB

        log_system_info()

    def test_detect_response_format_performance(self, benchmark, gemini_cli_headers, tools_list_request_body):
        """Benchmark detect_response_format performance."""
        def run_detection():
            return self.handler.detect_response_format(gemini_cli_headers, tools_list_request_body)

        result = benchmark(run_detection)
        assert result == ResponseFormat.GEMINI_ARRAY_TOOLS

    def test_adapt_tools_response_performance(self, benchmark, sample_tools_object):
        """Benchmark adapt_tools_response performance."""
        def run_adaptation():
            return self.handler.adapt_tools_response(sample_tools_object, ResponseFormat.GEMINI_ARRAY_TOOLS)

        result = benchmark(run_adaptation)
        assert "tools" in result
        assert isinstance(result["tools"], list)

    def test_adapt_error_response_performance(self, benchmark, sample_error_response):
        """Benchmark adapt_error_response performance."""
        def run_adaptation():
            return self.handler.adapt_error_response(sample_error_response, ResponseFormat.GEMINI_NUMERIC_ERRORS)

        result = benchmark(run_adaptation)
        assert result["code"] == -32602  # INVALID_PARAMS

    def test_create_compatible_response_performance(self, benchmark, gemini_cli_headers, tools_list_request_body, sample_tools_object):
        """Benchmark create_compatible_response performance."""
        def run_creation():
            return self.handler.create_compatible_response(
                result={"tools": sample_tools_object},
                request_headers=gemini_cli_headers,
                request_body=tools_list_request_body
            )

        result = benchmark(run_creation)
        assert "tools" in result

    def test_large_tools_adaptation_performance(self, benchmark, large_tools_object):
        """Benchmark adaptation of large tools objects."""
        def run_large_adaptation():
            return self.handler.adapt_tools_response(large_tools_object, ResponseFormat.GEMINI_ARRAY_TOOLS)

        result = benchmark(run_large_adaptation)
        assert len(result["tools"]) == self.adaptive_tools_count

    def test_concurrent_requests_performance(self, benchmark):
        """Benchmark handling of concurrent requests."""
        import threading

        def worker():
            for _ in range(100):
                self.handler.detect_response_format(
                    {"user-agent": "node"},
                    {"method": "tools/list"}
                )

        def run_concurrent():
            threads = []
            for _ in range(self.adaptive_thread_count):
                t = threading.Thread(target=worker)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

        benchmark(run_concurrent)


class TestMemoryUsageBenchmarks:
    """Memory usage benchmarks for GeminiCompatibilityHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

        # Hardware-adaptive configuration
        self.hardware_profile = get_hardware_profile()
        self.adaptive_config = get_adaptive_config(self.hardware_profile)

        # Adaptive scaling based on hardware profile
        if self.hardware_profile == 'light':
            self.adaptive_tools_count = 100
            self.adaptive_thread_count = 2
            self.adaptive_payload_size = 1000  # 1KB
        elif self.hardware_profile == 'medium':
            self.adaptive_tools_count = 500
            self.adaptive_thread_count = 5
            self.adaptive_payload_size = 5000  # 5KB
        else:  # full
            self.adaptive_tools_count = 1000
            self.adaptive_thread_count = 10
            self.adaptive_payload_size = 10000  # 10KB

    def test_memory_usage_detect_response_format(self, gemini_cli_headers, tools_list_request_body):
        """Test memory usage for detect_response_format."""
        def operation():
            return self.handler.detect_response_format(gemini_cli_headers, tools_list_request_body)

        mem_usage = memory_usage(operation, interval=0.1)
        max_memory = max(mem_usage)

        # Should use less than 10MB
        assert max_memory < 10

    def test_memory_usage_large_tools_adaptation(self, large_tools_object):
        """Test memory usage for large tools adaptation."""
        def operation():
            return self.handler.adapt_tools_response(large_tools_object, ResponseFormat.GEMINI_ARRAY_TOOLS)

        mem_usage = memory_usage(operation, interval=0.1)
        max_memory = max(mem_usage)

        # Adaptive memory limits based on hardware profile
        if self.hardware_profile == 'light':
            memory_limit = 10  # MB for 100 tools
        elif self.hardware_profile == 'medium':
            memory_limit = 25  # MB for 500 tools
        else:  # full
            memory_limit = 50  # MB for 1000 tools

        assert max_memory < memory_limit

    def test_memory_usage_request_history(self):
        """Test memory usage with request history accumulation."""
        def operation():
            for i in range(20):
                self.handler.detect_response_format(
                    {"user-agent": "node"},
                    {"method": f"test{i}", "params": {"data": "x" * 1000}}
                )

        mem_usage = memory_usage(operation, interval=0.1)
        max_memory = max(mem_usage)

        # Should use less than 20MB
        assert max_memory < 20

    def test_memory_efficiency_repeated_operations(self, sample_tools_object):
        """Test memory efficiency of repeated operations."""
        def operation():
            for _ in range(1000):
                self.handler.adapt_tools_response(sample_tools_object, ResponseFormat.GEMINI_ARRAY_TOOLS)

        mem_usage = memory_usage(operation, interval=0.5)
        max_memory = max(mem_usage)

        # Should use less than 30MB for 1000 operations
        assert max_memory < 30


class TestThroughputBenchmarks:
    """Throughput benchmarks for high-load scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

        # Hardware-adaptive configuration
        self.hardware_profile = get_hardware_profile()
        self.adaptive_config = get_adaptive_config(self.hardware_profile)

        # Adaptive scaling based on hardware profile
        if self.hardware_profile == 'light':
            self.adaptive_tools_count = 100
            self.adaptive_thread_count = 2
            self.adaptive_payload_size = 1000  # 1KB
        elif self.hardware_profile == 'medium':
            self.adaptive_tools_count = 500
            self.adaptive_thread_count = 5
            self.adaptive_payload_size = 5000  # 5KB
        else:  # full
            self.adaptive_tools_count = 1000
            self.adaptive_thread_count = 10
            self.adaptive_payload_size = 10000  # 10KB

    def test_throughput_small_requests(self, benchmark):
        """Test throughput for small requests."""
        requests = [
            ({"user-agent": "node"}, {"method": "tools/list", "id": i})
            for i in range(100)
        ]

        def process_requests():
            results = []
            for headers, body in requests:
                result = self.handler.detect_response_format(headers, body)
                results.append(result)
            return results

        results = benchmark(process_requests)
        assert len(results) == 100
        assert all(r == ResponseFormat.GEMINI_ARRAY_TOOLS for r in results)

    def test_throughput_mixed_request_types(self, benchmark):
        """Test throughput for mixed request types."""
        requests = []
        for i in range(100):
            if i % 3 == 0:
                headers = {"user-agent": "node"}
                body = {"method": "tools/list", "id": i}
            elif i % 3 == 1:
                headers = {"user-agent": "node"}
                body = {"method": "tools/call", "id": i}
            else:
                headers = {"user-agent": "curl"}
                body = {"method": "initialize", "id": i}
            requests.append((headers, body))

        def process_mixed_requests():
            results = []
            for headers, body in requests:
                result = self.handler.detect_response_format(headers, body)
                results.append(result)
            return results

        results = benchmark(process_mixed_requests)
        assert len(results) == 100

    def test_throughput_large_payloads(self, benchmark):
        """Test throughput for large payloads."""
        large_requests = [
            (
                {"user-agent": "node"},
                {
                    "method": "tools/list",
                    "id": i,
                    "params": {"data": "x" * self.adaptive_payload_size}
                }
            )
            for i in range(50)
        ]

        def process_large_requests():
            results = []
            for headers, body in large_requests:
                result = self.handler.detect_response_format(headers, body)
                results.append(result)
            return results

        results = benchmark(process_large_requests)
        assert len(results) == 50
        assert all(r == ResponseFormat.GEMINI_ARRAY_TOOLS for r in results)


class TestScalabilityBenchmarks:
    """Scalability benchmarks for different data sizes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

        # Hardware-adaptive configuration
        self.hardware_profile = get_hardware_profile()
        self.adaptive_config = get_adaptive_config(self.hardware_profile)

        # Adaptive scaling based on hardware profile
        if self.hardware_profile == 'light':
            self.adaptive_tools_count = 100
            self.adaptive_thread_count = 2
            self.adaptive_payload_size = 1000  # 1KB
        elif self.hardware_profile == 'medium':
            self.adaptive_tools_count = 500
            self.adaptive_thread_count = 5
            self.adaptive_payload_size = 5000  # 5KB
        else:  # full
            self.adaptive_tools_count = 1000
            self.adaptive_thread_count = 10
            self.adaptive_payload_size = 10000  # 10KB

    @pytest.mark.parametrize("num_tools", [10, 100, 1000, 5000])
    def test_scalability_tools_adaptation(self, benchmark, num_tools):
        """Test scalability of tools adaptation with different sizes."""
        # Skip large tool counts on lower-end hardware
        if self.hardware_profile == 'light' and num_tools > self.adaptive_tools_count:
            pytest.skip(f"Skipping {num_tools} tools test on light hardware (max: {self.adaptive_tools_count})")
        elif self.hardware_profile == 'medium' and num_tools > self.adaptive_tools_count:
            pytest.skip(f"Skipping {num_tools} tools test on medium hardware (max: {self.adaptive_tools_count})")

        tools = {
            f"tool_{i}": {
                "name": f"tool_{i}",
                "description": f"Description for tool {i}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "param": {"type": "string"}
                    }
                }
            }
            for i in range(num_tools)
        }

        def adapt_tools():
            return self.handler.adapt_tools_response(tools, ResponseFormat.GEMINI_ARRAY_TOOLS)

        result = benchmark(adapt_tools)
        assert len(result["tools"]) == num_tools

    @pytest.mark.parametrize("data_size", [1000, 10000, 100000])
    def test_scalability_request_processing(self, benchmark, data_size):
        """Test scalability of request processing with different data sizes."""
        large_body = {
            "method": "tools/list",
            "params": {"data": "x" * data_size}
        }

        def process_large_request():
            return self.handler.detect_response_format({"user-agent": "node"}, large_body)

        result = benchmark(process_large_request)
        assert result == ResponseFormat.GEMINI_ARRAY_TOOLS

    def test_scalability_concurrent_users(self, benchmark):
        """Test scalability with simulated concurrent users."""
        import concurrent.futures

        def simulate_user_requests(user_id):
            results = []
            for i in range(50):
                headers = {"user-agent": f"node/{user_id}"}
                body = {"method": "tools/list", "id": f"{user_id}_{i}"}
                result = self.handler.detect_response_format(headers, body)
                results.append(result)
            return results

        def run_concurrent_users():
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.adaptive_thread_count) as executor:
                futures = [executor.submit(simulate_user_requests, i) for i in range(self.adaptive_thread_count)]
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.extend(future.result())
                return results

        results = benchmark(run_concurrent_users)
        expected_results = self.adaptive_thread_count * 50
        assert len(results) == expected_results
        assert all(r == ResponseFormat.GEMINI_ARRAY_TOOLS for r in results)


class TestResourceUtilizationBenchmarks:
    """Resource utilization benchmarks."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GeminiCompatibilityHandler()

        # Hardware-adaptive configuration
        self.hardware_profile = get_hardware_profile()
        self.adaptive_config = get_adaptive_config(self.hardware_profile)

        # Adaptive scaling based on hardware profile
        if self.hardware_profile == 'light':
            self.adaptive_tools_count = 100
            self.adaptive_thread_count = 2
            self.adaptive_payload_size = 1000  # 1KB
        elif self.hardware_profile == 'medium':
            self.adaptive_tools_count = 500
            self.adaptive_thread_count = 5
            self.adaptive_payload_size = 5000  # 5KB
        else:  # full
            self.adaptive_tools_count = 1000
            self.adaptive_thread_count = 10
            self.adaptive_payload_size = 10000  # 10KB

    def test_cpu_usage_during_load(self):
        """Test CPU usage during high load."""
        process = psutil.Process(os.getpid())
        initial_cpu = process.cpu_percent(interval=1)

        # Adaptive load generation based on hardware profile
        if self.hardware_profile == 'light':
            load_iterations = 200
            cpu_limit = 90  # Allow higher CPU for light hardware
        elif self.hardware_profile == 'medium':
            load_iterations = 500
            cpu_limit = 85
        else:  # full
            load_iterations = 1000
            cpu_limit = 80

        # Generate load
        for _ in range(load_iterations):
            self.handler.detect_response_format({"user-agent": "node"}, {"method": "tools/list"})

        final_cpu = process.cpu_percent(interval=1)

        # CPU usage should be reasonable
        assert final_cpu < cpu_limit

    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        import gc

        # Get initial memory
        initial_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # Adaptive operation count based on hardware profile
        if self.hardware_profile == 'light':
            operation_count = 2000
            memory_limit = 10  # MB increase
        elif self.hardware_profile == 'medium':
            operation_count = 5000
            memory_limit = 25  # MB increase
        else:  # full
            operation_count = 10000
            memory_limit = 50  # MB increase

        # Perform many operations
        for i in range(operation_count):
            self.handler.detect_response_format({"user-agent": "node"}, {"method": f"test{i}"})
            if i % 1000 == 0:
                gc.collect()  # Force garbage collection

        # Get final memory
        final_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # Memory increase should be reasonable
        assert (final_mem - initial_mem) < memory_limit

    def test_response_time_distribution(self, benchmark):
        """Test response time distribution under load."""
        times = []

        def timed_operation():
            start = time.perf_counter()
            result = self.handler.detect_response_format({"user-agent": "node"}, {"method": "tools/list"})
            end = time.perf_counter()
            times.append(end - start)
            return result

        # Run multiple times
        for _ in range(100):
            result = benchmark(timed_operation)
            assert result == ResponseFormat.GEMINI_ARRAY_TOOLS

        # Calculate statistics
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)

        # Performance assertions
        assert avg_time < 0.001  # Less than 1ms average
        assert max_time < 0.01   # Less than 10ms max
        assert min_time > 0       # Should be positive


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
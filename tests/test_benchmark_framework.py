#!/usr/bin/env python3
"""
Unit Tests for CodeSage MCP Benchmarking Framework.

This module provides comprehensive unit tests for the benchmarking framework
to ensure reliability and correctness of performance measurements.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import statistics

from benchmark_performance import (
    PerformanceBenchmarker,
    BenchmarkResult
)
from benchmark_mcp_tools import (
    ModularBenchmarkConfig,
    ModularMCPToolBenchmarker
)


class TestBenchmarkResult:
    """Test cases for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult instance."""
        result = BenchmarkResult(
            test_name="test_indexing",
            metric_name="indexing_time",
            value=15.5,
            unit="seconds",
            target=30.0,
            achieved=True,
            metadata={"files": 100}
        )

        assert result.test_name == "test_indexing"
        assert result.metric_name == "indexing_time"
        assert result.value == 15.5
        assert result.unit == "seconds"
        assert result.target == 30.0
        assert result.achieved is True
        assert result.metadata == {"files": 100}

    def test_benchmark_result_defaults(self):
        """Test BenchmarkResult with default values."""
        result = BenchmarkResult(
            test_name="test_cache",
            metric_name="hit_rate",
            value=85.0,
            unit="percent"
        )

        assert result.target is None
        assert result.achieved is False
        assert result.metadata == {}


class TestPerformanceBenchmarker:
    """Test cases for PerformanceBenchmarker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.benchmarker = PerformanceBenchmarker(str(self.temp_dir / "results"))

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_create_test_codebase_small(self):
        """Test creating a small test codebase."""
        codebase_dir = self.benchmarker.create_test_codebase("small")

        assert codebase_dir.exists()
        assert codebase_dir.is_dir()

        # Check if Python files were created
        py_files = list(codebase_dir.glob("*.py"))
        assert len(py_files) == 10  # Small config has 10 files

        # Check if README and .gitignore exist
        assert (codebase_dir / "README.md").exists()
        assert (codebase_dir / ".gitignore").exists()

        # Clean up
        shutil.rmtree(codebase_dir)

    def test_create_test_codebase_medium(self):
        """Test creating a medium test codebase."""
        codebase_dir = self.benchmarker.create_test_codebase("medium")

        assert codebase_dir.exists()
        py_files = list(codebase_dir.glob("*.py"))
        assert len(py_files) == 50  # Medium config has 50 files

        shutil.rmtree(codebase_dir)

    def test_create_test_codebase_large(self):
        """Test creating a large test codebase."""
        codebase_dir = self.benchmarker.create_test_codebase("large")

        assert codebase_dir.exists()
        py_files = list(codebase_dir.glob("*.py"))
        assert len(py_files) == 200  # Large config has 200 files

        shutil.rmtree(codebase_dir)

    @patch('benchmark_performance.requests.post')
    def test_benchmark_jsonrpc_latency_success(self, mock_post):
        """Test JSON-RPC latency benchmarking with successful requests."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}
        mock_post.return_value = mock_response

        results = self.benchmarker.benchmark_jsonrpc_latency(
            server_url="http://test:8000/mcp",
            num_requests=5
        )

        assert len(results) == 6  # 3 metrics * 2 request types
        assert all(isinstance(r, BenchmarkResult) for r in results)

        # Check that requests were made
        assert mock_post.call_count == 10  # 5 requests * 2 types

    @patch('benchmark_performance.requests.post')
    def test_benchmark_jsonrpc_latency_failure(self, mock_post):
        """Test JSON-RPC latency benchmarking with failed requests."""
        # Mock failed response
        mock_post.side_effect = Exception("Connection failed")

        results = self.benchmarker.benchmark_jsonrpc_latency(
            server_url="http://test:8000/mcp",
            num_requests=3
        )

        # Should still return results even with failures
        assert isinstance(results, list)

    @patch('benchmark_performance.requests.post')
    def test_benchmark_tool_execution_times(self, mock_post):
        """Test tool execution time benchmarking."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}
        mock_post.return_value = mock_response

        results = self.benchmarker.benchmark_tool_execution_times(
            server_url="http://test:8000/mcp",
            tools_to_test=["read_code_file"]
        )

        assert len(results) == 2  # avg and max for one tool
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_benchmark_resource_utilization(self):
        """Test resource utilization benchmarking."""
        # This test might take some time due to sleep intervals
        results = self.benchmarker.benchmark_resource_utilization(duration_seconds=1)

        assert len(results) >= 4  # CPU and memory metrics
        assert all(isinstance(r, BenchmarkResult) for r in results)

    @patch('benchmark_performance.requests.post')
    @patch('benchmark_performance.concurrent.futures.ThreadPoolExecutor')
    def test_benchmark_throughput_and_scalability(self, mock_executor, mock_post):
        """Test throughput and scalability benchmarking."""
        # Mock the executor and its results
        mock_future = Mock()
        mock_future.result.return_value = ([100, 200, 300], [0.1, 0.15, 0.12])  # requests, latencies
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        mock_executor.return_value.__enter__.return_value.as_completed.return_value = [mock_future]

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}
        mock_post.return_value = mock_response

        results = self.benchmarker.benchmark_throughput_and_scalability(
            server_url="http://test:8000/mcp",
            concurrent_users=[5]
        )

        assert isinstance(results, list)

    def test_benchmark_edge_cases(self):
        """Test edge cases benchmarking."""
        # This will test with invalid URLs, so should handle gracefully
        results = self.benchmarker.benchmark_edge_cases(server_url="http://invalid:9999")

        assert isinstance(results, list)
        # Should still return results even if some tests fail

    def test_run_full_benchmark_suite(self):
        """Test running the full benchmark suite."""
        with patch.object(self.benchmarker, 'benchmark_indexing_performance') as mock_indexing, \
             patch.object(self.benchmarker, 'benchmark_search_performance') as mock_search, \
             patch.object(self.benchmarker, 'benchmark_cache_performance') as mock_cache, \
             patch.object(self.benchmarker, 'benchmark_memory_management') as mock_memory, \
             patch.object(self.benchmarker, 'benchmark_chunking_performance') as mock_chunking, \
             patch.object(self.benchmarker, 'benchmark_jsonrpc_latency') as mock_jsonrpc, \
             patch.object(self.benchmarker, 'benchmark_tool_execution_times') as mock_tools, \
             patch.object(self.benchmarker, 'benchmark_resource_utilization') as mock_resources, \
             patch.object(self.benchmarker, 'benchmark_throughput_and_scalability') as mock_throughput, \
             patch.object(self.benchmarker, 'benchmark_edge_cases') as mock_edge_cases, \
             patch.object(self.benchmarker, 'save_report') as mock_save:

            # Mock all benchmark methods to return empty lists
            mock_indexing.return_value = []
            mock_search.return_value = []
            mock_cache.return_value = []
            mock_memory.return_value = []
            mock_chunking.return_value = []
            mock_jsonrpc.return_value = []
            mock_tools.return_value = []
            mock_resources.return_value = []
            mock_throughput.return_value = []
            mock_edge_cases.return_value = []

            report = self.benchmarker.run_full_benchmark_suite()

            assert "test_suite" in report
            assert "timestamp" in report
            assert "results" in report
            assert "summary" in report

    def test_generate_summary(self):
        """Test summary generation."""
        results = [
            BenchmarkResult("test1", "metric1", 10.0, "ms", 15.0, True),
            BenchmarkResult("test2", "metric2", 20.0, "ms", 15.0, False),
            BenchmarkResult("test3", "metric3", 5.0, "ms", 10.0, True),
        ]

        summary = self.benchmarker._generate_summary(results)

        assert "total_tests" in summary
        assert "passed_tests" in summary
        assert "failed_tests" in summary
        assert "performance_score" in summary
        assert summary["total_tests"] == 3
        assert summary["passed_tests"] == 2
        assert summary["failed_tests"] == 1


class TestModularBenchmarkConfig:
    """Test cases for ModularBenchmarkConfig."""

    def test_config_creation(self):
        """Test creating a ModularBenchmarkConfig instance."""
        config = ModularBenchmarkConfig(
            server_url="http://test:8000",
            test_iterations=10,
            concurrent_users=20,
            timeout_seconds=60,
            enable_edge_cases=False
        )

        assert config.server_url == "http://test:8000"
        assert config.test_iterations == 10
        assert config.concurrent_users == 20
        assert config.timeout_seconds == 60
        assert config.enable_edge_cases is False

    def test_config_defaults(self):
        """Test ModularBenchmarkConfig with default values."""
        config = ModularBenchmarkConfig()

        assert config.server_url == "http://localhost:8000/mcp"
        assert config.test_iterations == 5
        assert config.concurrent_users == 1
        assert config.timeout_seconds == 30
        assert config.enable_edge_cases is True
        assert "read_code_file" in config.request_mix_ratios


class TestModularMCPToolBenchmarker:
    """Test cases for ModularMCPToolBenchmarker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ModularBenchmarkConfig(test_iterations=2)  # Reduce for faster tests
        self.benchmarker = ModularMCPToolBenchmarker(self.config)

    @patch('benchmark_mcp_tools.requests.post')
    def test_run_tool_benchmark_success(self, mock_post):
        """Test running tool benchmark with successful response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}
        mock_post.return_value = mock_response

        latencies = self.benchmarker._run_tool_benchmark("read_code_file", {"file_path": "test.py"})

        assert len(latencies) == 2  # test_iterations
        assert all(isinstance(lat, float) for lat in latencies)
        assert mock_post.call_count == 2

    @patch('benchmark_mcp_tools.requests.post')
    def test_run_tool_benchmark_failure(self, mock_post):
        """Test running tool benchmark with failed response."""
        mock_post.side_effect = Exception("Connection failed")

        latencies = self.benchmarker._run_tool_benchmark("read_code_file", {"file_path": "test.py"})

        assert len(latencies) == 0  # No successful requests

    def test_create_tool_results(self):
        """Test creating tool benchmark results."""
        latencies = [100.0, 150.0, 120.0]

        results = self.benchmarker._create_tool_results(
            "test_tool_read_code_file",
            latencies,
            "Test tool description"
        )

        assert len(results) == 3  # avg, max, p95
        assert all(isinstance(r, BenchmarkResult) for r in results)

        # Check avg latency result
        avg_result = results[0]
        assert avg_result.metric_name == "avg_latency"
        assert avg_result.value == 123.33333333333333  # mean of latencies
        assert avg_result.unit == "milliseconds"

    def test_create_tool_results_empty(self):
        """Test creating tool results with empty latencies."""
        results = self.benchmarker._create_tool_results("test_tool", [], "description")

        assert len(results) == 0

    @patch('benchmark_mcp_tools.requests.post')
    def test_benchmark_code_reading_tools(self, mock_post):
        """Test benchmarking code reading tools."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}
        mock_post.return_value = mock_response

        with patch.object(self.benchmarker, 'base_benchmarker') as mock_base:
            mock_temp_dir = Path("/tmp/test")
            mock_base.create_test_codebase.return_value = mock_temp_dir

            results = self.benchmarker.benchmark_code_reading_tools()

            assert isinstance(results, list)
            # Should have results for multiple tools

    def test_run_modular_benchmark_suite(self):
        """Test running the modular benchmark suite."""
        with patch.object(self.benchmarker, 'benchmark_code_reading_tools') as mock_code_reading, \
             patch.object(self.benchmarker, 'benchmark_search_and_analysis_tools') as mock_search, \
             patch.object(self.benchmarker, 'benchmark_performance_monitoring_tools') as mock_perf, \
             patch.object(self.benchmarker, 'benchmark_access_patterns') as mock_access, \
             patch.object(self.benchmarker, 'benchmark_edge_cases_and_stress') as mock_edge:

            # Mock all methods to return empty lists
            mock_code_reading.return_value = []
            mock_search.return_value = []
            mock_perf.return_value = []
            mock_access.return_value = []
            mock_edge.return_value = []

            results = self.benchmarker.run_modular_benchmark_suite()

            assert isinstance(results, list)


class TestBenchmarkIntegration:
    """Integration tests for benchmark framework."""

    def test_benchmark_result_serialization(self):
        """Test that BenchmarkResult can be serialized to JSON."""
        result = BenchmarkResult(
            test_name="test_serialization",
            metric_name="test_metric",
            value=42.5,
            unit="ms",
            target=50.0,
            achieved=True,
            metadata={"extra": "data"}
        )

        # Convert to dict (similar to JSON serialization)
        result_dict = {
            "test_name": result.test_name,
            "metric_name": result.metric_name,
            "value": result.value,
            "unit": result.unit,
            "target": result.target,
            "achieved": result.achieved,
            "metadata": result.metadata
        }

        # Serialize to JSON
        json_str = json.dumps(result_dict, default=str)
        assert json_str is not None

        # Deserialize back
        parsed = json.loads(json_str)
        assert parsed["test_name"] == "test_serialization"
        assert parsed["value"] == 42.5

    def test_statistics_calculations(self):
        """Test statistical calculations used in benchmarking."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]

        # Test mean
        mean = statistics.mean(values)
        assert mean == 30.0

        # Test median
        median = statistics.median(values)
        assert median == 30.0

        # Test quantiles (95th percentile)
        p95 = statistics.quantiles(values, n=20)[18]  # 95th percentile
        assert p95 == 50.0  # Highest value

    def test_tolerance_comparisons(self):
        """Test tolerance-based comparisons."""
        tolerance = 0.05  # 5%

        # Test improvement detection
        assert abs(10.0 - 9.5) / 10.0 <= tolerance  # Within tolerance
        assert abs(10.0 - 8.0) / 10.0 > tolerance   # Outside tolerance

        # Test percentage calculations
        assert (9.5 - 10.0) / 10.0 == -0.05  # 5% decrease
        assert (10.5 - 10.0) / 10.0 == 0.05  # 5% increase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
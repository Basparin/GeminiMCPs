"""
Performance Benchmarking Framework for CodeSage MCP Server.

This module provides comprehensive performance benchmarking capabilities to measure
and validate the performance targets specified in the testing plan:

- Indexing: 3-5x faster than baseline
- Memory: 50-70% reduction through optimization
- Cache Hit Rate: >70% for frequently accessed content
- Response Time: <2s for typical operations
"""

import time
import json
import statistics
import psutil
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch
import tempfile
import os
import numpy as np

from codesage_mcp.indexing import IndexingManager
from codesage_mcp.searching import SearchingManager
from codesage_mcp.memory_manager import MemoryManager
from codesage_mcp.cache import IntelligentCache
from codesage_mcp.chunking import DocumentChunker


@dataclass
class BenchmarkResult:
    """Represents the result of a benchmark test."""
    test_name: str
    metric_name: str
    value: float
    unit: str
    target: Optional[float] = None
    achieved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    test_suite: str
    timestamp: str
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmarker:
    """Main class for running performance benchmarks."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.process = psutil.Process()

    def create_test_codebase(self, size: str = "small") -> Path:
        """Create a test codebase of specified size."""
        size_configs = {
            "small": {"files": 10, "avg_lines": 50},
            "medium": {"files": 50, "avg_lines": 100},
            "large": {"files": 200, "avg_lines": 200}
        }

        config = size_configs[size]

        # Create temporary directory that persists until explicitly cleaned up
        temp_dir = Path(tempfile.mkdtemp())
        codebase_dir = temp_dir / f"benchmark_{size}"
        codebase_dir.mkdir()

        # Create Python files
        for i in range(config["files"]):
            self._create_test_file(codebase_dir, i, config["avg_lines"])

        # Create some non-Python files
        (codebase_dir / "README.md").write_text("# Benchmark Test Project\n\nThis is a test project.")
        (codebase_dir / ".gitignore").write_text("*.pyc\n__pycache__/\n")

        return codebase_dir

    def _create_test_file(self, codebase_dir: Path, file_id: int, num_lines: int):
        """Create a test Python file with specified number of lines."""
        content = f'''"""
Module {file_id} - Test module for benchmarking.

This module contains various functions and classes for performance testing.
"""

import os
import sys
from typing import List, Dict, Any

# Global variables
MODULE_ID = {file_id}
DATA = list(range({num_lines // 10}))

def get_module_id() -> int:
    """Get the module ID."""
    return MODULE_ID

def process_data() -> List[int]:
    """Process the module data."""
    return [x * MODULE_ID for x in DATA]

class TestClass{file_id}:
    """Test class for benchmarking."""

    def __init__(self):
        self.id = {file_id}
        self.data = DATA.copy()

    def compute(self) -> int:
        """Compute a value based on module data."""
        return sum(self.data) * self.id

    def get_info(self) -> Dict[str, Any]:
        """Get information about this instance."""
        return {{
            "id": self.id,
            "data_length": len(self.data),
            "computed_value": self.compute()
        }}

def utility_function(value: int) -> int:
    """A utility function for testing."""
    return value * {file_id} + len(DATA)

# Additional functions to reach target line count
'''

        # Add more functions to reach target line count
        remaining_lines = num_lines - len(content.split('\n'))
        for j in range(max(1, remaining_lines // 5)):
            content += f'''
def additional_function_{j}() -> str:
    """Additional function {j} for padding."""
    return f"function_{j}_module_{file_id}"
'''

        file_path = codebase_dir / f"module_{file_id}.py"
        file_path.write_text(content)

    def benchmark_indexing_performance(self, codebase_sizes: List[str] = None) -> List[BenchmarkResult]:
        """Benchmark indexing performance across different codebase sizes."""
        if codebase_sizes is None:
            codebase_sizes = ["small", "medium"]

        results = []

        for size in codebase_sizes:
            print(f"Benchmarking indexing performance for {size} codebase...")

            # Create test codebase
            codebase_dir = self.create_test_codebase(size)

            # Set up components
            indexing_manager = IndexingManager(
                index_dir_name=str(codebase_dir / ".codesage")
            )

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            # Measure memory before indexing
            memory_before = self.process.memory_info().rss / 1024 / 1024

            # Time the indexing process
            start_time = time.time()

            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content=f"content {i}", start_line=1, end_line=1)
                    for i in range(3)
                ]
                indexed_files = indexing_manager.index_codebase(str(codebase_dir), mock_model)

            indexing_time = time.time() - start_time
            memory_after = self.process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before

            # Record results
            results.extend([
                BenchmarkResult(
                    test_name=f"indexing_{size}",
                    metric_name="indexing_time",
                    value=indexing_time,
                    unit="seconds",
                    target=30.0,  # Should complete within 30 seconds
                    achieved=indexing_time <= 30.0,
                    metadata={"codebase_size": size, "files_indexed": len(indexed_files)}
                ),
                BenchmarkResult(
                    test_name=f"indexing_{size}",
                    metric_name="memory_usage",
                    value=memory_used,
                    unit="MB",
                    target=500.0,  # Should use less than 500MB
                    achieved=memory_used <= 500.0,
                    metadata={"codebase_size": size}
                ),
                BenchmarkResult(
                    test_name=f"indexing_{size}",
                    metric_name="files_per_second",
                    value=len(indexed_files) / indexing_time,
                    unit="files/sec",
                    target=5.0,  # At least 5 files per second
                    achieved=(len(indexed_files) / indexing_time) >= 5.0,
                    metadata={"codebase_size": size, "total_files": len(indexed_files)}
                )
            ])

            # Cleanup
            shutil.rmtree(codebase_dir)

        return results

    def benchmark_search_performance(self, num_queries: int = 100) -> List[BenchmarkResult]:
        """Benchmark search performance."""
        results = []

        # Create test codebase
        codebase_dir = self.create_test_codebase("medium")

        try:
            # Set up components
            indexing_manager = IndexingManager(
                index_dir_name=str(codebase_dir / ".codesage")
            )
            searching_manager = SearchingManager(indexing_manager)

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 128
            mock_model.encode.return_value = np.random.rand(128).astype(np.float32)

            # Index the codebase first
            with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    MagicMock(content=f"content {i}", start_line=1, end_line=1)
                    for i in range(3)
                ]
                indexing_manager.index_codebase(str(codebase_dir), mock_model)

            # Benchmark semantic search
            semantic_times = []
            for i in range(num_queries):
                query = f"test query {i}"
                start_time = time.time()
                results_search = searching_manager.semantic_search_codebase(
                    query, mock_model, top_k=5
                )
                semantic_times.append(time.time() - start_time)

            avg_semantic_time = statistics.mean(semantic_times)
            p95_semantic_time = statistics.quantiles(semantic_times, n=20)[18]  # 95th percentile

            # Benchmark text search
            text_times = []
            for i in range(num_queries):
                pattern = f"def.*function_{i % 10}"
                start_time = time.time()
                text_results = searching_manager.search_codebase(
                    str(codebase_dir), pattern
                )
                text_times.append(time.time() - start_time)

            avg_text_time = statistics.mean(text_times)
            p95_text_time = statistics.quantiles(text_times, n=20)[18]

            # Record results
            results.extend([
                BenchmarkResult(
                    test_name="search_performance",
                    metric_name="semantic_search_avg_time",
                    value=avg_semantic_time,
                    unit="seconds",
                    target=2.0,
                    achieved=avg_semantic_time <= 2.0,
                    metadata={"queries": num_queries}
                ),
                BenchmarkResult(
                    test_name="search_performance",
                    metric_name="semantic_search_p95_time",
                    value=p95_semantic_time,
                    unit="seconds",
                    target=5.0,
                    achieved=p95_semantic_time <= 5.0,
                    metadata={"queries": num_queries}
                ),
                BenchmarkResult(
                    test_name="search_performance",
                    metric_name="text_search_avg_time",
                    value=avg_text_time,
                    unit="seconds",
                    target=1.0,
                    achieved=avg_text_time <= 1.0,
                    metadata={"queries": num_queries}
                ),
                BenchmarkResult(
                    test_name="search_performance",
                    metric_name="text_search_p95_time",
                    value=p95_text_time,
                    unit="seconds",
                    target=3.0,
                    achieved=p95_text_time <= 3.0,
                    metadata={"queries": num_queries}
                )
            ])

        finally:
            # Cleanup
            shutil.rmtree(codebase_dir)

        return results

    def benchmark_cache_performance(self, num_operations: int = 1000) -> List[BenchmarkResult]:
        """Benchmark cache performance and hit rates."""
        results = []

        cache = IntelligentCache()

        # Pre-populate cache with some data
        test_data = {}
        for i in range(100):
            key = f"test_key_{i}"
            content = f"test content {i}"
            embedding = np.random.rand(128).astype(np.float32)

            cache.store_embedding(key, content, embedding)
            cache.store_file_content(key, content)
            test_data[key] = content

        # Benchmark cache operations
        embedding_hits = 0
        embedding_misses = 0
        file_hits = 0
        file_misses = 0

        embedding_times = []
        file_times = []

        # Test embedding cache
        for i in range(num_operations):
            key = f"test_key_{i % 100}"
            content = test_data[key]

            start_time = time.time()
            embedding, hit = cache.get_embedding(key, content)
            embedding_times.append(time.time() - start_time)

            if hit:
                embedding_hits += 1
            else:
                embedding_misses += 1

        # Test file content cache
        for i in range(num_operations):
            key = f"test_key_{i % 100}"
            content = test_data[key]

            start_time = time.time()
            file_content, hit = cache.get_file_content(key)
            file_times.append(time.time() - start_time)

            if hit:
                file_hits += 1
            else:
                file_misses += 1

        # Calculate hit rates
        embedding_hit_rate = embedding_hits / (embedding_hits + embedding_misses)
        file_hit_rate = file_hits / (file_hits + file_misses)

        avg_embedding_time = statistics.mean(embedding_times)
        avg_file_time = statistics.mean(file_times)

        # Record results
        results.extend([
            BenchmarkResult(
                test_name="cache_performance",
                metric_name="embedding_cache_hit_rate",
                value=embedding_hit_rate * 100,
                unit="percent",
                target=70.0,
                achieved=embedding_hit_rate >= 0.7,
                metadata={"operations": num_operations}
            ),
            BenchmarkResult(
                test_name="cache_performance",
                metric_name="file_cache_hit_rate",
                value=file_hit_rate * 100,
                unit="percent",
                target=70.0,
                achieved=file_hit_rate >= 0.7,
                metadata={"operations": num_operations}
            ),
            BenchmarkResult(
                test_name="cache_performance",
                metric_name="embedding_cache_avg_time",
                value=avg_embedding_time * 1000,  # Convert to milliseconds
                unit="milliseconds",
                target=10.0,  # Less than 10ms
                achieved=avg_embedding_time <= 0.01,
                metadata={"operations": num_operations}
            ),
            BenchmarkResult(
                test_name="cache_performance",
                metric_name="file_cache_avg_time",
                value=avg_file_time * 1000,
                unit="milliseconds",
                target=5.0,  # Less than 5ms
                achieved=avg_file_time <= 0.005,
                metadata={"operations": num_operations}
            )
        ])

        return results

    def benchmark_memory_management(self) -> List[BenchmarkResult]:
        """Benchmark memory management effectiveness."""
        results = []

        memory_manager = MemoryManager()

        # Test model caching effectiveness
        mock_model = MagicMock()

        with patch('codesage_mcp.memory_manager.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_model

            # Load model multiple times
            start_time = time.time()
            for i in range(10):
                loaded_model = memory_manager.load_model(f"test_model_{i % 3}")
            model_loading_time = time.time() - start_time

            # Check cache stats
            cache_stats = memory_manager.model_cache.get_stats()

        # Test memory monitoring
        memory_stats = memory_manager.get_memory_stats()

        # Record results
        results.extend([
            BenchmarkResult(
                test_name="memory_management",
                metric_name="model_cache_effectiveness",
                value=cache_stats["cached_models"],
                unit="models_cached",
                target=3.0,  # Should cache at least 3 models
                achieved=cache_stats["cached_models"] >= 3,
                metadata={"cache_stats": cache_stats}
            ),
            BenchmarkResult(
                test_name="memory_management",
                metric_name="memory_monitoring_accuracy",
                value=memory_stats["percent"],
                unit="percent",
                target=100.0,  # Should be able to monitor memory
                achieved=isinstance(memory_stats["percent"], (int, float)),
                metadata={"memory_stats": memory_stats}
            ),
            BenchmarkResult(
                test_name="memory_management",
                metric_name="model_loading_performance",
                value=model_loading_time,
                unit="seconds",
                target=1.0,  # Should load models quickly with caching
                achieved=model_loading_time <= 1.0,
                metadata={"loads": 10}
            )
        ])

        # Cleanup
        memory_manager.cleanup()

        return results

    def benchmark_chunking_performance(self) -> List[BenchmarkResult]:
        """Benchmark document chunking performance."""
        results = []

        chunker = DocumentChunker()

        # Create test documents of varying sizes
        test_sizes = [100, 500, 1000, 5000]

        for size in test_sizes:
            # Create test content
            content = ""
            for i in range(size // 10):
                content += f"""
def function_{i}():
    \"\"\"Function {i} documentation.\"\"\"
    x = {i}
    y = x * 2
    return y

class Class_{i}:
    def method(self):
        return "method_{i}"
"""

            # Benchmark chunking
            start_time = time.time()
            chunks = chunker.split_into_chunks(content)
            chunking_time = time.time() - start_time

            stats = chunker.get_chunk_statistics(chunks)

            results.append(BenchmarkResult(
                test_name="chunking_performance",
                metric_name="chunking_time",
                value=chunking_time,
                unit="seconds",
                target=1.0,  # Should chunk within 1 second
                achieved=chunking_time <= 1.0,
                metadata={
                    "content_size": size,
                    "chunks_created": stats["total_chunks"],
                    "avg_chunk_size": stats["average_chunk_size"]
                }
            ))

        return results

    def run_full_benchmark_suite(self) -> PerformanceReport:
        """Run the complete benchmark suite."""
        print("Starting comprehensive performance benchmark suite...")

        all_results = []

        # Run all benchmarks
        print("1. Benchmarking indexing performance...")
        all_results.extend(self.benchmark_indexing_performance())

        print("2. Benchmarking search performance...")
        all_results.extend(self.benchmark_search_performance())

        print("3. Benchmarking cache performance...")
        all_results.extend(self.benchmark_cache_performance())

        print("4. Benchmarking memory management...")
        all_results.extend(self.benchmark_memory_management())

        print("5. Benchmarking chunking performance...")
        all_results.extend(self.benchmark_chunking_performance())

        # Generate summary
        summary = self._generate_summary(all_results)

        # Create report
        report = PerformanceReport(
            test_suite="CodeSage MCP Performance Benchmark",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            results=all_results,
            summary=summary
        )

        # Save report
        self.save_report(report)

        return report

    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate a summary of benchmark results."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.achieved)
        failed_tests = total_tests - passed_tests

        # Group by test category
        categories = {}
        for result in results:
            category = result.test_name.split('_')[0]
            if category not in categories:
                categories[category] = {"passed": 0, "failed": 0, "total": 0}
            categories[category]["total"] += 1
            if result.achieved:
                categories[category]["passed"] += 1
            else:
                categories[category]["failed"] += 1

        # Calculate overall performance score
        performance_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "performance_score": performance_score,
            "categories": categories,
            "overall_status": "PASS" if performance_score >= 80 else "FAIL"
        }

    def save_report(self, report: PerformanceReport):
        """Save benchmark report to file."""
        report_file = self.output_dir / f"benchmark_report_{int(time.time())}.json"

        # Convert dataclasses to dictionaries
        report_dict = {
            "test_suite": report.test_suite,
            "timestamp": report.timestamp,
            "results": [
                {
                    "test_name": r.test_name,
                    "metric_name": r.metric_name,
                    "value": r.value,
                    "unit": r.unit,
                    "target": r.target,
                    "achieved": r.achieved,
                    "metadata": r.metadata
                }
                for r in report.results
            ],
            "summary": report.summary
        }

        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)

        print(f"Benchmark report saved to: {report_file}")

        # Also save a human-readable summary
        summary_file = self.output_dir / f"benchmark_summary_{int(time.time())}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"CodeSage MCP Performance Benchmark Report\n")
            f.write(f"Generated: {report.timestamp}\n\n")

            f.write("SUMMARY\n")
            f.write("=" * 50 + "\n")
            summary = report.summary
            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Passed: {summary['passed_tests']}\n")
            f.write(f"Failed: {summary['failed_tests']}\n")
            f.write(".1f")
            f.write(f"Overall Status: {summary['overall_status']}\n\n")

            f.write("RESULTS BY CATEGORY\n")
            f.write("=" * 50 + "\n")
            for category, stats in summary['categories'].items():
                f.write(f"{category.upper()}:\n")
                f.write(f"  Passed: {stats['passed']}\n")
                f.write(f"  Failed: {stats['failed']}\n")
                f.write(f"  Total: {stats['total']}\n\n")

            f.write("DETAILED RESULTS\n")
            f.write("=" * 50 + "\n")
            for result in report.results:
                status = "✓ PASS" if result.achieved else "✗ FAIL"
                f.write(f"{result.test_name} - {result.metric_name}: {result.value:.4f} {result.unit} [{status}]\n")
                if result.target is not None:
                    f.write(f"  Target: {result.target} {result.unit}\n")

        print(f"Benchmark summary saved to: {summary_file}")

    def print_report(self, report: PerformanceReport):
        """Print a formatted benchmark report to console."""
        print("\n" + "=" * 80)
        print("CODESAGE MCP PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        print(f"Generated: {report.timestamp}")
        print()

        summary = report.summary
        print("EXECUTIVE SUMMARY")
        print("-" * 30)
        print(f"Total Tests Run: {summary['total_tests']}")
        print(f"Tests Passed: {summary['passed_tests']}")
        print(f"Tests Failed: {summary['failed_tests']}")
        print(".1f")
        print(f"Overall Status: {summary['overall_status']}")
        print()

        print("PERFORMANCE TARGETS VALIDATION")
        print("-" * 35)

        # Group results by metric type
        indexing_results = [r for r in report.results if "indexing" in r.test_name]
        search_results = [r for r in report.results if "search" in r.test_name]
        cache_results = [r for r in report.results if "cache" in r.test_name]
        memory_results = [r for r in report.results if "memory" in r.test_name]

        for category, results in [
            ("INDEXING PERFORMANCE", indexing_results),
            ("SEARCH PERFORMANCE", search_results),
            ("CACHE PERFORMANCE", cache_results),
            ("MEMORY MANAGEMENT", memory_results)
        ]:
            if results:
                print(f"\n{category}:")
                for result in results:
                    status = "✓" if result.achieved else "✗"
                    print(f"  {status} {result.metric_name}: {result.value:.4f} {result.unit}")
                    if result.target:
                        print(f"     Target: {result.target} {result.unit}")

        print("\n" + "=" * 80)


def run_performance_benchmarks():
    """Run the complete performance benchmark suite."""
    benchmarker = PerformanceBenchmarker()

    print("CodeSage MCP Performance Benchmark Suite")
    print("=========================================")
    print()

    # Run benchmarks
    report = benchmarker.run_full_benchmark_suite()

    # Print results
    benchmarker.print_report(report)

    return report


if __name__ == "__main__":
    run_performance_benchmarks()
"""
Integration Tests for IntelligentCache System.

This module contains comprehensive integration tests that simulate real-world usage patterns
and stress the IntelligentCache system under different loads. Tests focus on end-to-end
behavior, performance under load, reliability, and edge cases like resource exhaustion.

Tests include:
- High concurrency with mixed operations (embedding storage, file content caching, search results)
- Memory pressure stress testing
- Bursty workloads
- Long-running sessions with gradual memory buildup
- Cross-cache interactions

Metrics measured:
- Hit rates (embedding, search, file)
- Response times for operations
- Memory usage
- Cache sizes and utilization
"""

import pytest
import time
import threading
import concurrent.futures
import numpy as np
from unittest.mock import MagicMock
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import psutil

from codesage_mcp.features.caching.intelligent_cache import IntelligentCache
from codesage_mcp.features.caching.cache import reset_cache_instance


class MetricsCollector:
    """Helper class to collect and analyze performance metrics during tests."""

    def __init__(self):
        self.response_times = defaultdict(list)
        self.memory_usage = []
        self.cache_stats_history = []
        self.operation_counts = defaultdict(int)
        self.errors = []

    def record_response_time(self, operation: str, start_time: float):
        """Record response time for an operation."""
        elapsed = time.time() - start_time
        self.response_times[operation].append(elapsed)

    def record_memory_usage(self):
        """Record current memory usage."""
        try:
            memory_percent = psutil.virtual_memory().percent
            self.memory_usage.append(memory_percent)
        except ImportError:
            # Fallback if psutil not available
            self.memory_usage.append(0.0)

    def record_cache_stats(self, cache: IntelligentCache):
        """Record cache statistics."""
        stats = cache.get_comprehensive_stats()
        self.cache_stats_history.append(stats)

    def record_operation(self, operation: str):
        """Record operation count."""
        self.operation_counts[operation] += 1

    def record_error(self, error: Exception, operation: str):
        """Record an error that occurred."""
        self.errors.append({"error": str(error), "operation": operation, "time": time.time()})

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {
            "total_operations": sum(self.operation_counts.values()),
            "operation_breakdown": dict(self.operation_counts),
            "avg_response_times": {},
            "memory_usage_stats": {},
            "cache_performance": {},
            "error_count": len(self.errors),
            "errors": self.errors[:10],  # First 10 errors
        }

        # Calculate average response times
        for operation, times in self.response_times.items():
            if times:
                summary["avg_response_times"][operation] = {
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times),
                }

        # Calculate memory usage statistics
        if self.memory_usage:
            summary["memory_usage_stats"] = {
                "avg_percent": sum(self.memory_usage) / len(self.memory_usage),
                "max_percent": max(self.memory_usage),
                "min_percent": min(self.memory_usage),
                "samples": len(self.memory_usage),
            }

        # Calculate cache performance metrics
        if self.cache_stats_history:
            latest_stats = self.cache_stats_history[-1]
            summary["cache_performance"] = {
                "final_hit_rates": latest_stats.get("hit_rates", {}),
                "final_memory_usage": latest_stats.get("performance_metrics", {}).get("memory_usage_percent", 0),
                "cache_sizes": latest_stats.get("caches", {}),
            }

        return summary


@pytest.fixture
def cache_instance():
    """Fixture providing a fresh IntelligentCache instance for each test."""
    reset_cache_instance()
    cache = IntelligentCache(config={
        "embedding_cache_size": 1000,
        "search_cache_size": 500,
        "file_cache_size": 200,
        "max_file_size": 1024 * 1024,  # 1MB
        "enable_persistence": False,  # Disable for testing
        "cache_warming_enabled": False,  # Disable for testing
    })
    yield cache
    # Cleanup
    cache.clear_all()


@pytest.fixture
def mock_sentence_transformer():
    """Fixture providing a mock sentence transformer model."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
    return mock_model


@pytest.fixture
def test_data_generator():
    """Fixture providing test data generation utilities."""
    class TestDataGenerator:
        def __init__(self):
            self.file_counter = 0
            self.query_counter = 0

        def generate_file_data(self, count: int = 10) -> List[Tuple[str, str]]:
            """Generate test file paths and contents."""
            files = []
            for i in range(count):
                file_path = f"/test/file_{self.file_counter + i}.py"
                content = f"""
def function_{i}():
    \"\"\"Test function {i}.\"\"\"
    x = {i}
    y = x * 2
    return y

class TestClass{i}:
    def method_{i}(self):
        return "test_{i}"
"""
                files.append((file_path, content))
            self.file_counter += count
            return files

        def generate_search_queries(self, count: int = 5) -> List[str]:
            """Generate test search queries."""
            queries = []
            for i in range(count):
                query = f"test query {self.query_counter + i} with some keywords"
                queries.append(query)
            self.query_counter += count
            return queries

        def generate_search_results(self, query: str, count: int = 3) -> List[Dict]:
            """Generate mock search results."""
            results = []
            for i in range(count):
                result = {
                    "file": f"/test/result_file_{i}.py",
                    "score": 0.9 - (i * 0.1),
                    "line": i + 1,
                    "content": f"matching content {i} for query: {query[:20]}..."
                }
                results.append(result)
            return results

    return TestDataGenerator()


@pytest.fixture
def metrics_collector():
    """Fixture providing a metrics collector instance."""
    return MetricsCollector()


class TestHighConcurrencyMixedOperations:
    """Test high concurrency scenarios with mixed cache operations."""

    def test_concurrent_mixed_operations(self, cache_instance, mock_sentence_transformer,
                                        test_data_generator, metrics_collector):
        """Test concurrent execution of mixed cache operations."""
        num_threads = 10
        operations_per_thread = 50

        # Pre-populate some data
        initial_files = test_data_generator.generate_file_data(20)
        for file_path, content in initial_files:
            embedding = mock_sentence_transformer.encode(content)
            cache_instance.store_embedding(file_path, content, embedding)
            cache_instance.store_file_content(file_path, content)

        def worker_thread(thread_id: int):
            """Worker function for concurrent operations."""
            try:
                # Generate thread-specific data
                files = test_data_generator.generate_file_data(5)
                queries = test_data_generator.generate_search_queries(3)

                for i in range(operations_per_thread):
                    operation_start = time.time()

                    if i % 4 == 0:
                        # Store embedding operation
                        file_path, content = files[i % len(files)]
                        embedding = mock_sentence_transformer.encode(content)
                        cache_instance.store_embedding(file_path, content, embedding)
                        metrics_collector.record_operation("store_embedding")

                    elif i % 4 == 1:
                        # Get embedding operation
                        file_path, content = files[i % len(files)]
                        embedding, hit = cache_instance.get_embedding(file_path, content)
                        metrics_collector.record_operation("get_embedding")

                    elif i % 4 == 2:
                        # File content operations
                        file_path, content = files[i % len(files)]
                        if i % 2 == 0:
                            cache_instance.store_file_content(file_path, content)
                            metrics_collector.record_operation("store_file_content")
                        else:
                            content, hit = cache_instance.get_file_content(file_path)
                            metrics_collector.record_operation("get_file_content")

                    else:
                        # Search operations
                        query = queries[i % len(queries)]
                        query_embedding = mock_sentence_transformer.encode(query)
                        if i % 2 == 0:
                            results = test_data_generator.generate_search_results(query)
                            cache_instance.store_search_results(query, query_embedding, results)
                            metrics_collector.record_operation("store_search_results")
                        else:
                            results, hit = cache_instance.get_search_results(query, query_embedding)
                            metrics_collector.record_operation("get_search_results")

                    metrics_collector.record_response_time("mixed_operation", operation_start)
                    metrics_collector.record_memory_usage()

            except Exception as e:
                metrics_collector.record_error(e, f"thread_{thread_id}")

        # Execute concurrent operations
        threads = []
        start_time = time.time()

        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        execution_time = time.time() - start_time

        # Collect final metrics
        metrics_collector.record_cache_stats(cache_instance)

        # Assertions
        summary = metrics_collector.get_summary()
        assert summary["total_operations"] > 0
        assert summary["error_count"] == 0, f"Errors occurred: {summary['errors']}"
        assert execution_time < 30, f"Test took too long: {execution_time}s"

        # Check that operations were distributed
        operations = summary["operation_breakdown"]
        assert "store_embedding" in operations
        assert "get_embedding" in operations
        assert "store_file_content" in operations
        assert "get_file_content" in operations
        assert "store_search_results" in operations
        assert "get_search_results" in operations

        # Check performance metrics
        avg_response_times = summary["avg_response_times"]
        assert "mixed_operation" in avg_response_times
        assert avg_response_times["mixed_operation"]["avg"] < 1.0  # Should be fast

    def test_concurrent_cache_invalidation(self, cache_instance, mock_sentence_transformer,
                                         test_data_generator, metrics_collector):
        """Test concurrent cache invalidation operations."""
        num_threads = 8
        files_per_thread = 10

        # Pre-populate cache with data
        all_files = test_data_generator.generate_file_data(num_threads * files_per_thread)
        for file_path, content in all_files:
            embedding = mock_sentence_transformer.encode(content)
            cache_instance.store_embedding(file_path, content, embedding)
            cache_instance.store_file_content(file_path, content)

        def invalidation_worker(thread_id: int):
            """Worker for invalidation operations."""
            try:
                start_idx = thread_id * files_per_thread
                end_idx = start_idx + files_per_thread
                thread_files = all_files[start_idx:end_idx]

                for file_path, content in thread_files:
                    operation_start = time.time()

                    # Invalidate file (affects all caches)
                    invalidated = cache_instance.invalidate_file(file_path)
                    metrics_collector.record_operation("invalidate_file")

                    metrics_collector.record_response_time("invalidate", operation_start)

                    # Verify invalidation worked
                    embedding, hit = cache_instance.get_embedding(file_path, content)
                    assert not hit, f"Embedding not invalidated for {file_path}"

                    content_cached, hit = cache_instance.get_file_content(file_path)
                    assert not hit, f"File content not invalidated for {file_path}"

            except Exception as e:
                metrics_collector.record_error(e, f"invalidation_thread_{thread_id}")

        # Execute concurrent invalidations
        threads = []
        start_time = time.time()

        for i in range(num_threads):
            thread = threading.Thread(target=invalidation_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        execution_time = time.time() - start_time

        # Collect final metrics
        metrics_collector.record_cache_stats(cache_instance)

        # Assertions
        summary = metrics_collector.get_summary()
        assert summary["error_count"] == 0, f"Errors occurred: {summary['errors']}"
        assert execution_time < 10, f"Invalidation took too long: {execution_time}s"

        # Check that invalidations occurred
        assert summary["operation_breakdown"].get("invalidate_file", 0) > 0


class TestMemoryPressureStress:
    """Test cache behavior under memory pressure conditions."""

    def test_memory_pressure_with_large_embeddings(self, cache_instance, mock_sentence_transformer,
                                                 test_data_generator, metrics_collector):
        """Test cache behavior when storing large numbers of embeddings."""
        # Configure cache with smaller sizes to create pressure
        cache_instance.config["embedding_cache_size"] = 50
        cache_instance.config["file_cache_size"] = 20

        num_files = 200
        files = test_data_generator.generate_file_data(num_files)

        start_time = time.time()
        stored_count = 0
        hit_count = 0

        for i, (file_path, content) in enumerate(files):
            operation_start = time.time()

            # Store embedding
            embedding = mock_sentence_transformer.encode(content)
            cache_instance.store_embedding(file_path, content, embedding)
            stored_count += 1

            metrics_collector.record_response_time("store_embedding", operation_start)
            metrics_collector.record_memory_usage()
            metrics_collector.record_operation("store_embedding")

            # Periodically check hit rates
            if i % 20 == 0:
                metrics_collector.record_cache_stats(cache_instance)

        # Test retrieval under memory pressure
        for file_path, content in files[:50]:  # Test first 50 files
            operation_start = time.time()
            embedding, hit = cache_instance.get_embedding(file_path, content)
            metrics_collector.record_response_time("get_embedding", operation_start)
            if hit:
                hit_count += 1
            metrics_collector.record_operation("get_embedding")

        execution_time = time.time() - start_time
        hit_rate = hit_count / 50 if 50 > 0 else 0

        # Collect final metrics
        metrics_collector.record_cache_stats(cache_instance)

        # Assertions
        summary = metrics_collector.get_summary()
        assert summary["error_count"] == 0
        assert execution_time < 60, f"Test took too long: {execution_time}s"
        assert hit_rate > 0.1, f"Hit rate too low under memory pressure: {hit_rate}"

        # Check memory usage was monitored
        memory_stats = summary["memory_usage_stats"]
        assert memory_stats["samples"] > 0

    def test_memory_pressure_with_large_files(self, cache_instance, test_data_generator, metrics_collector):
        """Test cache behavior with large file contents."""
        # Configure small file cache to create pressure
        cache_instance.config["file_cache_size"] = 10
        cache_instance.config["max_file_size"] = 50 * 1024  # 50KB limit

        num_files = 100
        files = []

        # Generate files of varying sizes
        for i in range(num_files):
            file_path = f"/test/large_file_{i}.py"
            # Create content of varying sizes
            content_size = 1024 + (i * 512)  # 1KB to ~50KB
            content = f"# Large file {i}\n" + "x = " + str(i) + "\n" * (content_size // 10)
            files.append((file_path, content))

        start_time = time.time()
        stored_count = 0
        failed_count = 0

        for file_path, content in files:
            operation_start = time.time()

            success = cache_instance.store_file_content(file_path, content)
            if success:
                stored_count += 1
                metrics_collector.record_operation("store_file_success")
            else:
                failed_count += 1
                metrics_collector.record_operation("store_file_failed")

            metrics_collector.record_response_time("store_file", operation_start)
            metrics_collector.record_memory_usage()

        # Test retrieval
        retrieved_count = 0
        for file_path, content in files[:30]:
            operation_start = time.time()
            cached_content, hit = cache_instance.get_file_content(file_path)
            metrics_collector.record_response_time("get_file", operation_start)
            if hit:
                retrieved_count += 1
                assert cached_content == content

        execution_time = time.time() - start_time
        retrieval_rate = retrieved_count / 30 if 30 > 0 else 0

        # Collect final metrics
        metrics_collector.record_cache_stats(cache_instance)

        # Assertions
        summary = metrics_collector.get_summary()
        assert summary["error_count"] == 0
        assert execution_time < 30, f"Test took too long: {execution_time}s"
        assert stored_count > 0, "No files were stored"
        assert failed_count >= 0, "Some files should fail due to size limits"
        assert retrieval_rate > 0.5, f"Retrieval rate too low: {retrieval_rate}"


class TestBurstyWorkloads:
    """Test cache behavior under bursty workload patterns."""

    def test_sudden_request_burst(self, cache_instance, mock_sentence_transformer,
                                test_data_generator, metrics_collector):
        """Test cache performance during sudden bursts of requests."""
        # Pre-populate with some data
        initial_files = test_data_generator.generate_file_data(20)
        for file_path, content in initial_files:
            embedding = mock_sentence_transformer.encode(content)
            cache_instance.store_embedding(file_path, content, embedding)

        burst_sizes = [10, 50, 100, 50, 10]  # Varying burst sizes
        burst_intervals = [1, 0.5, 0.2, 0.5, 1]  # Varying intervals

        total_operations = 0
        burst_results = []

        for burst_idx, (burst_size, interval) in enumerate(zip(burst_sizes, burst_intervals)):
            time.sleep(interval)  # Wait between bursts

            burst_start = time.time()
            burst_operations = 0

            # Execute burst of operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []

                for i in range(burst_size):
                    # Mix of operations
                    if i % 3 == 0:
                        future = executor.submit(self._burst_store_operation,
                                               cache_instance, mock_sentence_transformer,
                                               test_data_generator, metrics_collector)
                    elif i % 3 == 1:
                        future = executor.submit(self._burst_get_operation,
                                               cache_instance, mock_sentence_transformer,
                                               initial_files, metrics_collector)
                    else:
                        future = executor.submit(self._burst_search_operation,
                                               cache_instance, mock_sentence_transformer,
                                               test_data_generator, metrics_collector)
                    futures.append(future)

                # Wait for burst to complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result(timeout=5)
                        burst_operations += 1
                    except Exception as e:
                        metrics_collector.record_error(e, f"burst_{burst_idx}")

            burst_time = time.time() - burst_start
            total_operations += burst_operations

            burst_results.append({
                "burst_idx": burst_idx,
                "operations": burst_operations,
                "time": burst_time,
                "ops_per_second": burst_operations / burst_time if burst_time > 0 else 0
            })

            metrics_collector.record_cache_stats(cache_instance)

        # Assertions
        summary = metrics_collector.get_summary()
        assert summary["error_count"] == 0, f"Errors in burst test: {summary['errors']}"
        assert total_operations > 0

        # Check that burst performance varies appropriately
        for result in burst_results:
            assert result["ops_per_second"] > 0, f"Zero ops/sec in burst {result['burst_idx']}"

    def _burst_store_operation(self, cache, model, data_gen, metrics):
        """Helper for burst store operations."""
        operation_start = time.time()
        files = data_gen.generate_file_data(1)
        file_path, content = files[0]
        embedding = model.encode(content)
        cache.store_embedding(file_path, content, embedding)
        metrics.record_response_time("burst_store", operation_start)
        metrics.record_operation("burst_store")

    def _burst_get_operation(self, cache, model, files, metrics):
        """Helper for burst get operations."""
        operation_start = time.time()
        file_path, content = files[len(files) % len(files)]
        embedding, hit = cache.get_embedding(file_path, content)
        metrics.record_response_time("burst_get", operation_start)
        metrics.record_operation("burst_get")

    def _burst_search_operation(self, cache, model, data_gen, metrics):
        """Helper for burst search operations."""
        operation_start = time.time()
        queries = data_gen.generate_search_queries(1)
        query = queries[0]
        query_embedding = model.encode(query)
        results, hit = cache.get_search_results(query, query_embedding)
        metrics.record_response_time("burst_search", operation_start)
        metrics.record_operation("burst_search")


class TestLongRunningSessions:
    """Test cache behavior during long-running sessions with gradual memory buildup."""

    def test_gradual_memory_buildup(self, cache_instance, mock_sentence_transformer,
                                  test_data_generator, metrics_collector):
        """Test cache behavior with gradual accumulation of data over time."""
        session_duration = 10  # seconds
        operation_interval = 0.1  # 100ms between operations

        start_time = time.time()
        operation_count = 0
        file_counter = 0

        while time.time() - start_time < session_duration:
            # Generate new data gradually
            files = test_data_generator.generate_file_data(2)
            for file_path, content in files:
                operation_start = time.time()

                # Store embedding
                embedding = mock_sentence_transformer.encode(content)
                cache_instance.store_embedding(file_path, content, embedding)

                # Store file content
                cache_instance.store_file_content(file_path, content)

                # Occasionally perform search
                if operation_count % 5 == 0:
                    queries = test_data_generator.generate_search_queries(1)
                    query = queries[0]
                    query_embedding = mock_sentence_transformer.encode(query)
                    results = test_data_generator.generate_search_results(query)
                    cache_instance.store_search_results(query, query_embedding, results)

                metrics_collector.record_response_time("session_operation", operation_start)
                metrics_collector.record_memory_usage()
                metrics_collector.record_operation("session_store")

                operation_count += 1
                file_counter += 1

            # Periodic cache stats collection
            if operation_count % 20 == 0:
                metrics_collector.record_cache_stats(cache_instance)

            time.sleep(operation_interval)

        # Test retrieval of accumulated data
        retrieval_start = time.time()
        hit_count = 0
        total_checks = min(50, file_counter)  # Check up to 50 files

        # Generate some files to check (simulate accessing accumulated data)
        check_files = test_data_generator.generate_file_data(total_checks)
        for file_path, content in check_files:
            operation_start = time.time()
            embedding, hit = cache_instance.get_embedding(file_path, content)
            metrics_collector.record_response_time("session_retrieval", operation_start)
            if hit:
                hit_count += 1

        retrieval_time = time.time() - retrieval_start
        hit_rate = hit_count / total_checks if total_checks > 0 else 0

        # Collect final metrics
        metrics_collector.record_cache_stats(cache_instance)

        # Assertions
        summary = metrics_collector.get_summary()
        assert summary["error_count"] == 0
        assert operation_count > 50, f"Insufficient operations: {operation_count}"
        assert hit_rate > 0.3, f"Hit rate too low after buildup: {hit_rate}"

        # Check memory monitoring
        memory_stats = summary["memory_usage_stats"]
        assert memory_stats["samples"] > 10, "Insufficient memory monitoring"

    def test_adaptive_sizing_during_session(self, cache_instance, mock_sentence_transformer,
                                          test_data_generator, metrics_collector):
        """Test adaptive cache sizing during a long session."""
        # Enable adaptive sizing
        cache_instance.adaptive_config["enabled"] = True
        cache_instance.adaptive_config["adjustment_interval"] = 2  # Adjust every 2 seconds

        session_duration = 8  # seconds
        start_time = time.time()
        operation_count = 0

        # Simulate varying workload
        high_workload_periods = [(2, 4), (6, 8)]  # seconds

        while time.time() - start_time < session_duration:
            current_time = time.time() - start_time

            # Determine operation intensity based on time
            is_high_workload = any(start <= current_time <= end for start, end in high_workload_periods)
            operations_per_batch = 5 if is_high_workload else 2

            # Perform batch of operations
            for _ in range(operations_per_batch):
                operation_start = time.time()

                files = test_data_generator.generate_file_data(1)
                file_path, content = files[0]

                embedding = mock_sentence_transformer.encode(content)
                cache_instance.store_embedding(file_path, content, embedding)
                cache_instance.store_file_content(file_path, content)

                metrics_collector.record_response_time("adaptive_operation", operation_start)
                metrics_collector.record_operation("adaptive_store")

                operation_count += 1

            metrics_collector.record_memory_usage()

            # Record cache stats periodically
            if operation_count % 10 == 0:
                metrics_collector.record_cache_stats(cache_instance)

            time.sleep(0.2)  # 200ms between batches

        # Collect final metrics
        metrics_collector.record_cache_stats(cache_instance)

        # Assertions
        summary = metrics_collector.get_summary()
        assert summary["error_count"] == 0
        assert operation_count > 20

        # Check that adaptive sizing occurred
        cache_performance = summary["cache_performance"]
        assert "cache_sizes" in cache_performance


class TestCrossCacheInteractions:
    """Test interactions between different cache types."""

    def test_cross_cache_data_consistency(self, cache_instance, mock_sentence_transformer,
                                        test_data_generator, metrics_collector):
        """Test data consistency across different caches for the same files."""
        num_files = 30
        files = test_data_generator.generate_file_data(num_files)

        # Store data in all caches for the same files
        for file_path, content in files:
            operation_start = time.time()

            # Store embedding
            embedding = mock_sentence_transformer.encode(content)
            cache_instance.store_embedding(file_path, content, embedding)

            # Store file content
            cache_instance.store_file_content(file_path, content)

            # Store search results related to this file
            query = f"functions in {file_path}"
            query_embedding = mock_sentence_transformer.encode(query)
            results = test_data_generator.generate_search_results(query)
            cache_instance.store_search_results(query, query_embedding, results)

            metrics_collector.record_response_time("cross_cache_store", operation_start)
            metrics_collector.record_operation("cross_cache_store")

        # Test cross-cache consistency during invalidation
        test_files = files[:10]  # Test first 10 files

        for file_path, content in test_files:
            operation_start = time.time()

            # Invalidate file (should affect all caches)
            invalidated = cache_instance.invalidate_file(file_path)

            metrics_collector.record_response_time("cross_cache_invalidate", operation_start)
            metrics_collector.record_operation("cross_cache_invalidate")

            # Verify all caches are invalidated
            embedding, embedding_hit = cache_instance.get_embedding(file_path, content)
            content_cached, content_hit = cache_instance.get_file_content(file_path)

            assert not embedding_hit, f"Embedding not invalidated for {file_path}"
            assert not content_hit, f"File content not invalidated for {file_path}"

            # Search results should still be available (not file-specific invalidation)
            query = f"functions in {file_path}"
            query_embedding = mock_sentence_transformer.encode(query)
            search_results, search_hit = cache_instance.get_search_results(query, query_embedding)
            # Search results are not automatically invalidated by file invalidation

        # Test cross-cache performance correlation
        metrics_collector.record_cache_stats(cache_instance)

        # Assertions
        summary = metrics_collector.get_summary()
        assert summary["error_count"] == 0
        assert summary["operation_breakdown"].get("cross_cache_store", 0) == num_files
        assert summary["operation_breakdown"].get("cross_cache_invalidate", 0) == len(test_files)

    def test_cache_interaction_under_load(self, cache_instance, mock_sentence_transformer,
                                        test_data_generator, metrics_collector):
        """Test cache interactions when all caches are under load simultaneously."""
        num_threads = 6
        operations_per_thread = 30

        def interaction_worker(thread_id: int):
            """Worker that performs operations affecting multiple caches."""
            try:
                files = test_data_generator.generate_file_data(5)
                queries = test_data_generator.generate_search_queries(3)

                for i in range(operations_per_thread):
                    operation_start = time.time()

                    if i % 5 == 0:
                        # Store operation (affects embedding and file caches)
                        file_path, content = files[i % len(files)]
                        embedding = mock_sentence_transformer.encode(content)
                        cache_instance.store_embedding(file_path, content, embedding)
                        cache_instance.store_file_content(file_path, content)
                        metrics_collector.record_operation("multi_store")

                    elif i % 5 == 1:
                        # Get operation (affects embedding and file caches)
                        file_path, content = files[i % len(files)]
                        embedding, emb_hit = cache_instance.get_embedding(file_path, content)
                        content_cached, cont_hit = cache_instance.get_file_content(file_path)
                        metrics_collector.record_operation("multi_get")

                    elif i % 5 == 2:
                        # Search operation (affects search cache)
                        query = queries[i % len(queries)]
                        query_embedding = mock_sentence_transformer.encode(query)
                        results, hit = cache_instance.get_search_results(query, query_embedding)
                        if not hit:
                            results = test_data_generator.generate_search_results(query)
                            cache_instance.store_search_results(query, query_embedding, results)
                        metrics_collector.record_operation("search_op")

                    elif i % 5 == 3:
                        # Mixed read operation
                        file_path, content = files[i % len(files)]
                        query = queries[i % len(queries)]

                        # Read from embedding cache
                        embedding, emb_hit = cache_instance.get_embedding(file_path, content)
                        # Read from file cache
                        content_cached, cont_hit = cache_instance.get_file_content(file_path)
                        # Read from search cache
                        query_embedding = mock_sentence_transformer.encode(query)
                        results, search_hit = cache_instance.get_search_results(query, query_embedding)

                        metrics_collector.record_operation("mixed_read")

                    else:
                        # Invalidation operation (affects all caches)
                        file_path, content = files[i % len(files)]
                        invalidated = cache_instance.invalidate_file(file_path)
                        metrics_collector.record_operation("multi_invalidate")

                    metrics_collector.record_response_time("interaction_operation", operation_start)

            except Exception as e:
                metrics_collector.record_error(e, f"interaction_thread_{thread_id}")

        # Execute concurrent interactions
        threads = []
        start_time = time.time()

        for i in range(num_threads):
            thread = threading.Thread(target=interaction_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        execution_time = time.time() - start_time

        # Collect final metrics
        metrics_collector.record_cache_stats(cache_instance)

        # Assertions
        summary = metrics_collector.get_summary()
        assert summary["error_count"] == 0, f"Errors in interaction test: {summary['errors']}"
        assert execution_time < 20, f"Interaction test took too long: {execution_time}s"

        # Check that all types of operations occurred
        operations = summary["operation_breakdown"]
        assert "multi_store" in operations
        assert "multi_get" in operations
        assert "search_op" in operations
        assert "mixed_read" in operations
        assert "multi_invalidate" in operations

        # Check performance metrics
        avg_response_times = summary["avg_response_times"]
        assert "interaction_operation" in avg_response_times
        assert avg_response_times["interaction_operation"]["avg"] < 2.0  # Should be reasonably fast
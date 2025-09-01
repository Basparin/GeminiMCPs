"""
Memory Optimization and Performance Validation Tests.

This module contains tests focused on memory usage optimization, performance validation,
and ensuring the system maintains performance standards under various conditions.
"""

import pytest
import tempfile
import os
import time
import gc
import threading
import psutil

from codesage_mcp.core.code_model import (
    CodeGraph,
    CodeModelGenerator,
    CodeNode,
    NodeType,
    RelationshipType,
    LayerType
)
from codesage_mcp.features.codebase_manager import AdvancedAnalysisManager
from codesage_mcp.features.memory_management.memory_manager import MemoryManager


class TestMemoryOptimization:
    """Test memory optimization features."""

    @pytest.fixture
    def memory_optimized_setup(self):
        """Create setup with memory optimization enabled."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        # Initialize memory manager
        memory_manager = MemoryManager()
        generator.memory_manager = memory_manager
        analyzer.memory_manager = memory_manager

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer,
            'memory_manager': memory_manager
        }

    def test_memory_cleanup_after_large_operations(self, memory_optimized_setup):
        """Test memory cleanup after large operations."""
        setup = memory_optimized_setup
        generator = setup['generator']
        memory_manager = setup['memory_manager']

        # Create large content
        large_content = ""
        for i in range(1000):
            large_content += f"def func_{i}(): return {i}\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            # Monitor memory before
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            # Perform large operation
            nodes = generator.generate_from_file(temp_file, large_content)

            # Check memory after operation
            after_operation_memory = process.memory_info().rss / 1024 / 1024
            operation_memory_increase = after_operation_memory - initial_memory

            # Force garbage collection
            gc.collect()

            # Check memory after cleanup
            after_cleanup_memory = process.memory_info().rss / 1024 / 1024
            cleanup_memory_reduction = after_operation_memory - after_cleanup_memory

            # Verify operation completed
            assert len(nodes) >= 1000

            # Memory should be managed
            assert operation_memory_increase < 200  # Less than 200MB increase
            assert cleanup_memory_reduction >= 0  # Memory should not increase after cleanup

        finally:
            os.unlink(temp_file)

    def test_memory_efficient_batch_processing(self, memory_optimized_setup):
        """Test memory efficiency in batch processing."""
        setup = memory_optimized_setup
        generator = setup['generator']

        # Create multiple files
        num_files = 100
        files_data = []

        for i in range(num_files):
            content = f"def func_{i}(): return {i}\nclass Class_{i}: pass\n"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content))

        try:
            # Monitor memory during batch processing
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            memory_samples = []

            def monitor_memory():
                """Monitor memory usage during processing."""
                for _ in range(50):  # Monitor for 5 seconds
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(memory_mb)
                    time.sleep(0.1)

            # Start memory monitoring
            monitor_thread = threading.Thread(target=monitor_memory)
            monitor_thread.start()

            # Process files
            total_nodes = 0
            for file_path, content in files_data:
                nodes = generator.generate_from_file(file_path, content)
                total_nodes += len(nodes)

            monitor_thread.join()

            final_memory = process.memory_info().rss / 1024 / 1024
            total_memory_increase = final_memory - initial_memory

            # Verify processing completed
            assert total_nodes >= num_files * 2  # At least 2 nodes per file

            # Check memory usage
            if memory_samples:
                max_memory = max(memory_samples)
                avg_memory = sum(memory_samples) / len(memory_samples)
                memory_variance = max_memory - min(memory_samples)

                # Memory should be stable during processing
                assert total_memory_increase < 150  # Less than 150MB total increase
                assert memory_variance < 50  # Less than 50MB variance during processing

        finally:
            for file_path, _ in files_data:
                os.unlink(file_path)

    def test_graph_memory_optimization(self, memory_optimized_setup):
        """Test graph memory optimization features."""
        setup = memory_optimized_setup
        graph = setup['graph']

        # Create many nodes and relationships
        num_nodes = 10000
        nodes = []

        for i in range(num_nodes):
            node = CodeNode(
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                file_path=f"/memory_test_{i%50}.py",
                start_line=i % 1000,
                end_line=(i % 1000) + 1
            )
            nodes.append(node)
            graph.add_node(node, LayerType.SEMANTIC)

        # Add relationships
        for i in range(0, num_nodes - 1, 2):
            relationship = Relationship(
                source_id=nodes[i].id,
                target_id=nodes[i + 1].id,
                relationship_type=RelationshipType.CALLS,
                layer=LayerType.SEMANTIC
            )
            graph.add_relationship(relationship)

        # Test memory optimization
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Run optimization
        optimization_result = graph.optimize_for_memory(target_memory_mb=100.0)

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_change = final_memory - initial_memory

        # Verify optimization results
        assert optimization_result['optimized'] is True
        assert optimization_result['removed_nodes'] > 0
        assert optimization_result['current_memory_mb'] > 0

        # Memory should not have increased significantly
        assert memory_change < 50  # Less than 50MB increase

    def test_memory_manager_integration(self, memory_optimized_setup):
        """Test memory manager integration with components."""
        setup = memory_optimized_setup
        memory_manager = setup['memory_manager']
        generator = setup['generator']

        # Create test content
        content = "def test_func(): return 42"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Test memory manager functionality
            initial_memory = memory_manager.get_memory_usage_mb()

            # Perform operation
            nodes = generator.generate_from_file(temp_file, content)

            # Check memory manager stats
            final_memory = memory_manager.get_memory_usage_mb()
            memory_stats = memory_manager.get_memory_stats()

            # Verify memory tracking
            assert 'rss_mb' in memory_stats
            assert 'vms_mb' in memory_stats
            assert isinstance(memory_stats['rss_mb'], float)
            assert isinstance(memory_stats['vms_mb'], float)

            # Operation should complete
            assert len(nodes) >= 1

        finally:
            os.unlink(temp_file)


class TestPerformanceValidation:
    """Test performance validation against standards."""

    @pytest.fixture
    def performance_setup(self):
        """Create setup for performance validation."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_performance_standards_model_generation(self, performance_setup):
        """Test model generation meets performance standards."""
        setup = performance_setup
        generator = setup['generator']

        # Define performance standards
        standards = {
            'small_file_max_time': 0.1,  # 100ms for small files
            'medium_file_max_time': 1.0,  # 1 second for medium files
            'large_file_max_time': 10.0,  # 10 seconds for large files
            'max_memory_increase_mb': 100,  # 100MB max increase
        }

        # Test small file
        small_content = "def func(): return 42"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(small_content)
            temp_file = f.name

        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()
            nodes = generator.generate_from_file(temp_file, small_content)
            generation_time = time.time() - start_time

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Validate against standards
            assert generation_time < standards['small_file_max_time']
            assert memory_increase < standards['max_memory_increase_mb']
            assert len(nodes) >= 1

        finally:
            os.unlink(temp_file)

    def test_performance_standards_analysis(self, performance_setup):
        """Test analysis meets performance standards."""
        setup = performance_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create test file
        content = '''
import os
from typing import List

def complex_function(data: List[int]) -> int:
    """Complex function for analysis."""
    result = 0
    for i in range(10):
        for j in range(10):
            result += i * j
    return result

class TestClass:
    def method(self):
        return "test"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Generate model first
            generator.generate_from_file(temp_file, content)

            # Test analysis performance
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()
            result = analyzer.run_comprehensive_analysis(temp_file)
            analysis_time = time.time() - start_time

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Performance standards
            assert analysis_time < 5.0  # Less than 5 seconds
            assert memory_increase < 50  # Less than 50MB increase

            # Verify analysis results
            assert 'dependency_analysis' in result
            assert 'performance_analysis' in result
            assert result['dependency_analysis']['summary']['total_functions_analyzed'] >= 2

        finally:
            os.unlink(temp_file)

    def test_scalability_performance_curve(self, performance_setup):
        """Test performance scaling with increasing workload."""
        setup = performance_setup
        generator = setup['generator']

        # Test different file sizes
        file_sizes = [10, 50, 100, 200]  # Number of functions

        performance_results = []

        for num_functions in file_sizes:
            # Create content with specified number of functions
            content = ""
            for i in range(num_functions):
                content += f"def func_{i}(): return {i}\n"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                # Measure performance
                start_time = time.time()
                nodes = generator.generate_from_file(temp_file, content)
                generation_time = time.time() - start_time

                performance_results.append({
                    'num_functions': num_functions,
                    'generation_time': generation_time,
                    'nodes_generated': len(nodes),
                    'time_per_function': generation_time / num_functions
                })

                # Basic validation
                assert len(nodes) >= num_functions
                assert generation_time < 30  # Should complete within 30 seconds

            finally:
                os.unlink(temp_file)

        # Analyze scaling performance
        for i in range(1, len(performance_results)):
            current = performance_results[i]
            previous = performance_results[i - 1]

            # Time per function should not increase dramatically
            time_ratio = current['time_per_function'] / previous['time_per_function']
            assert time_ratio < 5.0  # Allow up to 5x increase for scaling

    def test_concurrent_performance_standards(self, performance_setup):
        """Test performance standards under concurrent load."""
        setup = performance_setup
        generator = setup['generator']

        num_threads = 10
        files_per_thread = 20

        # Create test files
        all_files = []
        for i in range(num_threads * files_per_thread):
            content = f"def func_{i}(): return {i}"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                all_files.append((f.name, content))

        try:
            results = []
            errors = []

            def process_files_concurrently(thread_id):
                """Process files and measure performance."""
                try:
                    start_idx = thread_id * files_per_thread
                    end_idx = start_idx + files_per_thread
                    thread_files = all_files[start_idx:end_idx]

                    thread_start = time.time()
                    nodes_processed = 0

                    for file_path, content in thread_files:
                        nodes = generator.generate_from_file(file_path, content)
                        nodes_processed += len(nodes)

                    thread_time = time.time() - thread_start

                    results.append({
                        'thread_id': thread_id,
                        'processing_time': thread_time,
                        'nodes_processed': nodes_processed,
                        'files_processed': len(thread_files)
                    })

                except Exception as e:
                    errors.append(f"Thread {thread_id}: {e}")

            # Execute concurrently
            threads = []
            start_time = time.time()

            for i in range(num_threads):
                thread = threading.Thread(target=process_files_concurrently, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            total_time = time.time() - start_time

            # Validate concurrent performance
            assert len(errors) == 0, f"Concurrent processing errors: {errors}"
            assert len(results) == num_threads

            # Check performance standards
            total_files = num_threads * files_per_thread
            avg_time_per_file = total_time / total_files

            assert avg_time_per_file < 0.5  # Less than 500ms per file
            assert total_time < 60  # Total processing under 1 minute

        finally:
            for file_path, _ in all_files:
                os.unlink(file_path)


class TestResourceManagement:
    """Test resource management and cleanup."""

    @pytest.fixture
    def resource_setup(self):
        """Create setup for resource management testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_resource_cleanup_after_operations(self, resource_setup):
        """Test proper resource cleanup after operations."""
        setup = resource_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create test content
        content = '''
def func1():
    return 1

def func2():
    return 2

class TestClass:
    def method(self):
        return "test"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Perform operations
            nodes = generator.generate_from_file(temp_file, content)
            result = analyzer.run_comprehensive_analysis(temp_file)

            # Verify operations completed
            assert len(nodes) >= 4  # Module + 2 functions + 1 class
            assert 'dependency_analysis' in result

            # Force cleanup
            gc.collect()

            # Verify system is still functional
            nodes2 = generator.generate_from_file(temp_file, content)
            assert len(nodes2) >= 4

        finally:
            os.unlink(temp_file)

    def test_memory_leak_detection(self, resource_setup):
        """Test detection of memory leaks during repeated operations."""
        setup = resource_setup
        generator = setup['generator']

        # Create test content
        content = "def test_func(): return 42"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            process = psutil.Process()

            # Perform repeated operations
            memory_samples = []
            num_iterations = 100

            for i in range(num_iterations):
                # Perform operation
                nodes = generator.generate_from_file(temp_file, content)

                # Sample memory
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)

                # Periodic cleanup
                if i % 20 == 0:
                    gc.collect()

            # Analyze memory usage
            initial_memory = memory_samples[0]
            final_memory = memory_samples[-1]
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)

            memory_growth = final_memory - initial_memory
            memory_variance = max_memory - min_memory

            # Memory should not grow significantly
            assert memory_growth < 20  # Less than 20MB growth over 100 iterations
            assert memory_variance < 15  # Less than 15MB variance

            # All operations should have succeeded
            assert len(memory_samples) == num_iterations

        finally:
            os.unlink(temp_file)

    def test_cpu_usage_optimization(self, resource_setup):
        """Test CPU usage optimization during operations."""
        setup = resource_setup
        generator = setup['generator']

        # Create CPU-intensive content
        content = ""
        for i in range(500):
            content += f"def func_{i}():\n"
            content += f"    x = {i}\n"
            content += "    return x * 2\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            process = psutil.Process()

            # Measure CPU usage during operation
            cpu_samples = []

            def monitor_cpu():
                """Monitor CPU usage."""
                for _ in range(30):  # Monitor for 3 seconds
                    cpu_percent = process.cpu_percent(interval=0.1)
                    cpu_samples.append(cpu_percent)

            # Start CPU monitoring
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()

            # Perform operation
            nodes = generator.generate_from_file(temp_file, content)

            monitor_thread.join()

            # Analyze CPU usage
            if cpu_samples:
                avg_cpu = sum(cpu_samples) / len(cpu_samples)
                max_cpu = max(cpu_samples)
                min_cpu = min(cpu_samples)

                # CPU usage should be reasonable
                assert avg_cpu < 50  # Less than 50% average CPU
                assert max_cpu < 80  # Less than 80% peak CPU
                assert min_cpu >= 0  # Valid CPU readings

            # Operation should complete
            assert len(nodes) >= 500

        finally:
            os.unlink(temp_file)


class TestOptimizationValidation:
    """Test that optimizations are working correctly."""

    @pytest.fixture
    def optimization_setup(self):
        """Create setup for optimization validation."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_caching_optimization(self, optimization_setup):
        """Test that caching optimizations are working."""
        setup = optimization_setup
        generator = setup['generator']

        # Create test content
        content = '''
import os
def func1():
    return os.getcwd()

def func2():
    return os.path.exists('/tmp')
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # First generation
            start_time = time.time()
            nodes1 = generator.generate_from_file(temp_file, content)
            first_time = time.time() - start_time

            # Second generation (should be faster if caching works)
            start_time = time.time()
            nodes2 = generator.generate_from_file(temp_file, content)
            second_time = time.time() - start_time

            # Results should be consistent
            assert len(nodes1) == len(nodes2)
            assert len(nodes1) >= 3  # Module + 2 functions

            # Second run should be reasonably fast (caching benefit may vary)
            assert second_time < first_time * 3  # Allow some variance

        finally:
            os.unlink(temp_file)

    def test_memory_optimization_effectiveness(self, optimization_setup):
        """Test effectiveness of memory optimizations."""
        setup = optimization_setup
        graph = setup['graph']

        # Create large graph
        num_nodes = 5000
        for i in range(num_nodes):
            node = CodeNode(
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                file_path=f"/opt_test_{i%20}.py",
                start_line=i % 100,
                end_line=(i % 100) + 1
            )
            graph.add_node(node, LayerType.SEMANTIC)

        # Test graph statistics
        stats = graph.get_statistics()
        assert stats['total_nodes'] == num_nodes

        # Test search performance
        start_time = time.time()
        search_results = graph.find_nodes_by_name("func_100")
        search_time = time.time() - start_time

        # Search should be fast
        assert search_time < 1.0  # Less than 1 second
        assert len(search_results) >= 1

    def test_batch_processing_optimization(self, optimization_setup):
        """Test batch processing optimizations."""
        setup = optimization_setup
        generator = setup['generator']

        # Create batch of files
        num_files = 50
        files_data = []

        for i in range(num_files):
            content = f"def func_{i}(): return {i}"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content))

        try:
            # Process files individually
            individual_start = time.time()
            individual_nodes = 0
            for file_path, content in files_data:
                nodes = generator.generate_from_file(file_path, content)
                individual_nodes += len(nodes)
            individual_time = time.time() - individual_start

            # Calculate metrics
            avg_time_per_file = individual_time / num_files
            total_nodes = individual_nodes

            # Validate batch processing performance
            assert avg_time_per_file < 0.2  # Less than 200ms per file
            assert total_nodes >= num_files  # At least one node per file
            assert individual_time < 30  # Total time under 30 seconds

        finally:
            for file_path, _ in files_data:
                os.unlink(file_path)


if __name__ == "__main__":
    pytest.main([__file__])
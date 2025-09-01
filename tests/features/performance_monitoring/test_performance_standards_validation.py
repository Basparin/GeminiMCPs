"""
Performance Standards Validation Tests.

This module contains tests that validate the system maintains performance standards
and properly integrates with existing systems and components.
"""

import pytest
import tempfile
import os
import time
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
from codesage_mcp.config.config import ENABLE_CACHING


class TestPerformanceStandardsValidation:
    """Test that the system maintains performance standards."""

    @pytest.fixture
    def performance_validation_setup(self):
        """Create setup for performance validation."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_baseline_performance_standards(self, performance_validation_setup):
        """Test that baseline performance standards are met."""
        setup = performance_validation_setup
        generator = setup['generator']

        # Define performance standards
        standards = {
            'model_generation_time_per_kb': 0.01,  # 10ms per KB
            'memory_usage_per_node': 1024,  # 1KB per node
            'analysis_time_per_function': 0.05,  # 50ms per function
            'max_memory_growth_mb': 50,  # 50MB max growth
        }

        # Test with various file sizes
        test_cases = [
            (1, "def func(): return 1"),  # 1KB
            (5, "def func():\n" + "    x = 1\n" * 100),  # ~5KB
            (10, "def func():\n" + "    x = 1\n" * 200),  # ~10KB
        ]

        for size_kb, content in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                # Measure performance
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024

                start_time = time.time()
                nodes = generator.generate_from_file(temp_file, content)
                generation_time = time.time() - start_time

                final_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = final_memory - initial_memory

                # Validate against standards
                expected_max_time = size_kb * standards['model_generation_time_per_kb']
                assert generation_time <= expected_max_time * 2, f"Generation too slow for {size_kb}KB file"

                expected_max_memory = len(nodes) * standards['memory_usage_per_node'] / (1024 * 1024)
                assert memory_growth <= standards['max_memory_growth_mb'], f"Memory growth too high: {memory_growth}MB"

                assert len(nodes) > 0, "Should generate at least one node"

            finally:
                os.unlink(temp_file)

    def test_analysis_performance_standards(self, performance_validation_setup):
        """Test that analysis meets performance standards."""
        setup = performance_validation_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create test file with multiple functions
        num_functions = 20
        content = ""
        for i in range(num_functions):
            content += f"""
def function_{i}(data):
    result = []
    for item in data:
        result.append(item * {i})
    return result
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Generate model first
            generator.generate_from_file(temp_file, content)

            # Measure analysis performance
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()
            result = analyzer.run_comprehensive_analysis(temp_file)
            analysis_time = time.time() - start_time

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory

            # Validate performance
            expected_max_time = num_functions * 0.05  # 50ms per function
            assert analysis_time <= expected_max_time * 2, f"Analysis too slow: {analysis_time}s for {num_functions} functions"

            assert memory_growth <= 30, f"Memory growth too high: {memory_growth}MB"

            # Validate results
            assert 'dependency_analysis' in result
            assert 'performance_analysis' in result
            assert result['dependency_analysis']['summary']['total_functions_analyzed'] >= num_functions

        finally:
            os.unlink(temp_file)

    def test_scalability_under_load(self, performance_validation_setup):
        """Test system scalability under increasing load."""
        setup = performance_validation_setup
        generator = setup['generator']

        # Test scalability with increasing file counts
        file_counts = [10, 25, 50, 100]

        scalability_results = []

        for num_files in file_counts:
            files_data = []

            # Create test files
            for i in range(num_files):
                content = f"def func_{i}(): return {i}\n"
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(content)
                    files_data.append((f.name, content))

            try:
                # Measure performance
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024

                start_time = time.time()

                total_nodes = 0
                for file_path, content in files_data:
                    nodes = generator.generate_from_file(file_path, content)
                    total_nodes += len(nodes)

                total_time = time.time() - start_time
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = final_memory - initial_memory

                scalability_results.append({
                    'num_files': num_files,
                    'total_time': total_time,
                    'time_per_file': total_time / num_files,
                    'total_nodes': total_nodes,
                    'memory_growth': memory_growth
                })

                # Basic validation
                assert total_time < 60, f"Processing {num_files} files took too long: {total_time}s"
                assert total_nodes >= num_files, f"Should generate at least {num_files} nodes"
                assert memory_growth < 100, f"Memory growth too high: {memory_growth}MB"

            finally:
                for file_path, _ in files_data:
                    os.unlink(file_path)

        # Check scalability trends
        for i in range(1, len(scalability_results)):
            current = scalability_results[i]
            previous = scalability_results[i - 1]

            # Time per file should not increase dramatically
            time_ratio = current['time_per_file'] / previous['time_per_file']
            assert time_ratio < 3.0, f"Poor scalability at {current['num_files']} files: {time_ratio}x slower per file"

    def test_resource_efficiency_standards(self, performance_validation_setup):
        """Test resource efficiency against standards."""
        setup = performance_validation_setup
        generator = setup['generator']

        # Test with large content
        large_content = "x = 1\n" * 10000  # 10,000 lines

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            # Monitor resources
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            initial_cpu = process.cpu_percent(interval=1)

            start_time = time.time()
            nodes = generator.generate_from_file(temp_file, large_content)
            processing_time = time.time() - start_time

            final_memory = process.memory_info().rss / 1024 / 1024
            final_cpu = process.cpu_percent(interval=1)

            memory_growth = final_memory - initial_memory

            # Resource efficiency standards
            assert processing_time < 10, f"Processing too slow: {processing_time}s"
            assert memory_growth < 200, f"Memory growth too high: {memory_growth}MB"
            assert final_cpu < 80, f"CPU usage too high: {final_cpu}%"

            # Should handle large files
            assert len(nodes) >= 1

        finally:
            os.unlink(temp_file)


class TestSystemIntegrationValidation:
    """Test integration with existing systems and components."""

    @pytest.fixture
    def integration_setup(self):
        """Create setup for integration testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_cache_integration_validation(self, integration_setup):
        """Test that caching integration works correctly."""
        setup = integration_setup
        generator = setup['generator']

        if not ENABLE_CACHING:
            pytest.skip("Caching is disabled")

        # Create test content
        content = "def test_func(): return 42"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # First generation
            start_time = time.time()
            nodes1 = generator.generate_from_file(temp_file, content)
            first_time = time.time() - start_time

            # Second generation (should be faster due to cache)
            start_time = time.time()
            nodes2 = generator.generate_from_file(temp_file, content)
            second_time = time.time() - start_time

            # Results should be consistent
            assert len(nodes1) == len(nodes2)
            assert len(nodes1) >= 1

            # Second run should be faster (cache benefit)
            assert second_time <= first_time, f"Cache not working: {second_time} vs {first_time}"

        finally:
            os.unlink(temp_file)

    def test_memory_manager_integration(self, integration_setup):
        """Test memory manager integration."""
        setup = integration_setup
        generator = setup['generator']

        # Create memory-intensive content
        content = ""
        for i in range(1000):
            content += f"def func_{i}():\n    data = list(range(100))\n    return sum(data)\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Monitor memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            nodes = generator.generate_from_file(temp_file, content)

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory

            # Should handle memory-intensive operations
            assert len(nodes) >= 1000
            assert memory_growth < 300  # Reasonable memory growth

        finally:
            os.unlink(temp_file)

    def test_graph_persistence_integration(self, integration_setup):
        """Test graph persistence and loading integration."""
        setup = integration_setup
        graph = setup['graph']

        # Create some nodes and relationships
        for i in range(10):
            node = CodeNode(
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                file_path="/test.py",
                start_line=i,
                end_line=i+1
            )
            graph.add_node(node, LayerType.SEMANTIC)

        # Add relationships
        for i in range(9):
            relationship = Relationship(
                source_id=f"func_{i}",
                target_id=f"func_{i+1}",
                relationship_type=RelationshipType.CALLS,
                layer=LayerType.SEMANTIC
            )
            graph.add_relationship(relationship)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Save graph
            graph.save_to_file(temp_file)

            # Create new graph and load
            new_graph = CodeGraph()
            new_graph.load_from_file(temp_file)

            # Verify data integrity
            original_stats = graph.get_statistics()
            loaded_stats = new_graph.get_statistics()

            assert loaded_stats['total_nodes'] == original_stats['total_nodes']
            assert loaded_stats['total_relationships'] == original_stats['total_relationships']

        finally:
            os.unlink(temp_file)

    def test_concurrent_access_integration(self, integration_setup):
        """Test concurrent access to integrated components."""
        setup = integration_setup
        graph = setup['graph']
        generator = setup['generator']

        num_threads = 8
        operations_per_thread = 25

        results = []
        errors = []

        def concurrent_operations(thread_id):
            """Perform concurrent operations."""
            try:
                thread_results = []

                for i in range(operations_per_thread):
                    # Add node
                    node = CodeNode(
                        node_type=NodeType.FUNCTION,
                        name=f"concurrent_func_{thread_id}_{i}",
                        qualified_name=f"concurrent_func_{thread_id}_{i}",
                        file_path=f"/concurrent_{thread_id}.py",
                        start_line=i,
                        end_line=i+1
                    )
                    graph.add_node(node, LayerType.SEMANTIC)

                    # Query graph
                    if i % 5 == 0:
                        stats = graph.get_statistics()
                        thread_results.append(stats['total_nodes'])

                results.append(thread_results)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Execute concurrently
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=concurrent_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Validate concurrent access
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == num_threads

        # Check final graph state
        final_stats = graph.get_statistics()
        expected_nodes = num_threads * operations_per_thread
        assert final_stats['total_nodes'] == expected_nodes


class TestEndToEndWorkflowValidation:
    """Test end-to-end workflows to ensure complete integration."""

    @pytest.fixture
    def workflow_setup(self):
        """Create setup for end-to-end workflow testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_complete_codebase_workflow(self, workflow_setup):
        """Test complete workflow from file to analysis."""
        setup = workflow_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create a complete codebase
        codebase = {
            'main.py': '''
from utils import helper
from data import process

def main():
    data = [1, 2, 3, 4, 5]
    processed = process(data)
    result = helper(processed)
    return result
''',
            'utils.py': '''
def helper(data):
    return sum(data)

class Utils:
    @staticmethod
    def validate(data):
        return len(data) > 0
''',
            'data.py': '''
import os

def process(data):
    # Nested loop (potential bottleneck)
    result = []
    for i in range(len(data)):
        for j in range(3):
            result.append(data[i] * j)
    return result
'''
        }

        temp_files = []
        try:
            # Create temporary files
            for filename, content in codebase.items():
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(content)
                    temp_files.append(f.name)

            # Process all files
            all_nodes = []
            for file_path in temp_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                nodes = generator.generate_from_file(file_path, content)
                all_nodes.extend(nodes)

            # Run comprehensive analysis
            analysis_results = []
            for file_path in temp_files:
                result = analyzer.run_comprehensive_analysis(file_path)
                analysis_results.append(result)

            # Validate complete workflow
            assert len(all_nodes) >= 6  # At least 6 nodes across all files
            assert len(analysis_results) == len(temp_files)

            # Check analysis quality
            total_functions = 0
            total_bottlenecks = 0

            for result in analysis_results:
                dep_analysis = result.get('dependency_analysis', {})
                perf_analysis = result.get('performance_analysis', {})

                total_functions += dep_analysis.get('summary', {}).get('total_functions_analyzed', 0)
                total_bottlenecks += len(perf_analysis.get('bottlenecks', []))

            assert total_functions >= 4  # Should find all functions
            assert total_bottlenecks >= 1  # Should detect nested loops in data.py

        finally:
            for file_path in temp_files:
                os.unlink(file_path)

    def test_workflow_performance_standards(self, workflow_setup):
        """Test that complete workflows meet performance standards."""
        setup = workflow_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create test files
        num_files = 20
        files_data = []

        for i in range(num_files):
            content = f"""
import os
def func_{i}():
    data = list(range(10))
    return sum(data)

class Class_{i}:
    def method(self):
        return {i}
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content))

        try:
            # Measure complete workflow performance
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()

            # Process all files
            total_nodes = 0
            for file_path, content in files_data:
                nodes = generator.generate_from_file(file_path, content)
                total_nodes += len(nodes)

            # Analyze subset
            analysis_sample = files_data[:5]
            for file_path, _ in analysis_sample:
                analyzer.run_comprehensive_analysis(file_path)

            total_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory

            # Performance standards for complete workflow
            assert total_time < 30, f"Complete workflow too slow: {total_time}s"
            assert memory_growth < 150, f"Memory growth too high: {memory_growth}MB"
            assert total_nodes >= num_files * 2, f"Insufficient nodes generated: {total_nodes}"

        finally:
            for file_path, _ in files_data:
                os.unlink(file_path)


if __name__ == "__main__":
    pytest.main([__file__])
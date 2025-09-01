"""
Performance Tests for Large Codebase Analysis.

This module contains performance tests specifically designed to test the system's
ability to handle large codebases efficiently, including memory usage, processing
speed, and scalability.
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
    Relationship,
    RelationshipType,
    LayerType
)
from codesage_mcp.features.codebase_manager import AdvancedAnalysisManager
from codesage_mcp.features.memory_management.memory_manager import MemoryManager


class TestLargeCodebasePerformance:
    """Performance tests for large codebase handling."""

    @pytest.fixture
    def large_codebase_setup(self):
        """Create setup optimized for large codebase testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        # Configure for large codebase handling
        memory_manager = MemoryManager()
        generator.memory_manager = memory_manager
        analyzer.memory_manager = memory_manager

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer,
            'memory_manager': memory_manager
        }

    def test_1000_file_codebase_generation(self, large_codebase_setup):
        """Test code model generation for a 1000-file codebase."""
        setup = large_codebase_setup
        generator = setup['generator']

        num_files = 1000
        files_data = []

        # Create 1000 files
        for i in range(num_files):
            content = f"""
def function_{i}():
    return {i}

class Class_{i}:
    def method(self):
        return "method_{i}"
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content))

        try:
            # Monitor memory and time
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()

            # Generate models for all files
            for file_path, content in files_data:
                generator.generate_from_file(file_path, content)

            generation_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Performance assertions
            assert generation_time < 300  # Less than 5 minutes
            assert memory_increase < 1000  # Less than 1GB increase

            # Verify graph contains all nodes
            stats = generator.graph.get_statistics()
            assert stats['total_nodes'] >= num_files * 2  # At least 2 nodes per file
            assert stats['total_files'] == num_files

        finally:
            # Cleanup files
            for file_path, _ in files_data:
                os.unlink(file_path)

    def test_1000_file_codebase_analysis(self, large_codebase_setup):
        """Test analysis of a 1000-file codebase."""
        setup = large_codebase_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        num_files = 500  # Reduced for analysis testing
        files_data = []

        # Create files with analyzable content
        for i in range(num_files):
            content = f"""
import os
from typing import List

def complex_function_{i}(data: List[int]) -> int:
    result = 0
    for item in data:
        for j in range(10):  # Nested loop
            result += item * j
    return result

class Processor_{i}:
    def __init__(self):
        self.data = [0] * 100

    def process(self):
        return sum(self.data)
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content))

        try:
            # Generate models first
            for file_path, content in files_data:
                generator.generate_from_file(file_path, content)

            # Analyze subset of files
            analysis_files = files_data[:100]  # Analyze first 100 files

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()

            results = []
            for file_path, _ in analysis_files:
                result = analyzer.run_comprehensive_analysis(file_path)
                results.append(result)

            analysis_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Performance assertions
            assert analysis_time < 180  # Less than 3 minutes
            assert memory_increase < 500  # Less than 500MB increase

            # Verify results
            assert len(results) == 100
            for result in results:
                assert 'dependency_analysis' in result
                assert 'performance_analysis' in result

                # Should have detected dependencies and bottlenecks
                deps = result['dependency_analysis']['summary']['total_functions_analyzed']
                bottlenecks = len(result['performance_analysis']['bottlenecks'])

                assert deps >= 2  # At least 2 functions per file
                assert bottlenecks >= 1  # Should detect nested loops

        finally:
            for file_path, _ in files_data:
                os.unlink(file_path)

    def test_memory_optimization_large_graph(self, large_codebase_setup):
        """Test memory optimization with large graphs."""
        setup = large_codebase_setup
        graph = setup['graph']
        memory_manager = setup['memory_manager']

        # Create a large graph
        num_nodes = 50000
        nodes = []

        for i in range(num_nodes):
            node = CodeNode(
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                file_path=f"/large_test_{i%100}.py",
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
        memory_reduction = initial_memory - final_memory

        # Verify optimization worked
        assert optimization_result['optimized'] is True
        assert optimization_result['removed_nodes'] > 0
        assert memory_reduction >= 0  # Memory should not increase

    def test_concurrent_large_codebase_processing(self, large_codebase_setup):
        """Test concurrent processing of large codebases."""
        setup = large_codebase_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        num_files = 200
        num_threads = 8
        files_per_thread = num_files // num_threads

        # Create files
        all_files = []
        for i in range(num_files):
            content = f"""
def func_{i}():
    return {i}

class Class_{i}:
    def method(self):
        return "method_{i}"
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                all_files.append((f.name, content))

        try:
            results = {}
            errors = []

            def process_files_concurrently(thread_id):
                """Process files for a specific thread."""
                try:
                    start_idx = thread_id * files_per_thread
                    end_idx = start_idx + files_per_thread
                    thread_files = all_files[start_idx:end_idx]

                    thread_results = []
                    for file_path, content in thread_files:
                        # Generate model
                        generator.generate_from_file(file_path, content)

                        # Run analysis
                        result = analyzer.run_comprehensive_analysis(file_path)
                        thread_results.append(result)

                    results[thread_id] = thread_results

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

            concurrent_time = time.time() - start_time

            # Verify results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == num_threads

            total_results = sum(len(thread_results) for thread_results in results.values())
            assert total_results == num_files

            # Performance check
            assert concurrent_time < 120  # Less than 2 minutes

        finally:
            for file_path, _ in all_files:
                os.unlink(file_path)


class TestScalabilityBenchmarks:
    """Scalability benchmarks for different codebase sizes."""

    @pytest.fixture
    def scalability_setup(self):
        """Create setup for scalability testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    @pytest.mark.parametrize("num_files", [10, 50, 100, 200])
    def test_scalability_file_count(self, scalability_setup, num_files):
        """Test scalability with different numbers of files."""
        setup = scalability_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create test files
        files_data = []
        for i in range(num_files):
            content = f"""
import os

def func_{i}():
    return os.getcwd()

class Class_{i}:
    def method(self):
        return {i}
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content))

        try:
            # Measure generation time
            start_time = time.time()
            for file_path, content in files_data:
                generator.generate_from_file(file_path, content)
            generation_time = time.time() - start_time

            # Measure analysis time (sample of files)
            analysis_sample = min(10, num_files)
            analysis_files = files_data[:analysis_sample]

            start_time = time.time()
            for file_path, _ in analysis_files:
                analyzer.run_comprehensive_analysis(file_path)
            analysis_time = time.time() - start_time

            # Calculate per-file metrics
            generation_per_file = generation_time / num_files
            analysis_per_file = analysis_time / analysis_sample

            # Scalability assertions
            assert generation_per_file < 1.0  # Less than 1 second per file
            assert analysis_per_file < 5.0  # Less than 5 seconds per file

            # Memory check
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            assert memory_mb < 1000  # Less than 1GB total

        finally:
            for file_path, _ in files_data:
                os.unlink(file_path)

    @pytest.mark.parametrize("file_size_kb", [1, 10, 50, 100])
    def test_scalability_file_size(self, scalability_setup, file_size_kb):
        """Test scalability with different file sizes."""
        setup = scalability_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create files of different sizes
        num_functions = file_size_kb * 10  # Roughly 10 functions per KB

        content = "import os\nfrom typing import List\n\n"
        for i in range(num_functions):
            content += f"""
def function_{i}(data: List[int]) -> int:
    result = 0
    for item in data:
        result += item * {i}
    return result

"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Measure processing time
            start_time = time.time()
            generator.generate_from_file(temp_file, content)
            generation_time = time.time() - start_time

            start_time = time.time()
            result = analyzer.run_comprehensive_analysis(temp_file)
            analysis_time = time.time() - start_time

            # Scalability assertions
            assert generation_time < 10  # Less than 10 seconds regardless of size
            assert analysis_time < 30  # Less than 30 seconds regardless of size

            # Verify analysis quality
            deps = result['dependency_analysis']['summary']['total_functions_analyzed']
            assert deps >= num_functions * 0.8  # Should find most functions

        finally:
            os.unlink(temp_file)


class TestMemoryStressTests:
    """Memory stress tests for large codebases."""

    @pytest.fixture
    def memory_stress_setup(self):
        """Create setup for memory stress testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        memory_manager = MemoryManager()
        generator.memory_manager = memory_manager
        analyzer.memory_manager = memory_manager

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer,
            'memory_manager': memory_manager
        }

    def test_memory_pressure_with_many_relationships(self, memory_stress_setup):
        """Test memory pressure with many relationships."""
        setup = memory_stress_setup
        graph = setup['graph']

        # Create many nodes and relationships
        num_nodes = 10000
        nodes = []

        # Create nodes
        for i in range(num_nodes):
            node = CodeNode(
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                file_path=f"/stress_test_{i%50}.py",
                start_line=i % 100,
                end_line=(i % 100) + 1
            )
            nodes.append(node)
            graph.add_node(node, LayerType.SEMANTIC)

        # Create many relationships (each node calls 3 others)
        for i in range(num_nodes):
            for j in range(3):
                target_idx = (i + j + 1) % num_nodes
                relationship = Relationship(
                    source_id=nodes[i].id,
                    target_id=nodes[target_idx].id,
                    relationship_type=RelationshipType.CALLS,
                    layer=LayerType.SEMANTIC
                )
                graph.add_relationship(relationship)

        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Test graph operations under memory pressure
        stats = graph.get_statistics()
        search_results = graph.find_nodes_by_name("func_5")

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # Verify functionality
        assert stats['total_nodes'] == num_nodes
        assert stats['total_relationships'] >= num_nodes * 2  # At least 2 relationships per node
        assert len(search_results) >= 50  # Should find nodes across files

        # Memory should be reasonable
        assert memory_increase < 500  # Less than 500MB increase

    def test_graph_serialization_large_dataset(self, memory_stress_setup):
        """Test graph serialization with large datasets."""
        setup = memory_stress_setup
        graph = setup['graph']

        # Create large graph
        num_nodes = 5000
        for i in range(num_nodes):
            node = CodeNode(
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                file_path=f"/large_test_{i%20}.py",
                start_line=i % 50,
                end_line=(i % 50) + 1
            )
            graph.add_node(node, LayerType.SEMANTIC)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Test serialization
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()
            graph.save_to_file(temp_file)
            save_time = time.time() - start_time

            save_memory = process.memory_info().rss / 1024 / 1024
            save_memory_increase = save_memory - initial_memory

            # Check file size
            file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
            assert file_size_mb < 50  # Less than 50MB

            # Test deserialization
            new_graph = CodeGraph()
            load_start = time.time()
            new_graph.load_from_file(temp_file)
            load_time = time.time() - load_start

            load_memory = process.memory_info().rss / 1024 / 1024
            load_memory_increase = load_memory - save_memory

            # Verify loaded data
            loaded_stats = new_graph.get_statistics()
            assert loaded_stats['total_nodes'] == num_nodes

            # Performance checks
            assert save_time < 30  # Less than 30 seconds to save
            assert load_time < 30  # Less than 30 seconds to load
            assert save_memory_increase < 200  # Less than 200MB during save
            assert load_memory_increase < 200  # Less than 200MB during load

        finally:
            os.unlink(temp_file)


class TestPerformanceRegressionDetection:
    """Tests to detect performance regressions."""

    @pytest.fixture
    def performance_baseline(self):
        """Establish performance baseline."""
        return {
            'model_generation_per_file': 0.1,  # seconds
            'analysis_per_file': 1.0,  # seconds
            'memory_increase_mb': 50,  # MB
            'max_processing_time': 60  # seconds for 100 files
        }

    def test_performance_regression_model_generation(self, performance_baseline, large_codebase_setup):
        """Test for performance regression in model generation."""
        setup = large_codebase_setup
        generator = setup['generator']

        # Create test files
        num_files = 100
        files_data = []

        for i in range(num_files):
            content = f"def func_{i}(): return {i}"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content))

        try:
            # Measure performance
            start_time = time.time()
            for file_path, content in files_data:
                generator.generate_from_file(file_path, content)
            total_time = time.time() - start_time

            avg_time_per_file = total_time / num_files

            # Check against baseline
            assert avg_time_per_file < performance_baseline['model_generation_per_file'] * 2  # Allow 2x regression
            assert total_time < performance_baseline['max_processing_time']

        finally:
            for file_path, _ in files_data:
                os.unlink(file_path)

    def test_performance_regression_analysis(self, performance_baseline, large_codebase_setup):
        """Test for performance regression in analysis."""
        setup = large_codebase_setup
        generator = setup['generator']
        analyzer = setup['analyzer']

        # Create test files
        num_files = 50
        files_data = []

        for i in range(num_files):
            content = f"""
import os
def func_{i}():
    return os.getcwd()
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content))

        try:
            # Generate models
            for file_path, content in files_data:
                generator.generate_from_file(file_path, content)

            # Measure analysis performance
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            start_time = time.time()
            for file_path, _ in files_data:
                analyzer.run_comprehensive_analysis(file_path)
            total_time = time.time() - start_time

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            avg_time_per_file = total_time / num_files

            # Check against baseline
            assert avg_time_per_file < performance_baseline['analysis_per_file'] * 2  # Allow 2x regression
            assert memory_increase < performance_baseline['memory_increase_mb'] * 2  # Allow 2x memory increase

        finally:
            for file_path, _ in files_data:
                os.unlink(file_path)


if __name__ == "__main__":
    pytest.main([__file__])
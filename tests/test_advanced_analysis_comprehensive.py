"""
Comprehensive Unit Tests for Advanced Analysis Components.

This module contains extensive tests covering edge cases, error handling, performance,
memory optimization, and integration scenarios for the advanced analysis components.
"""

import pytest
import tempfile
import os
import time
import threading
import ast
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import psutil
import numpy as np
from typing import Dict, List, Any, Optional

from codesage_mcp.advanced_analysis import (
    AdvancedDependencyAnalyzer,
    PerformancePredictor,
    AdvancedAnalysisManager
)
from codesage_mcp.code_model import CodeGraph, CodeNode, NodeType, RelationshipType, LayerType
from codesage_mcp.exceptions import BaseMCPError


class TestAdvancedDependencyAnalyzerEdgeCases:
    """Test edge cases for AdvancedDependencyAnalyzer."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample code graph for testing."""
        graph = CodeGraph()
        return graph

    @pytest.fixture
    def analyzer(self, sample_graph):
        """Create analyzer instance."""
        return AdvancedDependencyAnalyzer(sample_graph)

    def test_empty_file_analysis(self, analyzer):
        """Test analysis of empty files."""
        # Create empty file content
        empty_content = ""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(empty_content)
            temp_file = f.name

        try:
            # Add a module node for the empty file
            module_node = CodeNode(
                node_type=NodeType.MODULE,
                name="empty_module",
                qualified_name="empty_module",
                file_path=temp_file,
                start_line=1,
                end_line=1,
                content=empty_content
            )
            analyzer.graph.add_node(module_node, LayerType.SEMANTIC)

            result = analyzer.analyze_function_dependencies(temp_file)

            # Should handle gracefully
            assert "dependencies" in result
            assert "summary" in result
            assert result["summary"]["total_functions_analyzed"] == 0

        finally:
            os.unlink(temp_file)

    def test_malformed_code_analysis(self, analyzer):
        """Test analysis of malformed code."""
        malformed_codes = [
            "def broken(\n    return",  # Syntax error
            "class Broken:\n    def method(self\n        pass",  # Indentation error
            "import nonexistent_module",  # Import error
        ]

        for malformed_code in malformed_codes:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(malformed_code)
                temp_file = f.name

            try:
                # Add a function node with malformed content
                func_node = CodeNode(
                    node_type=NodeType.FUNCTION,
                    name="broken_func",
                    qualified_name="broken_func",
                    file_path=temp_file,
                    start_line=1,
                    end_line=3,
                    content=malformed_code
                )
                analyzer.graph.add_node(func_node, LayerType.SEMANTIC)

                # Should handle malformed code gracefully
                deps = analyzer._analyze_single_function_dependencies(func_node)
                assert isinstance(deps, dict)
                assert "complexity_score" in deps

            finally:
                os.unlink(temp_file)

    def test_complex_dependency_patterns(self, analyzer):
        """Test analysis of complex dependency patterns."""
        complex_code = '''
import os
import sys
from typing import List, Dict
from collections import defaultdict
import numpy as np
import pandas as pd

def complex_function(data: List[Dict], config: dict):
    """Function with complex dependencies."""
    # Use standard library
    result = []
    for item in data:
        processed = os.path.join(config.get('path', '/'), str(item.get('id', 0)))
        result.append(processed)

    # Use external libraries
    if np and pd:  # Check if libraries are available
        arr = np.array(result)
        df = pd.DataFrame({'paths': result})

    # Use typing constructs
    typed_var: Dict[str, List[int]] = defaultdict(list)

    return result

class ComplexClass:
    def __init__(self):
        self.data = defaultdict(list)

    def process_with_dependencies(self, items):
        """Method with multiple dependencies."""
        # Use instance variable
        for item in items:
            self.data['processed'].append(item)

        # Use imported functions
        return len(self.data['processed'])
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_code)
            temp_file = f.name

        try:
            # Generate code model first
            from codesage_mcp.code_model import CodeModelGenerator
            generator = CodeModelGenerator(analyzer.graph)
            nodes = generator.generate_from_file(temp_file, complex_code)

            # Analyze dependencies
            result = analyzer.analyze_function_dependencies(temp_file, "complex_function")

            assert "dependencies" in result
            deps = result["dependencies"]["complex_function"]

            # Should detect various dependencies
            assert len(deps["external_libraries"]) > 0
            assert len(deps["imports_used"]) > 0
            assert deps["complexity_score"] > 0

            # Check for specific libraries
            external_libs = deps["external_libraries"]
            assert any(lib in ['numpy', 'pandas', 'os', 'sys'] for lib in external_libs)

        finally:
            os.unlink(temp_file)

    def test_circular_dependency_detection(self, analyzer):
        """Test detection of circular dependencies."""
        # Create code with potential circular dependencies
        circular_code = '''
def func_a():
    return func_b()

def func_b():
    return func_c()

def func_c():
    return func_a()  # Circular call
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(circular_code)
            temp_file = f.name

        try:
            # Generate model and relationships
            from codesage_mcp.code_model import CodeModelGenerator
            generator = CodeModelGenerator(analyzer.graph)
            nodes = generator.generate_from_file(temp_file, circular_code)

            # Analyze dependencies
            result = analyzer.analyze_function_dependencies(temp_file)

            # Should handle circular dependencies gracefully
            assert "dependencies" in result
            assert len(result["dependencies"]) >= 3  # All three functions

            # Check that complexity scores are reasonable
            for func_name, deps in result["dependencies"].items():
                assert deps["complexity_score"] >= 0
                assert deps["complexity_score"] < 10  # Should not be excessively high

        finally:
            os.unlink(temp_file)

    def test_large_function_analysis(self, analyzer):
        """Test analysis of very large functions."""
        # Create a large function with many dependencies
        large_function = '''
def large_function():
    """A very large function with many dependencies."""
    import os, sys, json, re
    from collections import Counter, defaultdict
    from typing import List, Dict, Set
    import math, random, datetime

    # Many variables and operations
    data = []
    for i in range(1000):
        item = {
            'id': i,
            'value': random.random(),
            'timestamp': datetime.datetime.now(),
            'path': os.path.join('/tmp', str(i))
        }
        data.append(item)

    # Complex processing
    counter = Counter()
    grouped: Dict[str, List] = defaultdict(list)

    for item in data:
        key = str(item['id'] % 10)
        grouped[key].append(item)
        counter[key] += 1

    # Use regex
    pattern = re.compile(r'\d+')
    matches = [pattern.findall(str(item['id'])) for item in data]

    # Use math operations
    result = sum(math.sqrt(item['value']) for item in data)

    return {
        'result': result,
        'count': len(data),
        'groups': dict(grouped),
        'matches': matches
    }
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_function)
            temp_file = f.name

        try:
            # Generate model
            from codesage_mcp.code_model import CodeModelGenerator
            generator = CodeModelGenerator(analyzer.graph)
            nodes = generator.generate_from_file(temp_file, large_function)

            # Analyze dependencies
            result = analyzer.analyze_function_dependencies(temp_file, "large_function")

            assert "dependencies" in result
            deps = result["dependencies"]["large_function"]

            # Should detect many external libraries
            external_libs = deps["external_libraries"]
            expected_libs = ['os', 'sys', 'json', 're', 'math', 'random', 'datetime']
            found_libs = [lib for lib in expected_libs if lib in external_libs]
            assert len(found_libs) >= 3  # Should find several libraries

            # Should have reasonable complexity score
            assert deps["complexity_score"] > 5  # Complex function
            assert deps["complexity_score"] < 20  # Not excessively high

        finally:
            os.unlink(temp_file)


class TestPerformancePredictorEdgeCases:
    """Test edge cases for PerformancePredictor."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample code graph for testing."""
        graph = CodeGraph()
        return graph

    @pytest.fixture
    def predictor(self, sample_graph):
        """Create predictor instance."""
        return PerformancePredictor(sample_graph)

    def test_empty_file_prediction(self, predictor):
        """Test performance prediction on empty files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")
            temp_file = f.name

        try:
            result = predictor.predict_bottlenecks(temp_file)

            # Should handle empty files gracefully
            assert "bottlenecks" in result
            assert isinstance(result["bottlenecks"], list)
            assert len(result["bottlenecks"]) == 0  # No bottlenecks in empty file

        finally:
            os.unlink(temp_file)

    def test_deeply_nested_structures(self, predictor):
        """Test prediction on deeply nested code structures."""
        # Create deeply nested code
        nested_code = "x = 1\n"
        for i in range(15):  # 15 levels of nesting
            nested_code += "    " * i + f"if x == {i}:\n"
        nested_code += "    " * 15 + "pass\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(nested_code)
            temp_file = f.name

        try:
            # Add a function node
            func_node = CodeNode(
                node_type=NodeType.FUNCTION,
                name="nested_func",
                qualified_name="nested_func",
                file_path=temp_file,
                start_line=1,
                end_line=len(nested_code.split('\n')),
                content=nested_code
            )
            predictor.graph.add_node(func_node, LayerType.SEMANTIC)

            result = predictor.predict_bottlenecks(temp_file)

            # Should detect deeply nested structures
            assert "bottlenecks" in result
            bottlenecks = result["bottlenecks"]

            # Should find nested loops bottleneck
            nested_bottlenecks = [b for b in bottlenecks if b.get("type") == "nested_loops"]
            assert len(nested_bottlenecks) > 0

            # Should have high severity for deep nesting
            for bottleneck in nested_bottlenecks:
                assert bottleneck["severity_score"] >= 6

        finally:
            os.unlink(temp_file)

    def test_inefficient_operations_detection(self, predictor):
        """Test detection of various inefficient operations."""
        inefficient_code = '''
def inefficient_function():
    """Function with various inefficient operations."""
    # String concatenation in loop
    result = ""
    for i in range(1000):
        result += str(i) + ","

    # Large list comprehensions
    large_list = [x * 2 for x in range(10000)]

    # Nested comprehensions
    nested_comp = [[x * y for x in range(100)] for y in range(100)]

    # Recursive function without base case
    def recursive_func(n):
        if n > 0:
            return recursive_func(n - 1)
        # Missing return statement - infinite recursion potential

    return result
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(inefficient_code)
            temp_file = f.name

        try:
            # Generate model
            from codesage_mcp.code_model import CodeModelGenerator
            generator = CodeModelGenerator(predictor.graph)
            nodes = generator.generate_from_file(temp_file, inefficient_code)

            result = predictor.predict_bottlenecks(temp_file)

            assert "bottlenecks" in result
            bottlenecks = result["bottlenecks"]

            # Should detect multiple types of bottlenecks
            bottleneck_types = [b["type"] for b in bottlenecks]

            # Should find inefficient string operations
            assert "inefficient_string_operations" in bottleneck_types

            # Should find large data structures
            assert "large_data_structures" in bottleneck_types

            # Should find complex comprehensions
            assert "complex_comprehension" in bottleneck_types

            # Should find recursion issues
            assert "recursion" in bottleneck_types

        finally:
            os.unlink(temp_file)

    def test_performance_prediction_large_files(self, predictor):
        """Test performance prediction on large files."""
        # Create a large file with multiple functions
        large_content = ""
        for i in range(100):
            large_content += f"""
def function_{i}():
    result = []
    for j in range(100):  # Nested loop
        for k in range(50):  # Deep nesting
            result.append(j * k)
    return result
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            # Generate model
            from codesage_mcp.code_model import CodeModelGenerator
            generator = CodeModelGenerator(predictor.graph)
            nodes = generator.generate_from_file(temp_file, large_content)

            # Time the prediction
            start_time = time.time()
            result = predictor.predict_bottlenecks(temp_file)
            prediction_time = time.time() - start_time

            # Should complete in reasonable time
            assert prediction_time < 10  # Less than 10 seconds

            # Should find many bottlenecks
            assert "bottlenecks" in result
            bottlenecks = result["bottlenecks"]
            assert len(bottlenecks) >= 50  # Should find bottlenecks in most functions

            # Should have nested loops bottlenecks
            nested_bottlenecks = [b for b in bottlenecks if b.get("type") == "nested_loops"]
            assert len(nested_bottlenecks) >= 50  # Most functions have nested loops

        finally:
            os.unlink(temp_file)


class TestAdvancedAnalysisErrorHandling:
    """Test error handling in advanced analysis components."""

    @pytest.fixture
    def analysis_manager(self):
        """Create an AdvancedAnalysisManager instance."""
        graph = CodeGraph()
        return AdvancedAnalysisManager(graph)

    def test_corrupted_graph_handling(self, analysis_manager):
        """Test handling of corrupted graph data."""
        # Simulate corrupted graph by adding invalid nodes
        invalid_node = CodeNode(
            node_type="invalid_type",  # Invalid enum value
            name="invalid",
            qualified_name="invalid",
            file_path="/invalid.py",
            start_line=1,
            end_line=1
        )

        # Should handle gracefully when accessing invalid data
        try:
            analysis_manager.graph.add_node(invalid_node, LayerType.SEMANTIC)
            result = analysis_manager.run_comprehensive_analysis()
            # Should not crash
            assert isinstance(result, dict)
        except Exception as e:
            # If it does crash, should be a meaningful error
            assert isinstance(e, (ValueError, TypeError, AttributeError))

    def test_memory_exhaustion_during_analysis(self, analysis_manager):
        """Test handling of memory exhaustion during analysis."""
        # Create a very large codebase
        large_content = ""
        for i in range(500):
            large_content += f"""
def func_{i}():
    data = list(range(1000))  # Large data structures
    result = []
    for item in data:
        for j in range(100):  # Nested loops
            result.append(item * j)
    return result

class Class_{i}:
    def __init__(self):
        self.large_data = [0] * 10000  # Large instance data

    def complex_method(self):
        return sum(self.large_data) * {i}
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            # Generate model
            from codesage_mcp.code_model import CodeModelGenerator
            generator = CodeModelGenerator(analysis_manager.graph)
            nodes = generator.generate_from_file(temp_file, large_content)

            # Run analysis
            result = analysis_manager.run_comprehensive_analysis(temp_file)

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Should not cause excessive memory usage
            assert memory_increase < 300  # Less than 300MB increase
            assert result is not None

        finally:
            os.unlink(temp_file)

    def test_timeout_handling(self, analysis_manager):
        """Test handling of timeout scenarios."""
        # Create code that might cause analysis to take long
        complex_content = ""
        for i in range(200):
            complex_content += f"""
def complex_func_{i}():
    # Very complex nested structures
    result = []
    for a in range(10):
        for b in range(10):
            for c in range(10):
                for d in range(10):
                    result.append(a * b * c * d)
    return result
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_content)
            temp_file = f.name

        try:
            # Time the analysis
            start_time = time.time()
            result = analysis_manager.run_comprehensive_analysis(temp_file)
            analysis_time = time.time() - start_time

            # Should complete within reasonable time
            assert analysis_time < 30  # Less than 30 seconds
            assert result is not None

            # Should still produce valid results
            assert "dependency_analysis" in result
            assert "performance_analysis" in result

        finally:
            os.unlink(temp_file)


class TestAdvancedAnalysisPerformance:
    """Test performance aspects of advanced analysis."""

    @pytest.fixture
    def analysis_manager(self):
        """Create an AdvancedAnalysisManager instance."""
        graph = CodeGraph()
        return AdvancedAnalysisManager(graph)

    def test_batch_analysis_performance(self, analysis_manager):
        """Test performance of batch analysis operations."""
        # Create multiple files
        num_files = 10
        file_paths = []
        file_contents = []

        for i in range(num_files):
            content = f"""
import os, sys
from typing import List

def func_{i}(data: List[int]):
    result = []
    for item in data:
        processed = os.path.join('/tmp', str(item))
        result.append(processed)
    return result

class Class_{i}:
    def method(self):
        return sys.version
"""
            file_contents.append(content)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                file_paths.append(f.name)

        try:
            # Generate models for all files
            from codesage_mcp.code_model import CodeModelGenerator
            generator = CodeModelGenerator(analysis_manager.graph)

            model_start = time.time()
            for path, content in zip(file_paths, file_contents):
                generator.generate_from_file(path, content)
            model_time = time.time() - model_start

            # Run batch analysis
            analysis_start = time.time()
            results = []
            for path in file_paths:
                result = analysis_manager.run_comprehensive_analysis(path)
                results.append(result)
            analysis_time = time.time() - analysis_start

            # Performance checks
            assert model_time < 5  # Model generation should be fast
            assert analysis_time < 15  # Analysis should be reasonable

            # Verify results
            assert len(results) == num_files
            for result in results:
                assert "dependency_analysis" in result
                assert "performance_analysis" in result

        finally:
            for path in file_paths:
                os.unlink(path)

    def test_incremental_analysis_performance(self, analysis_manager):
        """Test performance of incremental analysis."""
        # Create initial file
        initial_content = """
import os

def func1():
    return os.getcwd()

def func2():
    return os.path.exists('/tmp')
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(initial_content)
            temp_file = f.name

        try:
            # Initial analysis
            from codesage_mcp.code_model import CodeModelGenerator
            generator = CodeModelGenerator(analysis_manager.graph)
            generator.generate_from_file(temp_file, initial_content)

            initial_result = analysis_manager.run_comprehensive_analysis(temp_file)

            # Modify file
            modified_content = initial_content + """

def func3():
    import sys
    return sys.version
"""
            with open(temp_file, 'w') as f:
                f.write(modified_content)

            # Re-analyze
            generator.generate_from_file(temp_file, modified_content)

            start_time = time.time()
            modified_result = analysis_manager.run_comprehensive_analysis(temp_file)
            incremental_time = time.time() - start_time

            # Should be fast
            assert incremental_time < 2  # Less than 2 seconds

            # Should have more functions in modified result
            initial_funcs = initial_result["dependency_analysis"]["summary"]["total_functions_analyzed"]
            modified_funcs = modified_result["dependency_analysis"]["summary"]["total_functions_analyzed"]
            assert modified_funcs > initial_funcs

        finally:
            os.unlink(temp_file)

    def test_memory_usage_during_analysis(self, analysis_manager):
        """Test memory usage patterns during analysis."""
        # Create moderately complex content
        complex_content = ""
        for i in range(50):
            complex_content += f"""
import os, sys, json
from collections import Counter

def complex_func_{i}():
    data = list(range(100))
    counter = Counter()
    for item in data:
        for j in range(10):  # Nested loop
            counter[str(item % j)] += 1
    return dict(counter)

class ComplexClass_{i}:
    def __init__(self):
        self.data = [0] * 1000

    def process(self):
        return sum(self.data)
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_content)
            temp_file = f.name

        try:
            # Generate model
            from codesage_mcp.code_model import CodeModelGenerator
            generator = CodeModelGenerator(analysis_manager.graph)
            generator.generate_from_file(temp_file, complex_content)

            # Monitor memory during analysis
            process = psutil.Process()
            memory_samples = []

            # Run analysis while monitoring memory
            analysis_thread = threading.Thread(target=self._monitor_memory_during_analysis,
                                             args=(analysis_manager, temp_file, memory_samples))
            analysis_thread.start()
            analysis_thread.join()

            # Check memory usage
            if memory_samples:
                max_memory = max(memory_samples)
                avg_memory = sum(memory_samples) / len(memory_samples)

                # Should not use excessive memory
                assert max_memory < 400  # Less than 400MB peak
                assert len(memory_samples) > 5  # Should have monitoring data

        finally:
            os.unlink(temp_file)

    def _monitor_memory_during_analysis(self, manager, file_path, samples):
        """Helper to monitor memory during analysis."""
        process = psutil.Process()

        # Start monitoring
        for _ in range(20):  # Monitor for 2 seconds
            memory_mb = process.memory_info().rss / 1024 / 1024
            samples.append(memory_mb)
            time.sleep(0.1)

        # Run analysis (this will happen concurrently with monitoring)
        manager.run_comprehensive_analysis(file_path)


class TestAdvancedAnalysisConcurrency:
    """Test concurrency aspects of advanced analysis."""

    @pytest.fixture
    def analysis_manager(self):
        """Create an AdvancedAnalysisManager instance."""
        graph = CodeGraph()
        return AdvancedAnalysisManager(graph)

    def test_concurrent_analysis_operations(self, analysis_manager):
        """Test concurrent execution of analysis operations."""
        num_threads = 8
        files_per_thread = 3

        # Create test files
        all_files = []
        for i in range(num_threads * files_per_thread):
            content = f"""
import os, sys

def func_{i}():
    return os.getcwd()

class Class_{i}:
    def method(self):
        return sys.version
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                all_files.append(f.name)

        try:
            results = {}
            errors = []

            def analyze_files(thread_id):
                """Analyze files for a specific thread."""
                try:
                    start_idx = thread_id * files_per_thread
                    end_idx = start_idx + files_per_thread
                    thread_files = all_files[start_idx:end_idx]

                    thread_results = {}
                    for file_path in thread_files:
                        # Generate model first
                        from codesage_mcp.code_model import CodeModelGenerator
                        generator = CodeModelGenerator(analysis_manager.graph)
                        with open(file_path, 'r') as f:
                            content = f.read()
                        generator.generate_from_file(file_path, content)

                        # Run analysis
                        result = analysis_manager.run_comprehensive_analysis(file_path)
                        thread_results[file_path] = result

                    results[thread_id] = thread_results

                except Exception as e:
                    errors.append(f"Thread {thread_id}: {e}")

            # Execute concurrent analysis
            threads = []
            start_time = time.time()

            for i in range(num_threads):
                thread = threading.Thread(target=analyze_files, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            concurrent_time = time.time() - start_time

            # Verify results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == num_threads
            assert concurrent_time < 20  # Should complete within 20 seconds

            # Check that all threads produced results
            total_results = sum(len(thread_results) for thread_results in results.values())
            assert total_results == num_threads * files_per_thread

        finally:
            for file_path in all_files:
                os.unlink(file_path)

    def test_shared_graph_thread_safety(self, analysis_manager):
        """Test thread safety when multiple threads access the same graph."""
        num_threads = 10
        operations_per_thread = 20

        def graph_operations(thread_id):
            """Perform various graph operations."""
            for i in range(operations_per_thread):
                try:
                    # Create and add node
                    node = CodeNode(
                        node_type=NodeType.FUNCTION,
                        name=f"func_{thread_id}_{i}",
                        qualified_name=f"func_{thread_id}_{i}",
                        file_path=f"/shared_{thread_id}.py",
                        start_line=i,
                        end_line=i+1
                    )
                    analysis_manager.graph.add_node(node, LayerType.SEMANTIC)

                    # Occasionally run analysis
                    if i % 5 == 0:
                        # This tests thread safety of analysis operations
                        analysis_manager.dependency_analyzer.analyze_function_dependencies(
                            f"/shared_{thread_id}.py"
                        )

                except Exception as e:
                    pytest.fail(f"Thread {thread_id} failed: {e}")

        # Execute concurrent operations
        threads = []
        start_time = time.time()

        for i in range(num_threads):
            thread = threading.Thread(target=graph_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        concurrent_time = time.time() - start_time

        # Verify graph state
        stats = analysis_manager.graph.get_statistics()
        expected_nodes = num_threads * operations_per_thread
        assert stats["total_nodes"] == expected_nodes
        assert concurrent_time < 10  # Should complete quickly


if __name__ == "__main__":
    pytest.main([__file__])
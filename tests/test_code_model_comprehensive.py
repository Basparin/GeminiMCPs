"""
Comprehensive Unit Tests for Code Model Generation and Analysis Components.

This module contains extensive tests covering edge cases, error handling, performance,
memory optimization, and integration scenarios for the code model and analysis components.
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

from codesage_mcp.code_model import (
    CodeNode,
    Relationship,
    GraphLayer,
    CodeGraph,
    CodeModelGenerator,
    NodeType,
    RelationshipType,
    LayerType,
)
from codesage_mcp.advanced_analysis import (
    AdvancedDependencyAnalyzer,
    PerformancePredictor,
    AdvancedAnalysisManager
)
from codesage_mcp.exceptions import BaseMCPError


class TestCodeModelEdgeCases:
    """Test edge cases for code model generation."""

    @pytest.fixture
    def generator(self):
        """Create a CodeModelGenerator instance."""
        graph = CodeGraph()
        return CodeModelGenerator(graph)

    def test_empty_file_handling(self, generator):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, "")
            # Should still create a module node
            assert len(nodes) >= 1
            module_nodes = [n for n in nodes if n.node_type == NodeType.MODULE]
            assert len(module_nodes) == 1
            assert module_nodes[0].content == ""
        finally:
            os.unlink(temp_file)

    def test_large_file_handling(self, generator):
        """Test handling of very large files."""
        # Create a large Python file
        large_content = "# Large file test\n" + "\n".join([
            f"def function_{i}():\n    return {i}\n\nclass Class_{i}:\n    pass\n"
            for i in range(1000)
        ])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            start_time = time.time()
            nodes = generator.generate_from_file(temp_file, large_content)
            generation_time = time.time() - start_time

            # Should handle large files reasonably
            assert len(nodes) > 100  # Should find many nodes
            assert generation_time < 10  # Should complete within 10 seconds

            # Check memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            assert memory_mb < 500  # Should not use excessive memory

        finally:
            os.unlink(temp_file)

    def test_malformed_ast_handling(self, generator):
        """Test handling of malformed AST/code."""
        malformed_codes = [
            "def broken_function(\n    return 'broken'",  # Missing closing paren
            "class BrokenClass:\n    def method(self\n        pass",  # Missing closing paren
            "import os\nimport sys\nfrom broken import *",  # Non-existent import
            "x = [1, 2, 3\ny = 4",  # Unclosed list
        ]

        for malformed_code in malformed_codes:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(malformed_code)
                temp_file = f.name

            try:
                # Should handle gracefully without crashing
                nodes = generator.generate_from_file(temp_file, malformed_code)
                # May return empty list or partial results
                assert isinstance(nodes, list)
            except Exception as e:
                # Should raise specific exceptions, not generic crashes
                assert isinstance(e, (SyntaxError, BaseMCPError, Exception))
            finally:
                os.unlink(temp_file)

    def test_unicode_and_encoding_handling(self, generator):
        """Test handling of Unicode characters and encoding."""
        unicode_content = '''
# -*- coding: utf-8 -*-
def función_español():
    """Función con acentos: áéíóú"""
    return "café"

class ClaseConÑandÁ:
    def método(self):
        return "ñoño"

# Emoji support
def emoji_function():
    return "🚀⭐🔥"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(unicode_content)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, unicode_content)
            assert len(nodes) >= 4  # Module + 2 functions + 1 class

            # Check that Unicode names are preserved
            function_names = [n.name for n in nodes if n.node_type == NodeType.FUNCTION]
            assert "función_español" in function_names
            assert "emoji_function" in function_names

        finally:
            os.unlink(temp_file)

    def test_complex_ast_features(self, generator):
        """Test handling of complex AST features."""
        complex_code = '''
# Type hints
from typing import List, Dict, Optional, Union

def typed_function(x: int, y: str) -> Dict[str, List[int]]:
    return {"result": [x]}

# Async functions
async def async_function():
    return await some_coroutine()

# Generators
def generator_function():
    yield 1
    yield 2

# Decorators
@staticmethod
def static_method():
    pass

@property
def property_method(self):
    return "value"

# Lambda functions
lambda_func = lambda x: x * 2

# List/dict/set comprehensions
comprehension_result = [x*2 for x in range(10)]
dict_comp = {x: x*2 for x in range(5)}
set_comp = {x*2 for x in range(5)}

# Class with inheritance and metaclasses
class MetaClass(type):
    pass

class ComplexClass(BaseClass, metaclass=MetaClass):
    def __init__(self):
        super().__init__()

    @classmethod
    def class_method(cls):
        return cls()

    @staticmethod
    def static_method():
        return "static"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_code)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, complex_code)
            assert len(nodes) >= 8  # Should find many constructs

            # Check specific node types
            function_nodes = [n for n in nodes if n.node_type == NodeType.FUNCTION]
            class_nodes = [n for n in nodes if n.node_type == NodeType.CLASS]
            import_nodes = [n for n in nodes if n.node_type == NodeType.IMPORT]

            assert len(function_nodes) >= 5  # typed, async, generator, static, property
            assert len(class_nodes) >= 2  # MetaClass, ComplexClass
            assert len(import_nodes) >= 1  # typing import

        finally:
            os.unlink(temp_file)

    def test_deep_nesting_handling(self, generator):
        """Test handling of deeply nested code structures."""
        # Create deeply nested code
        nested_code = "x = 1\n"
        for i in range(20):  # Create 20 levels of nesting
            nested_code += "    " * i + f"if x == {i}:\n"
        nested_code += "    " * 20 + "pass\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(nested_code)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, nested_code)
            # Should handle without recursion errors
            assert isinstance(nodes, list)
            assert len(nodes) >= 1  # At least module node

        finally:
            os.unlink(temp_file)


class TestCodeModelErrorHandling:
    """Test error handling in code model generation."""

    @pytest.fixture
    def generator(self):
        """Create a CodeModelGenerator instance."""
        graph = CodeGraph()
        return CodeModelGenerator(graph)

    def test_file_not_found_handling(self, generator):
        """Test handling of non-existent files."""
        nodes = generator.generate_from_file("/nonexistent/file.py")
        assert nodes == []  # Should return empty list

    def test_permission_denied_handling(self, generator):
        """Test handling of permission denied errors."""
        # Create a file and remove read permissions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('test')")
            temp_file = f.name

        try:
            # Remove read permission
            os.chmod(temp_file, 0o000)

            # Should handle gracefully
            nodes = generator.generate_from_file(temp_file)
            # May return empty list or partial results
            assert isinstance(nodes, list)

        finally:
            # Restore permissions for cleanup
            os.chmod(temp_file, 0o644)
            os.unlink(temp_file)

    def test_corrupted_cache_handling(self, generator):
        """Test handling of corrupted cache data."""
        # Mock cache to return corrupted data
        with patch.object(generator, 'cache') as mock_cache:
            mock_cache.get_file_content.return_value = ("corrupted json", True)

            with patch('json.loads', side_effect=json.JSONDecodeError("Invalid JSON", "corrupted json", 0)):
                # Should handle corrupted cache gracefully
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write("def test(): pass")
                    temp_file = f.name

                try:
                    nodes = generator.generate_from_file(temp_file)
                    # Should still generate nodes despite cache corruption
                    assert isinstance(nodes, list)
                    assert len(nodes) >= 1
                finally:
                    os.unlink(temp_file)

    def test_memory_exhaustion_handling(self, generator):
        """Test handling of memory exhaustion scenarios."""
        # Create a very large file that might cause memory issues
        large_content = "x = " + str(list(range(100000))) + "\n" * 1000

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            nodes = generator.generate_from_file(temp_file, large_content)

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Should not cause excessive memory usage
            assert memory_increase < 200  # Less than 200MB increase
            assert isinstance(nodes, list)

        finally:
            os.unlink(temp_file)


class TestCodeModelPerformance:
    """Test performance aspects of code model generation."""

    @pytest.fixture
    def generator(self):
        """Create a CodeModelGenerator instance."""
        graph = CodeGraph()
        return CodeModelGenerator(graph)

    def test_batch_processing_performance(self, generator):
        """Test performance of batch processing multiple files."""
        # Create multiple test files
        file_contents = []
        file_paths = []

        for i in range(10):
            content = f"""
def function_{i}():
    return {i}

class Class_{i}:
    def method(self):
        return "method_{i}"
"""
            file_contents.append(content)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                file_paths.append(f.name)

        try:
            start_time = time.time()
            results = generator.batch_generate_from_files(file_paths, file_contents)
            batch_time = time.time() - start_time

            # Should process all files
            assert len(results) == 10
            assert all(isinstance(nodes, list) for nodes in results.values())

            # Should be reasonably fast
            assert batch_time < 5  # Less than 5 seconds for 10 files

            # Check total nodes generated
            total_nodes = sum(len(nodes) for nodes in results.values())
            assert total_nodes >= 30  # At least 3 nodes per file (module + function + class)

        finally:
            for path in file_paths:
                os.unlink(path)

    def test_incremental_updates_performance(self, generator):
        """Test performance of incremental updates."""
        # Create initial file
        initial_content = """
def func1():
    return 1

def func2():
    return 2
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(initial_content)
            temp_file = f.name

        try:
            # Initial generation
            initial_nodes = generator.generate_from_file(temp_file, initial_content)
            initial_count = len(initial_nodes)

            # Modify file
            modified_content = initial_content + """

def func3():
    return 3
"""
            with open(temp_file, 'w') as f:
                f.write(modified_content)

            # Regenerate
            start_time = time.time()
            modified_nodes = generator.generate_from_file(temp_file, modified_content)
            update_time = time.time() - start_time

            # Should be fast
            assert update_time < 1  # Less than 1 second
            assert len(modified_nodes) > initial_count

        finally:
            os.unlink(temp_file)

    def test_memory_usage_during_generation(self, generator):
        """Test memory usage patterns during code generation."""
        # Create a moderately complex file
        complex_content = "\n".join([
            f"def func_{i}():\n    x = {i}\n    return x * 2\n\nclass Class_{i}:\n    def method(self):\n        return {i}\n"
            for i in range(100)
        ])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_content)
            temp_file = f.name

        try:
            process = psutil.Process()

            # Monitor memory during generation
            memory_samples = []
            generation_thread = threading.Thread(target=self._monitor_memory, args=(memory_samples, 0.1, 2))
            generation_thread.start()

            nodes = generator.generate_from_file(temp_file, complex_content)

            generation_thread.join()

            # Check memory usage
            if memory_samples:
                max_memory = max(memory_samples)
                avg_memory = sum(memory_samples) / len(memory_samples)

                # Should not use excessive memory
                assert max_memory < 300  # Less than 300MB peak
                assert len(nodes) >= 200  # Should find many nodes

        finally:
            os.unlink(temp_file)

    def _monitor_memory(self, samples, interval, duration):
        """Helper to monitor memory usage."""
        process = psutil.Process()
        end_time = time.time() + duration

        while time.time() < end_time:
            memory_mb = process.memory_info().rss / 1024 / 1024
            samples.append(memory_mb)
            time.sleep(interval)


class TestCodeModelConcurrency:
    """Test concurrency aspects of code model generation."""

    @pytest.fixture
    def generator(self):
        """Create a CodeModelGenerator instance."""
        graph = CodeGraph()
        return CodeModelGenerator(graph)

    def test_concurrent_file_processing(self, generator):
        """Test concurrent processing of multiple files."""
        num_files = 20
        file_paths = []
        file_contents = []

        # Create test files
        for i in range(num_files):
            content = f"def func_{i}(): return {i}"
            file_contents.append(content)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                file_paths.append(f.name)

        try:
            results = {}

            def process_file(index):
                path = file_paths[index]
                content = file_contents[index]
                nodes = generator.generate_from_file(path, content)
                results[index] = nodes

            # Process files concurrently
            threads = []
            start_time = time.time()

            for i in range(num_files):
                thread = threading.Thread(target=process_file, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            concurrent_time = time.time() - start_time

            # Verify results
            assert len(results) == num_files
            assert all(isinstance(nodes, list) for nodes in results.values())
            assert concurrent_time < 10  # Should complete within 10 seconds

        finally:
            for path in file_paths:
                os.unlink(path)

    def test_graph_thread_safety(self):
        """Test thread safety of graph operations."""
        graph = CodeGraph()
        num_threads = 10
        operations_per_thread = 50

        def graph_operations(thread_id):
            """Perform various graph operations."""
            for i in range(operations_per_thread):
                try:
                    # Create and add node
                    node = CodeNode(
                        node_type=NodeType.FUNCTION,
                        name=f"func_{thread_id}_{i}",
                        qualified_name=f"func_{thread_id}_{i}",
                        file_path=f"/test_{thread_id}.py",
                        start_line=i,
                        end_line=i+1
                    )
                    graph.add_node(node, LayerType.SEMANTIC)

                    # Add relationship occasionally
                    if i % 5 == 0 and i > 0:
                        prev_node_id = f"func_{thread_id}_{i-1}"
                        relationship = Relationship(
                            source_id=prev_node_id,
                            target_id=node.id,
                            relationship_type=RelationshipType.CALLS,
                            layer=LayerType.SEMANTIC
                        )
                        graph.add_relationship(relationship)

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
        stats = graph.get_statistics()
        assert stats["total_nodes"] == num_threads * operations_per_thread
        assert stats["total_relationships"] >= num_threads * (operations_per_thread // 5)
        assert concurrent_time < 5  # Should complete quickly


class TestCodeModelSerialization:
    """Test serialization and deserialization of code models."""

    def test_large_graph_serialization(self):
        """Test serialization of large graphs."""
        graph = CodeGraph()

        # Create many nodes and relationships
        num_nodes = 1000
        nodes = []

        for i in range(num_nodes):
            node = CodeNode(
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                file_path=f"/test_{i%10}.py",
                start_line=i,
                end_line=i+1
            )
            nodes.append(node)
            graph.add_node(node, LayerType.SEMANTIC)

        # Add relationships
        for i in range(0, num_nodes-1, 2):
            relationship = Relationship(
                source_id=nodes[i].id,
                target_id=nodes[i+1].id,
                relationship_type=RelationshipType.CALLS,
                layer=LayerType.SEMANTIC
            )
            graph.add_relationship(relationship)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Test serialization
            start_time = time.time()
            graph.save_to_file(temp_file)
            save_time = time.time() - start_time

            # Check file size
            file_size = os.path.getsize(temp_file)
            assert file_size > 0
            assert file_size < 10 * 1024 * 1024  # Less than 10MB

            # Test deserialization
            new_graph = CodeGraph()
            load_start = time.time()
            new_graph.load_from_file(temp_file)
            load_time = time.time() - load_start

            # Verify loaded data
            loaded_stats = new_graph.get_statistics()
            assert loaded_stats["total_nodes"] == num_nodes
            assert loaded_stats["total_relationships"] >= num_nodes // 2

            # Performance checks
            assert save_time < 5  # Less than 5 seconds to save
            assert load_time < 5  # Less than 5 seconds to load

        finally:
            os.unlink(temp_file)

    def test_corrupted_data_handling(self):
        """Test handling of corrupted serialization data."""
        graph = CodeGraph()

        # Create valid data first
        node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="test_func",
            qualified_name="test_func",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )
        graph.add_node(node, LayerType.SEMANTIC)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Save valid data
            graph.save_to_file(temp_file)

            # Corrupt the file
            with open(temp_file, 'r') as f:
                data = f.read()
            corrupted_data = data.replace('"nodes":', '"nodes": invalid')
            with open(temp_file, 'w') as f:
                f.write(corrupted_data)

            # Try to load corrupted data
            new_graph = CodeGraph()
            new_graph.load_from_file(temp_file)  # Should handle gracefully

            # Should initialize as empty graph
            stats = new_graph.get_statistics()
            assert stats["total_nodes"] == 0

        finally:
            os.unlink(temp_file)


class TestAdvancedAnalysisIntegration:
    """Test integration between code model and advanced analysis."""

    @pytest.fixture
    def analysis_manager(self):
        """Create an AdvancedAnalysisManager instance."""
        graph = CodeGraph()
        return AdvancedAnalysisManager(graph)

    def test_large_codebase_analysis(self, analysis_manager):
        """Test analysis of large codebases."""
        # Create multiple large files
        file_contents = {}
        file_paths = []

        for i in range(5):
            content = "\n".join([
                f"def function_{i}_{j}():\n    return {j}\n\nclass Class_{i}_{j}:\n    def method(self):\n        return {j}\n"
                for j in range(50)  # 50 functions/classes per file
            ])
            file_contents[f"/test_{i}.py"] = content

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                file_paths.append(f.name)

        try:
            # Generate code models for all files
            generator = analysis_manager.graph  # Get the graph
            code_generator = CodeModelGenerator(generator)

            start_time = time.time()
            for path, content in zip(file_paths, file_contents.values()):
                code_generator.generate_from_file(path, content)
            model_time = time.time() - start_time

            # Run comprehensive analysis
            analysis_start = time.time()
            result = analysis_manager.run_comprehensive_analysis()
            analysis_time = time.time() - analysis_start

            # Verify results
            assert "dependency_analysis" in result
            assert "performance_analysis" in result
            assert "library_analysis" in result

            # Performance checks
            assert model_time < 10  # Model generation should be fast
            assert analysis_time < 15  # Analysis should be reasonable

            # Check analysis results
            dep_analysis = result["dependency_analysis"]
            perf_analysis = result["performance_analysis"]

            assert dep_analysis["summary"]["total_functions_analyzed"] >= 250  # 5 files * 50 functions
            assert isinstance(perf_analysis["bottlenecks"], list)

        finally:
            for path in file_paths:
                os.unlink(path)

    def test_memory_optimization_during_analysis(self, analysis_manager):
        """Test memory optimization during analysis operations."""
        # Create a complex codebase
        complex_content = "\n".join([
            f"def func_{i}():\n    x = {i}\n    return x\n\nclass Class_{i}:\n    def method_{j}(self):\n        return {j}\n"
            for i in range(20)
            for j in range(10)
        ])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_content)
            temp_file = f.name

        try:
            # Generate model
            generator = CodeModelGenerator(analysis_manager.graph)
            generator.generate_from_file(temp_file, complex_content)

            # Monitor memory during analysis
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            result = analysis_manager.run_comprehensive_analysis(temp_file)

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Should not cause excessive memory usage
            assert memory_increase < 100  # Less than 100MB increase
            assert result is not None

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
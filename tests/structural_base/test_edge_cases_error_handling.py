"""
Edge Cases and Error Handling Tests for Code Model and Analysis Components.

This module contains comprehensive tests for edge cases, error handling scenarios,
and boundary conditions that the system should handle gracefully.
"""

import pytest
import tempfile
import os
import time
from unittest.mock import patch, MagicMock
import psutil

from codesage_mcp.core.code_model import (
    CodeGraph,
    CodeModelGenerator,
    CodeNode,
    Relationship,
    NodeType,
    RelationshipType,
    LayerType
)
from codesage_mcp.features.codebase_manager.advanced_analysis import (
    AdvancedAnalysisManager
)


class TestEdgeCaseInputs:
    """Test edge case inputs and boundary conditions."""

    @pytest.fixture
    def edge_case_setup(self):
        """Create setup for edge case testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_empty_string_input(self, edge_case_setup):
        """Test handling of empty string inputs."""
        setup = edge_case_setup
        generator = setup['generator']

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, "")
            # Should handle gracefully
            assert isinstance(nodes, list)
            # May create a module node or return empty list
            assert len(nodes) <= 1  # At most one module node

        finally:
            os.unlink(temp_file)

    def test_whitespace_only_input(self, edge_case_setup):
        """Test handling of whitespace-only inputs."""
        setup = edge_case_setup
        generator = setup['generator']

        whitespace_content = "\n\n   \n\t  \n   \n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(whitespace_content)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, whitespace_content)
            assert isinstance(nodes, list)
            # Should still create a module node
            module_nodes = [n for n in nodes if n.node_type == NodeType.MODULE]
            assert len(module_nodes) == 1

        finally:
            os.unlink(temp_file)

    def test_very_long_lines(self, edge_case_setup):
        """Test handling of very long lines."""
        setup = edge_case_setup
        generator = setup['generator']

        # Create a line with 100,000 characters
        long_line = "x = " + str(list(range(10000)))  # Very long list
        content = f"""
def func():
    {long_line}
    return x
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, content)
            assert isinstance(nodes, list)
            assert len(nodes) >= 2  # Module + function

        finally:
            os.unlink(temp_file)

    def test_unicode_edge_cases(self, edge_case_setup):
        """Test handling of Unicode edge cases."""
        setup = edge_case_setup
        generator = setup['generator']

        unicode_content = '''
# Various Unicode scenarios
def función_con_ñ():
    return "café"

def emoji_func():
    return "🚀⭐🔥"

def mixed_unicode():
    return "α + β = ∑"  # Greek letters and math symbols

class ClaseConCarácteresEspeciales:
    def método(self):
        return "ñoño"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', encoding='utf-8', delete=False) as f:
            f.write(unicode_content)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, unicode_content)
            assert isinstance(nodes, list)
            assert len(nodes) >= 5  # Module + 3 functions + 1 class

            # Check that Unicode names are preserved
            function_names = [n.name for n in nodes if n.node_type == NodeType.FUNCTION]
            assert "función_con_ñ" in function_names
            assert "emoji_func" in function_names
            assert "mixed_unicode" in function_names

        finally:
            os.unlink(temp_file)

    def test_binary_file_like_content(self, edge_case_setup):
        """Test handling of binary-like content."""
        setup = edge_case_setup
        generator = setup['generator']

        # Content with null bytes and binary-like data
        binary_like_content = "print('hello')\x00\x01\x02binary data here\x00"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(binary_like_content)
            temp_file = f.name

        try:
            # Should handle gracefully (may raise SyntaxError but shouldn't crash)
            nodes = generator.generate_from_file(temp_file, binary_like_content)
            assert isinstance(nodes, list)

        finally:
            os.unlink(temp_file)

    def test_extremely_nested_code(self, edge_case_setup):
        """Test handling of extremely nested code structures."""
        setup = edge_case_setup
        generator = setup['generator']

        # Create 50 levels of nesting
        nested_code = "result = 1\n"
        for i in range(50):
            nested_code += "    " * i + f"if result > {i}:\n"
        nested_code += "    " * 50 + "result = 42\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(nested_code)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, nested_code)
            assert isinstance(nodes, list)
            # Should handle without recursion errors
            assert len(nodes) >= 1

        finally:
            os.unlink(temp_file)

    def test_circular_import_scenarios(self, edge_case_setup):
        """Test handling of circular import scenarios."""
        setup = edge_case_setup
        generator = setup['generator']

        # Create files with circular imports
        file_a_content = '''
from file_b import func_b

def func_a():
    return func_b()
'''

        file_b_content = '''
from file_a import func_a

def func_b():
    return func_a()
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(file_a_content)
            file_a = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(file_b_content)
            file_b = f.name

        try:
            # Generate models for both files
            nodes_a = generator.generate_from_file(file_a, file_a_content)
            nodes_b = generator.generate_from_file(file_b, file_b_content)

            assert isinstance(nodes_a, list)
            assert isinstance(nodes_b, list)
            assert len(nodes_a) >= 2  # Module + function
            assert len(nodes_b) >= 2  # Module + function

        finally:
            os.unlink(file_a)
            os.unlink(file_b)


class TestErrorHandlingScenarios:
    """Test various error handling scenarios."""

    @pytest.fixture
    def error_setup(self):
        """Create setup for error handling testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_syntax_error_recovery(self, error_setup):
        """Test recovery from syntax errors."""
        setup = error_setup
        generator = setup['generator']

        # Various syntax errors
        syntax_errors = [
            "def broken_function(\n    return 'incomplete'",  # Missing closing paren
            "class BrokenClass:\n    def method(self\n        pass",  # Missing closing paren
            "if True:\n    print('missing indentation'",  # Indentation error
            "def func():\n    return\n    print('unreachable')",  # Unreachable code
        ]

        for error_code in syntax_errors:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(error_code)
                temp_file = f.name

            try:
                # Should handle syntax errors gracefully
                nodes = generator.generate_from_file(temp_file, error_code)
                assert isinstance(nodes, list)
                # May return partial results or empty list

            finally:
                os.unlink(temp_file)

    def test_file_system_errors(self, error_setup):
        """Test handling of file system errors."""
        setup = error_setup
        generator = setup['generator']

        # Test non-existent file
        nodes = generator.generate_from_file("/nonexistent/path/file.py")
        assert nodes == []

        # Test file with permission issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('test')")
            temp_file = f.name

        try:
            # Remove read permission
            os.chmod(temp_file, 0o000)

            # Should handle permission denied gracefully
            nodes = generator.generate_from_file(temp_file)
            assert isinstance(nodes, list)

        finally:
            # Restore permissions for cleanup
            os.chmod(temp_file, 0o644)
            os.unlink(temp_file)

    def test_memory_allocation_errors(self, error_setup):
        """Test handling of memory allocation errors."""
        setup = error_setup
        generator = setup['generator']

        # Create extremely large content that might cause memory issues
        large_content = "x = " + str([i for i in range(100000)]) + "\n" * 1000

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
            assert memory_increase < 500  # Less than 500MB
            assert isinstance(nodes, list)

        finally:
            os.unlink(temp_file)

    def test_corrupted_ast_handling(self, error_setup):
        """Test handling of corrupted AST data."""
        setup = error_setup
        generator = setup['generator']

        # Mock AST parsing to return corrupted data
        with patch('ast.parse') as mock_parse:
            mock_parse.side_effect = Exception("AST corruption")

            content = "print('test')"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                nodes = generator.generate_from_file(temp_file, content)
                # Should handle AST corruption gracefully
                assert isinstance(nodes, list)

            finally:
                os.unlink(temp_file)

    def test_encoding_errors(self, error_setup):
        """Test handling of encoding errors."""
        setup = error_setup
        generator = setup['generator']

        # Create file with invalid UTF-8 content
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as f:
            # Write some valid UTF-8, then invalid bytes
            f.write("print('hello')\n".encode('utf-8'))
            f.write(b'\xff\xfe\xfd')  # Invalid UTF-8
            temp_file = f.name

        try:
            # Should handle encoding errors gracefully
            nodes = generator.generate_from_file(temp_file)
            assert isinstance(nodes, list)

        finally:
            os.unlink(temp_file)


class TestBoundaryConditions:
    """Test boundary conditions and limits."""

    @pytest.fixture
    def boundary_setup(self):
        """Create setup for boundary condition testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_maximum_file_size(self, boundary_setup):
        """Test handling of maximum file size limits."""
        setup = boundary_setup
        generator = setup['generator']

        # Create a very large file (10MB)
        large_content = "print('line')\n" * 100000  # ~1MB
        large_content = large_content * 10  # ~10MB

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            file_size_mb = len(large_content) / (1024 * 1024)
            assert file_size_mb > 5  # At least 5MB

            start_time = time.time()
            nodes = generator.generate_from_file(temp_file, large_content)
            processing_time = time.time() - start_time

            # Should handle large files within reasonable time
            assert processing_time < 60  # Less than 1 minute
            assert isinstance(nodes, list)

        finally:
            os.unlink(temp_file)

    def test_maximum_nodes_per_file(self, boundary_setup):
        """Test handling of files with maximum number of nodes."""
        setup = boundary_setup
        generator = setup['generator']

        # Create file with 1000 functions and classes
        content = ""
        for i in range(500):
            content += f"""
def function_{i}():
    return {i}

class Class_{i}:
    def method(self):
        return "method_{i}"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, content)

            # Should handle many nodes
            assert isinstance(nodes, list)
            assert len(nodes) >= 1000  # At least 1000 nodes

            # Check node types
            function_nodes = [n for n in nodes if n.node_type == NodeType.FUNCTION]
            class_nodes = [n for n in nodes if n.node_type == NodeType.CLASS]

            assert len(function_nodes) >= 500
            assert len(class_nodes) >= 500

        finally:
            os.unlink(temp_file)

    def test_maximum_relationships(self, boundary_setup):
        """Test handling of maximum number of relationships."""
        setup = boundary_setup
        graph = setup['graph']

        # Create many nodes and relationships
        num_nodes = 1000
        nodes = []

        for i in range(num_nodes):
            node = CodeNode(
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                file_path="/boundary_test.py",
                start_line=i,
                end_line=i+1
            )
            nodes.append(node)
            graph.add_node(node, LayerType.SEMANTIC)

        # Create relationships (each node calls 5 others)
        relationships_created = 0
        for i in range(num_nodes):
            for j in range(5):
                target_idx = (i + j + 1) % num_nodes
                relationship = Relationship(
                    source_id=nodes[i].id,
                    target_id=nodes[target_idx].id,
                    relationship_type=RelationshipType.CALLS,
                    layer=LayerType.SEMANTIC
                )
                graph.add_relationship(relationship)
                relationships_created += 1

        # Verify graph can handle many relationships
        stats = graph.get_statistics()
        assert stats['total_nodes'] == num_nodes
        assert stats['total_relationships'] >= relationships_created

        # Test graph operations still work
        search_results = graph.find_nodes_by_name("func_5")
        assert len(search_results) >= 1

    def test_deep_inheritance_chains(self, boundary_setup):
        """Test handling of deep inheritance chains."""
        setup = boundary_setup
        generator = setup['generator']

        # Create deep inheritance hierarchy
        content = "class Base:\n    pass\n\n"

        for i in range(1, 50):  # 50 levels of inheritance
            content += f"class Child{i}(Child{i-1} if i > 1 else Base):\n    pass\n\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, content)
            assert isinstance(nodes, list)
            assert len(nodes) >= 50  # 49 child classes + base + module

            # Check class nodes
            class_nodes = [n for n in nodes if n.node_type == NodeType.CLASS]
            assert len(class_nodes) >= 50

        finally:
            os.unlink(temp_file)

    def test_maximum_import_depth(self, boundary_setup):
        """Test handling of maximum import depth."""
        setup = boundary_setup
        generator = setup['generator']

        # Create deeply nested import structure
        content = ""
        for i in range(20):
            content += f"from module_{i} import sub_module_{i}\n"

        content += "\ndef func():\n    return 'deep imports'\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, content)
            assert isinstance(nodes, list)

            # Check import nodes
            import_nodes = [n for n in nodes if n.node_type == NodeType.IMPORT]
            assert len(import_nodes) >= 20

        finally:
            os.unlink(temp_file)


class TestConcurrencyEdgeCases:
    """Test edge cases in concurrent scenarios."""

    @pytest.fixture
    def concurrent_setup(self):
        """Create setup for concurrent testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_concurrent_file_modifications(self, concurrent_setup):
        """Test concurrent modifications to the same file."""
        setup = concurrent_setup
        generator = setup['generator']

        # Create initial file
        initial_content = "def func1(): return 1"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(initial_content)
            temp_file = f.name

        try:
            import threading

            results = []
            errors = []

            def modify_and_generate(thread_id):
                """Modify file and generate model concurrently."""
                try:
                    # Modify file content
                    new_content = f"def func{thread_id}(): return {thread_id}"
                    with open(temp_file, 'w') as file:
                        file.write(new_content)

                    # Generate model
                    nodes = generator.generate_from_file(temp_file, new_content)
                    results.append((thread_id, len(nodes)))

                except Exception as e:
                    errors.append(f"Thread {thread_id}: {e}")

            # Execute concurrently
            threads = []
            for i in range(10):
                thread = threading.Thread(target=modify_and_generate, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Verify results
            assert len(errors) == 0, f"Concurrent errors: {errors}"
            assert len(results) == 10

            # Each thread should have generated some nodes
            for thread_id, node_count in results:
                assert node_count >= 1

        finally:
            os.unlink(temp_file)

    def test_concurrent_graph_operations(self, concurrent_setup):
        """Test concurrent operations on the graph."""
        setup = concurrent_setup
        graph = setup['graph']

        import threading

        num_threads = 20
        operations_per_thread = 50

        errors = []

        def graph_operations(thread_id):
            """Perform concurrent graph operations."""
            try:
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

                    # Occasionally add relationship
                    if i % 3 == 0 and i > 0:
                        # Find a previous node to link to
                        prev_node_id = f"concurrent_func_{thread_id}_{i-1}"
                        relationship = Relationship(
                            source_id=prev_node_id,
                            target_id=node.id,
                            relationship_type=RelationshipType.CALLS,
                            layer=LayerType.SEMANTIC
                        )
                        graph.add_relationship(relationship)

                    # Occasionally query graph
                    if i % 10 == 0:
                        stats = graph.get_statistics()
                        assert stats['total_nodes'] >= 0

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Execute concurrent operations
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=graph_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Concurrent graph errors: {errors}"

        # Check final graph state
        stats = graph.get_statistics()
        expected_nodes = num_threads * operations_per_thread
        assert stats['total_nodes'] == expected_nodes


class TestResourceExhaustion:
    """Test handling of resource exhaustion scenarios."""

    @pytest.fixture
    def resource_setup(self):
        """Create setup for resource testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_disk_space_exhaustion_simulation(self, resource_setup):
        """Test handling when disk space is exhausted."""
        setup = resource_setup
        generator = setup['generator']

        # Create content
        content = "def func(): return 42"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Mock file write to simulate disk full
            with patch('builtins.open') as mock_open:
                mock_file = MagicMock()
                mock_file.write.side_effect = OSError("No space left on device")
                mock_open.return_value.__enter__.return_value = mock_file

                # Should handle disk full gracefully
                nodes = generator.generate_from_file(temp_file, content)
                assert isinstance(nodes, list)

        finally:
            os.unlink(temp_file)

    def test_cpu_exhaustion_simulation(self, resource_setup):
        """Test handling under CPU exhaustion."""
        setup = resource_setup
        generator = setup['generator']

        # Create CPU-intensive content
        content = ""
        for i in range(1000):
            content += f"def func_{i}():\n"
            content += "    result = 0\n"
            for j in range(100):
                content += f"    result += {i} * {j}\n"
            content += "    return result\n\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Measure CPU usage during processing
            process = psutil.Process()

            start_time = time.time()
            initial_cpu = process.cpu_percent(interval=1)

            nodes = generator.generate_from_file(temp_file, content)

            processing_time = time.time() - start_time
            final_cpu = process.cpu_percent(interval=1)

            # Should complete within reasonable time and CPU usage
            assert processing_time < 30  # Less than 30 seconds
            assert final_cpu < 90  # Less than 90% CPU usage
            assert isinstance(nodes, list)

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
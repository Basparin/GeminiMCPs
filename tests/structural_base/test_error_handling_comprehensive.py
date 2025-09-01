"""
Comprehensive Error Handling Tests for Code Model and Analysis Components.

This module contains extensive tests for error handling scenarios including
exceptions, timeouts, resource limits, and recovery mechanisms.
"""

import pytest
import tempfile
import os
import time
import threading
from unittest.mock import patch
import psutil
import json

from codesage_mcp.core.code_model import (
    CodeGraph,
    CodeModelGenerator,
    CodeNode,
    NodeType,
    RelationshipType,
    LayerType
)
from codesage_mcp.features.codebase_manager import (
    AdvancedAnalysisManager
)
from codesage_mcp.core.exceptions import (
    BaseMCPError,
    ToolExecutionError,
    InvalidRequestError,
    IndexingError,
    ConfigurationError
)


class TestExceptionHandling:
    """Test exception handling in various scenarios."""

    @pytest.fixture
    def exception_setup(self):
        """Create setup for exception handling testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_ast_parsing_exceptions(self, exception_setup):
        """Test handling of AST parsing exceptions."""
        setup = exception_setup
        generator = setup['generator']

        # Test various malformed Python code that causes AST errors
        malformed_codes = [
            "def func(\n    return",  # Unclosed parenthesis
            "class Test:\n    def method(self\n        pass",  # Unclosed parenthesis
            "if True:\n    print('test'",  # Missing closing quote
            "def func():\n    x = [1, 2, 3\n    return x",  # Unclosed list
            "import os\nfrom broken import *",  # Non-existent module
            "x = \\\n    1",  # Line continuation issue
        ]

        for malformed_code in malformed_codes:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(malformed_code)
                temp_file = f.name

            try:
                # Should handle AST parsing errors gracefully
                nodes = generator.generate_from_file(temp_file, malformed_code)
                assert isinstance(nodes, list)  # Should return list even on error

            finally:
                os.unlink(temp_file)

    def test_file_io_exceptions(self, exception_setup):
        """Test handling of file I/O exceptions."""
        setup = exception_setup
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

    def test_memory_allocation_exceptions(self, exception_setup):
        """Test handling of memory allocation exceptions."""
        setup = exception_setup
        generator = setup['generator']

        # Create extremely large content that might cause memory issues
        large_content = "x = " + str([i for i in range(100000)]) + "\n" * 1000

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            # Should handle large content gracefully
            nodes = generator.generate_from_file(temp_file, large_content)
            assert isinstance(nodes, list)

        finally:
            os.unlink(temp_file)

    def test_encoding_exceptions(self, exception_setup):
        """Test handling of encoding exceptions."""
        setup = exception_setup
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

    def test_corrupted_cache_exceptions(self, exception_setup):
        """Test handling of corrupted cache data."""
        setup = exception_setup
        generator = setup['generator']

        # Mock cache to return corrupted data
        with patch.object(generator, 'cache') as mock_cache:
            mock_cache.get_file_content.return_value = ("corrupted json data", True)

            with patch('json.loads', side_effect=json.JSONDecodeError("Invalid JSON", "corrupted json data", 0)):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write("def test(): pass")
                    temp_file = f.name

                try:
                    # Should handle corrupted cache gracefully
                    nodes = generator.generate_from_file(temp_file)
                    assert isinstance(nodes, list)
                    assert len(nodes) >= 1  # Should still generate nodes

                finally:
                    os.unlink(temp_file)


class TestTimeoutHandling:
    """Test timeout handling in operations."""

    @pytest.fixture
    def timeout_setup(self):
        """Create setup for timeout testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_operation_timeout_simulation(self, timeout_setup):
        """Test handling of operation timeouts."""
        setup = timeout_setup
        generator = setup['generator']

        # Create content that might take time to process
        large_content = ""
        for i in range(10000):
            large_content += f"def func_{i}():\n    return {i}\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            # Mock a timeout scenario
            with patch('time.sleep') as mock_sleep:
                # Simulate timeout by making sleep raise exception
                mock_sleep.side_effect = TimeoutError("Operation timed out")

                # Should handle timeout gracefully
                nodes = generator.generate_from_file(temp_file, large_content)
                assert isinstance(nodes, list)

        finally:
            os.unlink(temp_file)

    def test_concurrent_timeout_handling(self, timeout_setup):
        """Test timeout handling in concurrent operations."""
        setup = timeout_setup
        generator = setup['generator']

        num_threads = 5
        results = []
        errors = []

        def process_with_timeout(thread_id):
            """Process files with potential timeout."""
            try:
                content = f"def func_{thread_id}(): return {thread_id}"
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(content)
                    temp_file = f.name

                try:
                    # Simulate potential timeout
                    if thread_id == 2:  # Make one thread timeout
                        time.sleep(0.1)  # Simulate delay
                        raise TimeoutError(f"Thread {thread_id} timed out")

                    nodes = generator.generate_from_file(temp_file, content)
                    results.append((thread_id, len(nodes)))

                finally:
                    os.unlink(temp_file)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Execute concurrently
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=process_with_timeout, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should handle timeouts gracefully
        assert len(results) >= 3  # At least some threads should succeed
        assert len(errors) <= 2   # At most 2 threads should fail

    def test_analysis_timeout_handling(self, timeout_setup):
        """Test timeout handling in analysis operations."""
        setup = timeout_setup
        analyzer = setup['analyzer']

        # Create complex content for analysis
        complex_content = ""
        for i in range(1000):
            complex_content += f"""
def complex_func_{i}():
    result = []
    for j in range(100):
        for k in range(10):
            result.append(j * k)
    return result
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_content)
            temp_file = f.name

        try:
            # Generate model first
            setup['generator'].generate_from_file(temp_file, complex_content)

            # Mock timeout in analysis
            with patch.object(analyzer.dependency_analyzer, '_analyze_single_function_dependencies') as mock_analyze:
                mock_analyze.side_effect = TimeoutError("Analysis timed out")

                # Should handle analysis timeout gracefully
                result = analyzer.run_comprehensive_analysis(temp_file)
                assert isinstance(result, dict)
                # Should still return partial results
                assert 'dependency_analysis' in result

        finally:
            os.unlink(temp_file)


class TestResourceLimitHandling:
    """Test handling of resource limits."""

    @pytest.fixture
    def resource_setup(self):
        """Create setup for resource limit testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_memory_limit_handling(self, resource_setup):
        """Test handling when memory limits are reached."""
        setup = resource_setup
        generator = setup['generator']

        # Create content that uses significant memory
        memory_intensive_content = ""
        for i in range(5000):
            memory_intensive_content += f"""
def memory_func_{i}():
    # Create large data structures
    data = list(range(1000))
    result = {{}}
    for item in data:
        result[str(item)] = item * {i}
    return result
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(memory_intensive_content)
            temp_file = f.name

        try:
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            nodes = generator.generate_from_file(temp_file, memory_intensive_content)

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Should handle memory-intensive operations
            assert isinstance(nodes, list)
            assert len(nodes) >= 5000  # Should process all functions

            # Memory increase should be reasonable
            assert memory_increase < 500  # Less than 500MB

        finally:
            os.unlink(temp_file)

    def test_file_size_limit_handling(self, resource_setup):
        """Test handling of file size limits."""
        setup = resource_setup
        generator = setup['generator']

        # Create a very large file (several MB)
        large_content = "print('line')\n" * 50000  # ~1MB file

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            temp_file = f.name

        try:
            file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
            assert file_size_mb > 0.5  # At least 0.5MB

            # Should handle large files
            nodes = generator.generate_from_file(temp_file, large_content)
            assert isinstance(nodes, list)

        finally:
            os.unlink(temp_file)

    def test_node_count_limit_handling(self, resource_setup):
        """Test handling when node count limits are approached."""
        setup = resource_setup
        graph = setup['graph']

        # Create many nodes
        num_nodes = 10000
        for i in range(num_nodes):
            node = CodeNode(
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                file_path=f"/limit_test_{i%100}.py",
                start_line=i % 1000,
                end_line=(i % 1000) + 1
            )
            graph.add_node(node, LayerType.SEMANTIC)

        # Should handle large number of nodes
        stats = graph.get_statistics()
        assert stats['total_nodes'] == num_nodes

        # Graph operations should still work
        search_results = graph.find_nodes_by_name("func_100")
        assert len(search_results) >= 1

    def test_relationship_limit_handling(self, resource_setup):
        """Test handling when relationship limits are approached."""
        setup = resource_setup
        graph = setup['graph']

        # Create nodes and many relationships
        num_nodes = 1000
        nodes = []

        for i in range(num_nodes):
            node = CodeNode(
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                file_path="/rel_test.py",
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

        # Should handle many relationships
        stats = graph.get_statistics()
        assert stats['total_relationships'] >= relationships_created

        # Graph operations should still work
        node_relationships = graph.get_node_relationships(nodes[0].id)
        assert len(node_relationships) >= 5


class TestRecoveryMechanisms:
    """Test recovery mechanisms after errors."""

    @pytest.fixture
    def recovery_setup(self):
        """Create setup for recovery testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_error_recovery_and_continuation(self, recovery_setup):
        """Test recovery from errors and continuation of operations."""
        setup = recovery_setup
        generator = setup['generator']

        # Create multiple files, some with errors
        files_data = []

        # Valid file
        valid_content = "def valid_func(): return 42"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(valid_content)
            files_data.append((f.name, valid_content, True))

        # Invalid file
        invalid_content = "def invalid_func(\n    return"  # Syntax error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(invalid_content)
            files_data.append((f.name, invalid_content, False))

        # Another valid file
        valid_content2 = "def another_func(): return 24"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(valid_content2)
            files_data.append((f.name, valid_content2, True))

        try:
            results = []
            for file_path, content, should_succeed in files_data:
                nodes = generator.generate_from_file(file_path, content)
                results.append((should_succeed, len(nodes) > 0))

            # Should handle mixed success/failure gracefully
            assert len(results) == 3

            # Valid files should succeed
            valid_results = [success for should_succeed, success in results if should_succeed]
            assert all(valid_results)

            # Invalid file should be handled (may or may not produce nodes)
            invalid_results = [success for should_succeed, success in results if not should_succeed]
            assert len(invalid_results) == 1  # Should still return a result

        finally:
            for file_path, _, _ in files_data:
                os.unlink(file_path)

    def test_partial_failure_recovery(self, recovery_setup):
        """Test recovery from partial failures in batch operations."""
        setup = recovery_setup
        generator = setup['generator']

        # Create batch of files with mixed validity
        num_files = 10
        files_data = []

        for i in range(num_files):
            if i % 3 == 0:  # Every third file is invalid
                content = f"def broken_func_{i}(\n    return {i}"  # Syntax error
            else:
                content = f"def func_{i}(): return {i}"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                files_data.append((f.name, content, i % 3 != 0))  # Valid if not multiple of 3

        try:
            # Process all files
            total_nodes = 0
            successful_files = 0

            for file_path, content, is_valid in files_data:
                nodes = generator.generate_from_file(file_path, content)
                if len(nodes) > 0:
                    successful_files += 1
                    total_nodes += len(nodes)

            # Should process files despite some failures
            assert successful_files >= num_files * 0.6  # At least 60% success rate
            assert total_nodes > 0

        finally:
            for file_path, _, _ in files_data:
                os.unlink(file_path)

    def test_system_state_recovery(self, recovery_setup):
        """Test recovery of system state after errors."""
        setup = recovery_setup
        graph = setup['graph']
        generator = setup['generator']

        # Establish baseline
        initial_content = "def baseline_func(): return 1"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(initial_content)
            temp_file = f.name

        try:
            # Initial successful operation
            nodes1 = generator.generate_from_file(temp_file, initial_content)
            initial_stats = graph.get_statistics()

            # Simulate error condition
            error_content = "def error_func(\n    return"  # Syntax error
            with open(temp_file, 'w') as file:
                file.write(error_content)

            # Error operation
            nodes2 = generator.generate_from_file(temp_file, error_content)

            # Recovery operation
            recovery_content = "def recovery_func(): return 2"
            with open(temp_file, 'w') as file:
                file.write(recovery_content)

            nodes3 = generator.generate_from_file(temp_file, recovery_content)
            recovery_stats = graph.get_statistics()

            # System should recover and continue functioning
            assert len(nodes3) >= 2  # Should generate nodes after recovery
            assert recovery_stats['total_nodes'] >= initial_stats['total_nodes']

        finally:
            os.unlink(temp_file)


class TestCustomExceptionHandling:
    """Test handling of custom MCP exceptions."""

    @pytest.fixture
    def custom_exception_setup(self):
        """Create setup for custom exception testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_base_mcperror_handling(self, custom_exception_setup):
        """Test handling of BaseMCPError."""
        setup = custom_exception_setup
        analyzer = setup['analyzer']

        # Create a scenario that might raise BaseMCPError
        with patch.object(analyzer.dependency_analyzer, 'analyze_function_dependencies') as mock_analyze:
            mock_analyze.side_effect = BaseMCPError("Test MCP error", error_code="TEST_ERROR")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write("def test(): pass")
                temp_file = f.name

            try:
                # Should handle BaseMCPError gracefully
                result = analyzer.run_comprehensive_analysis(temp_file)
                assert isinstance(result, dict)
                # Should still return result structure
                assert 'dependency_analysis' in result

            finally:
                os.unlink(temp_file)

    def test_tool_execution_error_handling(self, custom_exception_setup):
        """Test handling of ToolExecutionError."""
        setup = custom_exception_setup
        analyzer = setup['analyzer']

        with patch.object(analyzer.performance_predictor, 'predict_bottlenecks') as mock_predict:
            mock_predict.side_effect = ToolExecutionError(
                "Tool execution failed",
                tool_name="performance_predictor",
                context={"operation": "bottleneck_detection"}
            )

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write("def test(): pass")
                temp_file = f.name

            try:
                result = analyzer.run_comprehensive_analysis(temp_file)
                assert isinstance(result, dict)
                assert 'performance_analysis' in result

            finally:
                os.unlink(temp_file)

    def test_invalid_request_error_handling(self, custom_exception_setup):
        """Test handling of InvalidRequestError."""
        setup = custom_exception_setup
        analyzer = setup['analyzer']

        with patch.object(analyzer.dependency_analyzer, 'analyze_external_library_usage') as mock_analyze:
            mock_analyze.side_effect = InvalidRequestError(
                "Invalid request in analysis",
                request_id="test_req_001",
                context={"field": "file_path", "expected": "string"}
            )

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write("def test(): pass")
                temp_file = f.name

            try:
                result = analyzer.run_comprehensive_analysis(temp_file)
                assert isinstance(result, dict)
                assert 'library_analysis' in result

            finally:
                os.unlink(temp_file)

    def test_indexing_error_handling(self, custom_exception_setup):
        """Test handling of IndexingError."""
        setup = custom_exception_setup
        generator = setup['generator']

        with patch('ast.parse') as mock_parse:
            mock_parse.side_effect = IndexingError(
                "Indexing failed due to AST error",
                file_path="/test/file.py",
                context={"ast_error": "Invalid syntax"}
            )

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write("def test(): pass")
                temp_file = f.name

            try:
                nodes = generator.generate_from_file(temp_file)
                assert isinstance(nodes, list)

            finally:
                os.unlink(temp_file)

    def test_configuration_error_handling(self, custom_exception_setup):
        """Test handling of ConfigurationError."""
        setup = custom_exception_setup
        analyzer = setup['analyzer']

        with patch.object(analyzer, 'run_comprehensive_analysis') as mock_run:
            mock_run.side_effect = ConfigurationError(
                "Configuration error in analysis",
                config_key="ANALYSIS_TIMEOUT",
                context={"expected": "integer", "received": "string"}
            )

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write("def test(): pass")
                temp_file = f.name

            try:
                result = analyzer.run_comprehensive_analysis(temp_file)
                assert isinstance(result, dict)

            finally:
                os.unlink(temp_file)


class TestErrorPropagation:
    """Test error propagation through the system."""

    @pytest.fixture
    def propagation_setup(self):
        """Create setup for error propagation testing."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)
        analyzer = AdvancedAnalysisManager(graph)

        return {
            'graph': graph,
            'generator': generator,
            'analyzer': analyzer
        }

    def test_error_context_preservation(self, propagation_setup):
        """Test that error context is preserved through propagation."""
        setup = propagation_setup
        analyzer = setup['analyzer']

        # Create a chain of operations that might propagate errors
        with patch.object(analyzer.dependency_analyzer, 'analyze_function_dependencies') as mock_dep:
            mock_dep.side_effect = BaseMCPError(
                "Dependency analysis failed",
                error_code="DEPENDENCY_ERROR",
                context={"file": "test.py", "function": "test_func"}
            )

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write("def test_func(): pass")
                temp_file = f.name

            try:
                result = analyzer.run_comprehensive_analysis(temp_file)

                # Error should be handled gracefully
                assert isinstance(result, dict)
                assert 'dependency_analysis' in result

                # Check if error information is preserved in result
                dep_analysis = result['dependency_analysis']
                if 'error' in dep_analysis:
                    assert 'DEPENDENCY_ERROR' in str(dep_analysis['error'])

            finally:
                os.unlink(temp_file)

    def test_error_recovery_strategies(self, propagation_setup):
        """Test different error recovery strategies."""
        setup = propagation_setup
        generator = setup['generator']

        # Test retry mechanism simulation
        retry_count = 0
        max_retries = 3

        def failing_operation():
            nonlocal retry_count
            retry_count += 1
            if retry_count < max_retries:
                raise IOError("Temporary I/O error")
            return ["success"]

        with patch.object(generator, 'generate_from_file') as mock_generate:
            mock_generate.side_effect = failing_operation

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write("def test(): pass")
                temp_file = f.name

            try:
                # Simulate retry logic
                result = None
                for attempt in range(max_retries):
                    try:
                        result = generator.generate_from_file(temp_file)
                        break
                    except IOError:
                        if attempt == max_retries - 1:
                            result = []  # Fallback result
                        continue

                assert result is not None

            finally:
                os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
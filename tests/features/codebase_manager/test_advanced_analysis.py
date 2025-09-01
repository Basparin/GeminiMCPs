"""Tests for Advanced Analysis Module."""

import pytest
import tempfile
import os
from unittest.mock import Mock

from codesage_mcp.features.codebase_manager import (
    AdvancedDependencyAnalyzer,
    PerformancePredictor,
    AdvancedAnalysisManager
)
from codesage_mcp.core.code_model import CodeGraph, CodeNode, NodeType, LayerType


class TestAdvancedDependencyAnalyzer:
    """Test cases for AdvancedDependencyAnalyzer."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample code graph for testing."""
        graph = CodeGraph()

        # Create sample nodes
        func_node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="test_function",
            qualified_name="test_function",
            file_path="/test/file.py",
            start_line=1,
            end_line=10,
            content="def test_function():\n    return 'test'"
        )

        lib_node = CodeNode(
            node_type=NodeType.IMPORT,
            name="import os",
            qualified_name="import_os",
            file_path="/test/file.py",
            start_line=1,
            end_line=1,
            content="import os"
        )

        graph.add_node(func_node, LayerType.SEMANTIC)
        graph.add_node(lib_node, LayerType.SEMANTIC)

        return graph

    @pytest.fixture
    def analyzer(self, sample_graph):
        """Create analyzer instance."""
        return AdvancedDependencyAnalyzer(sample_graph)

    def test_analyze_function_dependencies(self, analyzer):
        """Test function dependency analysis."""
        result = analyzer.analyze_function_dependencies("/test/file.py", "test_function")

        assert "dependencies" in result
        assert "summary" in result
        assert "test_function" in result["dependencies"]

        deps = result["dependencies"]["test_function"]
        assert "direct_calls" in deps
        assert "external_libraries" in deps
        assert "complexity_score" in deps

    def test_analyze_external_library_usage(self, analyzer):
        """Test external library usage analysis."""
        result = analyzer.analyze_external_library_usage("/test/file.py")

        assert "library_usage" in result
        assert "summary" in result

    def test_calculate_dependency_complexity(self, analyzer):
        """Test dependency complexity calculation."""
        dependencies = {
            "direct_calls": [{"name": "func1"}, {"name": "func2"}],
            "external_libraries": ["os", "sys"],
            "indirect_calls": []
        }

        score = analyzer._calculate_dependency_complexity(dependencies)
        assert isinstance(score, float)
        assert score >= 0

    def test_is_external_library(self, analyzer):
        """Test external library detection."""
        assert analyzer._is_external_library("os") is True
        assert analyzer._is_external_library("numpy") is True
        assert analyzer._is_external_library("some_internal_module") is False


class TestPerformancePredictor:
    """Test cases for PerformancePredictor."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample code graph for testing."""
        graph = CodeGraph()

        # Create a function node with nested loops
        func_node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="nested_loops_function",
            qualified_name="nested_loops_function",
            file_path="/test/file.py",
            start_line=1,
            end_line=20,
            content="def nested_loops_function():\n    for i in range(10):\n        for j in range(10):\n            pass"
        )

        graph.add_node(func_node, LayerType.SEMANTIC)
        return graph

    @pytest.fixture
    def predictor(self, sample_graph):
        """Create predictor instance."""
        return PerformancePredictor(sample_graph)

    def test_predict_bottlenecks(self, predictor):
        """Test bottleneck prediction."""
        result = predictor.predict_bottlenecks("/test/file.py")

        assert "bottlenecks" in result
        assert "summary" in result
        assert isinstance(result["bottlenecks"], list)

    def test_check_nested_loops(self, predictor):
        """Test nested loops detection."""
        func_node = Mock()
        func_node.name = "test_func"
        func_node.file_path = "/test/file.py"
        func_node.start_line = 1

        content = """
def test_func():
    for i in range(10):
        for j in range(10):
            for k in range(10):
                pass
"""

        bottlenecks = predictor._check_nested_loops(func_node, content)
        assert len(bottlenecks) > 0
        assert bottlenecks[0]["type"] == "nested_loops"
        assert bottlenecks[0]["severity_score"] >= 6  # 3 nested loops = depth 3, score = 3 * 2 = 6

    def test_check_inefficient_string_operations(self, predictor):
        """Test inefficient string operations detection."""
        func_node = Mock()
        func_node.name = "test_func"
        func_node.file_path = "/test/file.py"
        func_node.start_line = 1

        content = """
def test_func():
    result = ""
    for item in items:
        result += str(item)
"""

        bottlenecks = predictor._check_inefficient_operations(func_node, content)
        assert len(bottlenecks) > 0
        assert bottlenecks[0]["type"] == "inefficient_string_operations"


class TestAdvancedAnalysisManager:
    """Test cases for AdvancedAnalysisManager."""

    @pytest.fixture
    def manager(self):
        """Create manager instance."""
        graph = CodeGraph()
        return AdvancedAnalysisManager(graph)

    def test_run_comprehensive_analysis(self, manager):
        """Test comprehensive analysis."""
        result = manager.run_comprehensive_analysis()

        assert "dependency_analysis" in result
        assert "library_analysis" in result
        assert "performance_analysis" in result
        assert "summary" in result

    def test_get_analysis_stats(self, manager):
        """Test analysis statistics."""
        stats = manager.get_analysis_stats()

        assert "graph_stats" in stats
        assert "supported_analyses" in stats
        assert isinstance(stats["supported_analyses"], list)


class TestIntegration:
    """Integration tests for advanced analysis."""

    def test_end_to_end_analysis(self):
        """Test end-to-end analysis workflow."""
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import os
import sys
from typing import List

def process_data(items: List[str]) -> str:
    result = ""
    for item in items:
        for i in range(10):
            result += item + str(i)
    return result

def analyze_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        content = f.read()
    return {"content": content, "length": len(content)}
""")
            temp_file = f.name

        try:
            # Create graph and generate model
            graph = CodeGraph()
            from codesage_mcp.core.code_model import CodeModelGenerator
            generator = CodeModelGenerator(graph)
            nodes = generator.generate_from_file(temp_file)

            # Create analyzer and run analysis
            analyzer = AdvancedAnalysisManager(graph)
            result = analyzer.run_comprehensive_analysis(temp_file)

            # Verify results
            assert result is not None
            assert "dependency_analysis" in result
            assert "performance_analysis" in result

            # Check for detected bottlenecks
            bottlenecks = result["performance_analysis"]["bottlenecks"]
            assert isinstance(bottlenecks, list)

        finally:
            # Clean up
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
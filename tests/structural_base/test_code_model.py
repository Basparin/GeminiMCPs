"""
Unit tests for the Code Model components.

This module contains comprehensive tests for the multi-layered graph architecture
including CodeNode, Relationship, GraphLayer, CodeGraph, and CodeModelGenerator.
"""

import pytest
import tempfile
import os

from codesage_mcp.core.code_model import (
    CodeNode,
    Relationship,
    GraphLayer,
    CodeGraph,
    CodeModelGenerator,
    NodeType,
    RelationshipType,
    LayerType,
)


class TestCodeNode:
    """Test cases for CodeNode class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = CodeNode(
            id="test_id",
            node_type=NodeType.FUNCTION,
            name="test_function",
            qualified_name="module.test_function",
            file_path="/path/to/file.py",
            start_line=10,
            end_line=20,
            content="def test_function(): pass"
        )

        assert node.id == "test_id"
        assert node.node_type == NodeType.FUNCTION
        assert node.name == "test_function"
        assert node.qualified_name == "module.test_function"
        assert node.file_path == "/path/to/file.py"
        assert node.start_line == 10
        assert node.end_line == 20
        assert node.content == "def test_function(): pass"

    def test_node_auto_id_generation(self):
        """Test automatic ID generation when not provided."""
        node = CodeNode(
            node_type=NodeType.CLASS,
            name="TestClass",
            qualified_name="module.TestClass",
            file_path="/path/to/file.py",
            start_line=5,
            end_line=15
        )

        assert node.id is not None
        assert len(node.id) > 0

    def test_node_serialization(self):
        """Test node serialization and deserialization."""
        original_node = CodeNode(
            id="test_id",
            node_type=NodeType.FUNCTION,
            name="test_function",
            qualified_name="module.test_function",
            file_path="/path/to/file.py",
            start_line=10,
            end_line=20,
            content="def test_function(): pass",
            metadata={"test": "value"}
        )

        # Serialize
        data = original_node.to_dict()

        # Deserialize
        restored_node = CodeNode.from_dict(data)

        assert restored_node.id == original_node.id
        assert restored_node.node_type == original_node.node_type
        assert restored_node.name == original_node.name
        assert restored_node.qualified_name == original_node.qualified_name
        assert restored_node.file_path == original_node.file_path
        assert restored_node.start_line == original_node.start_line
        assert restored_node.end_line == original_node.end_line
        assert restored_node.content == original_node.content
        assert restored_node.metadata == original_node.metadata

    def test_node_update_content(self):
        """Test content update functionality."""
        node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="test_function",
            qualified_name="module.test_function",
            file_path="/path/to/file.py",
            start_line=10,
            end_line=20,
            content="def test_function(): pass"
        )

        old_updated_at = node.updated_at

        # Update content
        node.update_content("def test_function(): return True")

        assert node.content == "def test_function(): return True"
        assert node.updated_at > old_updated_at


class TestRelationship:
    """Test cases for Relationship class."""

    def test_relationship_creation(self):
        """Test basic relationship creation."""
        relationship = Relationship(
            source_id="source_id",
            target_id="target_id",
            relationship_type=RelationshipType.CALLS,
            layer=LayerType.SEMANTIC,
            metadata={"call_type": "direct"},
            weight=1.0
        )

        assert relationship.source_id == "source_id"
        assert relationship.target_id == "target_id"
        assert relationship.relationship_type == RelationshipType.CALLS
        assert relationship.layer == LayerType.SEMANTIC
        assert relationship.metadata == {"call_type": "direct"}
        assert relationship.weight == 1.0

    def test_relationship_serialization(self):
        """Test relationship serialization and deserialization."""
        original_rel = Relationship(
            source_id="source_id",
            target_id="target_id",
            relationship_type=RelationshipType.INHERITS,
            layer=LayerType.SEMANTIC,
            metadata={"inheritance_type": "single"},
            weight=2.0
        )

        # Serialize
        data = original_rel.to_dict()

        # Deserialize
        restored_rel = Relationship.from_dict(data)

        assert restored_rel.source_id == original_rel.source_id
        assert restored_rel.target_id == original_rel.target_id
        assert restored_rel.relationship_type == original_rel.relationship_type
        assert restored_rel.layer == original_rel.layer
        assert restored_rel.metadata == original_rel.metadata
        assert restored_rel.weight == original_rel.weight


class TestGraphLayer:
    """Test cases for GraphLayer class."""

    def test_layer_creation(self):
        """Test basic layer creation."""
        layer = GraphLayer(LayerType.SEMANTIC)

        assert layer.layer_type == LayerType.SEMANTIC
        assert len(layer.nodes) == 0
        assert len(layer.relationships) == 0

    def test_add_node(self):
        """Test adding nodes to layer."""
        layer = GraphLayer(LayerType.SEMANTIC)
        node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="test_func",
            qualified_name="test_func",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )

        layer.add_node(node)

        assert len(layer.nodes) == 1
        assert node.id in layer.nodes
        assert layer.nodes[node.id] == node

    def test_add_relationship(self):
        """Test adding relationships to layer."""
        layer = GraphLayer(LayerType.SEMANTIC)

        # Add nodes first
        source_node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="caller",
            qualified_name="caller",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )
        target_node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="callee",
            qualified_name="callee",
            file_path="/test.py",
            start_line=6,
            end_line=10
        )

        layer.add_node(source_node)
        layer.add_node(target_node)

        # Add relationship
        relationship = Relationship(
            source_id=source_node.id,
            target_id=target_node.id,
            relationship_type=RelationshipType.CALLS,
            layer=LayerType.SEMANTIC
        )

        layer.add_relationship(relationship)

        assert len(layer.relationships) == 1
        assert layer.relationships[0] == relationship

        # Check relationship tracking
        source_rels = layer.get_node_relationships(source_node.id)
        target_rels = layer.get_node_relationships(target_node.id)

        assert len(source_rels) == 1
        assert len(target_rels) == 1
        assert source_rels[0] == relationship
        assert target_rels[0] == relationship

    def test_remove_node(self):
        """Test removing nodes and their relationships."""
        layer = GraphLayer(LayerType.SEMANTIC)

        # Add nodes and relationship
        node1 = CodeNode(
            node_type=NodeType.FUNCTION,
            name="func1",
            qualified_name="func1",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )
        node2 = CodeNode(
            node_type=NodeType.FUNCTION,
            name="func2",
            qualified_name="func2",
            file_path="/test.py",
            start_line=6,
            end_line=10
        )

        layer.add_node(node1)
        layer.add_node(node2)

        relationship = Relationship(
            source_id=node1.id,
            target_id=node2.id,
            relationship_type=RelationshipType.CALLS,
            layer=LayerType.SEMANTIC
        )
        layer.add_relationship(relationship)

        # Remove node
        layer.remove_node(node1.id)

        assert node1.id not in layer.nodes
        assert len(layer.relationships) == 0  # Relationship should be removed

    def test_get_nodes_by_type(self):
        """Test filtering nodes by type."""
        layer = GraphLayer(LayerType.SEMANTIC)

        func_node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="func",
            qualified_name="func",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )
        class_node = CodeNode(
            node_type=NodeType.CLASS,
            name="MyClass",
            qualified_name="MyClass",
            file_path="/test.py",
            start_line=6,
            end_line=15
        )

        layer.add_node(func_node)
        layer.add_node(class_node)

        functions = layer.get_nodes_by_type(NodeType.FUNCTION)
        classes = layer.get_nodes_by_type(NodeType.CLASS)

        assert len(functions) == 1
        assert functions[0] == func_node
        assert len(classes) == 1
        assert classes[0] == class_node

    def test_layer_serialization(self):
        """Test layer serialization and deserialization."""
        layer = GraphLayer(LayerType.SEMANTIC)

        # Add some nodes and relationships
        node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="test_func",
            qualified_name="test_func",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )
        layer.add_node(node)

        # Serialize
        data = layer.to_dict()

        # Deserialize
        restored_layer = GraphLayer.from_dict(data)

        assert restored_layer.layer_type == layer.layer_type
        assert len(restored_layer.nodes) == len(layer.nodes)
        assert len(restored_layer.relationships) == len(layer.relationships)


class TestCodeGraph:
    """Test cases for CodeGraph class."""

    def test_graph_creation(self):
        """Test basic graph creation."""
        graph = CodeGraph()

        assert len(graph.layers) == 5  # All layer types
        assert len(graph.file_nodes) == 0
        assert len(graph.node_files) == 0

    def test_add_and_get_node(self):
        """Test adding and retrieving nodes."""
        graph = CodeGraph()
        node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="test_func",
            qualified_name="test_func",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )

        graph.add_node(node, LayerType.SEMANTIC)

        # Get from specific layer
        retrieved = graph.get_node(node.id, LayerType.SEMANTIC)
        assert retrieved == node

        # Get from any layer
        retrieved_any = graph.get_node(node.id)
        assert retrieved_any == node

    def test_add_relationship(self):
        """Test adding relationships."""
        graph = CodeGraph()

        node1 = CodeNode(
            node_type=NodeType.FUNCTION,
            name="caller",
            qualified_name="caller",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )
        node2 = CodeNode(
            node_type=NodeType.FUNCTION,
            name="callee",
            qualified_name="callee",
            file_path="/test.py",
            start_line=6,
            end_line=10
        )

        graph.add_node(node1, LayerType.SEMANTIC)
        graph.add_node(node2, LayerType.SEMANTIC)

        relationship = Relationship(
            source_id=node1.id,
            target_id=node2.id,
            relationship_type=RelationshipType.CALLS,
            layer=LayerType.SEMANTIC
        )

        graph.add_relationship(relationship)

        # Check relationships
        rels = graph.get_node_relationships(node1.id, LayerType.SEMANTIC)
        assert len(rels) == 1
        assert rels[0] == relationship

    def test_file_node_tracking(self):
        """Test file-to-node tracking."""
        graph = CodeGraph()

        node1 = CodeNode(
            node_type=NodeType.FUNCTION,
            name="func1",
            qualified_name="func1",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )
        node2 = CodeNode(
            node_type=NodeType.CLASS,
            name="MyClass",
            qualified_name="MyClass",
            file_path="/test.py",
            start_line=6,
            end_line=15
        )

        graph.add_node(node1, LayerType.SEMANTIC)
        graph.add_node(node2, LayerType.SEMANTIC)

        # Check file nodes
        file_nodes = graph.get_file_nodes("/test.py", LayerType.SEMANTIC)
        assert len(file_nodes) == 2
        assert node1 in file_nodes
        assert node2 in file_nodes

    def test_remove_file_nodes(self):
        """Test removing all nodes for a file."""
        graph = CodeGraph()

        node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="test_func",
            qualified_name="test_func",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )

        graph.add_node(node, LayerType.SEMANTIC)
        assert len(graph.get_file_nodes("/test.py")) == 1

        # Remove file nodes
        graph.remove_file_nodes("/test.py")
        assert len(graph.get_file_nodes("/test.py")) == 0
        assert node.id not in graph.node_files

    def test_find_nodes_by_name(self):
        """Test searching nodes by name."""
        graph = CodeGraph()

        node1 = CodeNode(
            node_type=NodeType.FUNCTION,
            name="test_function",
            qualified_name="module.test_function",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )
        node2 = CodeNode(
            node_type=NodeType.CLASS,
            name="TestClass",
            qualified_name="module.TestClass",
            file_path="/test.py",
            start_line=6,
            end_line=15
        )

        graph.add_node(node1, LayerType.SEMANTIC)
        graph.add_node(node2, LayerType.SEMANTIC)

        # Search by partial name
        results = graph.find_nodes_by_name("test")
        assert len(results) == 2  # Both contain "test"

        # Search by exact name
        results = graph.find_nodes_by_name("TestClass")
        assert len(results) == 1
        assert results[0] == node2

    def test_get_statistics(self):
        """Test getting graph statistics."""
        graph = CodeGraph()

        node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="test_func",
            qualified_name="test_func",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )

        graph.add_node(node, LayerType.SEMANTIC)

        stats = graph.get_statistics()

        assert stats["total_files"] == 1
        assert stats["total_nodes"] == 1
        assert stats["total_relationships"] == 0
        assert LayerType.SEMANTIC.value in stats["layers"]
        assert stats["layers"][LayerType.SEMANTIC.value]["nodes"] == 1

    def test_save_and_load(self):
        """Test saving and loading graph to/from file."""
        graph = CodeGraph()

        node = CodeNode(
            node_type=NodeType.FUNCTION,
            name="test_func",
            qualified_name="test_func",
            file_path="/test.py",
            start_line=1,
            end_line=5
        )

        graph.add_node(node, LayerType.SEMANTIC)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            # Save
            graph.save_to_file(temp_file)

            # Create new graph and load
            new_graph = CodeGraph()
            new_graph.load_from_file(temp_file)

            # Check loaded data
            loaded_node = new_graph.get_node(node.id, LayerType.SEMANTIC)
            assert loaded_node is not None
            assert loaded_node.name == node.name
            assert loaded_node.node_type == node.node_type

        finally:
            os.unlink(temp_file)


class TestCodeModelGenerator:
    """Test cases for CodeModelGenerator class."""

    def test_generator_creation(self):
        """Test basic generator creation."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)

        assert generator.graph == graph

    def test_generate_from_simple_function(self):
        """Test generating model from simple function."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)

        code = """
def simple_function():
    return "Hello World"
"""

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(code)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, code)

            # Should have at least module and function nodes
            assert len(nodes) >= 2

            # Check for function node
            func_nodes = [n for n in nodes if n.node_type == NodeType.FUNCTION]
            assert len(func_nodes) == 1

            func_node = func_nodes[0]
            assert func_node.name == "simple_function"
            assert func_node.start_line == 2
            assert "def simple_function():" in func_node.content

        finally:
            os.unlink(temp_file)

    def test_generate_from_class(self):
        """Test generating model from class definition."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)

        code = """
class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value
"""

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(code)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, code)

            # Should have module, class, and methods
            assert len(nodes) >= 4

            # Check for class node
            class_nodes = [n for n in nodes if n.node_type == NodeType.CLASS]
            assert len(class_nodes) == 1

            class_node = class_nodes[0]
            assert class_node.name == "MyClass"
            assert "class MyClass:" in class_node.content

            # Check for method nodes
            func_nodes = [n for n in nodes if n.node_type == NodeType.FUNCTION]
            assert len(func_nodes) == 2  # __init__ and get_value

        finally:
            os.unlink(temp_file)

    def test_generate_from_import(self):
        """Test generating model from import statements."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)

        code = """
import os
from pathlib import Path
from typing import List, Dict
"""

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(code)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, code)

            # Should have module and import nodes
            import_nodes = [n for n in nodes if n.node_type == NodeType.IMPORT]
            assert len(import_nodes) == 3  # Three import statements

        finally:
            os.unlink(temp_file)

    def test_handle_syntax_error(self):
        """Test handling of syntax errors."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)

        # Invalid Python code
        code = """
def broken_function(
    return "broken"
"""

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(code)
            temp_file = f.name

        try:
            nodes = generator.generate_from_file(temp_file, code)

            # Should return empty list for syntax errors
            assert len(nodes) == 0

        finally:
            os.unlink(temp_file)

    def test_handle_missing_file(self):
        """Test handling of missing files."""
        graph = CodeGraph()
        generator = CodeModelGenerator(graph)

        nodes = generator.generate_from_file("/nonexistent/file.py")

        # Should return empty list for missing files
        assert len(nodes) == 0


if __name__ == "__main__":
    pytest.main([__file__])
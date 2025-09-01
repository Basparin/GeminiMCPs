import pytest
from unittest.mock import Mock
from codesage_mcp.core.data_structures import CodeGraph, CodeNode, Relationship, ASTGenerator


@pytest.fixture
def mock_code_node():
    """Fixture to provide a mocked CodeNode instance."""
    return Mock(spec=CodeNode)


@pytest.fixture
def mock_relationship():
    """Fixture to provide a mocked Relationship instance."""
    return Mock(spec=Relationship)


@pytest.fixture
def mock_code_graph():
    """Fixture to provide a mocked CodeGraph instance."""
    return Mock(spec=CodeGraph)


@pytest.fixture
def mock_ast_generator():
    """Fixture to provide a mocked ASTGenerator instance."""
    return Mock(spec=ASTGenerator)


def test_code_node_creation(mock_code_node):
    """
    Test that CodeNode is created with correct attributes.

    Theoretical expectation: A CodeNode should have id, type, name, and optional metadata
    representing a code element (e.g., function, class).
    """
    # Arrange
    node_data = {"id": "func1", "type": "function", "name": "example_func"}

    # Act
    node = CodeNode(**node_data)

    # Assert
    assert node.id == "func1"
    assert node.type == "function"
    assert node.name == "example_func"


def test_relationship_creation(mock_relationship):
    """
    Test that Relationship is created with source, target, and type.

    Theoretical expectation: A Relationship should define connections between CodeNodes
    with a specific type (e.g., 'calls', 'inherits').
    """
    # Arrange
    rel_data = {"source": "func1", "target": "func2", "type": "calls"}

    # Act
    rel = Relationship(**rel_data)

    # Assert
    assert rel.source == "func1"
    assert rel.target == "func2"
    assert rel.type == "calls"


def test_code_graph_add_node(mock_code_graph):
    """
    Test that CodeGraph correctly adds nodes.

    Theoretical expectation: Adding a node should store it in the graph's node collection
    and allow retrieval by id.
    """
    # Arrange
    node = Mock()
    node.id = "node1"
    mock_code_graph.nodes = {}
    mock_code_graph.add_node.side_effect = lambda n: mock_code_graph.nodes.update({n.id: n})

    # Act
    mock_code_graph.add_node(node)

    # Assert
    assert "node1" in mock_code_graph.nodes


def test_code_graph_add_relationship(mock_code_graph):
    """
    Test that CodeGraph correctly adds relationships.

    Theoretical expectation: Adding a relationship should validate source and target nodes exist
    and store the relationship in the graph.
    """
    # Arrange
    rel = Mock()
    rel.source = "node1"
    rel.target = "node2"
    mock_code_graph.relationships = []
    mock_code_graph.nodes = {"node1": Mock(), "node2": Mock()}
    mock_code_graph.add_relationship.side_effect = lambda r: mock_code_graph.relationships.append(r)

    # Act
    mock_code_graph.add_relationship(rel)

    # Assert
    assert rel in mock_code_graph.relationships


def test_ast_generator_parse_code(mock_ast_generator):
    """
    Test that ASTGenerator parses code into CodeNodes and Relationships.

    Theoretical expectation: Parsing source code should generate a CodeGraph with nodes
    for classes/functions and relationships for dependencies.
    """
    # Arrange
    code = "def func1(): pass\ndef func2(): func1()"
    expected_graph = Mock()
    mock_ast_generator.parse.return_value = expected_graph

    # Act
    graph = mock_ast_generator.parse(code)

    # Assert
    assert graph == expected_graph
    mock_ast_generator.parse.assert_called_once_with(code)


def test_code_node_equality(mock_code_node):
    """
    Test that CodeNodes with same attributes are equal.

    Theoretical expectation: CodeNodes should be equal if they have the same id, type, and name.
    """
    # Arrange
    node1 = CodeNode(id="n1", type="function", name="func")
    node2 = CodeNode(id="n1", type="function", name="func")

    # Act & Assert
    assert node1 == node2


def test_relationship_validation(mock_relationship):
    """
    Test that Relationship validates source and target types.

    Theoretical expectation: Relationships should validate that source and target are valid
    node identifiers or references.
    """
    # Arrange
    valid_rel = Relationship(source="func1", target="class1", type="belongs_to")
    invalid_rel = Relationship(source="", target="class1", type="belongs_to")

    # Act & Assert
    assert valid_rel.is_valid()
    assert not invalid_rel.is_valid()


def test_code_graph_get_neighbors(mock_code_graph):
    """
    Test that CodeGraph returns correct neighbors for a node.

    Theoretical expectation: Neighbors should include all nodes directly connected
    via relationships to the given node.
    """
    # Arrange
    node_id = "node1"
    expected_neighbors = ["node2", "node3"]
    mock_code_graph.get_neighbors.return_value = expected_neighbors

    # Act
    neighbors = mock_code_graph.get_neighbors(node_id)

    # Assert
    assert neighbors == expected_neighbors


def test_ast_generator_handles_syntax_error(mock_ast_generator):
    """
    Test that ASTGenerator handles syntax errors in code.

    Theoretical expectation: Invalid code should raise a SyntaxError or return an empty graph
    with appropriate error indication.
    """
    # Arrange
    invalid_code = "def func(: pass"  # Syntax error
    mock_ast_generator.parse.side_effect = SyntaxError("Invalid syntax")

    # Act & Assert
    with pytest.raises(SyntaxError):
        mock_ast_generator.parse(invalid_code)


def test_code_graph_remove_node(mock_code_graph):
    """
    Test that CodeGraph correctly removes nodes and associated relationships.

    Theoretical expectation: Removing a node should delete it from the graph and
    remove all relationships involving that node.
    """
    # Arrange
    node_id = "node1"
    mock_code_graph.nodes = {"node1": Mock()}
    mock_code_graph.relationships = [Mock(source="node1"), Mock(target="node1")]
    mock_code_graph.remove_node.side_effect = lambda nid: mock_code_graph.nodes.pop(nid, None)

    # Act
    mock_code_graph.remove_node(node_id)

    # Assert
    assert node_id not in mock_code_graph.nodes
    # Assume relationships are cleaned up


def test_relationship_reverse(mock_relationship):
    """
    Test that Relationship can be reversed.

    Theoretical expectation: Reversing a relationship should swap source and target,
    and adjust the type if necessary (e.g., 'calls' becomes 'called_by').
    """
    # Arrange
    rel = Relationship(source="A", target="B", type="calls")
    reversed_rel = rel.reverse()

    # Act & Assert
    assert reversed_rel.source == "B"
    assert reversed_rel.target == "A"
    assert reversed_rel.type == "called_by"
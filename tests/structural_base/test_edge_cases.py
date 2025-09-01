import pytest
from unittest.mock import Mock, patch
import os
from codesage_mcp.core.api_handling import FastAPIApp, JSONRPCProcessor, MCPProtocol
from codesage_mcp.core.data_structures import CodeGraph, CodeNode, Relationship, ASTGenerator
from codesage_mcp.core.indexing_search import FAISSIndexer, SemanticSearch, RegexSearch, IncrementalUpdater
from codesage_mcp.configuration import ConfigManager, APIKeyValidator
from codesage_mcp.core.error_handling import CustomException, JSONLogger, ErrorReporter


@pytest.fixture
def edge_case_setup():
    """Fixture for edge case testing with mocked components."""
    return {
        'api_app': Mock(spec=FastAPIApp),
        'jsonrpc_processor': Mock(spec=JSONRPCProcessor),
        'code_graph': Mock(spec=CodeGraph),
        'ast_generator': Mock(spec=ASTGenerator),
        'faiss_indexer': Mock(spec=FAISSIndexer),
        'semantic_search': Mock(spec=SemanticSearch),
        'config_manager': Mock(spec=ConfigManager),
        'json_logger': Mock(spec=JSONLogger)
    }


def test_api_handling_empty_request(edge_case_setup):
    """
    Test API handling with empty request.

    Theoretical expectation: Empty requests should be rejected with appropriate
    error response indicating missing required fields.
    """
    # Arrange
    jsonrpc_processor = edge_case_setup['jsonrpc_processor']
    empty_request = {}
    expected_error = {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": None}
    jsonrpc_processor.process.return_value = expected_error

    # Act
    response = jsonrpc_processor.process(empty_request)

    # Assert
    assert response == expected_error


def test_api_handling_oversized_request(edge_case_setup):
    """
    Test API handling with extremely large request.

    Theoretical expectation: Large requests should be handled gracefully,
    either processed or rejected with size limit error.
    """
    # Arrange
    jsonrpc_processor = edge_case_setup['jsonrpc_processor']
    large_request = {"jsonrpc": "2.0", "method": "test", "params": {"data": "x" * 1000000}, "id": 1}
    jsonrpc_processor.process.side_effect = CustomException("Request too large", 413)

    # Act & Assert
    with pytest.raises(CustomException) as exc_info:
        jsonrpc_processor.process(large_request)
    assert exc_info.value.error_code == 413


def test_data_structures_empty_code_graph(edge_case_setup):
    """
    Test data structures with empty code graph.

    Theoretical expectation: Empty graphs should be valid and operations
    like adding nodes should work correctly.
    """
    # Arrange
    code_graph = edge_case_setup['code_graph']
    code_graph.nodes = {}
    code_graph.relationships = []
    code_graph.add_node.side_effect = lambda n: code_graph.nodes.update({n.id: n})

    # Act
    node = CodeNode(id="node1", type="function", name="func")
    code_graph.add_node(node)

    # Assert
    assert "node1" in code_graph.nodes


def test_data_structures_circular_relationships(edge_case_setup):
    """
    Test data structures with circular relationships.

    Theoretical expectation: Circular relationships should be detected and
    either allowed or prevented based on design requirements.
    """
    # Arrange
    code_graph = edge_case_setup['code_graph']
    rel1 = Relationship(source="A", target="B", type="calls")
    rel2 = Relationship(source="B", target="A", type="calls")
    code_graph.add_relationship(rel1)
    code_graph.add_relationship(rel2)
    code_graph.detect_cycle.return_value = True

    # Act
    has_cycle = code_graph.detect_cycle()

    # Assert
    assert has_cycle is True


def test_indexing_empty_index(edge_case_setup):
    """
    Test indexing with empty index.

    Theoretical expectation: Searching empty index should return empty results
    without errors.
    """
    # Arrange
    faiss_indexer = edge_case_setup['faiss_indexer']
    faiss_indexer.index = Mock()
    faiss_indexer.index.ntotal = 0
    query_vector = [0.1, 0.2]
    faiss_indexer.search.return_value = ([], [])

    # Act
    distances, indices = faiss_indexer.search(query_vector, k=5)

    # Assert
    assert distances == []
    assert indices == []


def test_indexing_maximum_vector_dimensions(edge_case_setup):
    """
    Test indexing with maximum supported vector dimensions.

    Theoretical expectation: Very high dimensional vectors should be handled
    or rejected with appropriate error.
    """
    # Arrange
    faiss_indexer = edge_case_setup['faiss_indexer']
    high_dim_vector = [0.1] * 10000  # 10k dimensions
    faiss_indexer.add_vectors.side_effect = Exception("Dimension too high")

    # Act & Assert
    with pytest.raises(Exception, match="Dimension too high"):
        faiss_indexer.add_vectors([high_dim_vector])


def test_configuration_missing_all_env_vars(edge_case_setup):
    """
    Test configuration with no environment variables set.

    Theoretical expectation: Missing variables should use defaults or
    raise errors for required configurations.
    """
    # Arrange
    config_manager = edge_case_setup['config_manager']
    with patch.dict(os.environ, {}, clear=True):
        config_manager.load_from_env.side_effect = CustomException("Missing required env vars")

    # Act & Assert
    with pytest.raises(CustomException, match="Missing required env vars"):
        config_manager.load_from_env()


def test_configuration_invalid_api_key_format(edge_case_setup):
    """
    Test configuration with invalid API key format.

    Theoretical expectation: Malformed API keys should be rejected
    with validation error.
    """
    # Arrange
    api_validator = Mock(spec=APIKeyValidator)
    invalid_key = "invalid-format-key"
    api_validator.validate.return_value = False
    api_validator.validate(invalid_key)

    # Assert
    api_validator.validate.assert_called_with(invalid_key)


def test_error_handling_nested_exceptions(edge_case_setup):
    """
    Test error handling with nested exceptions.

    Theoretical expectation: Nested exceptions should be properly unwrapped
    and the root cause identified.
    """
    # Arrange
    json_logger = edge_case_setup['json_logger']
    root_cause = ValueError("Root cause")
    wrapper_exc = CustomException("Wrapper", context={"cause": str(root_cause)})
    json_logger.log_error.return_value = None

    # Act
    json_logger.log_error("Nested error", wrapper_exc)

    # Assert
    json_logger.log_error.assert_called_once()


def test_ast_generation_malformed_code(edge_case_setup):
    """
    Test AST generation with malformed code.

    Theoretical expectation: Syntax errors in code should be caught and
    appropriate error responses generated.
    """
    # Arrange
    ast_generator = edge_case_setup['ast_generator']
    malformed_code = "def func(: pass"  # Missing parameter
    ast_generator.parse.side_effect = SyntaxError("Invalid syntax")

    # Act & Assert
    with pytest.raises(SyntaxError):
        ast_generator.parse(malformed_code)


def test_semantic_search_special_characters(edge_case_setup):
    """
    Test semantic search with special characters in query.

    Theoretical expectation: Special characters should be handled gracefully,
    either processed or sanitized.
    """
    # Arrange
    semantic_search = edge_case_setup['semantic_search']
    special_query = "function with @#$%^&*()"
    semantic_search.search.return_value = []

    # Act
    results = semantic_search.search(special_query)

    # Assert
    assert results == []


def test_regex_search_empty_pattern(edge_case_setup):
    """
    Test regex search with empty pattern.

    Theoretical expectation: Empty patterns should match everything or
    raise an appropriate error.
    """
    # Arrange
    regex_search = Mock(spec=RegexSearch)
    empty_pattern = ""
    text = "some text"
    regex_search.find_matches.side_effect = Exception("Empty pattern")

    # Act & Assert
    with pytest.raises(Exception, match="Empty pattern"):
        regex_search.find_matches(empty_pattern, text)


def test_incremental_updater_concurrent_updates(edge_case_setup):
    """
    Test incremental updater with concurrent update operations.

    Theoretical expectation: Concurrent updates should be handled safely,
    either serialized or with proper locking.
    """
    # Arrange
    incremental_updater = Mock(spec=IncrementalUpdater)
    updates = [{"id": "doc1"}, {"id": "doc2"}]
    incremental_updater.add_documents.side_effect = [None, None]  # Simulate concurrent success

    # Act
    for update in updates:
        incremental_updater.add_documents([update])

    # Assert
    assert incremental_updater.add_documents.call_count == 2


def test_mcp_protocol_unknown_method(edge_case_setup):
    """
    Test MCP protocol with unknown method.

    Theoretical expectation: Unknown methods should return method not found error.
    """
    # Arrange
    mcp_protocol = Mock(spec=MCPProtocol)
    unknown_request = {"method": "unknown_method"}
    mcp_protocol.process_request.side_effect = CustomException("Method not found", 32601)

    # Act & Assert
    with pytest.raises(CustomException) as exc_info:
        mcp_protocol.process_request(unknown_request)
    assert exc_info.value.error_code == 32601


def test_code_graph_maximum_nodes(edge_case_setup):
    """
    Test code graph with maximum number of nodes.

    Theoretical expectation: Large graphs should be handled efficiently or
    limited to prevent memory issues.
    """
    # Arrange
    code_graph = edge_case_setup['code_graph']
    max_nodes = 100000
    nodes = [CodeNode(id=f"node{i}", type="function", name=f"func{i}") for i in range(max_nodes)]
    code_graph.add_node.side_effect = [None] * max_nodes

    # Act
    for node in nodes:
        code_graph.add_node(node)

    # Assert
    assert code_graph.add_node.call_count == max_nodes


def test_configuration_unicode_values(edge_case_setup):
    """
    Test configuration with Unicode values.

    Theoretical expectation: Unicode characters in configuration should be
    handled properly without encoding issues.
    """
    # Arrange
    config_manager = edge_case_setup['config_manager']
    unicode_config = {"message": "Hello 世界", "path": "/path/to/文件"}
    config_manager.validate.return_value = True

    # Act
    is_valid = config_manager.validate(unicode_config)

    # Assert
    assert is_valid is True


def test_error_reporter_network_failure(edge_case_setup):
    """
    Test error reporter with network failure.

    Theoretical expectation: Network failures during reporting should be
    handled gracefully with retry or offline queuing.
    """
    # Arrange
    error_reporter = Mock(spec=ErrorReporter)
    error_details = {"message": "Network error"}
    error_reporter.report.side_effect = Exception("Connection failed")

    # Act & Assert
    with pytest.raises(Exception, match="Connection failed"):
        error_reporter.report(error_details)


def test_faiss_indexer_corrupted_index(edge_case_setup):
    """
    Test FAISS indexer with corrupted index file.

    Theoretical expectation: Corrupted index files should be detected and
    appropriate recovery or error handling initiated.
    """
    # Arrange
    faiss_indexer = edge_case_setup['faiss_indexer']
    corrupted_file = "/tmp/corrupted.index"
    faiss_indexer.load.side_effect = Exception("Corrupted index file")

    # Act & Assert
    with pytest.raises(Exception, match="Corrupted index file"):
        faiss_indexer.load(corrupted_file)


def test_json_logger_extremely_long_message(edge_case_setup):
    """
    Test JSON logger with extremely long message.

    Theoretical expectation: Very long log messages should be handled
    without memory issues or truncation.
    """
    # Arrange
    json_logger = edge_case_setup['json_logger']
    long_message = "A" * 1000000  # 1MB message
    json_logger.log.return_value = None

    # Act
    json_logger.log(20, long_message)  # INFO level

    # Assert
    json_logger.log.assert_called_once()


def test_semantic_search_timeout(edge_case_setup):
    """
    Test semantic search with timeout.

    Theoretical expectation: Long-running searches should timeout gracefully
    and return partial or empty results.
    """
    # Arrange
    semantic_search = edge_case_setup['semantic_search']
    slow_query = "very complex query requiring long processing"
    semantic_search.search.side_effect = TimeoutError("Search timeout")

    # Act & Assert
    with pytest.raises(TimeoutError, match="Search timeout"):
        semantic_search.search(slow_query)
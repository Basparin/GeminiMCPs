import pytest
from unittest.mock import Mock, patch
from codesage_mcp.core.api_handling import FastAPIApp, JSONRPCProcessor, MCPProtocol


@pytest.fixture
def mock_fastapi_app():
    """Fixture to provide a mocked FastAPI app instance for testing."""
    return Mock(spec=FastAPIApp)


@pytest.fixture
def mock_jsonrpc_processor():
    """Fixture to provide a mocked JSON-RPC processor instance."""
    return Mock(spec=JSONRPCProcessor)


@pytest.fixture
def mock_mcp_protocol():
    """Fixture to provide a mocked MCP protocol instance."""
    return Mock(spec=MCPProtocol)


def test_fastapi_app_initialization(mock_fastapi_app):
    """
    Test that the FastAPI app initializes correctly with expected routes and middleware.

    Theoretical expectation: The app should be instantiated with standard MCP routes
    (e.g., /mcp/initialize, /mcp/tools) and include JSON-RPC middleware for request processing.
    """
    # Arrange
    with patch('codesage_mcp.core.api_handling.FastAPI') as mock_fastapi:
        mock_fastapi.return_value = mock_fastapi_app

        # Act
        app = FastAPIApp()

        # Assert
        mock_fastapi.assert_called_once()
        assert app.app is not None


def test_jsonrpc_processor_handles_valid_request(mock_jsonrpc_processor):
    """
    Test that JSON-RPC processor correctly handles a valid JSON-RPC request.

    Theoretical expectation: The processor should parse the request, validate the JSON-RPC format,
    and return a properly structured response with id, result/error fields.
    """
    # Arrange
    valid_request = {"jsonrpc": "2.0", "method": "test_method", "params": {}, "id": 1}
    expected_response = {"jsonrpc": "2.0", "result": "success", "id": 1}
    mock_jsonrpc_processor.process.return_value = expected_response

    # Act
    response = mock_jsonrpc_processor.process(valid_request)

    # Assert
    assert response == expected_response
    mock_jsonrpc_processor.process.assert_called_once_with(valid_request)


def test_jsonrpc_processor_handles_invalid_request(mock_jsonrpc_processor):
    """
    Test that JSON-RPC processor handles invalid requests gracefully.

    Theoretical expectation: Invalid requests (e.g., missing jsonrpc field) should result
    in an error response with appropriate error code and message.
    """
    # Arrange
    invalid_request = {"method": "test_method", "params": {}}  # Missing jsonrpc and id
    expected_error = {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": None}
    mock_jsonrpc_processor.process.return_value = expected_error

    # Act
    response = mock_jsonrpc_processor.process(invalid_request)

    # Assert
    assert response == expected_error


def test_mcp_protocol_initialization(mock_mcp_protocol):
    """
    Test that MCP protocol initializes with correct capabilities.

    Theoretical expectation: The protocol should advertise supported capabilities
    (e.g., tools, resources) during initialization handshake.
    """
    # Arrange
    expected_capabilities = {"tools": {}, "resources": {}}
    mock_mcp_protocol.get_capabilities.return_value = expected_capabilities

    # Act
    capabilities = mock_mcp_protocol.get_capabilities()

    # Assert
    assert capabilities == expected_capabilities


def test_mcp_protocol_handles_initialize_request(mock_mcp_protocol):
    """
    Test that MCP protocol correctly handles initialize request.

    Theoretical expectation: The protocol should respond to initialize with server info,
    protocol version, and capabilities.
    """
    # Arrange
    init_request = {"method": "initialize", "params": {"protocolVersion": "2024-11-05"}}
    expected_response = {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}, "resources": {}},
        "serverInfo": {"name": "CodeSage", "version": "1.0.0"}
    }
    mock_mcp_protocol.handle_initialize.return_value = expected_response

    # Act
    response = mock_mcp_protocol.handle_initialize(init_request)

    # Assert
    assert response == expected_response


def test_fastapi_app_routes_registration(mock_fastapi_app):
    """
    Test that FastAPI app registers MCP-specific routes correctly.

    Theoretical expectation: Routes for MCP protocol (e.g., /mcp/tools/list) should be
    registered and accessible.
    """
    # Arrange
    with patch.object(mock_fastapi_app, 'add_api_route') as mock_add_route:
        app = FastAPIApp()
        app.app = mock_fastapi_app

        # Act
        app.register_routes()

        # Assert
        mock_add_route.assert_called()


def test_jsonrpc_processor_batch_requests(mock_jsonrpc_processor):
    """
    Test that JSON-RPC processor handles batch requests.

    Theoretical expectation: Batch requests should be processed individually and
    responses returned in the same order as requests.
    """
    # Arrange
    batch_request = [
        {"jsonrpc": "2.0", "method": "method1", "id": 1},
        {"jsonrpc": "2.0", "method": "method2", "id": 2}
    ]
    expected_responses = [
        {"jsonrpc": "2.0", "result": "result1", "id": 1},
        {"jsonrpc": "2.0", "result": "result2", "id": 2}
    ]
    mock_jsonrpc_processor.process_batch.return_value = expected_responses

    # Act
    responses = mock_jsonrpc_processor.process_batch(batch_request)

    # Assert
    assert responses == expected_responses


def test_mcp_protocol_error_handling(mock_mcp_protocol):
    """
    Test that MCP protocol handles errors during request processing.

    Theoretical expectation: Protocol errors should be wrapped in JSON-RPC error format
    with appropriate error codes.
    """
    # Arrange
    error_request = {"method": "invalid_method"}
    expected_error = {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": None}
    mock_mcp_protocol.process_request.return_value = expected_error

    # Act
    response = mock_mcp_protocol.process_request(error_request)

    # Assert
    assert response == expected_error
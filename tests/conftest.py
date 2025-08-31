import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Existing fixtures
@pytest.fixture
def mock_sentence_transformer_model():
    """
    Pytest fixture that provides a mock SentenceTransformer model.
    """
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 128
    mock_model.encode.return_value = np.random.rand(128).astype(np.float32)
    return mock_model

@pytest.fixture
def mock_chunk_file():
    """
    Pytest fixture that provides a mock for codesage_mcp.indexing.chunk_file.
    """
    with patch('codesage_mcp.indexing.chunk_file') as mock_chunk:
        mock_chunk.return_value = [
            MagicMock(content="mocked content", start_line=1, end_line=1)
        ]
        yield mock_chunk

@pytest.fixture(autouse=True)
def reset_global_cache():
    """
    Fixture to reset the global cache instance before and after each test.
    This ensures test isolation for the IntelligentCache singleton.
    """
    from codesage_mcp.cache import reset_cache_instance
    reset_cache_instance() # Reset before test
    yield
    reset_cache_instance() # Reset after test

# Gemini Compatibility Fixtures
@pytest.fixture
def sample_tools_object():
    """Sample tools object for testing."""
    return {
        "code_analysis": {
            "name": "code_analysis",
            "description": "Analyze code for issues and improvements",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string"}
                }
            }
        },
        "suggest_improvements": {
            "name": "suggest_improvements",
            "description": "Suggest code improvements",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                }
            }
        }
    }

@pytest.fixture
def large_tools_object():
    """Large tools object for performance testing."""
    return {
        f"tool_{i}": {
            "name": f"tool_{i}",
            "description": f"Description for tool {i} with extensive documentation and examples",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": f"Parameter 1 for tool {i}"},
                    "param2": {"type": "number", "description": f"Parameter 2 for tool {i}"},
                    "param3": {"type": "boolean", "description": f"Parameter 3 for tool {i}"}
                }
            }
        }
        for i in range(1000)
    }

@pytest.fixture
def gemini_cli_headers():
    """Headers simulating Gemini CLI requests."""
    return {
        "user-agent": "node/18.17.0",
        "content-type": "application/json",
        "accept": "application/json"
    }

@pytest.fixture
def standard_mcp_headers():
    """Headers for standard MCP requests."""
    return {
        "user-agent": "curl/7.88.1",
        "content-type": "application/json"
    }

@pytest.fixture
def tools_list_request_body():
    """Sample tools/list request body."""
    return {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1,
        "params": {}
    }

@pytest.fixture
def tools_call_request_body():
    """Sample tools/call request body."""
    return {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 2,
        "params": {
            "name": "code_analysis",
            "arguments": {"code": "print('hello')", "language": "python"}
        }
    }

@pytest.fixture
def initialize_request_body():
    """Sample initialize request body."""
    return {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 3,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

@pytest.fixture
def sample_error_response():
    """Sample error response for testing."""
    return {
        "code": "INVALID_PARAMS",
        "message": "Invalid parameters provided",
        "data": {"field": "code", "expected": "string"}
    }

@pytest.fixture
def corrupted_json_strings():
    """Various corrupted JSON strings for testing."""
    return [
        '{"method": "tools/list", "id": 1',  # Missing closing brace
        '{"method": "tools/list", "id": }',  # Invalid value
        '{"method": "tools/list", "id": 1, "params": {unclosed}',  # Unclosed object
        'not json at all',  # Not JSON
        '{"method": "tools/list", "id": 1, "extra": }',  # Invalid value
    ]

@pytest.fixture
def mock_fastapi_request():
    """Mock FastAPI request object."""
    request = MagicMock()
    request.headers = {
        "user-agent": "node/18.17.0",
        "content-type": "application/json"
    }
    request.method = "POST"
    request.url = MagicMock()
    request.url.path = "/mcp"
    return request

@pytest.fixture
def mock_fastapi_response():
    """Mock FastAPI response object."""
    response = MagicMock()
    response.status_code = 200
    response.headers = {"content-type": "application/json"}
    return response
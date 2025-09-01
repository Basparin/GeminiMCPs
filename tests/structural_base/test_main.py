import pytest
from fastapi.testclient import TestClient
import shutil
from unittest.mock import patch
from codesage_mcp.main import app

# Test client fixture
@pytest.fixture
def test_client():
    """Provide test client for all tests."""
    return TestClient(app)

# Common test data
MCP_BASE_PAYLOAD = {"jsonrpc": "2.0"}
TOOL_CALL_PAYLOAD = {**MCP_BASE_PAYLOAD, "method": "tools/call"}

client = TestClient(app)


# Helper functions for common test patterns
def make_mcp_request(method, params=None, request_id="1"):
    """Create a standardized MCP JSON-RPC request."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": request_id
    }


def assert_mcp_success_response(response_data, expected_id="1"):
    """Assert that MCP response indicates success."""
    assert response_data["jsonrpc"] == "2.0"
    assert response_data["id"] == expected_id
    assert "result" in response_data
    assert "error" not in response_data


def assert_mcp_error_response(response_data, expected_id="1"):
    """Assert that MCP response indicates error."""
    assert response_data["jsonrpc"] == "2.0"
    assert response_data["id"] == expected_id
    assert "error" in response_data
    assert "result" not in response_data


@pytest.fixture
def temp_dir(tmp_path):
    """Test Temp dir.

    Creates a temporary directory with a test file for testing purposes.
    The directory and its contents are automatically cleaned up after the test.

    Args:
        tmp_path: Pytest fixture that provides a temporary directory.
    """
    d = tmp_path / "test_dir"
    d.mkdir()
    (d / "file1.txt").write_text("hello")
    yield d
    shutil.rmtree(d)


@pytest.fixture
def temp_code_file_for_docs(tmp_path):
    """Creates a temporary Python file with documented and undocumented functions
    for testing."""
    code_file = tmp_path / "test_module.py"
    code_content = '''
"""A test module docstring."""

def documented_function(arg1: str) -> str:
    """This function is documented.

    Args:
        arg1: A string argument.

    Returns:
        A string return value.
    """
    return arg1

def undocumented_function(x, y):
    # This function lacks a docstring.
    return x + y

class DocumentedClass:
    """This class is documented."""

    def documented_method(self):
        """This method is documented."""
        pass

    def undocumented_method(self):
        # This method lacks a docstring.
        pass

# A standalone comment
some_var = 10

def another_undocumented_function():
    pass
'''
    code_file.write_text(code_content)
    yield code_file
    # Cleanup is handled by tmp_path fixture


def test_root_endpoint():
    """Test root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "CodeSage MCP Server is running!"}


def test_mcp_initialize():
    """Test MCP initialization process."""
    request = make_mcp_request("initialize", {}, "1")
    response = client.post("/mcp", json=request)
    assert response.status_code == 200

    data = response.json()
    assert_mcp_success_response(data, "1")
    assert data["result"]["serverInfo"]["name"] == "CodeSage MCP Server"


def test_mcp_tools_list():
    """Test tools list endpoint returns available tools."""
    request = make_mcp_request("tools/list", {}, "2")
    response = client.post("/mcp", json=request)
    assert response.status_code == 200

    data = response.json()
    assert_mcp_success_response(data, "2")
    assert len(data["result"]) > 0


def test_mcp_tool_call_success(temp_dir):
    """Test successful MCP tool call execution."""
    # Index the codebase first
    index_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "index_codebase",
                "arguments": {"path": str(temp_dir)},
            },
            "id": "index_for_get_structure_test",
        },
    )
    assert index_response.status_code == 200
    index_data = index_response.json()
    assert index_data["jsonrpc"] == "2.0"
    assert "result" in index_data
    assert "indexed successfully" in index_data["result"]["message"]

    # Call get_file_structure
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_file_structure",
                "arguments": {
                    "codebase_path": str(temp_dir),
                    "file_path": "file1.txt",
                },
            },
            "id": "3",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "3"
    assert "structure" in data["result"]
    assert any("file1.txt" in s for s in data["result"]["structure"])


def test_mcp_tool_call_not_found():
    """Test tool call with non-existent tool returns error."""
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "non_existent_tool", "arguments": {}},
            "id": "4",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "4"
    assert "error" in data
    assert data["error"]["code"] == -32001
    assert "Tool not found" in data["error"]["message"]


def test_mcp_tool_call_list_undocumented_functions(temp_code_file_for_docs):
    """Test calling the list_undocumented_functions tool via the MCP endpoint."""
    file_path = str(temp_code_file_for_docs)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "list_undocumented_functions",
                "arguments": {"file_path": file_path},
            },
            "id": "list_undoc_test_1",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "list_undoc_test_1"

    result = data["result"]
    assert "message" in result
    assert "undocumented_functions" in result
    assert result["message"].startswith("Found")

    # Verify the list of undocumented functions
    undocumented_funcs = result["undocumented_functions"]
    func_names = {func["name"] for func in undocumented_funcs}

    expected_undocumented = {
        "undocumented_function",
        "another_undocumented_function",
        "undocumented_method",
    }
    assert (
        func_names == expected_undocumented
    ), f"Expected {expected_undocumented}, but got {func_names}"

    # Verify line numbers are present (basic check)
    for func in undocumented_funcs:
        assert "line_number" in func and isinstance(func["line_number"], int)


# ... (other imports are already at the top) ...

# --- New Tests for Semantic Search Tool Function ---


@patch("codesage_mcp.tools.llm_analysis.codebase_manager")
def test_semantic_search_codebase_tool_success(mock_codebase_manager):
    """Test the semantic_search_codebase_tool function for a successful search."""
    # Arrange: Mock the manager's method to return a specific result
    mock_results = [
        {"file_path": "/path/to/file1.py", "score": 0.1},
        {"file_path": "/path/to/file2.py", "score": 0.2},
    ]
    mock_codebase_manager.searching_manager.semantic_search_codebase.return_value = (
        mock_results
    )

    # Act: Call the tool function
    from codesage_mcp.tools import semantic_search_codebase_tool

    result = semantic_search_codebase_tool(
        codebase_path="/test/codebase", query="test query", top_k=2
    )

    # Assert: Check the structure and content of the result
    assert "message" in result
    assert "Found 2 semantically similar code snippets" in result["message"]
    assert "results" in result
    assert result["results"] == mock_results

    # Verify the mock was called correctly
    mock_codebase_manager.searching_manager.semantic_search_codebase.assert_called_once_with(
        "test query", mock_codebase_manager.sentence_transformer_model, 2
    )


@patch("codesage_mcp.tools.llm_analysis.codebase_manager")
def test_semantic_search_codebase_tool_error(mock_codebase_manager):
    """Test the semantic_search_codebase_tool function when the manager
    raises an error."""
    # Arrange: Mock the manager's method to raise an exception
    mock_codebase_manager.searching_manager.semantic_search_codebase.side_effect = (
        Exception("Test error from FAISS")
    )

    # Act: Call the tool function
    from codesage_mcp.tools import semantic_search_codebase_tool

    result = semantic_search_codebase_tool(
        codebase_path="/test/codebase", query="test query", top_k=2
    )

    # Assert: Check the structure and content of the error result
    assert "error" in result
    assert result["error"]["code"] == -32012
    assert "Test error from FAISS" in result["error"]["message"]

    # Verify the mock was called
    mock_codebase_manager.searching_manager.semantic_search_codebase.assert_called_once_with(
        "test query", mock_codebase_manager.sentence_transformer_model, 2
    )


# --- New Tests for Semantic Search Tool Function ---


# --- New Integration Tests for Find Duplicate Code MCP Endpoint ---


def test_mcp_tool_call_find_duplicate_code_success(temp_dir):
    """Test calling the find_duplicate_code tool via the MCP endpoint successfully."""
    # Step 1: Index the temporary codebase
    index_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "index_codebase",
                "arguments": {"path": str(temp_dir)},
            },
            "id": "index_for_duplicate_code_test",
        },
    )
    assert index_response.status_code == 200
    index_result = index_response.json()["result"]
    assert "message" in index_result
    assert "indexed successfully" in index_result["message"]

    # Step 2: Perform duplicate code detection
    # Note: In a real test, we'd mock sentence-transformers and faiss
    # for predictability.
    # Here, we test the endpoint mechanics and assume the underlying logic
    # is tested in unit tests.
    duplicate_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "find_duplicate_code",
                "arguments": {
                    "codebase_path": str(temp_dir),
                    "min_similarity": 0.8,
                    "min_lines": 3,
                },
            },
            "id": "find_duplicate_code_test_1",
        },
    )
    assert duplicate_response.status_code == 200
    duplicate_data = duplicate_response.json()
    assert duplicate_data["jsonrpc"] == "2.0"
    assert duplicate_data["id"] == "find_duplicate_code_test_1"

    result = duplicate_data["result"]
    assert "message" in result
    assert "duplicate code sections" in result["message"]
    assert "duplicates" in result
    # We can't assert specific results without mocking, but we can check it's a list
    assert isinstance(result["duplicates"], list)


# --- New Integration Tests for Get Configuration MCP Endpoint ---


def test_mcp_tool_call_get_configuration_success():
    """Test calling the get_configuration tool via the MCP endpoint successfully."""
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_configuration",
                "arguments": {},
            },
            "id": "get_configuration_test_1",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "get_configuration_test_1"

    result = data["result"]
    assert "message" in result
    assert "configuration" in result
    assert result["message"] == "Current configuration retrieved successfully."

    # Check the configuration structure
    config = result["configuration"]
    assert "groq_api_key" in config
    assert "openrouter_api_key" in config
    assert "google_api_key" in config

    # Check that the API keys are masked
    # Patch all API keys to ensure consistent test values
    with patch('codesage_mcp.config.GROQ_API_KEY', 'gsk_long_test_key_for_masking_2riH'), \
         patch('codesage_mcp.config.OPENROUTER_API_KEY', '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'), \
         patch('codesage_mcp.config.GOOGLE_API_KEY', 'AIzaSyBRmsd7sKtVviULWkI6fR3iNQrZBOKdqmA'):
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "get_configuration",
                    "arguments": {},
                },
                "id": "get_configuration_test_1",
            },
        )
    data = response.json()
    config = data["result"]["configuration"]
    assert config["groq_api_key"] == "gsk_...2riH" # Expected masked value
    assert config["openrouter_api_key"] == "sk-o...94a7"
    assert config["google_api_key"] == "AIza...dqmA"


# --- New Integration Tests for Analyze Codebase Improvements MCP Endpoint ---


def test_mcp_tool_call_analyze_codebase_improvements_success(temp_dir):
    """Test calling the analyze_codebase_improvements tool via the MCP
    endpoint successfully."""
    # Step 1: Index the temporary codebase
    index_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "index_codebase",
                "arguments": {"path": str(temp_dir)},
            },
            "id": "index_for_analysis_test",
        },
    )
    assert index_response.status_code == 200
    index_result = index_response.json()["result"]
    assert "message" in index_result
    assert "indexed successfully" in index_result["message"]

    # Step 2: Perform codebase analysis
    # Note: In a real test, we'd mock sentence-transformers and faiss
    # for predictability.
    # Here, we test the endpoint mechanics and assume the underlying logic
    # is tested in unit tests.
    analysis_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "analyze_codebase_improvements",
                "arguments": {
                    "codebase_path": str(temp_dir),
                },
            },
            "id": "analyze_codebase_improvements_test_1",
        },
    )
    assert analysis_response.status_code == 200
    analysis_data = analysis_response.json()
    assert analysis_data["jsonrpc"] == "2.0"
    assert analysis_data["id"] == "analyze_codebase_improvements_test_1"

    result = analysis_data["result"]
    assert "message" in result
    assert "analysis" in result
    assert result["message"] == "Codebase analysis completed successfully."

    # Check the analysis structure
    analysis = result["analysis"]
    assert "total_files" in analysis
    assert "python_files" in analysis
    assert "todo_comments" in analysis
    assert "fixme_comments" in analysis
    assert "undocumented_functions" in analysis
    assert "potential_duplicates" in analysis
    assert "large_files" in analysis
    assert "suggestions" in analysis

    # We can't assert specific values without mocking, but we can check the structure
    assert isinstance(analysis["total_files"], int)
    assert isinstance(analysis["python_files"], int)
    assert isinstance(analysis["todo_comments"], int)
    assert isinstance(analysis["fixme_comments"], int)
    assert isinstance(analysis["undocumented_functions"], int)
    assert isinstance(analysis["potential_duplicates"], int)
    assert isinstance(analysis["large_files"], list)
    assert isinstance(analysis["suggestions"], list)


# --- New Integration Tests for Profile Code Performance MCP Endpoint ---


def test_mcp_tool_call_profile_code_performance_success(temp_dir):
    """Test calling the profile_code_performance tool via the MCP endpoint
    successfully."""

    # Create a simple test Python file in the temp directory
    test_file_path = temp_dir / "test_module.py"
    with open(test_file_path, "w") as f:
        f.write("""
def simple_function():
    return 1 + 1

def another_function():
    return simple_function() * 2

if __name__ == "__main__":
    result = another_function()
    print(result)
""")

    # Test profiling the entire file
    profile_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "profile_code_performance",
                "arguments": {
                    "file_path": str(test_file_path),
                },
            },
            "id": "profile_code_performance_test_1",
        },
    )
    assert profile_response.status_code == 200
    profile_data = profile_response.json()
    assert profile_data["jsonrpc"] == "2.0"
    assert profile_data["id"] == "profile_code_performance_test_1"

    result = profile_data["result"]
    assert "message" in result
    assert "total_functions_profiled" in result
    assert "top_bottlenecks" in result
    assert "raw_stats" in result
    assert f"Performance profiling completed for {test_file_path}" in result["message"]

    # Check the structure of the results
    assert isinstance(result["total_functions_profiled"], int)
    assert isinstance(result["top_bottlenecks"], list)
    assert isinstance(result["raw_stats"], str)

    # Test profiling a specific function
    profile_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "profile_code_performance",
                "arguments": {
                    "file_path": str(test_file_path),
                    "function_name": "simple_function",
                },
            },
            "id": "profile_code_performance_test_2",
        },
    )
    assert profile_response.status_code == 200
    profile_data = profile_response.json()
    assert profile_data["jsonrpc"] == "2.0"
    assert profile_data["id"] == "profile_code_performance_test_2"

    result = profile_data["result"]
    assert "message" in result
    assert "total_functions_profiled" in result
    assert "top_bottlenecks" in result
    assert "raw_stats" in result
    expected_msg = (
        f"Performance profiling completed for {test_file_path} "
        f"function 'simple_function'"
    )
    assert expected_msg in result["message"]


def test_mcp_tool_call_profile_code_performance_error():
    """Test calling the profile_code_performance tool via the MCP endpoint
    with an error."""
    # Test with a non-existent file
    profile_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "profile_code_performance",
                "arguments": {
                    "file_path": "/test/nonexistent/file.py",
                },
            },
            "id": "profile_code_performance_test_3",
        },
    )
    assert profile_response.status_code == 200
    profile_data = profile_response.json()
    assert profile_data["jsonrpc"] == "2.0"
    assert profile_data["id"] == "profile_code_performance_test_3"

    # Check that we get an error in the result field
    assert "result" in profile_data
    result = profile_data["result"]
    assert "error" in result
    error = result["error"]
    assert error is not None
# --- Temporary Unit Tests for Uncovered Functionality ---


def test_jsonrpc_response_model_dump():
    """Test JSONRPCResponse.model_dump method for uncovered lines."""
    from codesage_mcp.main import JSONRPCResponse

    # Test with result
    response = JSONRPCResponse(jsonrpc="2.0", result={"test": "data"}, id="123")
    data = response.model_dump()
    assert "jsonrpc" in data
    assert "result" in data
    assert "id" in data
    assert data["result"] == {"test": "data"}

    # Test with error
    response = JSONRPCResponse(jsonrpc="2.0", error={"code": -32000, "message": "test error"}, id="123")
    data = response.model_dump()
    assert "error" in data
    assert "result" not in data  # Should be excluded when error is present


def test_jsonrpc_response_dict():
    """Test JSONRPCResponse.dict method for uncovered lines."""
    from codesage_mcp.main import JSONRPCResponse

    response = JSONRPCResponse(jsonrpc="2.0", result={"test": "data"}, id="123")
    data = response.dict()
    assert data == {"jsonrpc": "2.0", "result": {"test": "data"}, "id": "123"}


def test_jsonrpc_response_create_compatible_response():
    """Test JSONRPCResponse.create_compatible_response method for uncovered lines."""
    from codesage_mcp.main import JSONRPCResponse
    from unittest.mock import patch

    with patch('codesage_mcp.main.get_compatibility_handler') as mock_get_handler:
        mock_handler = mock_get_handler.return_value
        mock_handler.adapt_error_response.return_value = {"code": -32000, "message": "adapted error"}
        mock_handler.create_compatible_response.return_value = {
            "jsonrpc": "2.0",
            "result": {"test": "adapted"},
            "id": "123"
        }

        # Test with error that needs adaptation
        response = JSONRPCResponse.create_compatible_response(
            result=None,
            error={"code": "string_code", "message": "test"},
            request_id="123"
        )

        mock_handler.adapt_error_response.assert_called_once()
        mock_handler.create_compatible_response.assert_called_once()


def test_metrics_endpoint():
    """Test /metrics endpoint for uncovered lines."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "codesage_mcp_info" in content
    assert "codesage_mcp_uptime_seconds" in content
    assert "codesage_mcp_requests_total" in content
    assert "codesage_mcp_performance_score" in content


def test_mcp_notifications_initialized():
    """Test notifications/initialized method for uncovered lines."""
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized", "id": "1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "1"
    # Notifications don't return a result field, only jsonrpc and id


def test_mcp_prompts_list():
    """Test prompts/list method for uncovered lines."""
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "prompts/list", "id": "1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "1"
    assert "result" in data
    assert "prompts" in data["result"]
    assert data["result"]["prompts"] == []


def test_mcp_tool_call_invalid_params():
    """Test tools/call with invalid params for uncovered lines."""
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "tools/call", "id": "1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "1"
    assert "error" in data
    assert data["error"]["code"] == -32602  # INVALID_PARAMS


def test_mcp_tool_call_invalid_params_dict():
    """Test tools/call with invalid params as non-dict for uncovered lines."""
    response = client.post(
        "/mcp", json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": "invalid_params",
            "id": "1"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "1"
    assert "error" in data


def test_mcp_tool_call_execution_error():
    """Test tools/call with tool execution error for uncovered lines."""
    from unittest.mock import patch

    # Mock a tool function to raise an exception
    with patch('codesage_mcp.main.TOOL_FUNCTIONS') as mock_tool_functions:
        mock_tool_functions.__contains__.return_value = True
        mock_tool_functions.__getitem__.return_value = lambda **kwargs: (_ for _ in ()).throw(Exception("Test execution error"))

        response = client.post(
            "/mcp", json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "test_tool", "arguments": {}},
                "id": "1"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "1"
        assert "error" in data
        assert "Error executing tool test_tool" in data["error"]["message"]




def test_mcp_unknown_method():
    """Test unknown JSON-RPC method for uncovered lines."""
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "unknown_method", "id": "1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "1"
    assert "error" in data
    assert "Unknown JSON-RPC method" in data["error"]["message"]
    assert "code" in data["error"]
def test_mcp_invalid_json():
    """Test invalid JSON request for uncovered lines."""
    response = client.post("/mcp", data="invalid json")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "Invalid JSON-RPC request" in data["error"]["message"]

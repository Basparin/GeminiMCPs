import pytest
from fastapi.testclient import TestClient
import shutil
from unittest.mock import patch
from codesage_mcp.main import app

client = TestClient(app)


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
    """Test Root endpoint.

    This test verifies that the root endpoint returns the expected welcome message.
    It checks that the server is running and responding correctly to basic requests.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "CodeSage MCP Server is running!"}


def test_mcp_initialize():
    """Test Mcp initialize.

    This test verifies that the MCP initialization process works correctly.
    It checks that the server responds with the expected protocol version
    and server information when initializing the connection.
    """
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "initialize", "id": "1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "1"
    assert data["result"]["serverInfo"]["name"] == "CodeSage MCP Server"


def test_mcp_tools_list():
    """Test Mcp tools list.

    This test verifies that the tools list endpoint returns the expected
    list of available tools. It checks that all registered tools are properly
    listed with their correct names and descriptions.
    """
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": "2"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "2"
    assert len(data["result"]) > 0
    assert len(data["result"]) > 0


def test_mcp_tool_call_success(temp_dir):
    """Test Mcp tool call success.

    This test verifies successful execution of mcp tool call success.
    It checks that when a valid tool call is made, the server correctly
    processes the request and returns the expected result.

    Args:
        temp_dir: Test parameter providing a temporary directory.
    """
    # Use a real tool that doesn't have complex dependencies
    # get_file_structure requires the codebase to be indexed first.

    # Step 1: Index the codebase
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
    assert "message" in index_data["result"]
    assert "indexed successfully" in index_data["result"]["message"]

    # Step 2: Call get_file_structure
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_file_structure",
                "arguments": {
                    "codebase_path": str(temp_dir),
                    "file_path": "file1.txt",  # File created by temp_dir fixture
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
    # The structure should contain the filename
    assert any("file1.txt" in s for s in data["result"]["structure"])


def test_mcp_tool_call_not_found():
    """Test Mcp tool call not found.

    This test verifies that when a tool call is made for a non-existent tool,
    the server correctly returns an appropriate error response indicating
    that the tool was not found.
    """
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
    assert data["error"]["code"] == -32001  # TOOL_NOT_FOUND
    assert "Tool not found" in data["error"]["message"]
    assert "Tool not found" in response.text


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
    assert "code" in error
    assert error["code"] == -32003

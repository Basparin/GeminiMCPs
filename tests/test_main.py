import pytest
from fastapi.testclient import TestClient
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from codesage_mcp.main import app

client = TestClient(app)


@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "test_dir"
    d.mkdir()
    (d / "file1.txt").write_text("hello")
    yield d
    shutil.rmtree(d)

@pytest.fixture
def temp_code_file_for_docs(tmp_path):
    """Creates a temporary Python file with documented and undocumented functions for testing."""
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
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "CodeSage MCP Server is running!"}


def test_mcp_initialize():
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "initialize", "id": "1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "1"
    assert data["result"]["serverInfo"]["name"] == "CodeSage MCP Server"


def test_mcp_tools_list():
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": "2"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "2"
    assert "tools" in data["result"]
    assert len(data["result"]["tools"]) > 0


def test_mcp_tool_call_success(temp_dir):
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
                    "file_path": "file1.txt", # File created by temp_dir fixture
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
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "non_existent_tool", "arguments": {}},
            "id": "4",
        },
    )
    assert response.status_code == 404
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
    
    expected_undocumented = {"undocumented_function", "another_undocumented_function", "undocumented_method"}
    assert func_names == expected_undocumented, f"Expected {expected_undocumented}, but got {func_names}"
    
    # Verify line numbers are present (basic check)
    for func in undocumented_funcs:
        assert "line_number" in func and isinstance(func["line_number"], int)

from unittest.mock import patch, MagicMock
# ... (other imports are already at the top) ...

# --- New Tests for Semantic Search Tool Function ---

@patch('codesage_mcp.tools.codebase_manager')
def test_semantic_search_codebase_tool_success(mock_codebase_manager):
    """Test the semantic_search_codebase_tool function for a successful search."""
    # Arrange: Mock the manager's method to return a specific result
    mock_results = [
        {"file_path": "/path/to/file1.py", "score": 0.1},
        {"file_path": "/path/to/file2.py", "score": 0.2}
    ]
    mock_codebase_manager.semantic_search_codebase.return_value = mock_results

    # Act: Call the tool function
    from codesage_mcp.tools import semantic_search_codebase_tool
    result = semantic_search_codebase_tool(
        codebase_path="/test/codebase",
        query="test query",
        top_k=2
    )

    # Assert: Check the structure and content of the result
    assert "message" in result
    assert "Found 2 semantically similar code snippets" in result["message"]
    assert "results" in result
    assert result["results"] == mock_results

    # Verify the mock was called correctly
    mock_codebase_manager.semantic_search_codebase.assert_called_once_with("test query", 2)

@patch('codesage_mcp.tools.codebase_manager')
def test_semantic_search_codebase_tool_error(mock_codebase_manager):
    """Test the semantic_search_codebase_tool function when the manager raises an error."""
    # Arrange: Mock the manager's method to raise an exception
    mock_codebase_manager.semantic_search_codebase.side_effect = Exception("Test error from FAISS")

    # Act: Call the tool function
    from codesage_mcp.tools import semantic_search_codebase_tool
    result = semantic_search_codebase_tool(
        codebase_path="/test/codebase",
        query="test query",
        top_k=2
    )

    # Assert: Check the structure and content of the error result
    assert "error" in result
    assert result["error"]["code"] == "SEMANTIC_SEARCH_ERROR"
    assert "Test error from FAISS" in result["error"]["message"]

    # Verify the mock was called
    mock_codebase_manager.semantic_search_codebase.assert_called_once_with("test query", 2)

# --- New Tests for Semantic Search Tool Function ---

@patch('codesage_mcp.tools.codebase_manager')
def test_semantic_search_codebase_tool_success(mock_codebase_manager):
    """Test the semantic_search_codebase_tool function for a successful search."""
    # Arrange: Mock the manager's method to return a specific result
    mock_results = [
        {"file_path": "/path/to/file1.py", "score": 0.1},
        {"file_path": "/path/to/file2.py", "score": 0.2}
    ]
    mock_codebase_manager.semantic_search_codebase.return_value = mock_results

    # Act: Call the tool function
    from codesage_mcp.tools import semantic_search_codebase_tool
    result = semantic_search_codebase_tool(
        codebase_path="/test/codebase",
        query="test query",
        top_k=2
    )

    # Assert: Check the structure and content of the result
    assert "message" in result
    assert "Found 2 semantically similar code snippets" in result["message"]
    assert "results" in result
    assert result["results"] == mock_results

    # Verify the mock was called correctly
    mock_codebase_manager.semantic_search_codebase.assert_called_once_with("test query", 2)

@patch('codesage_mcp.tools.codebase_manager')
def test_semantic_search_codebase_tool_error(mock_codebase_manager):
    """Test the semantic_search_codebase_tool function when the manager raises an error."""
    # Arrange: Mock the manager's method to raise an exception
    mock_codebase_manager.semantic_search_codebase.side_effect = Exception("Test error from FAISS")

    # Act: Call the tool function
    from codesage_mcp.tools import semantic_search_codebase_tool
    result = semantic_search_codebase_tool(
        codebase_path="/test/codebase",
        query="test query",
        top_k=2
    )

    # Assert: Check the structure and content of the error result
    assert "error" in result
    assert result["error"]["code"] == "SEMANTIC_SEARCH_ERROR"
    assert "Test error from FAISS" in result["error"]["message"]

    # Verify the mock was called
    mock_codebase_manager.semantic_search_codebase.assert_called_once_with("test query", 2)

# --- New Integration Tests for Semantic Search MCP Endpoint ---

def test_mcp_tool_call_semantic_search_success(temp_dir):
    """Test calling the semantic_search_codebase tool via the MCP endpoint successfully."""
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
            "id": "index_for_semantic_search_test",
        },
    )
    assert index_response.status_code == 200
    index_result = index_response.json()["result"]
    assert "message" in index_result
    assert "indexed successfully" in index_result["message"]

    # Step 2: Perform semantic search (using a generic query)
    # Note: In a real test, we'd mock sentence-transformers and faiss for predictability.
    # Here, we test the endpoint mechanics and assume the underlying logic is tested in unit tests.
    search_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "semantic_search_codebase",
                "arguments": {
                    "codebase_path": str(temp_dir),
                    "query": "find text",
                    "top_k": 2,
                },
            },
            "id": "semantic_search_test_1",
        },
    )
    assert search_response.status_code == 200
    search_data = search_response.json()
    assert search_data["jsonrpc"] == "2.0"
    assert search_data["id"] == "semantic_search_test_1"
    
    result = search_data["result"]
    assert "message" in result
    assert "semantically similar code snippets" in result["message"]
    assert "results" in result
    # We can't assert specific results without mocking, but we can check it's a list
    assert isinstance(result["results"], list)


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
    # Note: In a real test, we'd mock sentence-transformers and faiss for predictability.
    # Here, we test the endpoint mechanics and assume the underlying logic is tested in unit tests.
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
    # The default config has specific fake keys, so we can check the masked values
    assert config["groq_api_key"] == "gsk_...2riH"
    assert config["openrouter_api_key"] == "sk-o...94a7"
    assert config["google_api_key"] == "AIza...dqmA"


# --- New Integration Tests for Analyze Codebase Improvements MCP Endpoint ---

def test_mcp_tool_call_analyze_codebase_improvements_success(temp_dir):
    """Test calling the analyze_codebase_improvements tool via the MCP endpoint successfully."""
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


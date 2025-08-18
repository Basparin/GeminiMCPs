import pytest
from fastapi.testclient import TestClient
import shutil
from pathlib import Path
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

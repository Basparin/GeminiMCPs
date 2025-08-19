import pytest
from fastapi.testclient import TestClient
import shutil
from pathlib import Path

from codesage_mcp.main import app

client = TestClient(app)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


def test_mcp_tool_call_suggest_code_improvements_success(temp_dir):
    """Test calling the suggest_code_improvements tool via the MCP endpoint successfully."""
    # Create a simple test Python file in the temp directory
    test_file_path = temp_dir / "test_module.py"
    with open(test_file_path, "w") as f:
        f.write("""
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
""")
    
    # Test analyzing the entire file
    analysis_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "suggest_code_improvements",
                "arguments": {
                    "file_path": str(test_file_path),
                }
            },
            "id": "suggest_code_improvements_test_1",
        },
    )
    
    # Verify the response
    assert analysis_response.status_code == 200
    analysis_data = analysis_response.json()
    assert "id" in analysis_data
    assert analysis_data["id"] == "suggest_code_improvements_test_1"
    assert "result" in analysis_data
    result = analysis_data["result"]
    
    # Verify the result structure
    assert "message" in result
    assert "file_path" in result
    assert "suggestions" in result
    assert result["file_path"] == str(test_file_path)
    
    # Check that we got suggestions (could be from LLMs or static analysis)
    assert isinstance(result["suggestions"], list)


def test_mcp_tool_call_suggest_code_improvements_with_line_range(temp_dir):
    """Test calling the suggest_code_improvements tool with a line range via the MCP endpoint."""
    # Create a simple test Python file in the temp directory
    test_file_path = temp_dir / "test_module.py"
    with open(test_file_path, "w") as f:
        f.write("""# Line 1
# Line 2
def inefficient_function():
    result = 0
    for i in range(1000000):
        result += i
    return result

def efficient_function():
    return sum(range(1000000))

# Line 9
# Line 10
""")
    
    # Test analyzing a specific line range
    analysis_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "suggest_code_improvements",
                "arguments": {
                    "file_path": str(test_file_path),
                    "start_line": 3,
                    "end_line": 8
                }
            },
            "id": "suggest_code_improvements_test_2",
        },
    )
    
    # Verify the response
    assert analysis_response.status_code == 200
    analysis_data = analysis_response.json()
    assert "id" in analysis_data
    assert analysis_data["id"] == "suggest_code_improvements_test_2"
    assert "result" in analysis_data
    result = analysis_data["result"]
    
    # Verify the result structure
    assert "message" in result
    assert "file_path" in result
    assert "start_line" in result
    assert "end_line" in result
    assert "suggestions" in result
    assert result["file_path"] == str(test_file_path)
    assert result["start_line"] == 3
    assert result["end_line"] == 8
    
    # Check that we got suggestions
    assert isinstance(result["suggestions"], list)


def test_mcp_tool_call_suggest_code_improvements_error():
    """Test calling the suggest_code_improvements tool via the MCP endpoint with an error."""
    # Test with a non-existent file
    analysis_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "suggest_code_improvements",
                "arguments": {
                    "file_path": "/test/nonexistent/file.py",
                }
            },
            "id": "suggest_code_improvements_test_3",
        },
    )
    
    # Verify the response
    assert analysis_response.status_code == 200
    analysis_data = analysis_response.json()
    assert "id" in analysis_data
    assert analysis_data["id"] == "suggest_code_improvements_test_3"
    assert "result" in analysis_data
    result = analysis_data["result"]
    
    # Check that we get an error
    assert "error" in result
    assert result["error"]["code"] == "FILE_NOT_FOUND"
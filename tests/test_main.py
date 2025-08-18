import pytest
from fastapi.testclient import TestClient
import shutil
from codesage_mcp.main import app

client = TestClient(app)


@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "test_dir"
    d.mkdir()
    (d / "file1.txt").write_text("hello")
    yield d
    shutil.rmtree(d)


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
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_file_structure",
                "arguments": {"path": str(temp_dir)},
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

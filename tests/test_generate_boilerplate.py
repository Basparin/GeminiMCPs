import tempfile
import os
from codesage_mcp.tools import generate_boilerplate_tool


def test_generate_boilerplate_file_header():
    """Test generating a file header boilerplate."""
    result = generate_boilerplate_tool(boilerplate_type="file_header")

    # Check the structure of the result
    assert "boilerplate" in result
    assert "Brief description of the module" in result["boilerplate"]
    assert "import logging" in result["boilerplate"]


def test_generate_boilerplate_module():
    """Test generating a module boilerplate."""
    result = generate_boilerplate_tool(
        boilerplate_type="module", module_name="test_module"
    )

    # Check the structure of the result
    assert "boilerplate" in result
    assert "Test Module Module for CodeSage MCP Server" in result["boilerplate"]
    assert "import logging" in result["boilerplate"]


def test_generate_boilerplate_tool():
    """Test generating a tool function boilerplate."""
    result = generate_boilerplate_tool(
        boilerplate_type="tool", function_name="test_tool"
    )

    # Check the structure of the result
    assert "boilerplate" in result
    assert "def test_tool(param1: str = None) -> dict:" in result["boilerplate"]
    assert "Process the input parameters" in result["boilerplate"]


def test_generate_boilerplate_test():
    """Test generating a test file boilerplate."""
    result = generate_boilerplate_tool(
        boilerplate_type="test", module_name="test_module"
    )

    # Check the structure of the result
    assert "boilerplate" in result
    assert "import pytest" in result["boilerplate"]
    assert "def test_test_module():" in result["boilerplate"]
    assert "test_module_tool" in result["boilerplate"]


def test_generate_boilerplate_class():
    """Test generating a class boilerplate."""
    result = generate_boilerplate_tool(boilerplate_type="class", class_name="TestClass")

    # Check the structure of the result
    assert "boilerplate" in result
    assert "class TestClass:" in result["boilerplate"]
    assert "def __init__(self, param1: str = None):" in result["boilerplate"]


def test_generate_boilerplate_function():
    """Test generating a function boilerplate."""
    result = generate_boilerplate_tool(
        boilerplate_type="function", function_name="test_function"
    )

    # Check the structure of the result
    assert "boilerplate" in result
    assert "def test_function(param1: str = None) -> str:" in result["boilerplate"]
    assert "Do something with param1" in result["boilerplate"]


def test_generate_boilerplate_invalid_type():
    """Test generating boilerplate with an invalid type."""
    result = generate_boilerplate_tool(boilerplate_type="invalid_type")

    # Check that an error is returned
    assert "error" in result
    assert result["error"]["code"] == "INVALID_INPUT"
    assert "Unsupported boilerplate_type" in result["error"]["message"]


def test_generate_boilerplate_save_to_file():
    """Test saving boilerplate to a file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_file_path = f.name

    try:
        result = generate_boilerplate_tool(
            boilerplate_type="file_header", file_path=temp_file_path
        )

        # Check that the file was created and the result indicates success
        assert "message" in result
        assert f"Boilerplate saved to {temp_file_path}" in result["message"]

        # Check the content of the file
        with open(temp_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Brief description of the module" in content
            assert "import logging" in content

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

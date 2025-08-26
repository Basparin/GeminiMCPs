import pytest
from unittest.mock import patch
from codesage_mcp.tools import resolve_todo_fixme_tool


@pytest.fixture
def mock_llm_analysis_manager():
    with patch(
        "codesage_mcp.codebase_manager.codebase_manager.llm_analysis_manager",
        autospec=True,
    ) as mock_manager:
        yield mock_manager


@pytest.fixture
def temp_file_with_todos(tmp_path):
    file_content = """
# This is a regular comment
def my_function():
    # TODO: Implement this function
    pass

# Another comment
class MyClass:
    def __init__(self):
        # FIXME: Initialize correctly
        pass

"""
    file_path = tmp_path / "test_file.py"
    file_path.write_text(file_content)
    return file_path


@pytest.fixture
def temp_file_without_todos(tmp_path):
    file_content = """
def my_function():
    pass
"""
    file_path = tmp_path / "no_todo_file.py"
    file_path.write_text(file_content)
    return file_path


def test_resolve_todo_fixme_success(mock_llm_analysis_manager, temp_file_with_todos):
    # Mock the resolve_todo_fixme method to return a successful response
    mock_llm_analysis_manager.resolve_todo_fixme.return_value = {
        "message": "TODO/FIXME resolution suggested successfully.",
        "resolution": {
            "suggested_code": 'print("Hello, World!")',
            "explanation": "Replaced pass with a print statement.",
            "line_start": 3,
            "line_end": 4,
        },
        "original_comment": {
            "line_number": 3,
            "comment": "# TODO: Implement this function",
        },
    }

    result = resolve_todo_fixme_tool(str(temp_file_with_todos), line_number=3)

    assert "message" in result
    assert result["message"] == "TODO/FIXME resolution suggested successfully."
    assert "resolution" in result
    assert result["resolution"]["suggested_code"] == 'print("Hello, World!")'
    mock_llm_analysis_manager.resolve_todo_fixme.assert_called_once_with(
        str(temp_file_with_todos), 3
    )


def test_resolve_todo_fixme_file_not_found():
    result = resolve_todo_fixme_tool("/path/to/nonexistent_file.py")
    assert "error" in result
    assert result["error"]["code"] == "FILE_NOT_FOUND"


def test_resolve_todo_fixme_no_todo_found(
    mock_llm_analysis_manager, temp_file_without_todos
):
    mock_llm_analysis_manager.resolve_todo_fixme.side_effect = ValueError(
        "No TODO/FIXME comments found in no_todo_file.py"
    )
    result = resolve_todo_fixme_tool(str(temp_file_without_todos))
    assert "error" in result
    assert result["error"]["code"] == "INVALID_INPUT"
    assert "No TODO/FIXME comments found" in result["error"]["message"]


def test_resolve_todo_fixme_specific_line_not_found(
    mock_llm_analysis_manager, temp_file_with_todos
):
    mock_llm_analysis_manager.resolve_todo_fixme.side_effect = ValueError(
        "No TODO/FIXME comment found at line 100 in test_file.py"
    )
    result = resolve_todo_fixme_tool(str(temp_file_with_todos), line_number=100)
    assert "error" in result
    assert result["error"]["code"] == "INVALID_INPUT"
    assert "No TODO/FIXME comment found at line 100" in result["error"]["message"]


def test_resolve_todo_fixme_llm_error(mock_llm_analysis_manager, temp_file_with_todos):
    mock_llm_analysis_manager.resolve_todo_fixme.side_effect = Exception(
        "LLM API error"
    )
    result = resolve_todo_fixme_tool(str(temp_file_with_todos), line_number=3)
    assert "error" in result
    assert result["error"]["code"] == "TODO_FIXME_RESOLUTION_ERROR"
    assert "LLM API error" in result["error"]["message"]

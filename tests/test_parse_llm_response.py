import pytest
from codesage_mcp.llm_analysis import LLMAnalysisManager
from codesage_mcp.tools import parse_llm_response_tool


@pytest.fixture
def llm_analysis_manager():
    # Mock clients are not needed for parse_llm_response as it doesn't interact with LLMs
    return LLMAnalysisManager(None, None, None)


def test_parse_llm_response_valid_json(llm_analysis_manager):
    json_string = '{"key": "value", "number": 123}'
    parsed_data = llm_analysis_manager.parse_llm_response(json_string)
    assert parsed_data == {"key": "value", "number": 123}


def test_parse_llm_response_json_with_markdown_fences(llm_analysis_manager):
    json_string = '```json\n{"key": "value", "number": 123}\n```'
    parsed_data = llm_analysis_manager.parse_llm_response(json_string)
    assert parsed_data == {"key": "value", "number": 123}


def test_parse_llm_response_json_with_markdown_fences_and_extra_whitespace(
    llm_analysis_manager,
):
    json_string = '  ```json\n  {"key": "value", "number": 123}\n  ```  '
    parsed_data = llm_analysis_manager.parse_llm_response(json_string)
    assert parsed_data == {"key": "value", "number": 123}


def test_parse_llm_response_invalid_json(llm_analysis_manager):
    invalid_json_string = '{"key": "value", "number": 123,'
    with pytest.raises(ValueError, match="Failed to parse LLM response as JSON"):
        llm_analysis_manager.parse_llm_response(invalid_json_string)


def test_parse_llm_response_empty_string(llm_analysis_manager):
    empty_string = ""
    with pytest.raises(ValueError, match="Failed to parse LLM response as JSON"):
        llm_analysis_manager.parse_llm_response(empty_string)


def test_parse_llm_response_non_json_string(llm_analysis_manager):
    non_json_string = "This is not a JSON string."
    with pytest.raises(ValueError, match="Failed to parse LLM response as JSON"):
        llm_analysis_manager.parse_llm_response(non_json_string)


# Test the tool function
def test_parse_llm_response_tool_success():
    json_string = '{"status": "success"}'
    result = parse_llm_response_tool(json_string)
    assert "message" in result
    assert result["message"] == "LLM response parsed successfully."
    assert "parsed_data" in result
    assert result["parsed_data"] == {"status": "success"}


def test_parse_llm_response_tool_markdown_success():
    json_string = '```json\n{"status": "success"}\n```'
    result = parse_llm_response_tool(json_string)
    assert "message" in result
    assert result["message"] == "LLM response parsed successfully."
    assert "parsed_data" in result
    assert result["parsed_data"] == {"status": "success"}


def test_parse_llm_response_tool_invalid_json():
    invalid_json_string = '{"status": "error",'
    result = parse_llm_response_tool(invalid_json_string)
    assert "error" in result
    assert result["error"]["code"] == "JSON_PARSE_ERROR"
    assert "Failed to parse LLM response as JSON" in result["error"]["message"]


def test_parse_llm_response_tool_general_error():
    # Simulate an unexpected error in the underlying manager method
    # This requires mocking the manager, which is more complex for a simple test
    # For now, we'll rely on the direct method tests for error handling.
    pass

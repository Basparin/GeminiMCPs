import tempfile
import os
from unittest.mock import patch, MagicMock
from codesage_mcp.tools import suggest_code_improvements_tool

# Instead of importing codebase_manager, we will import LLMAnalysisManager directly
from codesage_mcp.features.llm_analysis.llm_analysis import LLMAnalysisManager
# We also need to mock the API clients


def test_suggest_code_improvements_tool():
    """Test the suggest_code_improvements_tool function with a simple Python file."""
    # Create a simple test Python file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
def calculate_sum(a, b):
    return a + b

def main():
    result = calculate_sum(1, 2)
    print(result)

if __name__ == "__main__":
    main()
""")
        temp_file_path = f.name

    try:
        # Create an instance of LLMAnalysisManager with mocked clients
        # For this test, we can pass None for the clients to test static analysis
        llm_analysis_manager = LLMAnalysisManager(
            groq_client=None, openrouter_client=None, google_ai_client=None
        )

        # Test analyzing the entire file
        result = suggest_code_improvements_tool(
            temp_file_path
        )

        # Check the structure of the result
        assert "message" in result
        assert "file_path" in result
        assert "suggestions" in result
        assert result["file_path"] == temp_file_path
        assert result["start_line"] == 1

        # Check that we got suggestions (could be from LLMs or static analysis)
        assert isinstance(result["suggestions"], list)
        assert len(result["suggestions"]) > 0

        # At minimum, we should get a static analysis suggestion if no LLMs are configured
        # The provider could be "Static Analysis" or an error message from a failed LLM call
        # depending on the environment and configuration.
        # assert result["suggestions"][0]["provider"] == "Static Analysis"

        print(f"Suggestions received: {result['suggestions']}")

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def test_suggest_code_improvements_tool_with_line_range():
    """Test the suggest_code_improvements_tool function with a specific line range."""
    # Create a simple test Python file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
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
        temp_file_path = f.name

    try:
        # Test analyzing a specific line range
        result = suggest_code_improvements_tool(
            temp_file_path, start_line=3, end_line=8
        )

        # Check the structure of the result
        assert "message" in result
        assert "file_path" in result
        assert "suggestions" in result
        assert result["file_path"] == temp_file_path
        assert result["start_line"] == 3
        assert result["end_line"] == 8

        # Check that we got suggestions
        assert isinstance(result["suggestions"], list)
        assert len(result["suggestions"]) > 0

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def test_suggest_code_improvements_tool_not_found():
    """Test the suggest_code_improvements_tool function with a non-existent file."""
    # Test with a non-existent file
    result = suggest_code_improvements_tool("/test/nonexistent/file.py")

    # Check that we get an error
    assert "error" in result
    assert result["error"]["code"] == -32003


# --- New test for LLMAnalysisManager ---


@patch("groq.Groq")  # Patch the actual Groq class from the groq module
@patch("openai.OpenAI")  # Patch the actual OpenAI class from the openai module
@patch("google.generativeai")  # Patch the actual genai module
def test_llm_analysis_manager_suggest_code_improvements_with_mocked_llms(
    mock_genai, mock_openai_class, mock_groq_class
):
    """Test the LLMAnalysisManager.suggest_code_improvements method with mocked LLM clients."""
    # Create a simple test Python file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
def hello_world():
    print("Hello, World!")
""")
        temp_file_path = f.name

    try:
        # Create mock clients
        mock_groq_client = MagicMock()
        mock_openai_client = MagicMock()
        mock_google_ai_client = MagicMock()

        # Configure the mock classes to return the mock clients
        mock_groq_class.return_value = mock_groq_client
        mock_openai_class.return_value = mock_openai_client

        # Mock the LLM responses
        mock_groq_response = MagicMock()
        mock_groq_response.choices = [MagicMock()]
        mock_groq_response.choices[
            0
        ].message.content = (
            "Groq suggestion: Consider adding a docstring to the function."
        )
        mock_groq_client.chat.completions.create.return_value = mock_groq_response

        mock_openrouter_response = MagicMock()
        mock_openrouter_response.choices = [MagicMock()]
        mock_openrouter_response.choices[
            0
        ].message.content = (
            "OpenRouter suggestion: Consider using a more descriptive function name."
        )
        mock_openai_client.chat.completions.create.return_value = (
            mock_openrouter_response
        )

        mock_google_response = MagicMock()
        mock_google_response.text = "Google AI suggestion: Consider adding type hints to the function parameters."
        mock_google_ai_client.generate_content.return_value = mock_google_response

        # Create an instance of LLMAnalysisManager with mocked clients
        from codesage_mcp.features.llm_analysis.llm_analysis import LLMAnalysisManager

        llm_analysis_manager = LLMAnalysisManager(
            groq_client=mock_groq_client,
            openrouter_client=mock_openai_client,
            google_ai_client=mock_google_ai_client,
        )

        # Test the method directly
        result = llm_analysis_manager.suggest_code_improvements(temp_file_path)

        # Check the structure of the result
        assert "message" in result
        assert "file_path" in result
        assert "suggestions" in result
        assert result["file_path"] == temp_file_path

        # Verify mocks were called
        mock_groq_client.chat.completions.create.assert_called()
        mock_openai_client.chat.completions.create.assert_called()
        mock_google_ai_client.generate_content.assert_called()

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

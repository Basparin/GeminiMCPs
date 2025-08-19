import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from codesage_mcp.tools import suggest_code_improvements_tool
from codesage_mcp.codebase_manager import codebase_manager


def test_suggest_code_improvements_tool():
    """Test the suggest_code_improvements_tool function with a simple Python file."""
    # Create a simple test Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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
        # Test analyzing the entire file
        result = suggest_code_improvements_tool(temp_file_path)
        
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
        suggestion = result["suggestions"][0]
        assert "provider" in suggestion
        
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def test_suggest_code_improvements_tool_with_line_range():
    """Test the suggest_code_improvements_tool function with a specific line range."""
    # Create a simple test Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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
        result = suggest_code_improvements_tool(temp_file_path, start_line=3, end_line=8)
        
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
    assert result["error"]["code"] == "FILE_NOT_FOUND"


@patch.object(codebase_manager, 'groq_client', MagicMock())
@patch.object(codebase_manager, 'openrouter_client', MagicMock())
@patch.object(codebase_manager, 'google_ai_client', MagicMock())
def test_suggest_code_improvements_with_mocked_llms():
    """Test the suggest_code_improvements method with mocked LLM clients."""
    # Create a simple test Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
def hello_world():
    print("Hello, World!")
""")
        temp_file_path = f.name
    
    try:
        # Mock the LLM responses
        mock_groq_response = MagicMock()
        mock_groq_response.choices = [MagicMock()]
        mock_groq_response.choices[0].message.content = "Groq suggestion: Consider adding a docstring to the function."
        codebase_manager.groq_client.chat.completions.create.return_value = mock_groq_response
        
        mock_openrouter_response = MagicMock()
        mock_openrouter_response.choices = [MagicMock()]
        mock_openrouter_response.choices[0].message.content = "OpenRouter suggestion: Consider using a more descriptive function name."
        codebase_manager.openrouter_client.chat.completions.create.return_value = mock_openrouter_response
        
        mock_google_response = MagicMock()
        mock_google_response.text = "Google AI suggestion: Consider adding type hints to the function parameters."
        codebase_manager.google_ai_client.GenerativeModel().generate_content.return_value = mock_google_response
        
        # Test the method directly
        result = codebase_manager.suggest_code_improvements(temp_file_path)
        
        # Check the structure of the result
        assert "message" in result
        assert "file_path" in result
        assert "suggestions" in result
        assert result["file_path"] == temp_file_path
        
        # Check that we got suggestions from all three providers
        assert len(result["suggestions"]) == 3
        
        # Check each provider's suggestion
        providers = [suggestion["provider"] for suggestion in result["suggestions"]]
        assert "Groq (Llama3)" in providers
        assert "OpenRouter (Gemini)" in providers
        assert "Google AI (Gemini)" in providers
        
        # Check that each suggestion has content
        for suggestion in result["suggestions"]:
            assert "suggestions" in suggestion or "error" in suggestion
            
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)
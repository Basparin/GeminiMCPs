import pytest
from codesage_mcp.llm_analysis import LLMAnalysisManager
from codesage_mcp.tools import generate_llm_api_wrapper_tool


@pytest.fixture
def llm_analysis_manager():
    """Test Llm analysis manager.

    Creates a mocked LLMAnalysisManager instance for testing purposes.
    This fixture provides a clean instance of the LLMAnalysisManager
    without requiring actual API clients, which allows testing of the
    generate_llm_api_wrapper functionality in isolation.

    Returns:
        LLMAnalysisManager: Mocked LLMAnalysisManager instance.
    """
    # Mock clients are not needed for generate_llm_api_wrapper as it generates code
    return LLMAnalysisManager(None, None, None)


def test_generate_llm_api_wrapper_groq(llm_analysis_manager):
    """Test Generate llm api wrapper groq.

    This test verifies that the LLM API wrapper generation correctly creates
    a Python class for the Groq provider. It checks that the generated code
    includes the necessary imports, class definition, and API key handling
    specific to the Groq service.

    Args:
        llm_analysis_manager: Mocked LLMAnalysisManager instance fixture.
    """
    generated_code = llm_analysis_manager.generate_llm_api_wrapper(
        llm_provider="groq",
        model_name="llama3-8b-8192",
        api_key_env_var="TEST_GROQ_API_KEY",
    )
    assert "from groq import Groq" in generated_code
    assert "class GroqLLMClient:" in generated_code
    assert 'os.getenv("TEST_GROQ_API_KEY")' in generated_code
    assert "self.client = Groq(api_key=self.api_key)" in generated_code
    assert "chat_completion = self.client.chat.completions.create(" in generated_code


def test_generate_llm_api_wrapper_openrouter(llm_analysis_manager):
    """Test Generate llm api wrapper openrouter.

    This test verifies that the LLM API wrapper generation correctly creates
    a Python class for the OpenRouter provider. It checks that the generated code
    includes the necessary imports, class definition, and API key handling
    specific to the OpenRouter service.

    Args:
        llm_analysis_manager: Mocked LLMAnalysisManager instance fixture.
    """
    generated_code = llm_analysis_manager.generate_llm_api_wrapper(
        llm_provider="openrouter",
        model_name="google/gemini-pro",
        api_key_env_var="TEST_OPENROUTER_API_KEY",
    )
    assert "from openai import OpenAI" in generated_code
    assert "class OpenRouterLLMClient:" in generated_code
    assert 'os.getenv("TEST_OPENROUTER_API_KEY")' in generated_code
    assert 'base_url="https://openrouter.ai/api/v1"' in generated_code
    assert "chat_completion = self.client.chat.completions.create(" in generated_code


def test_generate_llm_api_wrapper_google(llm_analysis_manager):
    """Test Generate llm api wrapper google.

    This test verifies that the LLM API wrapper generation correctly creates
    a Python class for the Google AI provider. It checks that the generated code
    includes the necessary imports, class definition, and API key handling
    specific to the Google AI service.

    Args:
        llm_analysis_manager: Mocked LLMAnalysisManager instance fixture.
    """
    generated_code = llm_analysis_manager.generate_llm_api_wrapper(
        llm_provider="google",
        model_name="gemini-pro",
        api_key_env_var="TEST_GOOGLE_API_KEY",
    )
    assert "import google.generativeai as genai" in generated_code
    assert "class GoogleLLMClient:" in generated_code
    assert 'os.getenv("TEST_GOOGLE_API_KEY")' in generated_code
    assert "genai.configure(api_key=self.api_key)" in generated_code
    assert "self.client = genai.GenerativeModel(self.model_name)" in generated_code
    assert "response = self.client.generate_content(prompt, **kwargs)" in generated_code


def test_generate_llm_api_wrapper_unsupported_provider(llm_analysis_manager):
    """Test Generate llm api wrapper unsupported provider.

    This test verifies that the LLM API wrapper generation correctly handles
    unsupported LLM providers by raising an appropriate ValueError. It ensures
    that the function properly validates provider names and rejects invalid ones.

    Args:
        llm_analysis_manager: Mocked LLMAnalysisManager instance fixture.
    """
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        llm_analysis_manager.generate_llm_api_wrapper(
            llm_provider="unsupported", model_name="any-model"
        )


def test_generate_llm_api_wrapper_tool_return_string(llm_analysis_manager):
    """Test Generate llm api wrapper tool return string.

    This test verifies that the LLM API wrapper generation tool correctly
    returns the generated code as a string when no output file path is specified.
    It ensures that the function properly generates and returns the wrapper code
    without saving it to a file.

    Args:
        llm_analysis_manager: Mocked LLMAnalysisManager instance fixture.
    """
    result = generate_llm_api_wrapper_tool(
        llm_provider="groq",
        model_name="llama3-8b-8192",
        api_key_env_var="TEST_GROQ_API_KEY",
    )
    assert "generated_code" in result
    assert "from groq import Groq" in result["generated_code"]


def test_generate_llm_api_wrapper_tool_save_to_file(llm_analysis_manager, tmp_path):
    """Test Generate llm api wrapper tool save to file.

    This test verifies that the LLM API wrapper generation tool correctly
    saves the generated code to a specified file when an output file path is provided.
    It ensures that the function properly generates the wrapper code and writes it
    to the specified file location.

    Args:
        llm_analysis_manager: Mocked LLMAnalysisManager instance fixture.
        tmp_path: Pytest fixture that provides a temporary directory.
    """
    output_file = tmp_path / "test_wrapper.py"
    result = generate_llm_api_wrapper_tool(
        llm_provider="google",
        model_name="gemini-pro",
        output_file_path=str(output_file),
        api_key_env_var="TEST_GOOGLE_API_KEY",
    )
    assert "message" in result
    assert f"LLM API wrapper saved to {output_file}" in result["message"]
    assert output_file.exists()
    content = output_file.read_text()
    assert "import google.generativeai as genai" in content


def test_generate_llm_api_wrapper_tool_invalid_provider(llm_analysis_manager):
    """Test Generate llm api wrapper tool invalid provider.

    This test verifies error handling in generate llm api wrapper tool invalid provider.
    It checks that when an invalid or unsupported LLM provider is specified,
    the tool correctly raises a ValueError with an appropriate error message.

    Args:
        llm_analysis_manager: Mocked LLMAnalysisManager instance fixture.
    """
    result = generate_llm_api_wrapper_tool(
        llm_provider="invalid", model_name="any-model"
    )
    assert "error" in result
    assert result["error"]["code"] == "INVALID_INPUT"
    assert "Unsupported LLM provider" in result["error"]["message"]

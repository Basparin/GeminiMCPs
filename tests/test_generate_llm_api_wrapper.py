import pytest
from codesage_mcp.llm_analysis import LLMAnalysisManager
from codesage_mcp.tools import generate_llm_api_wrapper_tool


@pytest.fixture
def llm_analysis_manager():
    # Mock clients are not needed for generate_llm_api_wrapper as it generates code
    return LLMAnalysisManager(None, None, None)


def test_generate_llm_api_wrapper_groq(llm_analysis_manager):
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
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        llm_analysis_manager.generate_llm_api_wrapper(
            llm_provider="unsupported", model_name="any-model"
        )


def test_generate_llm_api_wrapper_tool_return_string():
    result = generate_llm_api_wrapper_tool(
        llm_provider="groq",
        model_name="llama3-8b-8192",
        api_key_env_var="TEST_GROQ_API_KEY",
    )
    assert "generated_code" in result
    assert "from groq import Groq" in result["generated_code"]


def test_generate_llm_api_wrapper_tool_save_to_file(tmp_path):
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


def test_generate_llm_api_wrapper_tool_invalid_provider():
    result = generate_llm_api_wrapper_tool(
        llm_provider="invalid", model_name="any-model"
    )
    assert "error" in result
    assert result["error"]["code"] == "INVALID_INPUT"
    assert "Unsupported LLM provider" in result["error"]["message"]

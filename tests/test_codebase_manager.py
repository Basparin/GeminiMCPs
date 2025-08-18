import pytest
import json
from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch
from codesage_mcp.codebase_manager import CodebaseManager


@pytest.fixture
def temp_codebase(tmp_path):
    """Create a temporary codebase for testing."""
    codebase_dir = tmp_path / "test_project"
    codebase_dir.mkdir()

    (codebase_dir / "file1.py").write_text(
        'import os\n\n# A test comment\nprint("hello world")'
    )
    (codebase_dir / "file2.txt").write_text("this file should be ignored")

    subdir = codebase_dir / "subdir"
    subdir.mkdir()
    (subdir / "file3.js").write_text("console.log('this should also be ignored');")

    (codebase_dir / ".gitignore").write_text("*.txt\nsubdir/\n.codesage/")

    yield codebase_dir

    shutil.rmtree(codebase_dir)


def test_initialization_and_persistence(temp_codebase):
    """Test that the CodebaseManager initializes, creates index, and persists it."""
    manager = CodebaseManager()
    manager.index_dir = temp_codebase / ".codesage"
    manager.index_file = manager.index_dir / "codebase_index.json"
    manager.indexed_codebases = {}
    manager._initialize_index()

    manager.index_codebase(str(temp_codebase))

    assert manager.index_file.exists()

    with open(manager.index_file, "r") as f:
        index_data = json.load(f)

    abs_path_key = str(temp_codebase.resolve())
    # The index_data is a dict with 'indexed_codebases' and 'file_paths_map' keys
    assert abs_path_key in index_data["indexed_codebases"]
    assert "file1.py" in index_data["indexed_codebases"][abs_path_key]["files"]

    new_manager = CodebaseManager()
    new_manager.index_dir = temp_codebase / ".codesage"
    new_manager.index_file = new_manager.index_dir / "codebase_index.json"
    new_manager.indexed_codebases = {}
    new_manager._initialize_index()

    assert new_manager.indexed_codebases == index_data["indexed_codebases"]


def test_indexing_and_gitignore(temp_codebase):
    """Test that the indexing process correctly handles .gitignore files."""
    manager = CodebaseManager()
    manager.index_dir = temp_codebase / ".codesage"
    manager.index_file = manager.index_dir / "codebase_index.json"
    manager.indexed_codebases = {}
    manager._initialize_index()

    indexed_files = manager.index_codebase(str(temp_codebase))

    assert "file1.py" in indexed_files
    assert "file2.txt" not in indexed_files
    assert str(Path("subdir/file3.js")) not in indexed_files
    # .gitignore files are typically ignored by their own rules, so it should not be indexed.
    assert ".gitignore" not in indexed_files


def test_search_codebase(temp_codebase):
    """Test the search functionality."""
    manager = CodebaseManager()
    manager.index_dir = temp_codebase / ".codesage"
    manager.index_file = manager.index_dir / "codebase_index.json"
    manager.indexed_codebases = {}
    manager._initialize_index()

    manager.index_codebase(str(temp_codebase))

    search_results = manager.search_codebase(str(temp_codebase), "hello world")

    assert len(search_results) == 1
    assert search_results[0]["file_path"] == str(temp_codebase / "file1.py")
    assert search_results[0]["line_number"] == 4
    assert search_results[0]["line_content"] == 'print("hello world")'


def test_read_code_file(temp_codebase):
    """Test reading a file."""
    manager = CodebaseManager()
    file_path = temp_codebase / "file1.py"
    content = manager.read_code_file(str(file_path))

    assert "hello world" in content


@patch("codesage_mcp.codebase_manager.Groq")
def test_summarize_code_section_with_groq(mock_groq, temp_codebase):
    """Test the summarization feature with a mocked Groq API call."""
    mock_groq_client = MagicMock()
    mock_groq.return_value = mock_groq_client
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "This is a mock summary."
    mock_groq_client.chat.completions.create.return_value = mock_completion

    manager = CodebaseManager()
    manager.groq_client = mock_groq_client

    summary = manager.summarize_code_section(
        file_path=str(temp_codebase / "file1.py"),
        start_line=1,
        end_line=4,
        llm_model="llama3-8b-8192",
    )

    assert summary == "This is a mock summary."
    mock_groq_client.chat.completions.create.assert_called_once()
    call_args = mock_groq_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "llama3-8b-8192"


@patch("codesage_mcp.codebase_manager.OpenAI")
def test_summarize_code_section_with_openrouter(mock_openai, temp_codebase):
    """Test the summarization feature with a mocked OpenRouter API call."""
    mock_openai_client = MagicMock()
    mock_openai.return_value = mock_openai_client
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "This is a mock OpenRouter summary."
    mock_openai_client.chat.completions.create.return_value = mock_completion

    manager = CodebaseManager()
    manager.openrouter_client = mock_openai_client

    summary = manager.summarize_code_section(
        file_path=str(temp_codebase / "file1.py"),
        start_line=1,
        end_line=4,
        llm_model="openrouter/google/gemini-flash-1.5",
    )

    assert summary == "This is a mock OpenRouter summary."
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "google/gemini-flash-1.5"


@patch("codesage_mcp.codebase_manager.genai")
def test_summarize_code_section_with_google_ai(mock_genai, temp_codebase):
    """Test the summarization feature with a mocked Google AI API call."""
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    mock_response = MagicMock()
    mock_response.text = "This is a mock Google AI summary."
    mock_model.generate_content.return_value = mock_response

    manager = CodebaseManager()
    manager.google_ai_client = mock_genai

    summary = manager.summarize_code_section(
        file_path=str(temp_codebase / "file1.py"),
        start_line=1,
        end_line=4,
        llm_model="google/gemini-pro",
    )

    assert summary == "This is a mock Google AI summary."
    mock_genai.GenerativeModel.assert_called_once_with("gemini-pro")
    mock_model.generate_content.assert_called_once()
    call_args = mock_model.generate_content.call_args
    assert "Please summarize the following code snippet:" in call_args.args[0]
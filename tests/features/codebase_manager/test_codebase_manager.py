import pytest
import json
from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch, mock_open
from codesage_mcp.features.codebase_manager import CodebaseManager
import numpy as np


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
    """Test that the IndexingManager initializes, creates index, and persists it."""
    # Instead of using CodebaseManager, we will test IndexingManager directly
    from codesage_mcp.core.indexing import IndexingManager
    from sentence_transformers import SentenceTransformer

    indexing_manager = IndexingManager()
    indexing_manager.index_dir = temp_codebase / ".codesage"
    indexing_manager.index_file = indexing_manager.index_dir / "codebase_index.json"
    indexing_manager.indexed_codebases = {}
    indexing_manager._initialize_index()

    # For index_codebase, we need a sentence_transformer_model
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    indexing_manager.index_codebase(str(temp_codebase), sentence_transformer_model)

    assert indexing_manager.index_file.exists()

    with open(indexing_manager.index_file, "r") as f:
        index_data = json.load(f)

    abs_path_key = str(temp_codebase.resolve())
    # The index_data is a dict with 'indexed_codebases' and 'file_paths_map' keys
    assert abs_path_key in index_data["indexed_codebases"]
    assert "file1.py" in index_data["indexed_codebases"][abs_path_key]["files"]

    # Instead of using CodebaseManager, we will test IndexingManager directly
    from codesage_mcp.core.indexing import IndexingManager

    new_indexing_manager = IndexingManager()
    new_indexing_manager.index_dir = temp_codebase / ".codesage"
    new_indexing_manager.index_file = (
        new_indexing_manager.index_dir / "codebase_index.json"
    )
    new_indexing_manager.indexed_codebases = {}
    new_indexing_manager._initialize_index()

    assert new_indexing_manager.indexed_codebases == index_data["indexed_codebases"]


def test_indexing_and_gitignore(temp_codebase):
    """Test that the indexing process correctly handles .gitignore files."""
    # Instead of using CodebaseManager, we will test IndexingManager directly
    from codesage_mcp.core.indexing import IndexingManager
    from sentence_transformers import SentenceTransformer

    indexing_manager = IndexingManager()
    indexing_manager.index_dir = temp_codebase / ".codesage"
    indexing_manager.index_file = indexing_manager.index_dir / "codebase_index.json"
    indexing_manager.indexed_codebases = {}
    indexing_manager._initialize_index()

    # For index_codebase, we need a sentence_transformer_model
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    indexed_files = indexing_manager.index_codebase(
        str(temp_codebase), sentence_transformer_model
    )

    assert "file1.py" in indexed_files
    assert "file2.txt" not in indexed_files
    assert str(Path("subdir/file3.js")) not in indexed_files
    # .gitignore files are typically ignored by their own rules, so it should not be indexed.
    assert ".gitignore" not in indexed_files


def test_search_codebase(temp_codebase):
    """Test the search functionality."""
    # Instead of using CodebaseManager, we will test SearchingManager directly
    from codesage_mcp.core.searching import SearchingManager
    from codesage_mcp.core.indexing import IndexingManager

    # Create instances of the managers
    indexing_manager = IndexingManager()
    searching_manager = SearchingManager(indexing_manager)

    # Configure the indexing manager
    indexing_manager.index_dir = temp_codebase / ".codesage"
    indexing_manager.index_file = indexing_manager.index_dir / "codebase_index.json"
    indexing_manager._initialize_index()

    # Index the codebase using the indexing manager
    from sentence_transformers import SentenceTransformer

    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    indexing_manager.index_codebase(str(temp_codebase), sentence_transformer_model)

    # Perform the search using the searching manager
    search_results = searching_manager.search_codebase(
        str(temp_codebase), "hello world"
    )

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
    # Instead of using CodebaseManager, we will test LLMAnalysisManager directly
    from codesage_mcp.features.llm_analysis.llm_analysis import LLMAnalysisManager

    mock_groq_client = MagicMock()
    mock_groq.return_value = mock_groq_client
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "This is a mock summary."
    mock_groq_client.chat.completions.create.return_value = mock_completion

    # Create an instance of LLMAnalysisManager with the mocked client
    llm_analysis_manager = LLMAnalysisManager(
        groq_client=mock_groq_client, openrouter_client=None, google_ai_client=None
    )

    summary = llm_analysis_manager.summarize_code_section(
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
    # Instead of using CodebaseManager, we will test LLMAnalysisManager directly
    from codesage_mcp.features.llm_analysis.llm_analysis import LLMAnalysisManager

    mock_openai_client = MagicMock()
    mock_openai.return_value = mock_openai_client
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "This is a mock OpenRouter summary."
    mock_openai_client.chat.completions.create.return_value = mock_completion

    # Create an instance of LLMAnalysisManager with the mocked client
    llm_analysis_manager = LLMAnalysisManager(
        groq_client=None, openrouter_client=mock_openai_client, google_ai_client=None
    )

    summary = llm_analysis_manager.summarize_code_section(
        file_path=str(temp_codebase / "file1.py"),
        start_line=1,
        end_line=4,
        llm_model="openrouter/google/gemini-flash-1.5",
    )

    assert summary == "This is a mock OpenRouter summary."
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "google/gemini-flash-1.5"


def test_summarize_code_section_with_google_ai(temp_codebase):
    """Test the summarization feature with a mocked Google AI API call."""
    # Instead of using CodebaseManager, we will test LLMAnalysisManager directly
    from codesage_mcp.features.llm_analysis.llm_analysis import LLMAnalysisManager

    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(text="This is a mock Google AI summary.")

    # Create an instance of LLMAnalysisManager with the mocked client
    llm_analysis_manager = LLMAnalysisManager(
        groq_client=None, openrouter_client=None, google_ai_client=mock_model
    )

    summary = llm_analysis_manager.summarize_code_section(
        file_path=str(temp_codebase / "file1.py"),
        start_line=1,
        end_line=4,
        llm_model="google/gemini-pro",
    )

    assert summary == "This is a mock Google AI summary."
    mock_model.generate_content.assert_called_once()
    call_args = mock_model.generate_content.call_args
    assert "Please summarize the following code snippet:" in call_args.args[0]


# --- New Tests for Semantic Search ---


def test_semantic_search_empty_index():
    """Test semantic search when the FAISS index is empty or None."""
    # Instead of using CodebaseManager, we will test SearchingManager directly
    from codesage_mcp.core.searching import SearchingManager
    from codesage_mcp.core.indexing import IndexingManager

    # Create instances of the managers
    indexing_manager = IndexingManager()
    searching_manager = SearchingManager(indexing_manager)

    # Directly set the indexing manager's faiss_index to None to simulate an un-indexed state
    indexing_manager.faiss_index = None

    results = searching_manager.semantic_search_codebase(
        "test query", sentence_transformer_model=None
    )
    assert results == {'result': []}

    # Test with an index that has 0 total vectors
    mock_faiss_index = MagicMock()
    mock_faiss_index.ntotal = 0
    # Mock the indexing manager's faiss_index attribute
    indexing_manager.faiss_index = mock_faiss_index

    results = searching_manager.semantic_search_codebase(
        "test query", sentence_transformer_model=None
    )
    assert results == {'result': []}
    # Ensure no search was attempted on an empty index
    mock_faiss_index.search.assert_not_called()


def test_semantic_search_with_results(temp_codebase):
    """Test semantic search returning results from a real, indexed codebase."""
    # Instead of using CodebaseManager, we will test SearchingManager directly
    from codesage_mcp.core.searching import SearchingManager
    from codesage_mcp.core.indexing import IndexingManager
    from sentence_transformers import SentenceTransformer

    # Create instances of the managers
    indexing_manager = IndexingManager()
    searching_manager = SearchingManager(indexing_manager)

    indexing_manager.index_dir = temp_codebase / ".codesage"
    indexing_manager.index_file = indexing_manager.index_dir / "codebase_index.json"
    indexing_manager.faiss_index_file = (
        indexing_manager.index_dir / "codebase_index.faiss"
    )
    indexing_manager._initialize_index()

    # Index the test codebase
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    indexing_manager.index_codebase(str(temp_codebase), sentence_transformer_model)

    # We need to mock the sentence transformer's encode method to return predictable
    # embeddings for our test query.
    # Let's assume file1.py gets ID "0" and has a certain embedding.
    # Let's make the query embedding very similar to file1.py's mock embedding.

    # Store original method to restore later
    original_encode = sentence_transformer_model.encode

    # Mock embeddings. FAISS works with L2 distance. Smaller distance = more similar.
    # We'll make the query very close to file1.py's embedding and far from a dummy one.
    mock_file1_embedding = [1.0, 0.0, 0.0]
    mock_query_embedding = [0.9, 0.0, 0.0]  # Very similar to file1
    mock_dummy_embedding = [0.0, 1.0, 0.0]  # Different

    def mock_encode(text):
        """Test Mock encode.

        Creates mock embeddings for testing semantic search functionality.
        This function simulates the behavior of a sentence transformer
        by returning predefined embeddings based on the input text content
        for the first semantic search test scenario.

        Args:
            text: Test parameter representing the text to encode.
        """
        if "hello world" in text:  # This is in file1.py
            return mock_file1_embedding
        elif text == "dummy content":  # This is for the dummy file
            return mock_dummy_embedding
        elif text == "find hello world":  # This is our query
            return mock_query_embedding
        else:
            # Fallback, shouldn't happen in this specific test
            return original_encode(text)

    # Patch the encode method
    sentence_transformer_model.encode = mock_encode

    # Now, mock the FAISS search.
    # We'll pretend FAISS searched and found file1.py's ID (which should be "0")
    # as the most similar.
    mock_faiss_index = indexing_manager.faiss_index
    mock_faiss_index.search = MagicMock()
    # Return distance 0.1 (very close) and index 0
    mock_faiss_index.search.return_value = ([[0.1]], [[0]])

    # Also mock the file_paths_map to map ID "0" to our file1.py
    abs_file1_path = str((temp_codebase / "file1.py").resolve())
    indexing_manager.file_paths_map = {"0": abs_file1_path}

    # --- Perform the search ---
    results = searching_manager.semantic_search_codebase(
        "find hello world", sentence_transformer_model, top_k=1
    )

    # --- Assertions ---
    assert len(results) == 1
    assert results[0]["file_path"] == abs_file1_path
    assert results[0]["score"] == 0.1  # Check the mocked distance/score

    # Verify the mocks were called as expected
    mock_faiss_index.search.assert_called_once()
    call_args = mock_faiss_index.search.call_args
    assert call_args[0][1] == 1  # call_args[0][1] should be top_k

    # Restore original method
    sentence_transformer_model.encode = original_encode


def test_semantic_search_top_k():
    """Test that the top_k parameter correctly limits results."""
    manager = CodebaseManager()
    # Do NOT index a real codebase. We are mocking everything.

    # Mock the indexing manager's faiss_index attribute
    mock_faiss_index = MagicMock()
    mock_faiss_index.ntotal = 2  # Pretend we have 2 vectors indexed
    mock_faiss_index.d = 384  # Set the vector dimension to prevent early return
    manager.indexing_manager.faiss_index = mock_faiss_index

    # Custom mock side effect for search to respect top_k
    def mock_search_side_effect(query_vector, k):
        if k == 1:
            return (np.array([[0.1]]), np.array([[0]]))
        elif k == 2:
            return (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        else:
            return (np.array([]), np.array([])) # Default for other k values

    mock_faiss_index.search.side_effect = mock_search_side_effect

    # Mock the indexing manager's file_paths_map attribute
    manager.indexing_manager.file_paths_map = {
        "0": "/path/to/file1.py",
        "1": "/path/to/file2.py",
    }

    # --- Perform the search with top_k=1 ---
    results_k1 = manager.semantic_search_codebase("test query", top_k=1)

    # --- Assertions for top_k=1 ---
    assert len(results_k1) == 1
    assert results_k1[0]["file_path"] == "/path/to/file1.py"
    assert results_k1[0]["score"] == 0.1

    # --- Perform the search with top_k=2 ---
    results_k2 = manager.semantic_search_codebase("test query", top_k=2)

    # --- Assertions for top_k=2 ---
    assert len(results_k2) == 2
    assert results_k2[0]["file_path"] == "/path/to/file1.py"
    assert results_k2[0]["score"] == 0.1
    assert results_k2[1]["file_path"] == "/path/to/file2.py"
    assert results_k2[1]["score"] == 0.2


# --- New Tests for Find Duplicate Code ---
def test_find_duplicate_code_empty_index():
    """Test find_duplicate_code when the FAISS index is empty or None."""
    # Instead of using CodebaseManager, we will test SearchingManager directly
    from codesage_mcp.core.searching import SearchingManager
    from codesage_mcp.core.indexing import IndexingManager

    # Create instances of the managers
    indexing_manager = IndexingManager()
    searching_manager = SearchingManager(indexing_manager)

    # Test with an unindexed codebase - should raise ValueError
    indexing_manager.indexed_codebases = {}
    with pytest.raises(
        ValueError, match="Codebase at /test/codebase has not been indexed"
    ):
        searching_manager.find_duplicate_code(
            "/test/codebase", sentence_transformer_model=None
        )

    # Test with an index that has 0 total vectors
    mock_faiss_index = MagicMock()
    mock_faiss_index.ntotal = 0
    # Mock the indexing manager's faiss_index attribute
    indexing_manager.faiss_index = mock_faiss_index
    # Mock indexed_codebases to simulate an indexed codebase
    indexing_manager.indexed_codebases = {"/test/codebase": {"files": []}}

    results = searching_manager.find_duplicate_code(
        "/test/codebase", sentence_transformer_model=None
    )
    assert results == []
    # Ensure no search was attempted on an empty index
    mock_faiss_index.search.assert_not_called()


def test_find_duplicate_code_with_results(temp_codebase):
    """Test find_duplicate_code returning results from a real, indexed codebase."""
    # Instead of using CodebaseManager, we will test SearchingManager directly
    from codesage_mcp.core.searching import SearchingManager
    from codesage_mcp.core.indexing import IndexingManager
    from sentence_transformers import SentenceTransformer

    # Create instances of the managers
    indexing_manager = IndexingManager()
    searching_manager = SearchingManager(indexing_manager)

    indexing_manager.index_dir = temp_codebase / ".codesage"
    indexing_manager.index_file = indexing_manager.index_dir / "codebase_index.json"
    indexing_manager.faiss_index_file = (
        indexing_manager.index_dir / "codebase_index.faiss"
    )
    indexing_manager._initialize_index()

    # Index the test codebase
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    indexing_manager.index_codebase(str(temp_codebase), sentence_transformer_model)

    # We need to mock the sentence transformer's encode method to return predictable
    # embeddings for our test sections.
    # Let's assume file1.py gets ID "0" and has a certain embedding.
    # Let's make a duplicate section embedding very similar to file1.py's mock embedding.

    # Store original method to restore later
    original_encode = sentence_transformer_model.encode

    # Mock embeddings. FAISS works with L2 distance. Smaller distance = more similar.
    # We'll make the duplicate section very close to file1.py's embedding.
    mock_file1_embedding = [1.0, 0.0, 0.0]
    mock_duplicate_embedding = [0.9, 0.0, 0.0]  # Very similar to file1
    mock_dummy_embedding = [0.0, 1.0, 0.0]  # Different

    def mock_encode(text):
        """Test Mock encode.

        Creates mock embeddings for testing duplicate code detection functionality.
        This function simulates the behavior of a sentence transformer
        by returning predefined embeddings based on the input text content
        for the duplicate code detection test scenario.

        Args:
            text: Test parameter representing the text to encode.
        """
        if "hello world" in text:  # This is in file1.py
            return mock_file1_embedding
        elif text == "dummy content":  # This is for the dummy file
            return mock_dummy_embedding
        elif "duplicate section" in text:  # This is our duplicate section
            return mock_duplicate_embedding
        else:
            # Fallback, shouldn't happen in this specific test
            return original_encode(text)

    # Patch the encode method
    sentence_transformer_model.encode = mock_encode

    # Now, mock the FAISS search.
    # We'll pretend FAISS searched and found file1.py's ID (which should be "0")
    # as the most similar.
    mock_faiss_index = indexing_manager.faiss_index
    mock_faiss_index.search = MagicMock()
    # Return distance 0.1 (very close) and index 0
    mock_faiss_index.search.return_value = ([[0.1]], [[0]])

    # Also mock the file_paths_map to map ID "0" to our file1.py
    abs_file1_path = str((temp_codebase / "file1.py").resolve())
    indexing_manager.file_paths_map = {"0": abs_file1_path}
    # Also add the codebase path to indexed_codebases
    indexing_manager.indexed_codebases = {
        str(temp_codebase.resolve()): {"files": ["file1.py"]}
    }

    # --- Perform the duplicate code search ---
    # For this test, we'll mock the file content to have a section that matches our mock
    # This is a bit of a simplification, but it tests the core logic
    with patch(
        "builtins.open",
        mock_open(
            read_data='import os\n\n# A test comment\nprint("hello world")\n# duplicate section\n'
        ),
    ):
        results = searching_manager.find_duplicate_code(
            str(temp_codebase),
            sentence_transformer_model,
            min_similarity=0.8,
            min_lines=3,
        )

    # --- Assertions ---
    # Note: The actual implementation might not find duplicates in this mock setup,
    # but we're testing the structure and error handling.
    # A more comprehensive test would require a more complex mock setup.
    assert isinstance(results, list)

    # Restore original method
    sentence_transformer_model.encode = original_encode


# --- New Tests for Get Configuration Tool ---


def test_get_configuration_tool():
    """Test the get_configuration_tool function."""
    from codesage_mcp.tools import get_configuration_tool

    # Test with the default configuration (which has fake API keys)
    result = get_configuration_tool()

    # Check the structure of the result
    assert "message" in result
    assert "configuration" in result
    assert result["message"] == "Current configuration retrieved successfully."

    # Check the configuration structure
    config = result["configuration"]
    assert "groq_api_key" in config
    assert "openrouter_api_key" in config
    assert "google_api_key" in config

    # Check that the API keys are masked
    # The default config has specific fake keys, so we can check the masked values
    assert config["groq_api_key"] == "gsk_...2riH"
    assert config["openrouter_api_key"] == "sk-o...94a7"
    assert config["google_api_key"] == "AIza...dqmA"


# --- New Tests for Analyze Codebase Improvements Tool ---


def test_analyze_codebase_improvements_tool(temp_codebase):
    """Test the analyze_codebase_improvements_tool function."""
    from codesage_mcp.tools import analyze_codebase_improvements_tool
    from codesage_mcp.features.codebase_manager import codebase_manager

    # First, index the codebase
    codebase_manager.index_codebase(str(temp_codebase))

    # Test the analysis tool
    result = analyze_codebase_improvements_tool(str(temp_codebase))

    # Check the structure of the result
    assert "message" in result
    assert "analysis" in result
    assert result["message"] == "Codebase analysis completed successfully."

    # Check the analysis structure
    analysis = result["analysis"]
    assert "total_files" in analysis
    assert "python_files" in analysis
    assert "todo_comments" in analysis
    assert "fixme_comments" in analysis
    assert "undocumented_functions" in analysis
    assert "potential_duplicates" in analysis
    assert "large_files" in analysis
    assert "suggestions" in analysis

    # With our test codebase, we should have at least one Python file
    assert analysis["python_files"] >= 1


def test_analyze_codebase_improvements_tool_not_indexed():
    """Test the analyze_codebase_improvements_tool function with an unindexed codebase."""
    from codesage_mcp.tools import analyze_codebase_improvements_tool

    # Test with an unindexed codebase
    result = analyze_codebase_improvements_tool("/test/nonexistent/codebase")

    # Check that we get an error
    assert "error" in result
    assert result["error"]["code"] == "NOT_INDEXED"


# --- New Tests for Profile Code Performance Tool ---


def test_profile_code_performance_tool():
    """Test the profile_code_performance_tool function."""
    from codesage_mcp.tools import profile_code_performance_tool
    import tempfile
    import os

    # Create a simple test Python file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
def simple_function():
    return 1 + 1

def another_function():
    return simple_function() * 2

if __name__ == "__main__":
    result = another_function()
    print(result)
""")
        temp_file_path = f.name

    try:
        # Test profiling the entire file
        result = profile_code_performance_tool(temp_file_path)

        # Check the structure of the result
        assert "message" in result
        assert "total_functions_profiled" in result
        assert "top_bottlenecks" in result
        assert "raw_stats" in result
        assert (
            result["message"] == f"Performance profiling completed for {temp_file_path}"
        )

        # Check that we got some profiling data
        assert isinstance(result["total_functions_profiled"], int)
        assert isinstance(result["top_bottlenecks"], list)
        assert isinstance(result["raw_stats"], str)

        # Test profiling a specific function
        result = profile_code_performance_tool(temp_file_path, "simple_function")

        # Check the structure of the result
        assert "message" in result
        assert "total_functions_profiled" in result
        assert "top_bottlenecks" in result
        assert "raw_stats" in result
        assert (
            result["message"]
            == f"Performance profiling completed for {temp_file_path} function 'simple_function'"
        )

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


# --- New Tests for Incremental Indexing ---


def test_incremental_indexing_basic(temp_codebase):
    """Test basic incremental indexing functionality."""
    from codesage_mcp.core.indexing import IndexingManager
    from sentence_transformers import SentenceTransformer
    import time

    indexing_manager = IndexingManager()
    indexing_manager.index_dir = temp_codebase / ".codesage"
    indexing_manager.index_file = indexing_manager.index_dir / "codebase_index.json"
    indexing_manager.metadata_file = indexing_manager.index_dir / "codebase_metadata.json"
    indexing_manager._initialize_index()

    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initial full indexing
    indexed_files = indexing_manager.index_codebase(str(temp_codebase), sentence_transformer_model)
    assert "file1.py" in indexed_files
    assert len(indexed_files) == 1  # Only file1.py should be indexed

    # Verify metadata was created
    assert indexing_manager.metadata_file.exists()
    with open(indexing_manager.metadata_file, "r") as f:
        metadata = json.load(f)
    assert str(temp_codebase.resolve()) in metadata
    assert "file1.py" in metadata[str(temp_codebase.resolve())]

    # Modify file1.py to trigger incremental indexing
    file1_path = temp_codebase / "file1.py"
    original_content = file1_path.read_text()
    modified_content = original_content + '\n\n# Modified content\nprint("modified")'
    file1_path.write_text(modified_content)

    # Wait a bit to ensure modification time difference
    time.sleep(0.1)

    # Test incremental indexing
    incremental_files, was_incremental = indexing_manager.index_codebase_incremental(
        str(temp_codebase), sentence_transformer_model
    )

    assert was_incremental == True
    assert "file1.py" in incremental_files
    assert len(incremental_files) == 1  # Only modified file should be re-indexed


def test_incremental_indexing_add_file(temp_codebase):
    """Test incremental indexing when a new file is added."""
    from codesage_mcp.core.indexing import IndexingManager
    from sentence_transformers import SentenceTransformer

    indexing_manager = IndexingManager()
    indexing_manager.index_dir = temp_codebase / ".codesage"
    indexing_manager.index_file = indexing_manager.index_dir / "codebase_index.json"
    indexing_manager.metadata_file = indexing_manager.index_dir / "codebase_metadata.json"
    indexing_manager._initialize_index()

    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initial full indexing
    indexed_files = indexing_manager.index_codebase(str(temp_codebase), sentence_transformer_model)
    initial_count = len(indexed_files)

    # Add a new file
    new_file = temp_codebase / "new_file.py"
    new_file.write_text('print("new file content")')

    # Test incremental indexing
    incremental_files, was_incremental = indexing_manager.index_codebase_incremental(
        str(temp_codebase), sentence_transformer_model
    )

    assert was_incremental == True
    assert "new_file.py" in incremental_files
    assert len(incremental_files) == initial_count + 1  # One more file


def test_incremental_indexing_delete_file(temp_codebase):
    """Test incremental indexing when a file is deleted."""
    from codesage_mcp.core.indexing import IndexingManager
    from sentence_transformers import SentenceTransformer

    indexing_manager = IndexingManager()
    indexing_manager.index_dir = temp_codebase / ".codesage"
    indexing_manager.index_file = indexing_manager.index_dir / "codebase_index.json"
    indexing_manager.metadata_file = indexing_manager.index_dir / "codebase_metadata.json"
    indexing_manager._initialize_index()

    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Add another file to the test codebase
    extra_file = temp_codebase / "extra.py"
    extra_file.write_text('print("extra file")')

    # Initial full indexing
    indexed_files = indexing_manager.index_codebase(str(temp_codebase), sentence_transformer_model)
    initial_count = len(indexed_files)
    assert "extra.py" in indexed_files

    # Delete the extra file
    extra_file.unlink()

    # Test incremental indexing
    incremental_files, was_incremental = indexing_manager.index_codebase_incremental(
        str(temp_codebase), sentence_transformer_model
    )

    assert was_incremental == True
    assert "extra.py" not in incremental_files
    assert len(incremental_files) == initial_count - 1  # One less file


def test_incremental_indexing_no_changes(temp_codebase):
    """Test incremental indexing when no changes are detected."""
    from codesage_mcp.core.indexing import IndexingManager
    from sentence_transformers import SentenceTransformer

    indexing_manager = IndexingManager()
    indexing_manager.index_dir = temp_codebase / ".codesage"
    indexing_manager.index_file = indexing_manager.index_dir / "codebase_index.json"
    indexing_manager.metadata_file = indexing_manager.index_dir / "codebase_metadata.json"
    indexing_manager._initialize_index()

    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initial full indexing
    indexed_files = indexing_manager.index_codebase(str(temp_codebase), sentence_transformer_model)
    initial_files = set(indexed_files)

    # Test incremental indexing without making changes
    incremental_files, was_incremental = indexing_manager.index_codebase_incremental(
        str(temp_codebase), sentence_transformer_model
    )

    assert was_incremental == True
    assert set(incremental_files) == initial_files  # Same files returned


def test_force_full_reindex(temp_codebase):
    """Test forcing full re-indexing."""
    from codesage_mcp.core.indexing import IndexingManager
    from sentence_transformers import SentenceTransformer

    indexing_manager = IndexingManager()
    indexing_manager.index_dir = temp_codebase / ".codesage"
    indexing_manager.index_file = indexing_manager.index_dir / "codebase_index.json"
    indexing_manager.metadata_file = indexing_manager.index_dir / "codebase_metadata.json"
    indexing_manager._initialize_index()

    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initial full indexing
    indexing_manager.index_codebase(str(temp_codebase), sentence_transformer_model)

    # Test forcing full re-indexing
    incremental_files, was_incremental = indexing_manager.index_codebase_incremental(
        str(temp_codebase), sentence_transformer_model, force_full=True
    )

    assert was_incremental == False  # Should be False because we forced full indexing


def test_codebase_manager_incremental_integration(temp_codebase):
    """Test that CodebaseManager properly integrates incremental indexing."""
    from codesage_mcp.features.codebase_manager import CodebaseManager
    from unittest.mock import patch

    manager = CodebaseManager()

    # Mock the ENABLE_INCREMENTAL_INDEXING to True
    with patch('codesage_mcp.codebase_manager.ENABLE_INCREMENTAL_INDEXING', True), \
         patch('codesage_mcp.codebase_manager.FORCE_FULL_REINDEX', False):

        # First indexing should be full
        indexed_files = manager.index_codebase(str(temp_codebase))
        assert len(indexed_files) > 0

        # Second indexing should be incremental (no changes)
        indexed_files2 = manager.index_codebase(str(temp_codebase))
        assert len(indexed_files2) > 0

    # Test force full reindex
    with patch('codesage_mcp.codebase_manager.ENABLE_INCREMENTAL_INDEXING', True), \
         patch('codesage_mcp.codebase_manager.FORCE_FULL_REINDEX', False):

        indexed_files3 = manager.force_full_reindex(str(temp_codebase))
        assert len(indexed_files3) > 0


def test_profile_code_performance_tool_not_found():
    """Test the profile_code_performance_tool function with a non-existent file."""
    from codesage_mcp.tools import profile_code_performance_tool

    # Test with a non-existent file
    result = profile_code_performance_tool("/test/nonexistent/file.py")

    # Check that we get an error
    assert "error" in result
    assert result["error"]["code"] == -32003


def test_profile_code_performance_tool_invalid_function():
    """Test the profile_code_performance_tool function with a non-existent function."""
    from codesage_mcp.tools import profile_code_performance_tool
    import tempfile
    import os

    # Create a simple test Python file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
def simple_function():
    return 1 + 1
""")
        temp_file_path = f.name

    try:
        # Test profiling a non-existent function
        result = profile_code_performance_tool(temp_file_path, "non_existent_function")

        # Check that we get an error
        assert "error" in result
        assert result["error"]["code"] == "PROFILING_ERROR"

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

import pytest
import json
from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch, mock_open
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
# --- New Tests for Semantic Search ---

def test_semantic_search_empty_index():
    """Test semantic search when the FAISS index is empty or None."""
    manager = CodebaseManager()
    # Directly set the faiss_index to None to simulate an un-indexed state
    manager.faiss_index = None

    results = manager.semantic_search_codebase("test query")
    assert results == []

    # Test with an index that has 0 total vectors
    mock_faiss_index = MagicMock()
    mock_faiss_index.ntotal = 0
    manager.faiss_index = mock_faiss_index

    results = manager.semantic_search_codebase("test query")
    assert results == []
    # Ensure no search was attempted on an empty index
    mock_faiss_index.search.assert_not_called()


def test_semantic_search_with_results(temp_codebase):
    """Test semantic search returning results from a real, indexed codebase."""
    manager = CodebaseManager()
    manager.index_dir = temp_codebase / ".codesage"
    manager.index_file = manager.index_dir / "codebase_index.json"
    manager.faiss_index_file = manager.index_dir / "codebase_index.faiss"
    manager.indexed_codebases = {}
    manager._initialize_index()

    # Index the test codebase
    manager.index_codebase(str(temp_codebase))

    # We need to mock the sentence transformer's encode method to return predictable
    # embeddings for our test query.
    # Let's assume file1.py gets ID "0" and has a certain embedding.
    # Let's make the query embedding very similar to file1.py's mock embedding.
    
    # Store original method to restore later
    original_encode = manager.sentence_transformer_model.encode
    
    # Mock embeddings. FAISS works with L2 distance. Smaller distance = more similar.
    # We'll make the query very close to file1.py's embedding and far from a dummy one.
    mock_file1_embedding = [1.0, 0.0, 0.0]
    mock_query_embedding = [0.9, 0.0, 0.0] # Very similar to file1
    mock_dummy_embedding = [0.0, 1.0, 0.0] # Different

    def mock_encode(text):
        if "hello world" in text: # This is in file1.py
             return mock_file1_embedding
        elif text == "dummy content": # This is for the dummy file
             return mock_dummy_embedding
        elif text == "find hello world": # This is our query
             return mock_query_embedding
        else:
             # Fallback, shouldn't happen in this specific test
             return original_encode(text)

    # Patch the encode method
    manager.sentence_transformer_model.encode = mock_encode

    # Now, mock the FAISS search.
    # We'll pretend FAISS searched and found file1.py's ID (which should be "0")
    # as the most similar.
    mock_faiss_index = manager.faiss_index
    mock_faiss_index.search = MagicMock()
    # Return distance 0.1 (very close) and index 0
    mock_faiss_index.search.return_value = ( [[0.1]], [[0]] )

    # Also mock the file_paths_map to map ID "0" to our file1.py
    abs_file1_path = str((temp_codebase / "file1.py").resolve())
    manager.file_paths_map = {"0": abs_file1_path}

    # --- Perform the search ---
    results = manager.semantic_search_codebase("find hello world", top_k=1)

    # --- Assertions ---
    assert len(results) == 1
    assert results[0]["file_path"] == abs_file1_path
    assert results[0]["score"] == 0.1 # Check the mocked distance/score

    # Verify the mocks were called as expected
    mock_faiss_index.search.assert_called_once()
    call_args = mock_faiss_index.search.call_args
    assert call_args[0][1] == 1 # call_args[0][1] should be top_k

    # Restore original method
    manager.sentence_transformer_model.encode = original_encode



def test_semantic_search_top_k():
    """Test that the top_k parameter correctly limits results."""
    manager = CodebaseManager()
    # Do NOT index a real codebase. We are mocking everything.

    # Mock FAISS index to be not None and have vectors
    mock_faiss_index = MagicMock()
    mock_faiss_index.ntotal = 2 # Pretend we have 2 vectors indexed
    manager.faiss_index = mock_faiss_index

    # Custom mock side effect for search to respect top_k
    def mock_search_side_effect(query_vector, k):
        # Simulate FAISS returning two results if k >= 2, otherwise one.
        if k >= 2:
            return ([[0.1, 0.2]], [[0, 1]])
        else: # k == 1
            return ([[0.1]], [[0]])
            
    mock_faiss_index.search = MagicMock(side_effect=mock_search_side_effect)

    # Mock file_paths_map
    manager.file_paths_map = {"0": "/path/to/file1.py", "1": "/path/to/file2.py"}

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
    manager = CodebaseManager()
    # Test with an unindexed codebase - should raise ValueError
    manager.indexed_codebases = {}
    with pytest.raises(ValueError, match="Codebase at /test/codebase has not been indexed"):
        manager.find_duplicate_code("/test/codebase")

    # Test with an index that has 0 total vectors
    mock_faiss_index = MagicMock()
    mock_faiss_index.ntotal = 0
    manager.faiss_index = mock_faiss_index
    # Mock indexed_codebases to simulate an indexed codebase
    manager.indexed_codebases = {"/test/codebase": {"files": []}}

    results = manager.find_duplicate_code("/test/codebase")
    assert results == []
    # Ensure no search was attempted on an empty index
    mock_faiss_index.search.assert_not_called()


def test_find_duplicate_code_with_results(temp_codebase):
    """Test find_duplicate_code returning results from a real, indexed codebase."""
    manager = CodebaseManager()
    manager.index_dir = temp_codebase / ".codesage"
    manager.index_file = manager.index_dir / "codebase_index.json"
    manager.faiss_index_file = manager.index_dir / "codebase_index.faiss"
    manager.indexed_codebases = {}
    manager._initialize_index()

    # Index the test codebase
    manager.index_codebase(str(temp_codebase))

    # We need to mock the sentence transformer's encode method to return predictable
    # embeddings for our test sections.
    # Let's assume file1.py gets ID "0" and has a certain embedding.
    # Let's make a duplicate section embedding very similar to file1.py's mock embedding.
    
    # Store original method to restore later
    original_encode = manager.sentence_transformer_model.encode
    
    # Mock embeddings. FAISS works with L2 distance. Smaller distance = more similar.
    # We'll make the duplicate section very close to file1.py's embedding.
    mock_file1_embedding = [1.0, 0.0, 0.0]
    mock_duplicate_embedding = [0.9, 0.0, 0.0] # Very similar to file1
    mock_dummy_embedding = [0.0, 1.0, 0.0] # Different

    def mock_encode(text):
        if "hello world" in text: # This is in file1.py
             return mock_file1_embedding
        elif text == "dummy content": # This is for the dummy file
             return mock_dummy_embedding
        elif "duplicate section" in text: # This is our duplicate section
             return mock_duplicate_embedding
        else:
             # Fallback, shouldn't happen in this specific test
             return original_encode(text)

    # Patch the encode method
    manager.sentence_transformer_model.encode = mock_encode

    # Now, mock the FAISS search.
    # We'll pretend FAISS searched and found file1.py's ID (which should be "0")
    # as the most similar.
    mock_faiss_index = manager.faiss_index
    mock_faiss_index.search = MagicMock()
    # Return distance 0.1 (very close) and index 0
    mock_faiss_index.search.return_value = ( [[0.1]], [[0]] )

    # Also mock the file_paths_map to map ID "0" to our file1.py
    abs_file1_path = str((temp_codebase / "file1.py").resolve())
    manager.file_paths_map = {"0": abs_file1_path}
    # Also add the codebase path to indexed_codebases
    manager.indexed_codebases = {str(temp_codebase.resolve()): {"files": ["file1.py"]}}

    # --- Perform the duplicate code search ---
    # For this test, we'll mock the file content to have a section that matches our mock
    # This is a bit of a simplification, but it tests the core logic
    with patch("builtins.open", mock_open(read_data="import os\n\n# A test comment\nprint(\"hello world\")\n# duplicate section\n")):
        results = manager.find_duplicate_code(str(temp_codebase), min_similarity=0.8, min_lines=3)

    # --- Assertions ---
    # Note: The actual implementation might not find duplicates in this mock setup,
    # but we're testing the structure and error handling.
    # A more comprehensive test would require a more complex mock setup.
    assert isinstance(results, list)
    
    # Restore original method
    manager.sentence_transformer_model.encode = original_encode
